from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from amcl.datasets import SSLDataset, get_base_dataset, split_digest, validate_split
from amcl.augmentations import IMAGENET_MEAN, IMAGENET_STD
from amcl.backbones import resnet_cifar
from amcl.models import BarlowTwins, MoCo, SimCLR, SimSiam
from amcl.train import save_checkpoint
from torchvision import transforms


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_from_checkpoint(saved: dict, method: str):
    amcl = bool(saved.get("amcl", False))
    heads = int(saved.get("heads", 1)) if amcl else 1
    proj_dim = int(saved.get("resolved_proj_dim", saved.get("proj_dim") or (2048 if method in {"simsiam", "barlow_twins"} else 128)))
    beta = saved.get("resolved_beta", saved.get("beta"))
    beta = float(beta if beta is not None else (1e-6 if method == 'barlow_twins' else 1e-4))
    tau_min = float(saved.get("tau_min", 1e-5))
    tau_max = float(saved.get("tau_max", 2.0))
    tau = saved.get("resolved_tau", saved.get("tau"))
    tau = float(tau if tau is not None else (0.5 if method == 'simclr' else 0.1))
    temp_hidden = int(saved.get("resolved_temperature_hidden_dim", saved.get("temperature_hidden_dim", 128)))
    depth = int(saved.get("depth", 18))
    common = dict(
        depth=depth,
        num_heads=heads,
        amcl=amcl,
        beta=beta,
        tau_min=tau_min,
        tau_max=tau_max,
    )
    if method == "simclr":
        return SimCLR(proj_dim=proj_dim, tau=tau, topk=int(saved.get("topk", 100)), temperature_hidden_dim=temp_hidden, **common)
    if method == "moco":
        return MoCo(
            proj_dim=proj_dim,
            tau=tau,
            topk=int(saved.get("topk", 100)),
            queue_size=int(saved.get("queue_size", 4096)),
            momentum=float(saved.get("momentum", 0.99)),
            temperature_hidden_dim=temp_hidden,
            **common,
        )
    if method == "simsiam":
        return SimSiam(proj_dim=2048, temperature_hidden_dim=temp_hidden, **common)
    return BarlowTwins(
        proj_dim=2048,
        batch_size=int(saved.get("batch_size", 512)),
        temperature_hidden_dim=temp_hidden,
        **common,
    )


def load_encoder(checkpoint, method):
    """Accept package checkpoints and the encoder bundles from recorded RunPod runs."""
    saved = dict(checkpoint.get('args') or checkpoint.get('config') or {})
    metadata = checkpoint.get('metadata', {})
    recorded_method = saved.get('method', metadata.get('method'))
    if recorded_method is not None and recorded_method != method:
        raise ValueError('Checkpoint method differs from --method')
    if 'eta' in saved:
        saved.update(tau_min=saved['eta'], tau_max=saved['eta'] + saved['iota'])
    if 'config' in checkpoint:
        saved.setdefault('heads', 3 if saved.get('amcl') else 1)
    if 'encoder' in checkpoint:
        encoder, dim = resnet_cifar(int(saved.get('depth', 18)))
        encoder.load_state_dict(checkpoint['encoder'], strict=True)
    else:
        model = build_from_checkpoint(saved, method)
        model.load_state_dict(checkpoint['model'], strict=True)
        encoder = model.encoder_q if method == 'moco' else model.encoder
        dim = model.encoder_dim
    return encoder, dim, saved


@torch.no_grad()
def accuracy(encoder, classifier, loader, device):
    encoder.eval(); classifier.eval()
    correct = total = 0
    for images, labels in loader:
        predictions = classifier(encoder(images.to(device))).argmax(1).cpu()
        correct += int((predictions == labels).sum())
        total += len(labels)
    return dict(correct=correct, examples=total, accuracy=correct / total)


@torch.no_grad()
def cache_features(encoder, loader, device):
    features, labels = [], []
    for images, targets in loader:
        features.append(encoder(images.to(device)).float())
        labels.append(targets.to(device))
    return torch.cat(features), torch.cat(labels)


def fit_probe(encoder, classifier, train_loader, val_loader, device, args, bundle, output):
    """Fit only the classifier; select on validation or retain the fixed final epoch."""
    encoder.eval().requires_grad_(False)
    cached_train = cache_features(encoder, train_loader, device) if args.cache_features else None
    cached_val = cache_features(encoder, val_loader, device) if args.cache_features and val_loader else None
    if args.cache_features:
        # Recorded MoCo probes reset the seed after feature extraction.
        seed_everything(args.seed)
        classifier.reset_parameters()
    optimizer = torch.optim.SGD(classifier.parameters(), lr=args.lr, momentum=.9,
                                weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    generator = torch.Generator().manual_seed(args.seed)
    history, best_correct = [], -1
    for epoch in range(1, args.epochs + 1):
        classifier.train()
        if cached_train:
            x, y = cached_train
            batches = ((x[ids.to(device)], y[ids.to(device)]) for ids in
                       torch.randperm(len(x), generator=generator).split(args.batch_size))
        else:
            batches = train_loader
        for inputs, labels in batches:
            with torch.no_grad():
                features = inputs if cached_train else encoder(inputs.to(device))
            loss = nn.functional.cross_entropy(classifier(features), labels.to(device))
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite probe loss at epoch {epoch}')
            optimizer.zero_grad(set_to_none=True)
            loss.backward(); optimizer.step()
        scheduler.step()
        row = dict(epoch=epoch)
        if val_loader is not None and (args.selection == 'best' or epoch == args.epochs):
            if cached_val:
                classifier.eval()
                with torch.no_grad():
                    correct = int((classifier(cached_val[0]).argmax(1) == cached_val[1]).sum())
                row.update(correct=correct, examples=len(cached_val[1]), accuracy=correct / len(cached_val[1]))
            else:
                row.update(accuracy(encoder, classifier, val_loader, device))
        history.append(row)
        selected = (args.selection == 'last' and epoch == args.epochs) or (
            args.selection == 'best' and row['correct'] > best_correct)
        if selected:
            best_correct = row.get('correct', -1)
            bundle['classifier'] = {k: v.detach().cpu().clone() for k, v in classifier.state_dict().items()}
            bundle['metadata']['linear_selection'] = dict(row,
                scope='validation' if args.selection == 'best' else 'fixed_final_epoch',
                tie_break='earliest_epoch' if args.selection == 'best' else None)
            save_checkpoint(bundle, output / 'linear_probe.pt')
        (output / 'history.json').write_text(json.dumps(history, indent=2) + '\n')
        score = f" | val {100 * row['accuracy']:.2f}%" if 'accuracy' in row else ''
        print(f'Linear epoch {epoch:03d}/{args.epochs}{score}', flush=True)
    classifier.load_state_dict(bundle['classifier'], strict=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Frozen-encoder linear evaluation for an AMCL checkpoint")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset", required=True, choices=["cifar10", "cifar100", "stl10"])
    p.add_argument("--method", required=True, choices=["simclr", "moco", "simsiam", "barlow_twins"])
    p.add_argument("--data-root", default="./data")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument('--split-file', help='Saved CIFAR split; defaults to split.json beside the encoder')
    p.add_argument('--selection', choices=['last', 'best'], default='last')
    p.add_argument('--eval-split', choices=['validation', 'test'], default='test')
    p.add_argument('--cache-features', action='store_true', help='Fit on cached FP32 encoder features (MoCo protocol)')
    p.add_argument('--test-only', action='store_true', help='Load a saved classifier and run official-test inference only')
    p.add_argument('--output', help='Evaluation output directory')
    args = p.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0 or args.weight_decay < 0 or args.num_workers < 0:
        p.error('Invalid linear training parameters')
    if args.test_only and args.eval_split != 'test':
        p.error('--test-only requires --eval-split test')
    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    path = Path(args.checkpoint)
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    encoder, dim, saved = load_encoder(ckpt, args.method)
    metadata = ckpt.get('metadata', {})
    recorded_dataset = saved.get('dataset', metadata.get('dataset'))
    if recorded_dataset is not None and recorded_dataset != args.dataset:
        raise ValueError('Checkpoint dataset differs from --dataset')
    encoder.to(device).eval().requires_grad_(False)
    size = 96 if args.dataset == 'stl10' else 32
    tf = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(),
                             transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    output = Path(args.output) if args.output else (path.parent if args.test_only else path.parent / 'linear_evaluation')
    classes = 100 if args.dataset == 'cifar100' else 10
    if args.test_only:
        if 'classifier' not in ckpt:
            raise ValueError('--test-only requires a saved encoder/classifier bundle')
        classifier = nn.Linear(dim, classes).to(device)
        classifier.load_state_dict(ckpt['classifier'], strict=True)
    else:
        if (output / 'linear_probe.pt').exists():
            raise FileExistsError('Output already contains a classifier; choose another --output')
        split_path = Path(args.split_file) if args.split_file else path.parent / 'split.json'
        split = json.loads(split_path.read_text()) if split_path.exists() else None
        if args.split_file and split is None:
            raise FileNotFoundError(split_path)
        expected_hash = saved.get('split_sha256') or metadata.get('split_sha256') or (ckpt.get('selection') or {}).get('split_sha256')
        if expected_hash and (split is None or split_digest(split) != expected_hash):
            raise ValueError('A matching saved training split is required')
        if split is None and (args.selection == 'best' or args.eval_split == 'validation' or metadata.get('train_examples') == 45000 or 'config' in ckpt):
            raise ValueError('Provide --split-file for validation or historical RunPod checkpoints')
        base = get_base_dataset(args.dataset, args.data_root, 'train')
        val_ds = None
        if split is not None:
            validate_split(split, base, args.dataset)
            train_ds = SSLDataset(torch.utils.data.Subset(base, split['train']), tf)
            val_ds = SSLDataset(torch.utils.data.Subset(base, split['validation']), tf)
        else:
            base.transform = tf
            train_ds = base
        kwargs = dict(batch_size=args.batch_size, num_workers=args.num_workers)
        separate_rng = args.selection == 'best' and not args.cache_features
        train_loader = DataLoader(train_ds, shuffle=not args.cache_features,
            persistent_workers=separate_rng and args.num_workers > 0,
            generator=torch.Generator().manual_seed(args.seed) if separate_rng else None, **kwargs)
        val_loader = DataLoader(val_ds, shuffle=False,
            persistent_workers=separate_rng and args.num_workers > 0,
            generator=torch.Generator().manual_seed(args.seed + 1) if separate_rng else None, **kwargs) if val_ds is not None else None
        classifier = nn.Linear(dim, classes).to(device)
        output.mkdir(parents=True, exist_ok=True)
        metadata = dict(dataset=args.dataset, method=args.method, source_checkpoint=str(path),
            encoder_epoch=ckpt.get('epoch'), encoder_selection=ckpt.get('selection'),
            train_examples=len(train_ds), validation_examples=len(val_ds) if val_ds else 0,
            split_sha256=split_digest(split) if split else None,
            probe_epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay, batch_size=args.batch_size,
            seed=args.seed, cache_features=args.cache_features, momentum=.9, scheduler='cosine')
        bundle = dict(encoder={k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()},
                      args=saved, metadata=metadata)
        fit_probe(encoder, classifier, train_loader, val_loader, device, args, bundle, output)
        if split:
            (output / 'split.json').write_text(json.dumps(split) + '\n')
    # Official test images are loaded only after classifier training and selection.
    if args.eval_split == 'test':
        dataset = get_base_dataset(args.dataset, args.data_root, 'test')
        dataset.transform = tf
        if args.dataset.startswith('cifar') and len(dataset) != 10000:
            raise ValueError('Expected 10000 official test images')
        loader = DataLoader(dataset, args.batch_size, shuffle=False, num_workers=args.num_workers)
        result = accuracy(encoder, classifier, loader, device)
    else:
        result = accuracy(encoder, classifier, val_loader, device)
    result.update(scope='official_test' if args.eval_split == 'test' else 'validation', metadata=metadata)
    output.mkdir(parents=True, exist_ok=True)
    (output / (args.eval_split + '_result.json')).write_text(json.dumps(result, indent=2) + '\n')
    print(f"linear-probe top-1: {100 * result['accuracy']:.2f}%")


if __name__ == '__main__':
    main()
