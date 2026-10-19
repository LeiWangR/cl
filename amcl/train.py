from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .datasets import create_split, get_base_dataset, make_datasets, split_digest
from .evaluation import knn_accuracy
from .models import BarlowTwins, MoCo, SimCLR, SimSiam


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train the AMCL ECCV 2024 small-dataset reference implementation."
    )
    p.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100", "stl10"])
    p.add_argument("--data-root", default="./data")
    p.add_argument("--method", default="simclr", choices=["simclr", "moco", "simsiam", "barlow_twins"])
    p.add_argument("--amcl", action="store_true", help="Enable AMCL; omit for the corresponding baseline.")
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--heads", type=int, default=3)
    p.add_argument("--proj-dim", type=int, default=None)
    p.add_argument("--beta", type=float, default=None)
    p.add_argument("--tau", type=float, default=None, help="Override the baseline fixed temperature (SimCLR default 0.5; MoCo default 0.1).")
    p.add_argument("--tau-min", type=float, default=1e-5)
    p.add_argument("--tau-max", type=float, default=2.0)
    p.add_argument("--topk", type=int, default=100)
    p.add_argument("--temperature-hidden-dim", type=int, default=128, help="Hidden width of the shared temperature MLP; phi still maps d -> d.")
    p.add_argument("--queue-size", type=int, default=4096)
    p.add_argument("--momentum", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=None, help="Override the method-specific reference LR.")
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--aug", default="default", help="default, color, or a combination of a,b,c,d,e")
    p.add_argument("--stl10-unlabeled", action="store_true")
    p.add_argument("--output", default="./runs/amcl")
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--eval-k", type=int, default=200)
    p.add_argument("--disable-knn", action="store_true")
    p.add_argument("--max-steps", type=int, default=None, help="Stop after this many optimizer steps; useful for smoke tests.")
    p.add_argument("--no-fail-on-nonfinite", dest="fail_on_nonfinite", action="store_false", help="Do not abort on NaN/Inf (not recommended).")
    p.set_defaults(fail_on_nonfinite=True)
    p.add_argument("--grad-clip", type=float, default=1.0, help="Global gradient-norm clip. Set 0 to disable; this is a numerical safeguard, not part of the AMCL loss.")
    p.add_argument("--validation-size", type=int, choices=[0, 5000], default=0)
    p.add_argument("--split-file", help="Reuse a saved CIFAR train/validation split")
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--checkpoint-selection", choices=["last", "best"], default="last",
                   help="best selects validation kNN; requires a validation split")
    p.add_argument("--amp-initial-scale", type=float, default=65536.)
    p.add_argument("--temperature-lr-scale", type=float, default=1.)
    p.add_argument("--loader-seed", type=int, help="Independent training shuffle RNG (used by RunPod runs)")
    return p.parse_args()


def _reference_defaults(args: argparse.Namespace) -> tuple[int, float, float, float]:
    if args.method in {"simclr", "moco"}:
        proj_dim = 128
        beta = 1e-4
        lr = 0.06 if args.method == "simclr" else 0.08
        tau_default = 0.5 if args.method == "simclr" else 0.1
    elif args.method == "simsiam":
        proj_dim = 2048
        beta = 1e-4
        lr = 0.05
        tau_default = 0.0
    else:
        proj_dim = 2048
        beta = 1e-6
        lr = 0.11
        tau_default = 0.0
    return (
        proj_dim if args.proj_dim is None else args.proj_dim,
        beta if args.beta is None else args.beta,
        lr if args.lr is None else args.lr,
        tau_default if args.tau is None else args.tau,
    )


def _build_model(args: argparse.Namespace, proj_dim: int, beta: float, resolved_tau: float):
    heads = args.heads if args.amcl else 1
    common = dict(
        num_heads=heads,
        amcl=args.amcl,
        beta=beta,
        tau_min=args.tau_min,
        tau_max=args.tau_max,
    )
    if args.method == "simclr":
        return SimCLR(
            proj_dim=proj_dim,
            tau=resolved_tau,
            topk=args.topk,
            temperature_hidden_dim=args.temperature_hidden_dim,
            **common,
        )
    if args.method == "moco":
        return MoCo(
            proj_dim=proj_dim,
            tau=resolved_tau,
            topk=args.topk,
            queue_size=args.queue_size,
            momentum=args.momentum,
            temperature_hidden_dim=args.temperature_hidden_dim,
            **common,
        )
    if args.method == "simsiam":
        if proj_dim != 2048:
            raise ValueError("The paper-aligned SimSiam configuration uses proj_dim=2048")
        return SimSiam(proj_dim=proj_dim, temperature_hidden_dim=args.temperature_hidden_dim, **common)
    if proj_dim != 2048:
        raise ValueError("The paper-aligned Barlow Twins configuration uses proj_dim=2048")
    return BarlowTwins(
        proj_dim=proj_dim,
        batch_size=args.batch_size,
        temperature_hidden_dim=args.temperature_hidden_dim,
        **common,
    )


def _encoder(model):
    return model.encoder if hasattr(model, "encoder") else model.encoder_q


def _assert_finite(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        raise FloatingPointError(f"Non-finite {name} detected. Check loss/temperature settings and batch size.")


def optimizer_step(model, optimizer, scaler, grad_clip):
    """Skip overflowing AMP updates before clipping; retain GradScaler backoff."""
    scaler.unscale_(optimizer)
    bad = next((n for n, p in model.named_parameters()
                if p.grad is not None and not torch.isfinite(p.grad).all()), None)
    before = scaler.get_scale()
    if bad is not None and not scaler.is_enabled():
        raise FloatingPointError(f'Non-finite FP32 gradient for {bad}')
    if bad is None and grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip, error_if_nonfinite=True)
    scaler.step(optimizer)
    scaler.update()
    if bad is not None:
        optimizer.zero_grad(set_to_none=True)
    return bad is None, bad, before, scaler.get_scale()


def save_checkpoint(checkpoint, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def train(args: argparse.Namespace) -> Path:
    # Keep callers of train(Namespace(...)) compatible with the existing API.
    for name, default in dict(validation_size=0, split_file=None, split_seed=0,
                              checkpoint_selection='last', amp_initial_scale=65536.,
                              temperature_lr_scale=1., loader_seed=None).items():
        if not hasattr(args, name):
            setattr(args, name, default)
    seed_everything(args.seed)
    if args.batch_size < 2:
        raise ValueError("batch-size must be at least 2")
    if args.amcl and args.heads < 1:
        raise ValueError("heads must be >= 1")
    if args.eval_every < 1:
        raise ValueError("eval-every must be >= 1")
    if not np.isfinite(args.amp_initial_scale) or args.amp_initial_scale <= 0:
        raise ValueError('AMP initial scale must be finite and positive')
    if not np.isfinite(args.temperature_lr_scale) or args.temperature_lr_scale <= 0:
        raise ValueError('Temperature LR scale must be finite and positive')
    if not args.amcl and args.temperature_lr_scale != 1:
        raise ValueError('Temperature LR scale applies to AMCL only')
    if args.checkpoint_selection == 'best' and (args.disable_knn or not (args.validation_size or args.split_file)):
        raise ValueError('Best checkpoint selection requires validation data and kNN evaluation')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    proj_dim, beta, lr, resolved_tau = _reference_defaults(args)
    size = 96 if args.dataset == "stl10" else 32
    out = Path(args.output)
    if any((out / name).exists() for name in ('last.pt', 'best.pt')):
        raise FileExistsError('Output contains a checkpoint; use a new output directory')
    split = None
    if args.split_file:
        split = json.loads(Path(args.split_file).read_text())
    elif args.validation_size:
        split = create_split(args.dataset, get_base_dataset(args.dataset, args.data_root, 'train'), args.split_seed)
    ssl_ds, train_eval, test_eval = make_datasets(
        args.dataset,
        args.data_root,
        args.aug,
        size,
        args.stl10_unlabeled,
        **({'split': split} if split is not None else {}),
    )
    loader = DataLoader(
        ssl_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        generator=None if args.loader_seed is None else torch.Generator().manual_seed(args.loader_seed),
    )
    if len(loader) == 0:
        raise RuntimeError("The SSL loader has zero complete batches. Reduce batch-size or check the dataset.")

    train_eval_loader = DataLoader(
        train_eval,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )
    test_loader = DataLoader(
        test_eval,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )
    classes = 100 if args.dataset == "cifar100" else 10

    model = _build_model(args, proj_dim, beta, resolved_tau).to(device)
    parameters = model.parameters()
    if args.temperature_lr_scale != 1:
        named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        parameters = [
            {'params': [p for n, p in named if not n.startswith('criterion.temperature.')]},
            {'params': [p for n, p in named if n.startswith('criterion.temperature.')],
             'lr': lr * args.temperature_lr_scale},
        ]
    optimizer = torch.optim.SGD(
        parameters,
        lr=lr,
        momentum=0.9,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    use_amp = device.type == "cuda"
    if hasattr(torch.amp, 'GradScaler'):
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp, init_scale=args.amp_initial_scale)
    else:  # PyTorch 2.1/2.2 compatibility.
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp, init_scale=args.amp_initial_scale)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    history = []
    global_step = 0
    skipped_total = 0
    best_score = -float('inf')
    selection = None

    config = dict(vars(args))
    config.update({"resolved_proj_dim": proj_dim, "resolved_beta": beta, "resolved_lr": lr, "resolved_tau": resolved_tau, "resolved_temperature_hidden_dim": args.temperature_hidden_dim})
    config.update(amp_overflow_policy='skip_update_and_backoff', consecutive_overflow_limit=None,
                  evaluation_scope='validation' if split is not None else 'test')
    if split is not None:
        config['split_sha256'] = split_digest(split)
        config['validation_size'] = len(split['validation'])
        config['split_seed'] = split['seed']
        (out / 'split.json').write_text(json.dumps(split) + '\n')
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        steps = 0
        skipped_epoch = 0
        for views, _labels in loader:
            if args.max_steps is not None and global_step >= args.max_steps:
                break
            x1 = views[0].to(device, non_blocking=True)
            x2 = views[1].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                if args.method == "simclr":
                    *_, z1, z2 = model(x1, x2)
                    loss = model.loss(z1, z2)
                elif args.method == "moco":
                    q, k = model(x1, x2)
                    loss = model.loss(q, k)
                elif args.method == "simsiam":
                    *_, p1, z1, p2, z2 = model(x1, x2)
                    loss = model.loss(p1, z1, p2, z2)
                else:
                    *_, z1, z2 = model(x1, x2)
                    loss = model.loss(z1, z2)

            if args.fail_on_nonfinite:
                _assert_finite("loss", loss.detach())
            scaler.scale(loss).backward()
            updated, bad, old_scale, new_scale = optimizer_step(model, optimizer, scaler, args.grad_clip)
            if not updated:
                skipped_epoch += 1
                skipped_total += 1
                print(f'AMP overflow | epoch {epoch:04d} | skipped update | {bad} | scale {old_scale:g} -> {new_scale:g}', flush=True)
                continue
            if args.method == "moco":
                model.update_queue(k)
            if args.fail_on_nonfinite:
                for name, param in model.named_parameters():
                    _assert_finite(f"parameter {name}", param.detach())

            running += float(loss.detach().cpu())
            steps += 1
            global_step += 1

        if steps == 0:
            if skipped_epoch:
                raise FloatingPointError('No successful optimizer updates in this epoch')
            break
        scheduler.step()
        rec = {
            "epoch": epoch,
            "loss": running / steps,
            "lr": scheduler.get_last_lr()[0],
            "steps": steps,
            "amp_skipped_steps": skipped_epoch,
            "amp_scale": scaler.get_scale(),
        }
        improved = False
        if (epoch % args.eval_every == 0 or epoch == args.epochs) and not args.disable_knn:
            rec["knn_accuracy"] = knn_accuracy(
                _encoder(model),
                train_eval_loader,
                test_loader,
                classes,
                device,
                args.eval_k,
            )
            if args.checkpoint_selection == 'best':
                score = rec['knn_accuracy']
                if not np.isfinite(score) or not 0 <= score <= 1:
                    raise FloatingPointError('Invalid validation kNN accuracy')
                if score > best_score:
                    best_score, improved = score, True
                    selection = dict(scope='validation', metric='validation_knn_accuracy',
                                     accuracy=score, epoch=epoch, eval_every=args.eval_every,
                                     split_sha256=config['split_sha256'], tie_break='earliest_epoch')
            print(
                f"epoch {epoch:04d} | loss {rec['loss']:.6f} | "
                f"kNN {100 * rec['knn_accuracy']:.2f}%"
            )
        else:
            print(f"epoch {epoch:04d} | loss {rec['loss']:.6f}")
        history.append(rec)
        checkpoint = {
                "model": model.state_dict(),
                "args": config,
                "epoch": epoch,
                "global_step": global_step,
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(), "amp_skipped_steps": skipped_total,
                "rng_torch": torch.get_rng_state(),
                "rng_cuda": torch.cuda.get_rng_state_all() if use_amp else [],
                "rng_python": random.getstate(), "rng_numpy": np.random.get_state(),
            }
        save_checkpoint(checkpoint, out / 'last.pt')
        if improved:
            checkpoint['selection'] = selection
            save_checkpoint(checkpoint, out / 'best.pt')
            (out / 'checkpoint_selection.json').write_text(json.dumps(selection, indent=2) + '\n')
        (out / 'history.json').write_text(json.dumps(history, indent=2) + '\n')
        if args.max_steps is not None and global_step >= args.max_steps:
            break

    (out / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    return out / "last.pt"


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
