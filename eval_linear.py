from __future__ import annotations

import argparse
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from amcl.datasets import make_datasets
from amcl.models import BarlowTwins, MoCo, SimCLR, SimSiam


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
    beta = float(saved.get("resolved_beta", saved.get("beta", 0.01)))
    tau_min = float(saved.get("tau_min", 1e-5))
    tau_max = float(saved.get("tau_max", 2.0))
    tau = float(saved.get("resolved_tau", saved.get("tau", 0.5 if method == "simclr" else 0.1)))
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
    args = p.parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    saved = ckpt.get("args", {})
    model = build_from_checkpoint(saved, args.method)
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device).eval()
    encoder = model.encoder if hasattr(model, "encoder") else model.encoder_q
    for param in encoder.parameters():
        param.requires_grad = False

    size = 96 if args.dataset == "stl10" else 32
    _, train_ds, test_ds = make_datasets(args.dataset, args.data_root, "default", size)
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, args.batch_size, shuffle=False, num_workers=args.num_workers)
    classes = 100 if args.dataset == "cifar100" else 10
    feat_dim = model.encoder_dim
    classifier = nn.Linear(feat_dim, classes).to(device)
    opt = torch.optim.SGD(
        classifier.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        classifier.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                h = encoder(x)
            loss = criterion(classifier(h), y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite linear-evaluation loss at epoch {epoch}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sched.step()

    classifier.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in test_loader:
            pred = classifier(encoder(x.to(device))).argmax(1).cpu()
            correct += (pred == y).sum().item()
            total += y.numel()
    print(f"linear-probe top-1: {100.0 * correct / total:.2f}%")


if __name__ == "__main__":
    main()
