from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .datasets import make_datasets
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
        return SimSiam(proj_dim=proj_dim, **common)
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


def train(args: argparse.Namespace) -> Path:
    seed_everything(args.seed)
    if args.batch_size < 2:
        raise ValueError("batch-size must be at least 2")
    if args.amcl and args.heads < 1:
        raise ValueError("heads must be >= 1")
    if args.eval_every < 1:
        raise ValueError("eval-every must be >= 1")
    if args.method == "barlow_twins" and args.amcl and args.batch_size < 2:
        raise ValueError("AMCL Barlow Twins requires a fixed batch size >= 2")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    proj_dim, beta, lr, resolved_tau = _reference_defaults(args)
    size = 96 if args.dataset == "stl10" else 32
    ssl_ds, train_eval, test_eval = make_datasets(
        args.dataset,
        args.data_root,
        args.aug,
        size,
        args.stl10_unlabeled,
    )
    loader = DataLoader(
        ssl_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
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
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # AMCL losses force their numerically sensitive terms to float32, so AMP is
    # safe for the surrounding model on CUDA.
    use_amp = device.type == "cuda"
    # scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    history = []
    global_step = 0

    config = dict(vars(args))
    config.update({"resolved_proj_dim": proj_dim, "resolved_beta": beta, "resolved_lr": lr, "resolved_tau": resolved_tau, "resolved_temperature_hidden_dim": args.temperature_hidden_dim})
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        steps = 0
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
            scaler.unscale_(optimizer)
            if args.fail_on_nonfinite:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        _assert_finite(f"gradient for {name}", param.grad.detach())
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            if args.method == "moco":
                model.update_queue(k)
            if args.fail_on_nonfinite:
                for name, param in model.named_parameters():
                    _assert_finite(f"parameter {name}", param.detach())

            running += float(loss.detach().cpu())
            steps += 1
            global_step += 1

        scheduler.step()
        if steps == 0:
            break
        rec = {
            "epoch": epoch,
            "loss": running / steps,
            "lr": scheduler.get_last_lr()[0],
            "steps": steps,
        }
        if (epoch % args.eval_every == 0 or epoch == args.epochs) and not args.disable_knn:
            rec["knn_accuracy"] = knn_accuracy(
                _encoder(model),
                train_eval_loader,
                test_loader,
                classes,
                device,
                args.eval_k,
            )
            print(
                f"epoch {epoch:04d} | loss {rec['loss']:.6f} | "
                f"kNN {100 * rec['knn_accuracy']:.2f}%"
            )
        else:
            print(f"epoch {epoch:04d} | loss {rec['loss']:.6f}")
        history.append(rec)
        torch.save(
            {
                "model": model.state_dict(),
                "args": config,
                "epoch": epoch,
                "global_step": global_step,
            },
            out / "last.pt",
        )
        if args.max_steps is not None and global_step >= args.max_steps:
            break

    (out / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    return out / "last.pt"


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
