"""Short optimizer smoke test for every released training path."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from amcl.models import BarlowTwins, MoCo, SimCLR, SimSiam


def check_params(model, tag):
    for name, p in model.named_parameters():
        if not torch.isfinite(p).all():
            raise AssertionError(f"{tag}: non-finite parameter {name}")
        if p.grad is not None and not torch.isfinite(p.grad).all():
            raise AssertionError(f"{tag}: non-finite gradient {name}")


def train_steps(tag, model, steps=3, lr=1e-3, b=4):
    torch.manual_seed(123)
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    losses = []
    for step in range(steps):
        # Change the synthetic batch each step to exercise optimizer/state updates.
        x1 = torch.randn(b, 3, 32, 32)
        x2 = torch.randn(b, 3, 32, 32)
        opt.zero_grad(set_to_none=True)
        if tag.startswith("moco"):
            q, k = model(x1, x2)
            loss = model.loss(q, k)
        elif tag.startswith("simsiam"):
            *_, p1, z1, p2, z2 = model(x1, x2)
            loss = model.loss(p1, z1, p2, z2)
        else:
            *_, z1, z2 = model(x1, x2)
            loss = model.loss(z1, z2)
        if not torch.isfinite(loss):
            raise AssertionError(f"{tag}: non-finite loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        check_params(model, tag)
        opt.step()
        check_params(model, tag)
        if tag.startswith("moco"):
            model.update_queue(k)
        losses.append(float(loss.detach()))
    print(f"[OK] {tag:20s} losses=" + ", ".join(f"{x:.5g}" for x in losses))


def main():
    torch.set_num_threads(1)
    b = 4
    x1 = torch.randn(b, 3, 32, 32)
    x2 = torch.randn(b, 3, 32, 32)
    train_steps("simclr + AMCL", SimCLR(num_heads=2, amcl=True, topk=3), lr=0.06, b=b)
    train_steps("simclr baseline", SimCLR(num_heads=1, amcl=False), lr=0.06, b=b)
    train_steps("moco + AMCL", MoCo(num_heads=2, amcl=True, queue_size=16, topk=5), lr=0.08, b=b)
    train_steps("moco baseline", MoCo(num_heads=1, amcl=False, queue_size=16), lr=0.08, b=b)
    train_steps("simsiam + AMCL", SimSiam(num_heads=2, amcl=True), lr=0.05, b=b)
    train_steps("simsiam baseline", SimSiam(num_heads=1, amcl=False), lr=0.05, b=b)
    train_steps("barlow + AMCL", BarlowTwins(num_heads=1, amcl=True, batch_size=b, temperature_hidden_dim=16), lr=0.11, b=b)
    train_steps("barlow baseline", BarlowTwins(num_heads=1, amcl=False), lr=0.11, b=b)
    print("All multi-step optimizer smoke tests passed.")


if __name__ == "__main__":
    main()
