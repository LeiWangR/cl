"""Fast end-to-end numerical smoke tests for every released method."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from amcl.models import BarlowTwins, MoCo, SimCLR, SimSiam


def run(name, model, x1, x2):
    model.train()
    for p in model.parameters():
        p.grad = None
    if name.startswith("moco"):
        q, k = model(x1, x2)
        loss = model.loss(q, k)
    elif name.startswith("simsiam"):
        *_, p1, z1, p2, z2 = model(x1, x2)
        loss = model.loss(p1, z1, p2, z2)
    else:
        *_, z1, z2 = model(x1, x2)
        loss = model.loss(z1, z2)
    assert torch.isfinite(loss), (name, loss)
    loss.backward()
    if name.startswith("moco"):
        ptr_before = int(model.queue_ptr.item())
        model.update_queue(k)
        assert int(model.queue_ptr.item()) != ptr_before
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads), name
    print(f"[OK] {name:24s} loss={loss.item(): .6f}")


def main():
    torch.manual_seed(0)
    torch.set_num_threads(1)
    # Small tensor sizes keep this test fast; the production CLI uses the
    # paper's ResNet-18 / projection widths and batch size 512.
    b = 4
    x1 = torch.randn(b, 3, 32, 32)
    x2 = torch.randn(b, 3, 32, 32)

    run("simclr + AMCL", SimCLR(num_heads=2, amcl=True, topk=3), x1, x2)
    run("simclr baseline", SimCLR(num_heads=1, amcl=False), x1, x2)
    run("moco + AMCL", MoCo(num_heads=2, amcl=True, queue_size=16, topk=5), x1, x2)
    run("moco baseline", MoCo(num_heads=1, amcl=False, queue_size=16), x1, x2)

    # SimSiam/Barlow use 2048-d projections in the paper. BatchNorm needs >1
    # sample, hence b=4 here.
    run("simsiam + AMCL", SimSiam(num_heads=2, amcl=True), x1, x2)
    run("simsiam baseline", SimSiam(num_heads=1, amcl=False), x1, x2)
    run("barlow + AMCL", BarlowTwins(num_heads=2, amcl=True, batch_size=b, temperature_hidden_dim=16), x1, x2)
    run("barlow baseline", BarlowTwins(num_heads=1, amcl=False), x1, x2)
    print("All end-to-end smoke tests passed.")


if __name__ == "__main__":
    main()
