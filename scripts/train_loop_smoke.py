"""Exercise the actual amcl.train.train() orchestration without downloading data."""
from __future__ import annotations

import shutil
import sys
from argparse import Namespace
from pathlib import Path

import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import amcl.train as train_mod


class SyntheticSSL(Dataset):
    def __init__(self, n=8):
        self.x = torch.randn(n, 3, 32, 32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        x = self.x[i]
        # Deliberately generate distinct views while keeping the test fully local.
        return (x + 0.01 * torch.randn_like(x), x + 0.01 * torch.randn_like(x)), 0


class SyntheticEval(Dataset):
    def __init__(self, n=8):
        self.x = torch.randn(n, 3, 32, 32)
        self.y = torch.zeros(n, dtype=torch.long)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


def fake_datasets(*args, **kwargs):
    return SyntheticSSL(8), SyntheticEval(8), SyntheticEval(8)


def make_args(method: str, out: Path) -> Namespace:
    return Namespace(
        dataset="cifar10", data_root=".", method=method, amcl=True,
        epochs=2, batch_size=4, num_workers=0, heads=2,
        proj_dim=None, beta=None, tau=0.2, tau_min=1e-5, tau_max=2.0,
        topk=3, temperature_hidden_dim=16, queue_size=16, momentum=0.99, lr=None, weight_decay=5e-4,
        seed=0, aug="default", stl10_unlabeled=False, output=str(out),
        eval_every=10, eval_k=200, disable_knn=True, max_steps=2,
        fail_on_nonfinite=True, grad_clip=1.0,
    )


def main():
    train_mod.make_datasets = fake_datasets
    torch.set_num_threads(1)
    root = ROOT / ".release_train_smoke"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir()
    for method in ["simclr", "moco", "simsiam"]:
        out = root / method
        ckpt_path = train_mod.train(make_args(method, out))
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        assert "model" in ckpt and "args" in ckpt and ckpt["global_step"] == 2
        assert all(torch.isfinite(v).all() for v in ckpt["model"].values() if torch.is_floating_point(v))
        print(f"[OK] CLI train loop {method}")

    # Barlow Twins AMCL requires a fixed batch size; the synthetic loader obeys it.
    out = root / "barlow_twins"
    ckpt_path = train_mod.train(make_args("barlow_twins", out))
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert ckpt["global_step"] == 2
    print("[OK] CLI train loop barlow_twins")
    shutil.rmtree(root)
    print("Actual training-loop smoke tests passed for all four AMCL methods.")


if __name__ == "__main__":
    main()
