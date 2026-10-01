from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import resnet_cifar
from .losses import (
    AMCLBarlowTwinsLoss,
    AMCLMoCoLoss,
    AMCLNTXentLoss,
    AMCLSimSiamLoss,
    BarlowTwinsLoss,
    NTXentLoss,
    SimSiamLoss,
)


class SimCLRProjectionHead(nn.Module):
    """Two-layer SimCLR-style MLP used by the original Lightly baseline."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class SimSiamProjectionHead(nn.Module):
    """Three-layer SimSiam projection MLP matching the original Lightly style.

    BatchNorm is also applied to the output layer, as in the released baseline.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim, affine=False),
        )

    def forward(self, x):
        return self.net(x)


class SimSiamPredictor(nn.Module):
    """Two-layer SimSiam predictor: Linear-BN-ReLU-Linear."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class MultiHeadProjector(nn.Module):
    """Independent projection heads with a shared architecture."""

    def __init__(self, in_dim, hidden_dim, out_dim, num_heads, kind: str):
        super().__init__()
        if num_heads < 1:
            raise ValueError("num_heads must be >= 1")
        if kind == "simclr":
            head_cls = SimCLRProjectionHead
        elif kind == "simsiam":
            head_cls = SimSiamProjectionHead
        else:
            raise ValueError("kind must be 'simclr' or 'simsiam'")
        self.kind = kind
        self.heads = nn.ModuleList(
            [head_cls(in_dim, hidden_dim, out_dim) for _ in range(num_heads)]
        )

    def forward(self, x):
        if x.ndim == 2:
            return torch.stack([head(x) for head in self.heads], dim=1)
        if x.ndim == 3 and x.size(1) == len(self.heads):
            return torch.stack([head(x[:, i]) for i, head in enumerate(self.heads)], dim=1)
        raise ValueError("MultiHeadProjector expects [B,D] or [B,C,D]")


class SimCLR(nn.Module):
    def __init__(
        self,
        depth=18,
        num_heads=1,
        amcl=False,
        proj_dim=128,
        beta=1e-4,
        tau=0.5,
        tau_min=1e-5,
        tau_max=2.0,
        topk=100,
        temperature_hidden_dim=128,
    ):
        super().__init__()
        self.encoder, dim = resnet_cifar(depth)
        self.encoder_dim = dim
        self.projector = MultiHeadProjector(dim, 512, proj_dim, num_heads, "simclr")
        self.amcl = bool(amcl)
        self.num_heads = int(num_heads)
        self.proj_dim = int(proj_dim)
        self.criterion = (
            AMCLNTXentLoss(
                proj_dim,
                num_heads,
                beta,
                tau_min,
                tau_max,
                topk,
                temperature_hidden_dim,
            )
            if amcl
            else NTXentLoss(tau)
        )

    def forward(self, x1, x2):
        h1, h2 = self.encoder(x1), self.encoder(x2)
        return h1, h2, self.projector(h1), self.projector(h2)

    def loss(self, z1, z2):
        if self.amcl:
            return self.criterion(z1, z2)
        return self.criterion(z1[:, 0], z2[:, 0])


class SimSiam(nn.Module):
    def __init__(
        self,
        depth=18,
        num_heads=1,
        amcl=False,
        proj_dim=2048,
        beta=1e-4,
        tau_min=1e-5,
        tau_max=2.0,
        temperature_hidden_dim=128,
    ):
        super().__init__()
        self.encoder, dim = resnet_cifar(depth)
        self.encoder_dim = dim
        self.projector = MultiHeadProjector(dim, 2048, proj_dim, num_heads, "simsiam")
        self.predictor = nn.ModuleList(
            [SimSiamPredictor(proj_dim, 512) for _ in range(num_heads)]
        )
        self.amcl = bool(amcl)
        self.num_heads = int(num_heads)
        self.proj_dim = int(proj_dim)
        self.criterion = (
            AMCLSimSiamLoss(
                proj_dim,
                num_heads,
                beta,
                tau_min,
                tau_max,
                temperature_hidden_dim,
            )
            if amcl
            else SimSiamLoss()
        )

    def _predict(self, z):
        return torch.stack([head(z[:, i]) for i, head in enumerate(self.predictor)], dim=1)

    def forward(self, x1, x2):
        h1, h2 = self.encoder(x1), self.encoder(x2)
        z1, z2 = self.projector(h1), self.projector(h2)
        p1, p2 = self._predict(z1), self._predict(z2)
        return h1, h2, p1, z1, p2, z2

    def loss(self, p1, z1, p2, z2):
        return self.criterion(p1, z2, p2, z1)


class BarlowTwins(nn.Module):
    def __init__(
        self,
        depth=18,
        num_heads=1,
        amcl=False,
        proj_dim=2048,
        beta=1e-6,
        lambd=5e-3,
        tau_min=1e-5,
        tau_max=2.0,
        batch_size=512,
        temperature_hidden_dim=128,
    ):
        super().__init__()
        self.encoder, dim = resnet_cifar(depth)
        self.encoder_dim = dim
        self.projector = MultiHeadProjector(dim, 2048, proj_dim, num_heads, "simsiam")
        self.amcl = bool(amcl)
        self.num_heads = int(num_heads)
        self.proj_dim = int(proj_dim)
        if amcl:
            self.criterion = AMCLBarlowTwinsLoss(
                proj_dim,
                batch_size=batch_size,
                num_heads=num_heads,
                beta=beta,
                lambd=lambd,
                tau_min=tau_min,
                tau_max=tau_max,
                temperature_hidden_dim=temperature_hidden_dim,
            )
        else:
            self.criterion = BarlowTwinsLoss(lambd)

    def forward(self, x1, x2):
        h1, h2 = self.encoder(x1), self.encoder(x2)
        return h1, h2, self.projector(h1), self.projector(h2)

    def loss(self, z1, z2):
        if self.amcl:
            return self.criterion(z1, z2)
        return sum(
            self.criterion(z1[:, h], z2[:, h]) for h in range(self.num_heads)
        ) / self.num_heads


class MoCo(nn.Module):
    """Symmetric MoCo-style two-view objective with a momentum key encoder."""

    def __init__(
        self,
        depth=18,
        num_heads=1,
        amcl=False,
        proj_dim=128,
        beta=1e-4,
        tau=0.1,
        tau_min=1e-5,
        tau_max=2.0,
        topk=100,
        queue_size=4096,
        momentum=0.99,
        temperature_hidden_dim=128,
    ):
        super().__init__()
        if queue_size < 2:
            raise ValueError("queue_size must be >= 2")
        if not (0.0 <= momentum < 1.0):
            raise ValueError("momentum must satisfy 0 <= momentum < 1")
        self.encoder_q, dim = resnet_cifar(depth)
        self.encoder_k = copy.deepcopy(self.encoder_q)
        self.projector_q = MultiHeadProjector(dim, 512, proj_dim, num_heads, "simclr")
        self.projector_k = copy.deepcopy(self.projector_q)
        for p in self.encoder_k.parameters():
            p.requires_grad = False
        for p in self.projector_k.parameters():
            p.requires_grad = False
        self.momentum = float(momentum)
        self.num_heads = int(num_heads)
        self.proj_dim = int(proj_dim)
        self.queue_size = int(queue_size)
        self.amcl = bool(amcl)
        queue = F.normalize(torch.randn(num_heads, proj_dim, queue_size), dim=1)
        self.register_buffer("queue", queue)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.criterion = (
            AMCLMoCoLoss(
                proj_dim,
                num_heads,
                beta,
                tau_min,
                tau_max,
                topk,
                temperature_hidden_dim,
            )
            if amcl
            else None
        )
        self.temperature = float(tau)

    @torch.no_grad()
    def _momentum_update(self):
        for q, k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            k.data.mul_(self.momentum).add_(q.data, alpha=1.0 - self.momentum)
        for q, k in zip(self.projector_q.parameters(), self.projector_k.parameters()):
            k.data.mul_(self.momentum).add_(q.data, alpha=1.0 - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor):
        keys = F.normalize(keys.float(), dim=-1)
        b = keys.size(0)
        if b > self.queue_size:
            keys = keys[-self.queue_size :]
            b = keys.size(0)
        ptr = int(self.queue_ptr.item())
        for i in range(b):
            self.queue[:, :, (ptr + i) % self.queue_size] = keys[i]
        self.queue_ptr[0] = (ptr + b) % self.queue_size

    def _encode_key(self, x):
        with torch.no_grad():
            return self.projector_k(self.encoder_k(x))

    def forward(self, x1, x2):
        q1 = self.projector_q(self.encoder_q(x1))
        with torch.no_grad():
            self._momentum_update()
            k2 = self._encode_key(x2)
            # A single momentum update is used for the iteration, matching the
            # standard MoCo parameter-update pattern.
        return q1, k2

    def loss(self, q, k):
        q = q.float()
        k = k.float()
        if self.amcl:
            return self.criterion(q, k, self.queue.detach())

        losses = []
        for h in range(self.num_heads):
            qq = F.normalize(q[:, h], dim=-1)
            kk = F.normalize(k[:, h], dim=-1)
            pos = (qq * kk).sum(-1, keepdim=True) / self.temperature
            neg = qq @ self.queue[h].detach() / self.temperature
            logits = torch.cat([pos, neg], dim=1)
            target = torch.zeros(qq.size(0), dtype=torch.long, device=qq.device)
            losses.append(F.cross_entropy(logits, target))
        # One head is used for the baseline; this mean is defensive for
        # multi-head debugging. Queue mutation is deliberately separate from
        # loss evaluation so autograd can finish before the buffer changes.
        return torch.stack(losses).mean()

    @torch.no_grad()
    def update_queue(self, k: torch.Tensor):
        self._dequeue_and_enqueue(k.detach())
