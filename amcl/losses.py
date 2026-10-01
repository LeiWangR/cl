from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def omega(tau: torch.Tensor, dim: int) -> torch.Tensor:
    """Temperature regularizer from Eq. (2.1)-(2.3):

    .. math:: Omega(tau) = d'/2 * log(tau) + 1/tau.
    """
    tau = tau.float()
    return 0.5 * float(dim) * torch.log(tau) + torch.reciprocal(tau)


class TemperatureHead(nn.Module):
    """Shared pair-adaptive temperature predictor from Eq. (3).

    The paper constrains the temperature with

        sigma(r) = iota / (1 + exp(r)) + eta,

    i.e. ``tau = tau_min + (tau_max - tau_min) * sigmoid(-score)``.
    This inverse-sigmoid direction makes high-similarity/easier pairs tend
    towards lower temperatures and low-similarity/uncertain pairs towards
    higher temperatures, as discussed in the paper.
    """

    def __init__(
        self,
        input_dim: int,
        tau_min: float = 1e-5,
        tau_max: float = 2.0,
        hidden_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        if not (0.0 < tau_min < tau_max):
            raise ValueError("Require 0 < tau_min < tau_max")
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        hidden_dim = input_dim if hidden_dim is None else int(hidden_dim)
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")

        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)
        self.phi = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, input_dim),
        )

    def score_to_temperature(self, score: torch.Tensor) -> torch.Tensor:
        # Work in fp32 because tau_min can be 1e-5, making 1/tau as large as
        # 1e5, which is unsafe to evaluate in fp16.
        score = score.float()
        return self.tau_min + (self.tau_max - self.tau_min) * torch.sigmoid(-score)

    def pair_score(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        p1 = self.phi(z1.float())
        p2 = self.phi(z2.float())
        return (p1 * p2).sum(dim=-1)

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        return self.score_to_temperature(self.pair_score(z1, z2))


class BatchVectorTemperatureHead(nn.Module):
    """Temperature map for Barlow Twins channel vectors.

    Eq. (2.3) defines temperatures from pairs of batch-wise channel vectors
    ``z_l: = [z_l1, ..., z_lN]^T``. Therefore phi maps R^N -> R^N and is
    shared across all heads. The training batch size must stay fixed so the
    input/output dimension of phi is well-defined.
    """

    def __init__(
        self,
        batch_size: int,
        tau_min: float = 1e-5,
        tau_max: float = 2.0,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        if batch_size < 2:
            raise ValueError("batch_size must be at least 2")
        self.batch_size = int(batch_size)
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if not (0.0 < tau_min < tau_max):
            raise ValueError("Require 0 < tau_min < tau_max")
        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)
        self.phi = nn.Sequential(
            nn.Linear(self.batch_size, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, self.batch_size),
        )

    def score_to_temperature(self, score: torch.Tensor) -> torch.Tensor:
        score = score.float()
        return self.tau_min + (self.tau_max - self.tau_min) * torch.sigmoid(-score)

    def pair_temperature_matrix(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return tau for every row-pair in x/y.

        Args:
            x, y: [num_channels, batch_size] matrices.
        Returns:
            [num_channels, num_channels] temperatures.
        """
        if x.ndim != 2 or y.ndim != 2:
            raise ValueError("Expected [channels, batch] matrices")
        if x.shape != y.shape or x.size(1) != self.batch_size:
            raise ValueError(
                f"Barlow Twins AMCL expects [channels, {self.batch_size}] inputs; got {tuple(x.shape)}"
            )
        px = self.phi(x.float())
        py = self.phi(y.float())
        scores = px @ py.transpose(0, 1)
        return self.score_to_temperature(scores)


class NTXentLoss(nn.Module):
    """Standard symmetric NT-Xent/SimCLR loss."""

    def __init__(self, temperature: float = 0.2) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = float(temperature)

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        z1, z2 = F.normalize(z1.float(), dim=-1), F.normalize(z2.float(), dim=-1)
        b = z1.size(0)
        if b < 2:
            raise ValueError("NT-Xent needs at least two samples per batch")
        z = torch.cat([z1, z2], dim=0)
        logits = z @ z.t() / self.temperature
        logits.fill_diagonal_(-torch.inf)
        targets = (torch.arange(2 * b, device=z.device) + b) % (2 * b)
        return F.cross_entropy(logits, targets)


class AMCLNTXentLoss(nn.Module):
    """AMCL Eq. (2.1) with the practical Top-k approximation.

    For SimCLR, negatives for each anchor are all other views from both
    batches: 2B-2 negatives. The hardest k similarities are retained, with
    the matching pair-adaptive temperatures. The same shared temperature
    network is reused for all projection heads.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 3,
        beta: float = 0.01,
        tau_min: float = 1e-5,
        tau_max: float = 2.0,
        topk: int = 100,
        temperature_hidden_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        if num_heads < 1:
            raise ValueError("num_heads must be >= 1")
        if topk < 1:
            raise ValueError("topk must be >= 1")
        self.num_heads = int(num_heads)
        self.beta = float(beta)
        self.topk = int(topk)
        self.temperature = TemperatureHead(dim, tau_min, tau_max, temperature_hidden_dim)
        self.dim = int(dim)

    def _pair_temperatures(
        self,
        anchor: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        """Compute all pair temperatures efficiently without duplicated phi calls."""
        # anchor [B,D], negative [B,N,D]
        pa = self.temperature.phi(anchor.float())
        pn = self.temperature.phi(negative.float().reshape(-1, negative.size(-1)))
        pn = pn.reshape(negative.size(0), negative.size(1), -1)
        scores = torch.einsum("bd,bnd->bn", pa, pn)
        return self.temperature.score_to_temperature(scores)

    def _direction(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negatives: torch.Tensor,
    ) -> torch.Tensor:
        anchor = F.normalize(anchor.float(), dim=-1)
        positive = F.normalize(positive.float(), dim=-1)
        negatives = F.normalize(negatives.float(), dim=-1)

        pos_sim = (anchor * positive).sum(-1)
        tau_pos = self.temperature(anchor, positive)

        sim_neg = (anchor[:, None, :] * negatives).sum(-1)
        tau_neg = self._pair_temperatures(anchor, negatives)

        k = min(self.topk, negatives.size(1))
        vals, idx = sim_neg.topk(k=k, dim=1, largest=True, sorted=False)
        tau_top = tau_neg.gather(1, idx)

        # Eq. (5) Top-k approximation: mean_k sim/tau.
        neg = (vals / tau_top).mean(dim=1)

        # Eq. (5) selected-set regularizer:
        #   log(prod tau_k) + sum(1/tau_k)
        reg_pos = self.beta * omega(tau_pos, self.dim)
        reg_neg = self.beta * (
            torch.log(tau_top).sum(dim=1) + torch.reciprocal(tau_top).sum(dim=1)
        )
        return (-pos_sim / tau_pos + neg + reg_pos - reg_neg).mean()

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        if z1.shape != z2.shape or z1.ndim != 3:
            raise ValueError("Expected z1,z2 with shape [batch, heads, dim]")
        b, c, d = z1.shape
        if c != self.num_heads:
            raise ValueError("Number of heads does not match the loss")
        if b < 2:
            raise ValueError("AMCL NT-Xent needs at least two samples per batch")

        eye = torch.eye(b, dtype=torch.bool, device=z1.device)
        losses = []
        for h in range(c):
            a, p = z1[:, h], z2[:, h]
            # For a -> p, include z1 of the other samples and z2 of the other
            # samples, giving 2B-2 negatives. Reverse symmetrically.
            same_view_other = z1[:, h].unsqueeze(0).expand(b, -1, -1)[~eye].view(b, b - 1, d)
            other_view_other = z2[:, h].unsqueeze(0).expand(b, -1, -1)[~eye].view(b, b - 1, d)
            neg_a = torch.cat([same_view_other, other_view_other], dim=1)
            neg_p = torch.cat([other_view_other, same_view_other], dim=1)
            losses.append(self._direction(a, p, neg_a))
            losses.append(self._direction(p, a, neg_p))

        # The paper's multi-head objective is a sum over heads. The two-view
        # directions are averaged as in standard symmetric contrastive losses.
        return torch.stack(losses).view(c, 2).mean(dim=1).sum()


class AMCLMoCoLoss(nn.Module):
    """AMCL Eq. (2.1) for MoCo queue negatives."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        beta: float = 0.01,
        tau_min: float = 1e-5,
        tau_max: float = 2.0,
        topk: int = 100,
        temperature_hidden_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.num_heads = int(num_heads)
        self.beta = float(beta)
        self.topk = int(topk)
        self.dim = int(dim)
        self.temperature = TemperatureHead(dim, tau_min, tau_max, temperature_hidden_dim)

    def _direction(self, anchor: torch.Tensor, positive: torch.Tensor, negative_bank: torch.Tensor) -> torch.Tensor:
        anchor = F.normalize(anchor.float(), dim=-1)
        positive = F.normalize(positive.float(), dim=-1)
        negative_bank = F.normalize(negative_bank.float(), dim=-1)
        pos_sim = (anchor * positive).sum(-1)
        tau_pos = self.temperature(anchor, positive)

        pa = self.temperature.phi(anchor.float())
        pn = self.temperature.phi(negative_bank.float())
        tau_neg = self.temperature.score_to_temperature(pa @ pn.transpose(0, 1))
        sim_neg = anchor @ negative_bank.transpose(0, 1)
        k = min(self.topk, negative_bank.size(0))
        vals, idx = sim_neg.topk(k=k, dim=1, largest=True, sorted=False)
        tau_top = tau_neg.gather(1, idx)
        neg = (vals / tau_top).mean(dim=1)
        reg_pos = self.beta * omega(tau_pos, self.dim)
        reg_neg = self.beta * (
            torch.log(tau_top).sum(dim=1) + torch.reciprocal(tau_top).sum(dim=1)
        )
        return (-pos_sim / tau_pos + neg + reg_pos - reg_neg).mean()

    def forward(self, q: torch.Tensor, k: torch.Tensor, queue: torch.Tensor) -> torch.Tensor:
        if q.shape != k.shape or q.ndim != 3:
            raise ValueError("Expected q,k with shape [batch, heads, dim]")
        if queue.ndim != 3 or queue.shape[0] != self.num_heads or queue.shape[1] != q.size(-1):
            raise ValueError("Expected queue with shape [heads, dim, queue_size]")
        per_head = []
        for h in range(self.num_heads):
            qh, kh = q[:, h], k[:, h]
            bank = queue[h].transpose(0, 1)
            # Symmetric two-view queue objective; the same queue is used as a
            # memory bank for both directions.
            per_head.append(0.5 * (self._direction(qh, kh, bank) + self._direction(kh, qh, bank)))
        return torch.stack(per_head).sum()


class SimSiamLoss(nn.Module):
    """Standard symmetric negative cosine SimSiam objective."""

    def forward(self, p1: torch.Tensor, z2: torch.Tensor, p2: torch.Tensor, z1: torch.Tensor) -> torch.Tensor:
        z1, z2 = z1.detach(), z2.detach()
        return -0.5 * (
            F.cosine_similarity(p1, z2, dim=-1).mean()
            + F.cosine_similarity(p2, z1, dim=-1).mean()
        )


class AMCLSimSiamLoss(nn.Module):
    """AMCL Eq. (2.2), applied symmetrically to the SimSiam branches."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 3,
        beta: float = 0.01,
        tau_min: float = 1e-5,
        tau_max: float = 2.0,
        temperature_hidden_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.beta = float(beta)
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.temperature = TemperatureHead(dim, tau_min, tau_max, temperature_hidden_dim)

    def forward(self, p1, z2, p2, z1):
        losses = []
        for h in range(self.num_heads):
            a = F.normalize(p1[:, h].float(), dim=-1)
            b = F.normalize(z2[:, h].detach().float(), dim=-1)
            c = F.normalize(p2[:, h].float(), dim=-1)
            d = F.normalize(z1[:, h].detach().float(), dim=-1)
            tau1 = self.temperature(a, b)
            tau2 = self.temperature(c, d)
            s1 = (a * b).sum(-1)
            s2 = (c * d).sum(-1)
            term = (
                -0.5 * s1 / tau1
                -0.5 * s2 / tau2
                + self.beta * omega(tau1, self.dim)
                + self.beta * omega(tau2, self.dim)
            )
            losses.append(term.mean())
        return torch.stack(losses).sum()


class BarlowTwinsLoss(nn.Module):
    def __init__(self, lambd: float = 5e-3):
        super().__init__()
        if lambd < 0:
            raise ValueError("lambd must be non-negative")
        self.lambd = float(lambd)

    def forward(self, z1, z2):
        z1, z2 = z1.float(), z2.float()
        b = z1.size(0)
        if b < 2:
            raise ValueError("Barlow Twins needs at least two samples per batch")
        a = (z1 - z1.mean(0)) / (z1.std(0, unbiased=True) + 1e-9)
        c = (z2 - z2.mean(0)) / (z2.std(0, unbiased=True) + 1e-9)
        corr = a.T @ c / b
        diag = corr.diag()
        off = corr - torch.diag(diag)
        return ((1 - diag) ** 2).sum() + self.lambd * (off ** 2).sum()


class AMCLBarlowTwinsLoss(nn.Module):
    """AMCL Eq. (2.3) for Barlow Twins.

    The correlation matrix is computed from batch-standardized projected
    features. The adaptive temperature map follows the paper exactly at the
    structural level: positive temperatures are derived from pairs of the
    batch-wise channel vectors z_l:, while negative temperatures are derived
    from z_l: / z_m:+ channel-vector pairs. The same phi is shared across
    projection heads.
    """

    def __init__(
        self,
        dim: int,
        batch_size: int,
        num_heads: int = 3,
        beta: float = 1e-4,
        lambd: float = 5e-3,
        tau_min: float = 1e-5,
        tau_max: float = 2.0,
        temperature_hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.batch_size = int(batch_size)
        self.num_heads = int(num_heads)
        self.beta = float(beta)
        self.lambd = float(lambd)
        self.temperature = BatchVectorTemperatureHead(
            batch_size=batch_size,
            tau_min=tau_min,
            tau_max=tau_max,
            hidden_dim=temperature_hidden_dim,
        )

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        if z1.shape != z2.shape or z1.ndim != 3:
            raise ValueError("Expected z1,z2 with shape [batch, heads, dim]")
        if z1.size(0) != self.batch_size:
            raise ValueError(
                f"Barlow Twins AMCL requires the fixed training batch size {self.batch_size}; got {z1.size(0)}"
            )
        if z1.size(1) != self.num_heads or z1.size(2) != self.dim:
            raise ValueError("Tensor shape does not match AMCL Barlow Twins configuration")

        losses = []
        for h in range(self.num_heads):
            z1h, z2h = z1[:, h].float(), z2[:, h].float()
            # Match the baseline cross-correlation construction.
            x = (z1h - z1h.mean(0)) / (z1h.std(0, unbiased=True) + 1e-9)
            y = (z2h - z2h.mean(0)) / (z2h.std(0, unbiased=True) + 1e-9)
            corr = x.T @ y / self.batch_size

            # Eq. (2.3): z_l: is the full N-dimensional channel vector.
            x_channels = z1h.transpose(0, 1)  # [D, N], z_l: from Eq. (2.3)
            y_channels = z2h.transpose(0, 1)  # [D, N]
            tau = self.temperature.pair_temperature_matrix(x_channels, y_channels)
            tau_pos = torch.diagonal(tau)
            diag = torch.diagonal(corr)

            pos = (1.0 - diag / tau_pos).pow(2).sum()

            off_mask = ~torch.eye(self.dim, dtype=torch.bool, device=z1.device)
            corr_neg = corr[off_mask]
            tau_neg = tau[off_mask]
            neg = self.lambd * (corr_neg.pow(2) / tau_neg).sum()

            reg_pos = omega(tau_pos, self.dim).sum()
            reg_neg = omega(tau_neg, self.dim).sum()
            losses.append(pos + neg + self.beta * reg_pos - self.beta * reg_neg)

        # Eq. (2.3) sums over heads.
        return torch.stack(losses).sum()
