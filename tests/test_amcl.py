from __future__ import annotations

import math

import torch

from amcl.losses import (
    AMCLBarlowTwinsLoss,
    AMCLMoCoLoss,
    AMCLNTXentLoss,
    AMCLSimSiamLoss,
    BarlowTwinsLoss,
    NTXentLoss,
    SimSiamLoss,
    TemperatureHead,
    omega,
)


def finite(x):
    assert torch.isfinite(x).all(), x


def test_omega_stationary_point():
    d = 128
    tau = torch.tensor(2.0 / d, requires_grad=True)
    omega(tau, d).backward()
    assert abs(float(tau.grad)) < 1e-5


def test_temperature_bounds_and_direction():
    head = TemperatureHead(8, tau_min=1e-5, tau_max=2.0, hidden_dim=16)
    scores = torch.tensor([-20.0, 0.0, 20.0])
    tau = head.score_to_temperature(scores)
    assert torch.all((tau >= 1e-5) & (tau <= 2.0))
    # Eq. (3) uses the inverse sigmoid: lower pair score -> higher tau.
    assert tau[0] > tau[1] > tau[2]


def test_ntxent_is_finite():
    loss = NTXentLoss(0.2)(torch.randn(8, 16), torch.randn(8, 16))
    finite(loss)


def test_amcl_ntxent_topk_and_half_safety():
    torch.manual_seed(0)
    loss_fn = AMCLNTXentLoss(16, num_heads=2, beta=1e-4, topk=3)
    z1 = torch.randn(6, 2, 16, dtype=torch.float16, requires_grad=True)
    z2 = torch.randn(6, 2, 16, dtype=torch.float16, requires_grad=True)
    loss = loss_fn(z1, z2)
    finite(loss)
    assert loss.dtype == torch.float32
    loss.backward()
    finite(z1.grad)
    finite(z2.grad)


def test_amcl_ntxent_negative_count():
    # With B samples, each anchor must see 2B-2 negatives.
    b, d = 5, 8
    fn = AMCLNTXentLoss(d, num_heads=1, beta=0.0, topk=100)
    z1 = torch.randn(b, 1, d)
    z2 = torch.randn(b, 1, d)
    eye = torch.eye(b, dtype=torch.bool)
    same = z1[:, 0].unsqueeze(0).expand(b, -1, -1)[~eye].view(b, b - 1, d)
    other = z2[:, 0].unsqueeze(0).expand(b, -1, -1)[~eye].view(b, b - 1, d)
    assert torch.cat([same, other], dim=1).shape[1] == 2 * b - 2
    finite(fn(z1, z2))


def test_amcl_moco_queue_loss():
    q = torch.randn(4, 2, 16, requires_grad=True)
    k = torch.randn(4, 2, 16)
    queue = torch.randn(2, 16, 32)
    fn = AMCLMoCoLoss(16, 2, beta=1e-4, topk=5)
    loss = fn(q, k, queue)
    finite(loss)
    loss.backward()
    finite(q.grad)


def test_simsiam_losses():
    b, c, d = 6, 2, 16
    p1, z1, p2, z2 = [torch.randn(b, c, d, requires_grad=True) for _ in range(4)]
    base = SimSiamLoss()(p1, z2, p2, z1)
    finite(base)
    base.backward(retain_graph=True)
    amcl = AMCLSimSiamLoss(d, c, beta=1e-4)
    loss = amcl(p1, z2, p2, z1)
    finite(loss)
    loss.backward()
    for x in (p1, p2):
        finite(x.grad)


def test_barlow_losses():
    b, c, d = 8, 2, 16
    z1 = torch.randn(b, c, d, requires_grad=True)
    z2 = torch.randn(b, c, d, requires_grad=True)
    base = BarlowTwinsLoss()(z1[:, 0], z2[:, 0])
    finite(base)
    base.backward(retain_graph=True)
    amcl = AMCLBarlowTwinsLoss(d, batch_size=b, num_heads=c, beta=1e-4, temperature_hidden_dim=8)
    loss = amcl(z1, z2)
    finite(loss)
    loss.backward()
    finite(z1.grad)
    finite(z2.grad)


def test_barlow_batch_size_is_enforced():
    fn = AMCLBarlowTwinsLoss(8, batch_size=8, num_heads=1, temperature_hidden_dim=4)
    z1 = torch.randn(7, 1, 8)
    z2 = torch.randn(7, 1, 8)
    try:
        fn(z1, z2)
    except ValueError as exc:
        assert "fixed training batch size" in str(exc)
    else:
        raise AssertionError("Expected a fixed-batch-size error")


def test_amcl_head_loss_is_summed_not_averaged():
    torch.manual_seed(7)
    z1 = torch.randn(5, 1, 12)
    z2 = torch.randn(5, 1, 12)
    single = AMCLNTXentLoss(12, num_heads=1, beta=1e-4, topk=2)
    dual = AMCLNTXentLoss(12, num_heads=2, beta=1e-4, topk=2)
    # Copy one head's temperature map into the shared dual-head map. Since the
    # objective is a sum over heads, duplicating the same feature head doubles it.
    dual.temperature.load_state_dict(single.temperature.state_dict())
    z1d = z1.expand(-1, 2, -1).contiguous()
    z2d = z2.expand(-1, 2, -1).contiguous()
    expected = 2.0 * single(z1, z2)
    actual = dual(z1d, z2d)
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_moco_loss_does_not_mutate_queue():
    q = torch.randn(4, 1, 8, requires_grad=True)
    k = torch.randn(4, 1, 8)
    queue = torch.randn(1, 8, 16)
    fn = AMCLMoCoLoss(8, 1, beta=1e-4, topk=3)
    queue_before = queue.clone()
    loss = fn(q, k, queue)
    finite(loss)
    assert torch.equal(queue, queue_before)


def test_shared_temperature_map_produces_pairwise_temperatures():
    torch.manual_seed(11)
    fn = AMCLNTXentLoss(16, num_heads=3, beta=1e-4, topk=2, temperature_hidden_dim=8)
    assert sum(p.numel() for p in fn.temperature.parameters()) > 0
    # There is exactly one temperature module for all heads/loss terms.
    z = torch.randn(6, 16)
    t = fn.temperature(z, torch.roll(z, 1, dims=0))
    assert t.shape == (6,)
    finite(t)
    assert float(t.std().detach()) > 0.0


def test_barlow_temperature_map_uses_full_channel_vectors():
    torch.manual_seed(12)
    from amcl.losses import AMCLBarlowTwinsLoss

    fn = AMCLBarlowTwinsLoss(12, batch_size=8, num_heads=1, beta=1e-6, temperature_hidden_dim=4)
    z1 = torch.randn(8, 1, 12)
    z2 = torch.randn(8, 1, 12)
    tau = fn.temperature.pair_temperature_matrix(z1[:, 0].T, z2[:, 0].T)
    assert tau.shape == (12, 12)
    finite(tau)
    # A scalar channel mean would not retain this full pairwise structure.
    assert float(tau.std().detach()) > 0.0
