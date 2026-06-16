"""Unit tests for batched (parallel multi-start) SPG."""

from __future__ import annotations

import pytest
import torch

from pga_toolbox import (
    BatchedHistory,
    pga_ascent_spg,
    pga_ascent_spg_batched,
    pga_descent_spg_batched,
    project_frobenius_ball,
    project_frobenius_ball_batched,
    project_total_power,
    project_total_power_batched,
)


def test_batched_equivalence_to_scalar_spg_when_B1():
    """B=1 batched run must match scalar SPG to tight tolerance."""
    target = torch.full((5,), 0.5, dtype=torch.float64)

    x_s = torch.zeros(5, dtype=torch.float64, requires_grad=True)
    hist_s = pga_ascent_spg(
        lambda: -((x_s - target) ** 2).sum(), [x_s], max_iter=50,
    )

    xb = torch.zeros(1, 5, dtype=torch.float64, requires_grad=True)
    res = pga_ascent_spg_batched(
        lambda: -((xb - target) ** 2).sum(dim=1), [xb], max_iter=50,
    )
    assert isinstance(res, BatchedHistory)
    assert res.best_obj.shape == (1,)
    assert res.best_obj.item() == pytest.approx(hist_s[-1], abs=1e-8)
    assert torch.allclose(xb.detach()[0], x_s.detach(), atol=1e-6)
    assert res.winner == 0


def test_batched_independence_no_cross_talk():
    """B=4 with four different targets; each element solves its own problem."""
    targets = torch.stack([
        torch.full((3,), v, dtype=torch.float64) for v in (0.0, 1.0, -2.0, 3.0)
    ])  # (4, 3)
    xb = torch.zeros(4, 3, dtype=torch.float64, requires_grad=True)
    res = pga_ascent_spg_batched(
        lambda: -((xb - targets) ** 2).sum(dim=1), [xb], max_iter=100,
    )
    assert torch.allclose(xb.detach(), targets, atol=1e-4)
    assert torch.allclose(res.best_obj, torch.zeros(4, dtype=torch.float64), atol=1e-7)
    # Returned params are the best iterate: closure there equals best_obj.
    final = -((xb.detach() - targets) ** 2).sum(dim=1)
    assert torch.allclose(final, res.best_obj, atol=1e-8)


def test_batched_complex_independence():
    """Complex (Wirtinger) batched problem with distinct targets."""
    gen = torch.Generator(device=torch.get_default_device()).manual_seed(7)
    targets = torch.randn(2, 3, 3, dtype=torch.complex128, generator=gen)
    Zb = torch.zeros(2, 3, 3, dtype=torch.complex128, requires_grad=True)

    def closure():
        diff = Zb - targets
        return -torch.real((diff.conj() * diff).flatten(1).sum(1))  # (2,)

    res = pga_ascent_spg_batched(closure, [Zb], max_iter=80)
    assert torch.allclose(Zb.detach(), targets, atol=1e-5)
    assert torch.allclose(res.best_obj, torch.zeros(2, dtype=torch.float64), atol=1e-8)


def test_batched_total_power_projection_matches_scalar_per_slice():
    gen = torch.Generator(device=torch.get_default_device()).manual_seed(1)
    A = torch.randn(3, 2, 2, dtype=torch.complex128, generator=gen) * 5.0
    out = project_total_power_batched([A], P=4.0)[0]
    tot = (out.abs() ** 2).flatten(1).sum(1)
    assert torch.all(tot <= 4.0 + 1e-9)
    for b in range(3):
        ref = project_total_power([A[b]], P=4.0)[0]
        assert torch.allclose(out[b], ref, atol=1e-10)


def test_batched_frobenius_projection_per_element():
    gen = torch.Generator(device=torch.get_default_device()).manual_seed(2)
    A = torch.randn(4, 3, 3, dtype=torch.float64, generator=gen) * 10.0
    out = project_frobenius_ball_batched(A, P=2.0)
    nrm = (out.abs() ** 2).flatten(1).sum(1)
    assert torch.all(nrm <= 2.0 + 1e-9)
    for b in range(4):
        ref = project_frobenius_ball(A[b], P=2.0)
        assert torch.allclose(out[b], ref, atol=1e-10)


def test_batched_constrained_each_element_feasible():
    """Per-element power constraint holds for every restart after the run."""
    gen = torch.Generator(device=torch.get_default_device()).manual_seed(3)
    init = torch.randn(5, 4, 4, dtype=torch.complex128, generator=gen)
    target = torch.full((4, 4), 3.0, dtype=torch.complex128)
    Fb = init.clone().requires_grad_(True)

    def closure():
        diff = Fb - target
        return -torch.real((diff.conj() * diff).flatten(1).sum(1))  # (5,)

    res = pga_ascent_spg_batched(
        closure, [Fb],
        projector=lambda ps: project_total_power_batched(ps, P=4.0),
        max_iter=120,
    )
    tot = (Fb.detach().abs() ** 2).flatten(1).sum(1)
    assert torch.all(tot <= 4.0 + 1e-8)
    assert res.best_obj.shape == (5,)


def test_batched_best_of_B_finds_better_basin():
    """On a tilted double-well, best-of-B must reach the better (positive) basin
    while at least one element stays stuck in the worse (negative) basin."""
    inits = torch.linspace(-5.0, 5.0, 8, dtype=torch.float64).reshape(8, 1)
    xb = inits.clone().requires_grad_(True)

    def closure():
        # phi(x) = -0.1 (x^2 - 9)^2 + 0.3 x : maxima near x=+3 (high) and x=-3 (low).
        return (-0.1 * (xb ** 2 - 9.0) ** 2 + 0.3 * xb).sum(dim=1)  # (8,)

    res = pga_ascent_spg_batched(
        closure, [xb], max_iter=300, alpha0=0.05, alpha_max=0.5,
    )
    # Winner sits in the better, positive-x basin.
    assert xb.detach()[res.winner].item() > 0.0
    assert res.best_obj.max().item() > 0.5
    # Multi-start is not vacuous here: some element is stuck in the worse basin.
    assert res.best_obj.min().item() < 0.0
    assert res.best_obj[res.winner].item() == pytest.approx(
        res.best_obj.max().item(), abs=1e-12
    )


def test_batched_descent_minimises_cost():
    targets = torch.stack([
        torch.full((4,), v, dtype=torch.float64) for v in (0.5, -1.0)
    ])
    xb = torch.zeros(2, 4, dtype=torch.float64, requires_grad=True)
    res = pga_descent_spg_batched(
        lambda: ((xb - targets) ** 2).sum(dim=1), [xb], max_iter=80,
    )
    assert torch.allclose(xb.detach(), targets, atol=1e-4)
    assert torch.allclose(res.best_obj, torch.zeros(2, dtype=torch.float64), atol=1e-8)


def test_batched_stationary_element_stays_put():
    """An element initialised at the optimum is retired; others still converge."""
    target = torch.zeros(3, dtype=torch.float64)
    xb = torch.stack([
        torch.zeros(3, dtype=torch.float64),          # already optimal
        torch.full((3,), 5.0, dtype=torch.float64),   # far away
    ]).requires_grad_(True)
    res = pga_ascent_spg_batched(
        lambda: -((xb - target) ** 2).sum(dim=1), [xb], max_iter=100,
    )
    assert torch.allclose(xb.detach(), torch.zeros(2, 3, dtype=torch.float64), atol=1e-4)
    assert torch.allclose(res.best_obj, torch.zeros(2, dtype=torch.float64), atol=1e-8)


def test_batched_history_is_TxB_grid():
    targets = torch.stack([
        torch.full((2,), v, dtype=torch.float64) for v in (1.0, 2.0, 3.0)
    ])
    xb = torch.zeros(3, 2, dtype=torch.float64, requires_grad=True)
    res = pga_ascent_spg_batched(
        lambda: -((xb - targets) ** 2).sum(dim=1), [xb], max_iter=50,
    )
    assert len(res.history) >= 1
    for h in res.history:
        assert h.shape == (3,)
    # Best-so-far is non-decreasing per element (ascent).
    grid = torch.stack(res.history, 0)  # (T, 3)
    diffs = grid[1:] - grid[:-1]
    assert torch.all(diffs >= -1e-10)


def test_batched_forward_budget_caps_evals():
    target = torch.zeros(4, dtype=torch.float64)
    xb = torch.full((6, 4), 10.0, dtype=torch.float64, requires_grad=True)
    res = pga_ascent_spg_batched(
        lambda: -((xb - target) ** 2).sum(dim=1), [xb],
        max_iter=10000, forward_budget=8,
    )
    # At least 2 forwards per outer iteration => few outer iterations logged.
    assert len(res.history) <= 8


def test_batched_invalid_args_raise():
    xb = torch.zeros(2, 3, dtype=torch.float64, requires_grad=True)

    # No requires_grad.
    yb = torch.zeros(2, 3, dtype=torch.float64)
    with pytest.raises(ValueError):
        pga_ascent_spg_batched(lambda: -(yb ** 2).sum(dim=1), [yb], max_iter=5)

    with pytest.raises(ValueError):
        pga_ascent_spg_batched(lambda: -(xb ** 2).sum(dim=1), [xb], max_iter=5, c=0.0)
    with pytest.raises(ValueError):
        pga_ascent_spg_batched(
            lambda: -(xb ** 2).sum(dim=1), [xb], max_iter=5, nm_window=0,
        )
    with pytest.raises(ValueError):
        pga_ascent_spg_batched(
            lambda: -(xb ** 2).sum(dim=1), [xb], max_iter=5,
            alpha_min=1.0, alpha_max=1.0,
        )

    # Inconsistent batch sizes across params.
    a = torch.zeros(2, 3, dtype=torch.float64, requires_grad=True)
    b = torch.zeros(3, 3, dtype=torch.float64, requires_grad=True)
    with pytest.raises(ValueError):
        pga_ascent_spg_batched(
            lambda: -(a.sum(dim=1) + b.sum(dim=1)[:2]), [a, b], max_iter=5,
        )
