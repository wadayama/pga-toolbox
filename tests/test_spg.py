"""Unit tests for the Spectral Projected Gradient (SPG) variants of PGA."""

from __future__ import annotations

import pytest
import torch

from pga_toolbox import (
    pga_ascent,
    pga_ascent_spg,
    pga_descent_spg,
    project_frobenius_ball,
    project_total_power,
)


def test_spg_ascent_real_quadratic_beats_tiny_fixed_step():
    """SPG must converge in far fewer iterations than a tiny fixed step."""
    target = torch.full((5,), 0.5, dtype=torch.float64)

    x_spg = torch.zeros(5, dtype=torch.float64, requires_grad=True)
    hist_spg = pga_ascent_spg(
        lambda: -((x_spg - target) ** 2).sum(),
        [x_spg],
        max_iter=50,
    )
    assert hist_spg[-1] == pytest.approx(0.0, abs=1e-10)
    assert len(hist_spg) <= 30
    # On return params hold the best iterate (the optimum here).
    assert torch.allclose(x_spg.detach(), target, atol=1e-5)

    x_fixed = torch.zeros(5, dtype=torch.float64, requires_grad=True)
    hist_fixed = pga_ascent(
        lambda: -((x_fixed - target) ** 2).sum(),
        [x_fixed],
        step_size=1e-3,
        num_iters=50,
    )
    assert hist_spg[-1] > hist_fixed[-1]


def test_spg_descent_real_quadratic():
    target = torch.full((5,), 0.5, dtype=torch.float64)

    def closure():
        return ((x - target) ** 2).sum()

    x = torch.zeros(5, dtype=torch.float64, requires_grad=True)
    hist = pga_descent_spg(closure, [x], max_iter=50)
    assert hist[-1] == pytest.approx(0.0, abs=1e-10)
    assert torch.allclose(x.detach(), target, atol=1e-4)
    # Returned params are the best (minimum-cost) point seen.
    assert closure().item() == pytest.approx(min(hist), abs=1e-10)


def test_spg_with_frobenius_projection():
    """Active ball constraint: iterate must land on the boundary optimum."""
    target = torch.full((4,), 10.0, dtype=torch.float64)

    def closure():
        return -((x - target) ** 2).sum()

    x = torch.zeros(4, dtype=torch.float64, requires_grad=True)

    def projector(params):
        return [project_frobenius_ball(p, P=1.0) for p in params]

    hist = pga_ascent_spg(closure, [x], projector=projector, max_iter=100)
    assert (x.detach() ** 2).sum().item() <= 1.0 + 1e-9
    expected = target / torch.linalg.norm(target)  # sqrt(P)=1, active constraint
    assert torch.allclose(x.detach(), expected, atol=1e-4)


def test_spg_with_total_power_projection():
    def closure():
        return -sum(((p - 5.0) ** 2).sum() for p in params)

    params = [
        torch.zeros(3, 3, dtype=torch.float64, requires_grad=True)
        for _ in range(3)
    ]
    hist = pga_ascent_spg(
        closure,
        params,
        projector=lambda ps: project_total_power(ps, P=4.0),
        max_iter=200,
    )
    total = sum((torch.linalg.norm(p.detach()) ** 2).item() for p in params)
    assert total <= 4.0 + 1e-9
    assert len(hist) >= 1
    # On return params are the best iterate: objective there == max(history).
    assert closure().item() == pytest.approx(max(hist), abs=1e-8)


def test_spg_complex_quadratic():
    """Complex problem with Wirtinger gradient."""
    gen = torch.Generator().manual_seed(42)
    target = torch.randn(3, 3, dtype=torch.complex128, generator=gen)

    def closure():
        diff = Z - target
        return -torch.real(torch.sum(diff.conj() * diff))

    Z = torch.zeros(3, 3, dtype=torch.complex128, requires_grad=True)
    hist = pga_ascent_spg(closure, [Z], max_iter=80)
    assert hist[-1] == pytest.approx(0.0, abs=1e-9)
    assert torch.allclose(Z.detach(), target, atol=1e-5)


def test_spg_returns_best_point_under_nonmonotone():
    """Final params must be the best-seen iterate even if history is nonmonotone."""
    target = torch.full((6,), 3.0, dtype=torch.float64)

    def closure():
        return -((x - target) ** 2).sum()

    x = torch.zeros(6, dtype=torch.float64, requires_grad=True)
    hist = pga_ascent_spg(closure, [x], max_iter=100)
    # The objective at the returned params equals the best recorded value.
    assert closure().item() == pytest.approx(max(hist), abs=1e-8)
    assert closure().item() >= hist[-1] - 1e-12


def test_spg_adapts_to_large_problem_scale():
    """Even from a small alpha0 the spectral step should reach a far optimum."""
    target = torch.full((4,), 100.0, dtype=torch.float64)
    x = torch.zeros(4, dtype=torch.float64, requires_grad=True)
    hist = pga_ascent_spg(
        lambda: -((x - target) ** 2).sum(), [x], max_iter=60, alpha0=1e-4,
    )
    assert hist[-1] == pytest.approx(0.0, abs=1e-6)


def test_spg_terminates_at_stationary_point():
    """Starting at the optimum, no acceptable step exists; history is empty."""
    target = torch.full((3,), 1.0, dtype=torch.float64)
    x = target.clone().requires_grad_(True)
    hist = pga_ascent_spg(
        lambda: -((x - target) ** 2).sum(), [x], max_iter=20,
    )
    assert hist == [] or hist[-1] == pytest.approx(0.0, abs=1e-15)
    # params unchanged (best == initial).
    assert torch.allclose(x.detach(), target, atol=1e-12)


def test_spg_forward_budget_caps_total_evals():
    target = torch.zeros(5, dtype=torch.float64)
    x = torch.full((5,), 10.0, dtype=torch.float64, requires_grad=True)
    hist = pga_ascent_spg(
        lambda: -((x - target) ** 2).sum(),
        [x],
        max_iter=1000,
        forward_budget=12,
    )
    assert len(hist) <= 12


def test_spg_invalid_params_raise():
    x = torch.zeros(2, dtype=torch.float64, requires_grad=True)

    # No requires_grad.
    y = torch.zeros(2, dtype=torch.float64)
    with pytest.raises(ValueError):
        pga_ascent_spg(lambda: -(y ** 2).sum(), [y], max_iter=5)

    # Invalid c.
    with pytest.raises(ValueError):
        pga_ascent_spg(lambda: -(x ** 2).sum(), [x], max_iter=5, c=0.0)
    with pytest.raises(ValueError):
        pga_ascent_spg(lambda: -(x ** 2).sum(), [x], max_iter=5, c=1.0)

    # Invalid shrink.
    with pytest.raises(ValueError):
        pga_ascent_spg(lambda: -(x ** 2).sum(), [x], max_iter=5, shrink=1.5)

    # Invalid nm_window.
    with pytest.raises(ValueError):
        pga_ascent_spg(lambda: -(x ** 2).sum(), [x], max_iter=5, nm_window=0)

    # Invalid step bounds.
    with pytest.raises(ValueError):
        pga_ascent_spg(
            lambda: -(x ** 2).sum(), [x], max_iter=5,
            alpha_min=1.0, alpha_max=1.0,
        )
    with pytest.raises(ValueError):
        pga_ascent_spg(lambda: -(x ** 2).sum(), [x], max_iter=0)


def test_spg_nonmonotone_window_allows_uphill_excursion():
    """A larger nonmonotone window should still converge (sanity: window>1)."""
    target = torch.full((5,), 2.0, dtype=torch.float64)
    x = torch.zeros(5, dtype=torch.float64, requires_grad=True)
    hist = pga_ascent_spg(
        lambda: -((x - target) ** 2).sum(), [x], max_iter=80, nm_window=5,
    )
    assert hist[-1] == pytest.approx(0.0, abs=1e-8)
