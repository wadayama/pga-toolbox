"""Unit tests for Armijo line search variants of PGA."""

from __future__ import annotations

import pytest
import torch

from pga_toolbox import (
    pga_ascent,
    pga_ascent_armijo,
    pga_descent_armijo,
    project_frobenius_ball,
    project_total_power,
)


def test_armijo_ascent_real_quadratic_beats_tiny_fixed_step():
    """On a simple quadratic, Armijo (persistent step) must converge in
    far fewer iterations than fixed-step PGA at a deliberately tiny step.
    This is the qualitative property that motivated the line search.
    """
    target = torch.full((5,), 0.5, dtype=torch.float64)

    # Armijo run from origin.
    x_armijo = torch.zeros(5, dtype=torch.float64, requires_grad=True)
    hist_armijo = pga_ascent_armijo(
        lambda: -((x_armijo - target) ** 2).sum(),
        [x_armijo],
        max_iter=50,
    )
    # Should converge to optimum well within 50 iters.
    assert hist_armijo[-1] == pytest.approx(0.0, abs=1e-10)
    # And in far fewer accepts than 1/step iters of a tiny fixed step.
    assert len(hist_armijo) <= 30

    # Tiny fixed-step PGA at same iter cap is nowhere near optimum.
    x_fixed = torch.zeros(5, dtype=torch.float64, requires_grad=True)
    hist_fixed = pga_ascent(
        lambda: -((x_fixed - target) ** 2).sum(),
        [x_fixed],
        step_size=1e-3,
        num_iters=50,
    )
    # The fixed-step run is clearly behind.
    assert hist_armijo[-1] > hist_fixed[-1]


def test_armijo_descent_real_quadratic():
    target = torch.full((5,), 0.5, dtype=torch.float64)

    def closure():
        return ((x - target) ** 2).sum()

    x = torch.zeros(5, dtype=torch.float64, requires_grad=True)
    hist = pga_descent_armijo(closure, [x], max_iter=50)
    assert hist[-1] == pytest.approx(0.0, abs=1e-10)
    assert torch.allclose(x.detach(), target, atol=1e-4)
    # Monotone (descent on cost should be non-increasing on accept).
    for a, b in zip(hist[:-1], hist[1:]):
        assert b <= a + 1e-12


def test_armijo_with_frobenius_projection():
    """Constraint forces the iterate onto the ball boundary."""
    target = torch.full((4,), 10.0, dtype=torch.float64)

    def closure():
        return -((x - target) ** 2).sum()

    x = torch.zeros(4, dtype=torch.float64, requires_grad=True)

    def projector(params):
        return [project_frobenius_ball(p, P=1.0) for p in params]

    hist = pga_ascent_armijo(closure, [x], projector=projector, max_iter=50)
    # Constraint holds.
    assert (x.detach() ** 2).sum().item() <= 1.0 + 1e-10
    # Optimum on ball is x = target / ||target|| * sqrt(P) (active constraint).
    expected_dir = target / torch.linalg.norm(target)
    expected = expected_dir
    assert torch.allclose(x.detach(), expected, atol=1e-4)
    # Final value monotone non-decreasing across accepts (ascent).
    for a, b in zip(hist[:-1], hist[1:]):
        assert b >= a - 1e-12


def test_armijo_with_total_power_projection():
    def closure():
        return -sum(((p - 5.0) ** 2).sum() for p in params)

    params = [
        torch.zeros(3, 3, dtype=torch.float64, requires_grad=True)
        for _ in range(3)
    ]
    hist = pga_ascent_armijo(
        closure,
        params,
        projector=lambda ps: project_total_power(ps, P=4.0),
        max_iter=80,
    )
    total = sum((torch.linalg.norm(p.detach()) ** 2).item() for p in params)
    assert total <= 4.0 + 1e-10
    assert len(hist) >= 1
    # Each accept improves the objective.
    for a, b in zip(hist[:-1], hist[1:]):
        assert b >= a - 1e-12


def test_armijo_complex_quadratic():
    """Complex problem with Wirtinger gradient."""
    gen = torch.Generator(device=torch.get_default_device()).manual_seed(42)
    target = torch.randn(3, 3, dtype=torch.complex128, generator=gen)

    def closure():
        diff = Z - target
        return -torch.real(torch.sum(diff.conj() * diff))

    Z = torch.zeros(3, 3, dtype=torch.complex128, requires_grad=True)
    hist = pga_ascent_armijo(closure, [Z], max_iter=50)
    assert hist[-1] == pytest.approx(0.0, abs=1e-10)
    assert torch.allclose(Z.detach(), target, atol=1e-6)


def test_armijo_terminates_at_stationary_point():
    """At the optimum the gradient is zero, no acceptable step exists."""
    target = torch.full((3,), 1.0, dtype=torch.float64)
    x = target.clone().requires_grad_(True)
    hist = pga_ascent_armijo(
        lambda: -((x - target) ** 2).sum(), [x], max_iter=20,
    )
    # We start at the optimum; gradient is zero, no acceptable step.
    # The first iteration computes objective (==0), backward, then fails
    # to find any forward-progress step. History should be empty.
    assert hist == [] or hist[-1] == pytest.approx(0.0, abs=1e-15)


def test_armijo_forward_budget_caps_total_evals():
    """forward_budget caps total evaluations including backtracks."""
    target = torch.zeros(5, dtype=torch.float64)
    x = torch.full((5,), 10.0, dtype=torch.float64, requires_grad=True)
    hist = pga_ascent_armijo(
        lambda: -((x - target) ** 2).sum(),
        [x],
        max_iter=1000,
        forward_budget=10,
    )
    # Budget is respected (loop exits early). Length must be modest.
    assert len(hist) <= 10


def test_armijo_invalid_params_raise():
    x = torch.zeros(2, dtype=torch.float64, requires_grad=True)

    # No requires_grad.
    y = torch.zeros(2, dtype=torch.float64)
    with pytest.raises(ValueError):
        pga_ascent_armijo(lambda: -(y ** 2).sum(), [y], max_iter=5)

    # Invalid c.
    with pytest.raises(ValueError):
        pga_ascent_armijo(lambda: -(x ** 2).sum(), [x], max_iter=5, c=0.0)
    with pytest.raises(ValueError):
        pga_ascent_armijo(lambda: -(x ** 2).sum(), [x], max_iter=5, c=1.5)

    # Invalid grow.
    with pytest.raises(ValueError):
        pga_ascent_armijo(lambda: -(x ** 2).sum(), [x], max_iter=5, grow=1.0)

    # Invalid shrink.
    with pytest.raises(ValueError):
        pga_ascent_armijo(lambda: -(x ** 2).sum(), [x], max_iter=5, shrink=1.5)


def test_armijo_persistent_step_grows_then_settles():
    """The persistent step e should be able to grow to match the problem
    scale rather than sticking at the small e0. We verify indirectly by
    checking that few iterations are needed for a problem scale much
    larger than e0.
    """
    target = torch.full((4,), 100.0, dtype=torch.float64)
    x = torch.zeros(4, dtype=torch.float64, requires_grad=True)
    hist = pga_ascent_armijo(
        lambda: -((x - target) ** 2).sum(), [x], max_iter=40, e0=1e-4,
    )
    # Even starting from very small e0, the step should grow and the
    # algorithm should reach the optimum within max_iter.
    assert hist[-1] == pytest.approx(0.0, abs=1e-8)
