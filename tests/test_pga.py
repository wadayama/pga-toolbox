"""Unit tests for fixed-step pga_ascent / pga_descent."""

from __future__ import annotations

import pytest
import torch

from pga_toolbox import (
    pga_ascent,
    pga_descent,
    project_frobenius_ball,
    project_total_power,
)


def _make_real_quadratic():
    """Convex problem: max -||x - x*||^2, optimum x* = ones / 2."""
    target = torch.full((5,), 0.5, dtype=torch.float64)

    def closure(params):
        (x,) = params
        return -((x - target) ** 2).sum()

    return closure, target


def test_real_quadratic_ascent_converges():
    closure_fn, target = _make_real_quadratic()
    x = torch.zeros(5, dtype=torch.float64, requires_grad=True)
    hist = pga_ascent(
        lambda: closure_fn([x]),
        [x],
        step_size=0.4,
        num_iters=200,
    )
    assert len(hist) == 200
    assert hist[-1] > hist[0]
    # Should approach the optimum (0).
    assert hist[-1] == pytest.approx(0.0, abs=1e-8)
    assert torch.allclose(x.detach(), target, atol=1e-4)


def test_real_quadratic_descent_converges():
    target = torch.full((5,), 0.5, dtype=torch.float64)

    def closure():
        return ((x - target) ** 2).sum()

    x = torch.zeros(5, dtype=torch.float64, requires_grad=True)
    hist = pga_descent(closure, [x], step_size=0.4, num_iters=200)
    assert len(hist) == 200
    assert hist[-1] < hist[0]
    assert hist[-1] == pytest.approx(0.0, abs=1e-8)


def test_projector_inplace_enforced():
    target = torch.zeros(4, dtype=torch.float64)

    def closure():
        return -((x - 10.0) ** 2).sum()  # pull away from origin

    x = torch.zeros(4, dtype=torch.float64, requires_grad=True)

    def proj_inplace(params):
        (p,) = params
        p.copy_(project_frobenius_ball(p, P=1.0))

    pga_ascent(
        closure, [x], step_size=0.1, num_iters=50, projector=proj_inplace,
    )
    # Constraint must hold after final iteration.
    assert (x.detach() ** 2).sum().item() <= 1.0 + 1e-10


def test_projector_functional_enforced():
    target = torch.zeros(4, dtype=torch.float64)

    def closure():
        return -((x - 10.0) ** 2).sum()

    x = torch.zeros(4, dtype=torch.float64, requires_grad=True)

    def proj_functional(params):
        return [project_frobenius_ball(p, P=1.0) for p in params]

    pga_ascent(
        closure, [x], step_size=0.1, num_iters=50, projector=proj_functional,
    )
    assert (x.detach() ** 2).sum().item() <= 1.0 + 1e-10


def test_total_power_functional_in_pga():
    def closure():
        return -sum(((p - 5.0) ** 2).sum() for p in params)

    params = [
        torch.zeros(3, 3, dtype=torch.float64, requires_grad=True)
        for _ in range(3)
    ]

    def proj(params):
        return project_total_power(params, P=2.0)

    pga_ascent(closure, params, step_size=0.05, num_iters=80, projector=proj)
    total = sum((torch.linalg.norm(p.detach()) ** 2).item() for p in params)
    assert total <= 2.0 + 1e-10


def test_complex_ascent_quadratic():
    """Complex problem: max -||Z - Z_target||_F^2."""
    target = torch.randn(
        3, 3, dtype=torch.complex128,
        generator=torch.Generator(device=torch.get_default_device()).manual_seed(0),
    )

    def closure():
        diff = Z - target
        return -torch.real(torch.sum(diff.conj() * diff))

    Z = torch.zeros(3, 3, dtype=torch.complex128, requires_grad=True)
    hist = pga_ascent(closure, [Z], step_size=0.5, num_iters=100)
    assert hist[-1] > hist[0]
    assert hist[-1] == pytest.approx(0.0, abs=1e-8)
    assert torch.allclose(Z.detach(), target, atol=1e-4)


def test_invalid_step_size_raises():
    x = torch.zeros(2, dtype=torch.float64, requires_grad=True)
    with pytest.raises(ValueError):
        pga_ascent(lambda: -(x ** 2).sum(), [x], step_size=0.0, num_iters=10)
    with pytest.raises(ValueError):
        pga_descent(lambda: (x ** 2).sum(), [x], step_size=-0.1, num_iters=10)


def test_invalid_num_iters_raises():
    x = torch.zeros(2, dtype=torch.float64, requires_grad=True)
    with pytest.raises(ValueError):
        pga_ascent(lambda: -(x ** 2).sum(), [x], step_size=0.1, num_iters=0)


def test_missing_requires_grad_raises():
    x = torch.zeros(2, dtype=torch.float64)  # no requires_grad
    with pytest.raises(ValueError):
        pga_ascent(lambda: -(x ** 2).sum(), [x], step_size=0.1, num_iters=5)


def test_param_unused_in_closure_raises():
    used = torch.zeros(2, dtype=torch.float64, requires_grad=True)
    ghost = torch.zeros(2, dtype=torch.float64, requires_grad=True)
    with pytest.raises(RuntimeError, match="received no gradient"):
        pga_ascent(
            lambda: -(used ** 2).sum(),
            [used, ghost],
            step_size=0.1,
            num_iters=3,
        )


def test_projector_length_mismatch_raises():
    x = torch.zeros(2, dtype=torch.float64, requires_grad=True)

    def bad_proj(params):
        return [params[0], params[0]]  # returns 2, expected 1

    with pytest.raises(ValueError, match="projector returned"):
        pga_ascent(
            lambda: -(x ** 2).sum(),
            [x],
            step_size=0.1,
            num_iters=2,
            projector=bad_proj,
        )
