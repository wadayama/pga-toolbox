"""Unit tests for closed-form projections."""

from __future__ import annotations

import pytest
import torch

from pga_toolbox import project_frobenius_ball, project_total_power


def test_frobenius_ball_inside_unchanged():
    A = torch.randn(3, 3, dtype=torch.complex128) * 0.1
    P = 100.0
    A_proj = project_frobenius_ball(A, P)
    assert torch.allclose(A_proj, A)


def test_frobenius_ball_outside_rescaled():
    A = torch.randn(4, 4, dtype=torch.complex128) * 5.0
    P = 1.0
    A_proj = project_frobenius_ball(A, P)
    sq = (torch.linalg.norm(A_proj) ** 2).item()
    assert sq <= P + 1e-10
    assert sq == pytest.approx(P, rel=1e-9)
    # Direction preserved (uniform rescale).
    scale = torch.linalg.norm(A_proj) / torch.linalg.norm(A)
    assert torch.allclose(A_proj, scale * A)


def test_frobenius_ball_invalid_P():
    A = torch.randn(2, 2, dtype=torch.complex128)
    with pytest.raises(ValueError):
        project_frobenius_ball(A, 0.0)
    with pytest.raises(ValueError):
        project_frobenius_ball(A, -1.0)


def test_total_power_inside_unchanged():
    params = [torch.randn(2, 2, dtype=torch.complex128) * 0.1 for _ in range(3)]
    P = 100.0
    out = project_total_power(params, P)
    for p_in, p_out in zip(params, out):
        assert torch.allclose(p_out, p_in)


def test_total_power_outside_rescaled_uniformly():
    g = torch.Generator().manual_seed(0)
    params = [
        torch.randn(3, 3, dtype=torch.complex128, generator=g) * 2.0
        for _ in range(4)
    ]
    P = 1.0
    out = project_total_power(params, P)
    total_sq = sum((torch.linalg.norm(p) ** 2).item() for p in out)
    assert total_sq <= P + 1e-10
    assert total_sq == pytest.approx(P, rel=1e-9)
    # Uniform scale across all matrices.
    scales = [
        (torch.linalg.norm(p_out) / torch.linalg.norm(p_in)).item()
        for p_in, p_out in zip(params, out)
    ]
    for s in scales[1:]:
        assert s == pytest.approx(scales[0], rel=1e-12)


def test_total_power_empty_raises():
    with pytest.raises(ValueError):
        project_total_power([], 1.0)


def test_real_tensors_pass_through():
    A = torch.randn(3, 3, dtype=torch.float64) * 5.0
    P = 2.0
    A_proj = project_frobenius_ball(A, P)
    assert torch.linalg.norm(A_proj).item() == pytest.approx(P ** 0.5, rel=1e-9)
