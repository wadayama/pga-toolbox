"""CUDA-specific checks: CPU/CUDA result agreement and device preservation.

Unlike the rest of the suite (which is device-agnostic and parametrised over the
default device by ``conftest.py``), these tests manage devices *explicitly*:
each builds one problem and solves it on both CPU and CUDA from bit-identical
inputs, then asserts the two runs agree. The whole module is skipped when no
CUDA device is present.

A note on tolerances: the Armijo and SPG drivers take data-dependent branches
(accept / backtrack on an objective comparison), so the tiny ordering
differences between CPU and CUDA reductions can in principle flip a branch and
make the *trajectories* diverge even when both runs are correct. We therefore
assert agreement on the converged result — both runs must reach the same
(analytic) optimum — rather than requiring step-by-step identical histories.
"""

from __future__ import annotations

import pytest
import torch

from pga_toolbox import (
    pga_ascent,
    pga_ascent_armijo,
    pga_ascent_spg,
    pga_ascent_spg_batched,
    project_total_power,
    project_total_power_batched,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)

DEVICE_PAIR = ("cpu", "cuda")


def test_fixed_step_cpu_cuda_agreement():
    """Fixed-step ascent on a real quadratic agrees across devices and reaches
    the analytic optimum on both."""
    target_cpu = torch.full((6,), 0.5, dtype=torch.float64, device="cpu")
    finals = {}
    hists = {}
    for dev in DEVICE_PAIR:
        target = target_cpu.to(dev)
        x = torch.zeros(6, dtype=torch.float64, device=dev, requires_grad=True)
        hist = pga_ascent(
            lambda: -((x - target) ** 2).sum(),
            [x],
            step_size=0.4,
            num_iters=200,
        )
        finals[dev] = x.detach().to("cpu")
        hists[dev] = hist
    # Both reach the optimum (objective 0, x == target).
    for dev in DEVICE_PAIR:
        assert hists[dev][-1] == pytest.approx(0.0, abs=1e-8)
    # CPU and CUDA agree.
    assert torch.allclose(finals["cpu"], finals["cuda"], atol=1e-8)
    assert finals["cuda"].device.type == "cpu"  # moved back for compare


def test_armijo_cpu_cuda_agreement_constrained():
    """Armijo with an active total-power projection lands on the same boundary
    optimum on CPU and CUDA."""
    target_cpu = torch.full((4, 4), 3.0, dtype=torch.float64, device="cpu")
    finals = {}
    for dev in DEVICE_PAIR:
        target = target_cpu.to(dev)
        x = torch.zeros(4, 4, dtype=torch.float64, device=dev, requires_grad=True)
        pga_ascent_armijo(
            lambda: -((x - target) ** 2).sum(),
            [x],
            projector=lambda ps: project_total_power(ps, P=1.0),
            max_iter=200,
        )
        # Constraint satisfied on each device.
        assert (x.detach() ** 2).sum().item() <= 1.0 + 1e-9
        finals[dev] = x.detach().to("cpu")
    assert torch.allclose(finals["cpu"], finals["cuda"], atol=1e-6)


def test_spg_complex_cpu_cuda_agreement():
    """SPG on a complex (Wirtinger) quadratic reaches the same target on both
    devices."""
    gen = torch.Generator().manual_seed(123)
    target_cpu = torch.randn(3, 3, dtype=torch.complex128, generator=gen, device="cpu")
    finals = {}
    for dev in DEVICE_PAIR:
        target = target_cpu.to(dev)
        Z = torch.zeros(3, 3, dtype=torch.complex128, device=dev, requires_grad=True)

        def closure():
            diff = Z - target
            return -torch.real(torch.sum(diff.conj() * diff))

        hist = pga_ascent_spg(closure, [Z], max_iter=100)
        assert hist[-1] == pytest.approx(0.0, abs=1e-9)
        finals[dev] = Z.detach().to("cpu")
    assert torch.allclose(finals["cpu"], finals["cuda"], atol=1e-6)


def test_spg_batched_cpu_cuda_agreement():
    """Batched multi-start SPG returns the same per-element optima and winner
    on CPU and CUDA from identical inits/targets."""
    gen = torch.Generator().manual_seed(7)
    targets_cpu = torch.randn(5, 4, dtype=torch.float64, generator=gen, device="cpu")
    best_objs = {}
    winners = {}
    finals = {}
    for dev in DEVICE_PAIR:
        targets = targets_cpu.to(dev)
        xb = torch.zeros(5, 4, dtype=torch.float64, device=dev, requires_grad=True)
        res = pga_ascent_spg_batched(
            lambda: -((xb - targets) ** 2).sum(dim=1), [xb], max_iter=150,
        )
        best_objs[dev] = res.best_obj.to("cpu")
        winners[dev] = res.winner
        finals[dev] = xb.detach().to("cpu")
    # Each element solved to its own target on both devices.
    assert torch.allclose(finals["cpu"], targets_cpu, atol=1e-4)
    assert torch.allclose(finals["cpu"], finals["cuda"], atol=1e-5)
    assert torch.allclose(best_objs["cpu"], best_objs["cuda"], atol=1e-7)
    assert winners["cpu"] == winners["cuda"]


def test_params_and_grads_stay_on_cuda():
    """Running on CUDA leaves params on CUDA and yields a Python-float history
    (the .item() calls in the drivers must not silently move data)."""
    target = torch.full((5,), 2.0, dtype=torch.float64, device="cuda")
    x = torch.zeros(5, dtype=torch.float64, device="cuda", requires_grad=True)
    hist = pga_ascent_spg(lambda: -((x - target) ** 2).sum(), [x], max_iter=80)
    assert x.device.type == "cuda"
    assert x.grad is not None and x.grad.device.type == "cuda"
    assert isinstance(hist[-1], float)
    assert hist[-1] == pytest.approx(0.0, abs=1e-9)


def test_projection_preserves_device():
    """Projections keep their output on the input tensor's device (the internal
    ``torch.tensor(P, device=...)`` must follow the data, not default to CPU)."""
    A = torch.randn(3, 3, dtype=torch.complex128, device="cuda") * 5.0
    out = project_total_power([A], P=1.0)[0]
    assert out.device.type == "cuda"
    assert (out.abs() ** 2).sum().item() == pytest.approx(1.0, rel=1e-9)
    out_b = project_total_power_batched([A.unsqueeze(0)], P=1.0)[0]
    assert out_b.device.type == "cuda"
