"""Benchmark: batched multi-start SPG on CPU vs GPU.

``pga_ascent_spg_batched`` runs ``B`` independent SPG optimisations — one per
random initial point — as a single vectorised computation over a leading batch
dimension. On SIMD / GPU hardware those ``B`` restarts cost ~the same wall-clock
as one, so multi-start is nearly free; on a CPU the cost grows roughly linearly
with ``B``. This script measures exactly that, sweeping ``B`` on each device and
printing the wall-clock and per-restart cost.

The benchmark problem is, for each batch element ``b``, a complex least-squares
fit ``max -||F_b - T_b||_F^2`` with distinct random targets ``T_b`` placed
*outside* a Frobenius / total-power ball, so the constraint is active and the
spectral step does real work (it is not a one-step solve).

Usage::

    uv run python examples/multistart_benchmark.py
    uv run python examples/multistart_benchmark.py --sizes 8 32 --batches 1 64 1024 4096
    uv run python examples/multistart_benchmark.py --devices cpu cuda --reps 5

Notes on interpretation:
  - GPU has a fixed per-iteration overhead (kernel launch + the host-device sync
    forced by the drivers' ``.item()`` calls), so for small ``B`` and small
    matrices the CPU wins. The GPU pulls ahead once ``B`` (and/or the matrix
    size) is large enough to amortise that overhead.
  - Timings use a warmup run, take the best of ``--reps``, and call
    ``torch.cuda.synchronize()`` around CUDA runs so the GPU work is actually
    finished before the clock stops.
"""

from __future__ import annotations

import argparse
import time

import torch

from pga_toolbox import pga_ascent_spg_batched, project_total_power_batched


def _make_run(B: int, n: int, device: torch.device, max_iter: int):
    """Build a zero-arg callable that runs one batched multi-start solve.

    Targets are drawn from a fixed CPU generator (so every device sees the same
    problem) and scaled outside the unit power ball to keep the constraint
    active.
    """
    g = torch.Generator(device="cpu").manual_seed(0)
    targets = (
        torch.randn(B, n, n, dtype=torch.complex128, generator=g) * 3.0
    ).to(device)

    def run():
        F = torch.zeros(
            B, n, n, dtype=torch.complex128, device=device, requires_grad=True
        )

        def closure():
            diff = F - targets
            return -torch.real((diff.conj() * diff).flatten(1).sum(1))  # (B,)

        return pga_ascent_spg_batched(
            closure,
            [F],
            projector=lambda ps: project_total_power_batched(ps, P=1.0),
            max_iter=max_iter,
        )

    return run


def _time(run, device: torch.device, reps: int) -> tuple[float, int]:
    """Return (best wall-clock seconds, outer-iteration count) for ``run``."""
    is_cuda = device.type == "cuda"
    res = run()  # warmup (also captures the iteration count)
    if is_cuda:
        torch.cuda.synchronize()
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        run()
        if is_cuda:
            torch.cuda.synchronize()
        best = min(best, time.perf_counter() - t0)
    return best, len(res.history)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=[8, 32],
        help="Matrix sizes n (complex n x n) to benchmark.",
    )
    parser.add_argument(
        "--batches", type=int, nargs="+", default=[1, 16, 64, 256, 1024, 4096],
        help="Numbers of restarts B to sweep.",
    )
    parser.add_argument(
        "--devices", type=str, nargs="+", default=["cpu", "cuda"],
        choices=["cpu", "cuda"],
        help="Devices to compare (cuda is skipped if unavailable).",
    )
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--reps", type=int, default=3)
    args = parser.parse_args()

    devices = []
    for d in args.devices:
        if d == "cuda" and not torch.cuda.is_available():
            print("[skip] cuda requested but not available; running CPU only.\n")
            continue
        devices.append(torch.device(d))
    if not devices:
        raise SystemExit("No usable device selected.")

    have_pair = len(devices) == 2

    for n in args.sizes:
        print(
            f"=== complex {n}x{n}, total-power projection, "
            f"max_iter={args.max_iter}, best of {args.reps} ==="
        )
        header = f"{'B':>6}"
        for dev in devices:
            header += f" {dev.type + '(ms)':>11} {dev.type + ' us/start':>14}"
        if have_pair:
            header += f" {'speedup':>9}"
        print(header)

        for B in args.batches:
            times = {}
            for dev in devices:
                secs, _iters = _time(_make_run(B, n, dev, args.max_iter), dev, args.reps)
                times[dev.type] = secs
            row = f"{B:>6}"
            for dev in devices:
                t = times[dev.type]
                row += f" {t * 1e3:>11.1f} {t / B * 1e6:>14.2f}"
            if have_pair:
                row += f" {times['cpu'] / times['cuda']:>8.1f}x"
            print(row)
        print()


if __name__ == "__main__":
    main()
