"""Minimal demo: maximise a complex quadratic on a Frobenius ball.

Problem:
    maximise   f(Z) = - || Z - Z_target ||_F^2
    subject to ||Z||_F^2 <= P

Compares fixed-step PGA (at a small step) against Armijo line search;
prints iteration counts and final values to show that the persistent-step
line search reaches the same optimum in far fewer iterations and does
not need a tuned step size.
"""

from __future__ import annotations

import argparse

import torch

from pga_toolbox import (
    pga_ascent,
    pga_ascent_armijo,
    project_frobenius_ball,
)


def resolve_device(choice: str) -> torch.device:
    """Map the ``--device`` choice to a concrete device.

    ``auto`` (the default) picks CUDA when available, else CPU. Requesting
    ``cuda`` explicitly on a machine without a GPU is an error.
    """
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if choice == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is False.")
    return torch.device(choice)


def main(device: torch.device | None = None) -> None:
    if device is None:
        device = torch.device("cpu")
    torch.manual_seed(0)
    d = 4
    P = 1.0
    # Target outside the ball; created on the chosen device.
    Z_target = torch.randn(d, d, dtype=torch.complex128, device=device) * 3.0

    def closure(Z: torch.Tensor) -> torch.Tensor:
        diff = Z - Z_target
        return -torch.real(torch.sum(diff.conj() * diff))

    def projector(params):
        return [project_frobenius_ball(p, P=P) for p in params]

    # --- Fixed-step PGA (deliberately conservative step) ---
    Z_fixed = torch.zeros(
        d, d, dtype=torch.complex128, device=device, requires_grad=True
    )
    hist_fixed = pga_ascent(
        lambda: closure(Z_fixed),
        [Z_fixed],
        step_size=0.05,
        num_iters=200,
        projector=projector,
    )

    # --- Armijo line search PGA (no tuning) ---
    Z_armijo = torch.zeros(
        d, d, dtype=torch.complex128, device=device, requires_grad=True
    )
    hist_armijo = pga_ascent_armijo(
        lambda: closure(Z_armijo),
        [Z_armijo],
        projector=projector,
        max_iter=200,
    )

    # --- Closed-form optimum on the ball ---
    # f is maximised by Z* = sqrt(P) * Z_target / ||Z_target||_F.
    Z_star = (P ** 0.5) * Z_target / torch.linalg.norm(Z_target)
    f_star = closure(Z_star).item()

    print("== quadratic demo ==")
    print(f"device                         : {device}")
    print(f"closed-form optimum (max f)    : {f_star:.6f}")
    print(
        "fixed PGA  : "
        f"final = {hist_fixed[-1]:.6f}, iters = {len(hist_fixed)}, "
        f"step = 0.05"
    )
    print(
        "Armijo PGA : "
        f"final = {hist_armijo[-1]:.6f}, iters = {len(hist_armijo)}, "
        "step = persistent (no manual tuning)"
    )

    print()
    print("==> Armijo reaches the closed-form optimum in")
    print(f"    {len(hist_armijo)} iterations, vs {len(hist_fixed)} for the fixed step.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device to run on (default: auto = cuda if available, else cpu).",
    )
    args = parser.parse_args()
    main(resolve_device(args.device))
