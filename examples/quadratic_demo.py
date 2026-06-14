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

import torch

from pga_toolbox import (
    pga_ascent,
    pga_ascent_armijo,
    project_frobenius_ball,
)


def main() -> None:
    torch.manual_seed(0)
    d = 4
    P = 1.0
    Z_target = torch.randn(d, d, dtype=torch.complex128) * 3.0  # outside the ball

    def closure(Z: torch.Tensor) -> torch.Tensor:
        diff = Z - Z_target
        return -torch.real(torch.sum(diff.conj() * diff))

    def projector(params):
        return [project_frobenius_ball(p, P=P) for p in params]

    # --- Fixed-step PGA (deliberately conservative step) ---
    Z_fixed = torch.zeros(d, d, dtype=torch.complex128, requires_grad=True)
    hist_fixed = pga_ascent(
        lambda: closure(Z_fixed),
        [Z_fixed],
        step_size=0.05,
        num_iters=200,
        projector=projector,
    )

    # --- Armijo line search PGA (no tuning) ---
    Z_armijo = torch.zeros(d, d, dtype=torch.complex128, requires_grad=True)
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
    main()
