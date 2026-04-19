#!/usr/bin/env python3
"""Print max-abs errors for 6D rotation round-trips (run: uv run python scripts/verify_rotation_6d_roundtrip.py)."""

from __future__ import annotations

import sys
from pathlib import Path

# Repo root on path when run as scripts/foo.py
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch

from vae_features.utils.angleFormat import (
    euler_to_6d,
    euler_to_rotation_matrix,
    rotation_6d_to_euler,
    rotation_6d_to_matrix,
    rotation_matrix_to_6d,
    rotation_matrix_to_euler_xyz,
)


def max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).abs().max().item()


def main() -> None:
    torch.manual_seed(42)

    print("=== euler -> R -> euler -> R ===")
    errs = []
    for _ in range(5):
        e = (torch.rand(500, 3) * 2 - 1) * 1.5
        r = euler_to_rotation_matrix(e, "XYZ")
        e2 = rotation_matrix_to_euler_xyz(r)
        r2 = euler_to_rotation_matrix(e2, "XYZ")
        errs.append(max_abs(r, r2))
    print(f"  max_abs: {max(errs):.3e}")

    print("=== R -> rotation_matrix_to_6d -> rotation_6d_to_matrix -> R ===")
    errs = []
    for _ in range(5):
        e = (torch.rand(500, 3) * 2 - 1) * 1.5
        r = euler_to_rotation_matrix(e, "XYZ")
        d6 = rotation_matrix_to_6d(r)
        r_gs = rotation_6d_to_matrix(d6)
        errs.append(max_abs(r, r_gs))
    print(f"  max_abs: {max(errs):.3e}")

    print("=== euler -> euler_to_6d -> rotation_6d_to_matrix -> R ===")
    errs = []
    for _ in range(5):
        e = (torch.rand(500, 3) * 2 - 1) * 1.5
        r = euler_to_rotation_matrix(e, "XYZ")
        d6 = euler_to_6d(e, "XYZ")
        r2 = rotation_6d_to_matrix(d6)
        errs.append(max_abs(r, r2))
    print(f"  max_abs: {max(errs):.3e}")

    print("=== euler -> euler_to_6d -> rotation_6d_to_euler -> R ===")
    errs = []
    for _ in range(5):
        e = (torch.rand(500, 3) * 2 - 1) * 1.5
        r = euler_to_rotation_matrix(e, "XYZ")
        d6 = euler_to_6d(e, "XYZ")
        e2 = rotation_6d_to_euler(d6)
        r2 = euler_to_rotation_matrix(e2, "XYZ")
        errs.append(max_abs(r, r2))
    print(f"  max_abs: {max(errs):.3e}")

    r = torch.eye(3).unsqueeze(0)
    d6 = rotation_matrix_to_6d(r)
    print(f"=== Identity 6D: {d6[0].tolist()} ===")
    r2 = rotation_6d_to_matrix(d6)
    print(f"  max_abs I vs GS: {max_abs(r, r2):.3e}")

    print("Done (expect ~1e-6 or better on rotation matrices).")


if __name__ == "__main__":
    main()
