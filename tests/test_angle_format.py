"""Round-trip tests for 6D rotation helpers."""

import torch

from vae_features.utils.angleFormat import (
    euler_to_6d,
    euler_to_rotation_matrix,
    rotation_6d_to_euler,
    rotation_6d_to_matrix,
    rotation_matrix_to_6d,
    rotation_matrix_to_euler_xyz,
    wrap_euler_angles_pi,
)


def test_rotation_matrix_to_6d_matches_rotation_6d_to_matrix():
    torch.manual_seed(0)
    e = (torch.rand(64, 3) * 2 - 1) * 1.5
    r = euler_to_rotation_matrix(e, "XYZ")
    d6 = rotation_matrix_to_6d(r)
    r2 = rotation_6d_to_matrix(d6)
    assert torch.allclose(r, r2, atol=1e-5)


def test_euler_to_6d_round_trip_matrix():
    torch.manual_seed(1)
    e = (torch.rand(32, 3) * 2 - 1) * 1.5
    d6 = euler_to_6d(e, "XYZ")
    e2 = rotation_6d_to_euler(d6)
    r1 = euler_to_rotation_matrix(e, "XYZ")
    r2 = euler_to_rotation_matrix(e2, "XYZ")
    assert torch.allclose(r1, r2, atol=1e-4)


def test_identity_columns_in_6d():
    r = torch.eye(3).unsqueeze(0)
    d6 = rotation_matrix_to_6d(r)
    assert torch.allclose(
        d6, torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]])
    )
    r2 = rotation_6d_to_matrix(d6)
    assert torch.allclose(r, r2)


def test_wrap_euler_angles_pi_near_zero():
    x = torch.tensor([0.0, 6.283185307179586, -6.283185307179586])
    w = wrap_euler_angles_pi(x)
    assert torch.allclose(w, torch.zeros_like(x), atol=1e-5)


def test_euler_matrix_extract_round_trip():
    e = (torch.rand(20, 3) * 2 - 1) * 1.2
    r = euler_to_rotation_matrix(e, "XYZ")
    e_back = rotation_matrix_to_euler_xyz(r)
    r2 = euler_to_rotation_matrix(e_back, "XYZ")
    assert torch.allclose(r, r2, atol=1e-5)
