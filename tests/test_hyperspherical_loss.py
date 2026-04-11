"""Tests for angle reconstruction loss variants."""

import torch

from vae_features.loss.hyperSphericalLoss import (
    HypersphericalVAELoss,
    _angle_reconstruction_loss_trig_euler,
)


def test_trig_euler_loss_zero_when_identical():
    x = torch.randn(4, 127, 3)
    assert _angle_reconstruction_loss_trig_euler(x, x).item() == 0.0


def test_trig_euler_loss_matches_manual():
    pred = torch.tensor([[[0.0, 1.0, -0.5]]])
    lab = torch.tensor([[[0.1, 1.1, -0.4]]])
    d_sin = torch.sin(pred) - torch.sin(lab)
    d_cos = torch.cos(pred) - torch.cos(lab)
    want = 0.5 * (d_sin.square().sum() + d_cos.square().sum())
    got = _angle_reconstruction_loss_trig_euler(pred, lab)
    assert torch.allclose(got, want)


def test_trig_option_requires_euler_mode():
    try:
        HypersphericalVAELoss(
            use_6d_rotation_format=True,
            device=torch.device("cpu"),
            use_trig_euler_angle_loss=True,
        )
    except ValueError as e:
        assert "Euler" in str(e)
    else:
        raise AssertionError("expected ValueError")
