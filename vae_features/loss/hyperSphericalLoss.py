import math
from pathlib import Path
import torch
from torch import nn
from power_spherical import PowerSpherical, HypersphericalUniform

def _reshape_6d_joints(x: torch.Tensor) -> torch.Tensor:
    """
    Predictions and labels are already 6D per joint; only reshape to (batch, n_joints, 6).

    Accepts (B, J, 6) or flattened (B, J*6), e.g. from FeedForwardVAE or MHRPoseDataset.
    """
    if x.ndim == 3:
        if x.shape[-1] != 6:
            raise ValueError(
                f"use_6d_rotations=True expects last dim 6, got shape {tuple(x.shape)}"
            )
        return x
    if x.ndim == 2:
        flat = x.shape[1]
        if flat % 6 != 0:
            raise ValueError(
                f"use_6d_rotations=True expects flattened dim divisible by 6, got {flat}"
            )
        return x.view(x.shape[0], -1, 6)
    raise ValueError(
        f"use_6d_rotations=True expects joint tensor of rank 2 or 3, got shape {tuple(x.shape)}"
    )


def _reshape_euler_joints(x: torch.Tensor) -> torch.Tensor:
    """Reshape flat (B, J*3) or (B, J, 3) to (B, J, 3)."""
    if x.ndim == 3:
        if x.shape[-1] != 3:
            raise ValueError(
                f"Euler mode expects last dim 3, got shape {tuple(x.shape)}"
            )
        return x
    if x.ndim == 2:
        flat = x.shape[1]
        if flat % 3 != 0:
            raise ValueError(
                f"Euler mode expects flattened dim divisible by 3, got {flat}"
            )
        return x.view(x.shape[0], -1, 3)
    raise ValueError(
        f"Euler mode expects joint tensor of rank 2 or 3, got shape {tuple(x.shape)}"
    )


def _angle_reconstruction_loss_mse(pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred - label).square().sum(dim=-1).sum(dim=-1))


def _angle_reconstruction_loss_trig_euler(pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    """
    Continuous angular loss on radians: avoids MSE ambiguity at ±π wraps.

    L = 1/2 * (||sin(θ̂) - sin(θ)||² + ||cos(θ̂) - cos(θ)||²), mean over batch.
    """
    d_sin = torch.sin(pred) - torch.sin(label)
    d_cos = torch.cos(pred) - torch.cos(label)
    per_sample = 0.5 * (
        d_sin.square().sum(dim=(-2, -1)) + d_cos.square().sum(dim=(-2, -1))
    )
    return torch.mean(per_sample)


class HypersphericalVAELoss(nn.Module):
    def __init__(
        self,
        use_6d_rotation_format: bool,
        device: torch.device,
        use_vertex_supervision: bool = False,
        mhr_model_path: str | None = None,
        vertex_loss_weight: float = 1.0,
        use_trig_euler_angle_loss: bool = False,
    ) -> None:
        super().__init__()
        self.use_6d_rotations = use_6d_rotation_format
        self.use_vertex_supervision = use_vertex_supervision
        self.vertex_loss_weight = vertex_loss_weight
        self.use_trig_euler_angle_loss = use_trig_euler_angle_loss
        self.mhr_model = None

        if use_trig_euler_angle_loss and use_6d_rotation_format:
            raise ValueError(
                "use_trig_euler_angle_loss=True applies only to Euler (3 angles per joint); "
                "set use_6d_rotation_format=False."
            )

        if self.use_vertex_supervision:
            if mhr_model_path is None:
                raise ValueError(
                    "mhr_model_path must be provided when use_vertex_supervision=True"
                )
            path = Path(mhr_model_path)
            self.mhr_model = torch.jit.load(str(path), map_location=device)
            self.mhr_model.eval()

    def forward(
        self,
        predicted_joint_angles: torch.Tensor,
        label_joint_angles: torch.Tensor,
        latent_distributions: torch.distributions.Distribution,
        kl_weight: float,
        reconstructed_mhr_params: torch.Tensor | None = None,
        target_mhr_params: torch.Tensor | None = None,
        vertex_loss_weight: float | None = None,
    ) -> dict[str, torch.Tensor]:
        if self.use_6d_rotations:
            label_joint_angles = _reshape_6d_joints(label_joint_angles)
            predicted_joint_angles = _reshape_6d_joints(predicted_joint_angles)
            angle_rec_loss = _angle_reconstruction_loss_mse(
                predicted_joint_angles, label_joint_angles
            )
        else:
            label_joint_angles = _reshape_euler_joints(label_joint_angles)
            predicted_joint_angles = _reshape_euler_joints(predicted_joint_angles)
            if self.use_trig_euler_angle_loss:
                angle_rec_loss = _angle_reconstruction_loss_trig_euler(
                    predicted_joint_angles, label_joint_angles
                )
            else:
                angle_rec_loss = _angle_reconstruction_loss_mse(
                    predicted_joint_angles, label_joint_angles
                )
        kl_loss = self._compute_kl_divergence(latent_distributions)
        vertex_loss = torch.tensor(0.0, device=predicted_joint_angles.device)

        vw = (
            self.vertex_loss_weight
            if vertex_loss_weight is None
            else float(vertex_loss_weight)
        )
        total_loss = angle_rec_loss + kl_weight * kl_loss
        if self.use_vertex_supervision:
            if reconstructed_mhr_params is None or target_mhr_params is None:
                raise ValueError(
                    "reconstructed_mhr_params and target_mhr_params are required when "
                    "vertex supervision is enabled"
                )
            vertex_loss = self._compute_vertex_loss(
                reconstructed_mhr_params, target_mhr_params
            )
            total_loss = total_loss + vw * vertex_loss

        return {
            "angle_rec_loss": angle_rec_loss,
            "kl_loss": kl_loss,
            "vertex_loss": vertex_loss,
            "total_loss": total_loss,
        }

    def _compute_vertex_loss(
        self,
        reconstructed_mhr_params: torch.Tensor,
        target_mhr_params: torch.Tensor,
    ) -> torch.Tensor:
        if self.mhr_model is None:
            raise ValueError("mhr_model must be initialized for vertex supervision")

        device = reconstructed_mhr_params.device
        batch_size = reconstructed_mhr_params.shape[0]
        id_coeffs = torch.zeros(batch_size, 45, device=device)
        expr_coeffs = torch.zeros(batch_size, 72, device=device)

        verts_pred, _ = self.mhr_model(
            id_coeffs, reconstructed_mhr_params, expr_coeffs
        )
        verts_target, _ = self.mhr_model(
            id_coeffs, target_mhr_params.to(device), expr_coeffs
        )

        return torch.mean((verts_pred - verts_target).square().sum(dim=-1).sum(dim=-1))

    def _compute_kl_divergence(self, distribution) -> torch.Tensor:
        """
        Compute KL divergence to uniform distribution on the sphere.

        Args:
            distribution: Either PowerSpherical or VonMisesFisher

        Returns:
            KL divergence value
        """
        if isinstance(distribution, torch.distributions.Normal):
            # Create a standard normal prior matching the device and shape of the input
            prior = torch.distributions.Normal(
                torch.zeros_like(distribution.loc), torch.ones_like(distribution.scale)
            )

            # Sum over the embedding dimensions (D), average over batch (B) if needed,
            # but usually kl_divergence returns shape (B, D).
            # We explicitly sum over D to get the KL per vector, then mean over batch.
            kl = torch.distributions.kl_divergence(distribution, prior)
            return kl.sum(dim=-1).mean()
        if isinstance(distribution, PowerSpherical):
            # Use built-in KL divergence for PowerSpherical
            return torch.distributions.kl_divergence(
                distribution, HypersphericalUniform(dim=distribution.loc.shape[-1])
            ).mean()
        else:
            # For VonMisesFisher, use entropy-based approximation
            # KL(p || uniform) = -H(p) - log(1/surface_area)
            # For uniform on d-dim sphere: log(surface_area) = log(2π^(d/2) / Γ(d/2))

            d = distribution.dim
            entropy = distribution.entropy()

            # Log surface area of unit sphere in d dimensions
            # S_d = 2π^(d/2) / Γ(d/2)

            if d == 2:
                log_surface_area = math.log(2 * math.pi)
            else:
                # Approximate using Stirling's approximation for large d
                log_surface_area = (d / 2) * math.log(2 * math.pi) - torch.lgamma(
                    torch.tensor(d / 2.0)
                ).item()

            kl = -entropy - (-log_surface_area)
            return kl.mean()
