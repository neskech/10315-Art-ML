import math
from pathlib import Path
import torch
from torch import nn
from power_spherical import PowerSpherical, HypersphericalUniform

from vae_features.model.graphVae import euler_to_6d


class HypersphericalVAELoss(nn.Module):
    def __init__(
        self,
        use_6d_rotation_format: bool,
        use_vertex_supervision: bool = False,
        mhr_model_path: str | None = None,
        vertex_loss_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.use_6d_rotations = use_6d_rotation_format
        self.use_vertex_supervision = use_vertex_supervision
        self.vertex_loss_weight = vertex_loss_weight
        self.mhr_model = None

        if self.use_vertex_supervision:
            if mhr_model_path is None:
                raise ValueError(
                    "mhr_model_path must be provided when use_vertex_supervision=True"
                )
            path = Path(mhr_model_path)
            self.mhr_model = torch.jit.load(str(path))
            self.mhr_model.eval()

    def forward(
        self,
        predicted_joint_angles: torch.Tensor,
        label_joint_angles: torch.Tensor,
        latent_distributions: torch.distributions.Distribution,
        kl_weight: float,
        reconstructed_mhr_params: torch.Tensor | None = None,
        target_mhr_params: torch.Tensor | None = None,
    ):
        if self.use_6d_rotations:
            # Predicted joint angles should already be in 6d if this
            # boolean has been set to true
            label_joint_angles = euler_to_6d(label_joint_angles)

        reconstruction_loss = torch.mean(
            (predicted_joint_angles - label_joint_angles).square().sum(dim=-1)
        )
        kl_loss = self._compute_kl_divergence(latent_distributions)

        total_loss = reconstruction_loss + kl_weight * kl_loss
        if self.use_vertex_supervision:
            if reconstructed_mhr_params is None or target_mhr_params is None:
                raise ValueError(
                    "reconstructed_mhr_params and target_mhr_params are required when "
                    "vertex supervision is enabled"
                )
            vertex_loss = self._compute_vertex_loss(
                reconstructed_mhr_params, target_mhr_params
            )
            total_loss = total_loss + self.vertex_loss_weight * vertex_loss

        return total_loss

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
