import math
import torch
from torch import nn
from power_spherical import PowerSpherical, HypersphericalUniform


class HypersphericalVAELoss(nn.Module):
    def __init__(self, use_6d_rotation_format: bool) -> None:
        super().__init__()
        self.use_6d_rotations = use_6d_rotation_format

    def forward(
        self,
        predicted_joint_angles: torch.Tensor,
        label_joint_angles: torch.Tensor,
        latent_distributions: torch.distributions.Distribution,
        kl_weight: float,
    ):
        reconstruction_loss = torch.mean(
            (predicted_joint_angles - label_joint_angles).sum(dim=-1).square()
        )
        kl_loss = self._compute_kl_divergence(latent_distributions)
        return reconstruction_loss + kl_weight * kl_loss

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
