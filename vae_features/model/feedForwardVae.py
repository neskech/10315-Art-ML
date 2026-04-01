from torch import nn
import torch

from vae_features.utils.feedForward import FeedForward, Norm
from vae_features.utils.skeletonFormat import SkeletonFormat
from power_spherical import PowerSpherical


class FeedForwardVAE(nn.Module):
    def __init__(
        self,
        skeletonFormat: SkeletonFormat,
        encoderSizes: list[int],
        dropout: float,
        use_residuals: bool,
        activation: nn.Module,
        normalization: Norm,
    ) -> None:
        super().__init__()

        # Features are all joint angles (each angle is 3 numbers)
        num_features = skeletonFormat.get_joint_count() * 3
        assert encoderSizes[0] == num_features
        self.latent_size = encoderSizes[-1]

        # Add 1 dimension for the concentration
        newEncoderSizes = encoderSizes.copy()
        newEncoderSizes[-1] += 1

        # We only take latent vector as input, so don't
        # include the +1 input dimension
        decoderSizes = encoderSizes[::-1]

        self.encoder = FeedForward(
            newEncoderSizes,
            dropout,
            use_residuals,
            activation,
            normalization,
            use_final_augmentations=True,
            bias=True,
        )
        self.decoder = FeedForward(
            decoderSizes,
            dropout,
            use_residuals,
            activation,
            normalization,
            use_final_augmentations=False,
            bias=True,
        )

    def encode(self, x: torch.Tensor) -> torch.distributions.Distribution:
        raw = self.encoder.forward(x)
        latent = raw[:, :-1]
        concentration = raw[:, -1]
        latent = latent / (torch.norm(latent, dim=-1) + 1e-6)
        return PowerSpherical(loc=latent, scale=concentration), latent, concentration

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder.forward(latent)

    def encode_and_reconstruct(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _, latent, _ = self.encode(x)
        reconstruction = self.decode(latent)
        return latent, reconstruction
