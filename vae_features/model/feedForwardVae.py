from torch import nn
import torch

from vae_features.utils.feedForward import FeedForward, Norm
from vae_features.utils.skeletonFormat import SkeletonFormat


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

        self.encoder = FeedForward(
            encoderSizes,
            dropout,
            use_residuals,
            activation,
            normalization,
            use_final_augmentations=True,
            bias=True,
        )
        self.decoder = FeedForward(
            encoderSizes[::-1],
            dropout,
            use_residuals,
            activation,
            normalization,
            use_final_augmentations=False,
            bias=True,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder.forward(x)
        latent = latent / (torch.norm(latent, dim=-1) + 1e-6)
        return latent

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder.forward(latent)

    def encode_and_reconstruct(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(x)
        reconstruction = self.decode(latent)
        return latent, reconstruction
