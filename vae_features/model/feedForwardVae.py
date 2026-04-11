from torch import nn
import torch

from vae_features.utils.feedForward import FeedForward, Norm
from vae_features.utils.skeletonFormat import SkeletonFormat
from power_spherical import PowerSpherical


class FeedForwardVAE(nn.Module):

    def __init__(self,
                 skeletonFormat: SkeletonFormat,
                 encoderSizes: list[int],
                 dropout: float,
                 use_residuals: bool,
                 activation: nn.Module,
                 normalization: Norm,
                 device: torch.device,
                 use_6d_rotation_format: bool,
                 initial_concentration: float = 0.0) -> None:
        super().__init__()
        self.use_6d_rotation_format = use_6d_rotation_format

        # Features are all joint angles (each angle is 3 numbers)
        num_features = skeletonFormat.get_joint_count() * (
            3 if not use_6d_rotation_format else 6)
        encoderSizes = [num_features, *encoderSizes]
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

        self.concentration_log_scale = nn.Parameter(
            torch.log(torch.tensor(initial_concentration, device=device)))

    def encode(self, x: torch.Tensor) -> torch.distributions.Distribution:
        raw = self.encoder.forward(x)
        latent = raw[:, :-1]
        log_concentration = raw[:, -1]

        concentration = torch.exp(self.concentration_log_scale +
                                  log_concentration)
        latent = latent / (torch.norm(latent, dim=-1, keepdim=True) + 1e-6)
        return PowerSpherical(loc=latent,
                              scale=concentration), latent, concentration

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        y = self.decoder.forward(latent)
        return y

    def encode_and_reconstruct(
            self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        distribution, mean_latent, concentration = self.encode(x)
        
        # If the model is in training mode, sample from the distribution!
        # rsample() ensures gradients can still flow backward through the noise.
        if self.training:
            z = distribution.rsample()
        else:
            # During evaluation/inference, it's standard to just use the mean
            z = mean_latent
            
        reconstruction = self.decode(z)
        return distribution,mean_latent, z, concentration, reconstruction
