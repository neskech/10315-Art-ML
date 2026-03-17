import math
import torch
from torch.distributions import Distribution


class ProjectedNormal(Distribution):
    """
    Projected Normal Distribution for modeling directional data.
    This distribution is defined by projecting a multivariate normal distribution onto the unit sphere.
    """

    def __init__(self, mu, sigma, validate_args=None):
        """
        Initialize the Projected Normal distribution.

        Args:
            mu (torch.Tensor): Mean vector of the underlying normal distribution.
            sigma (torch.Tensor): Standard deviation of the underlying normal distribution.
            validate_args (bool, optional): Whether to validate input arguments.
        """
        self.mu = mu
        self.sigma = sigma
        # Store the normalized mean direction as loc
        self._mean_direction = mu / mu.norm(dim=-1, keepdim=True)

        super(ProjectedNormal, self).__init__(validate_args=validate_args)
        self.underlying_normal = torch.distributions.Normal(mu, sigma)
        self.has_rsample = True

    def sample(self, sample_shape=torch.Size()):
        """
        Generate samples from the Projected Normal distribution.

        Args:
            sample_shape (torch.Size, optional): Shape of the samples to generate.
        """
        samples = self.underlying_normal.sample(sample_shape)
        return samples / samples.norm(dim=-1, keepdim=True)

    def rsample(self, sample_shape=torch.Size()):
        """
        Generate reparameterized samples from the Projected Normal distribution.

        Args:
            sample_shape (torch.Size, optional): Shape of the samples to generate.
        """
        samples = self.underlying_normal.rsample(sample_shape)
        return samples / samples.norm(dim=-1, keepdim=True)

    def log_prob(self, value):
        """
        Compute the log probability density function of the Projected Normal distribution.

        For a projected normal distribution, we compute the density on the unit sphere.
        The PDF involves normalizing the input, computing the Gaussian likelihood,
        and accounting for the Jacobian of the projection.

        Args:
            value (torch.Tensor): Points at which to evaluate the log PDF.
        """
        # Normalize value to unit sphere
        value_norm = value / value.norm(dim=-1, keepdim=True)

        # Compute log probability using the underlying normal distribution
        # For projected normal: log p(x) = log N(||mu|| * x | mu, sigma) - log Z
        # where Z is a normalization constant

        # Compute the dot product between mu and normalized value
        mu_norm = self.mu.norm(dim=-1, keepdim=True)
        dot_product = (self.mu * value_norm).sum(dim=-1, keepdim=True)

        # Log probability based on the distance from mean direction
        # This is a simplified form - the exact form depends on sigma structure
        log_sigma_sq = 2 * torch.log(self.sigma)

        # Compute exponent: -0.5 * ||mu||^2 / sigma^2 + (mu^T * x)^2 / sigma^2
        exponent = -0.5 * (mu_norm**2 / (self.sigma**2)).sum(dim=-1)
        exponent = exponent + 0.5 * ((dot_product**2) / (self.sigma**2)).sum(dim=-1)

        # Normalization constant (approximation)
        dim = self.mu.size(-1)
        log_norm = -0.5 * dim * math.log(2 * math.pi) - 0.5 * log_sigma_sq.sum(dim=-1)

        log_prob = log_norm + exponent

        return log_prob

    def entropy(self):
        """
        Compute the entropy of the Projected Normal distribution.
        """
        dim = self.mu.size(-1)
        # Sum of log sigmas across dimensions
        log_sigma_sum = torch.log(self.sigma).sum(dim=-1)
        entropy_normal = 0.5 * dim * (1 + math.log(2 * math.pi)) + log_sigma_sum
        return entropy_normal

    @property
    def mean(self):
        """
        Return the mean direction (normalized mu).
        """
        return self._mean_direction

    @property
    def mode(self):
        """
        Return the mode (same as mean direction for projected normal).
        """
        return self._mean_direction

    @property
    def loc(self):
        """
        Return the location parameter (normalized mu).
        """
        return self._mean_direction
