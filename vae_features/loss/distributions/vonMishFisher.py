import torch
import numpy as np
import mpmath
from torch.distributions import Distribution


class vMFLogPartition(torch.autograd.Function):
    """
    Evaluates log C_d(kappa) for vMF density.
    Allows autograd wrt kappa.

    This is adapted from vmf-lib/models.py
    """

    besseli = np.vectorize(mpmath.besseli)
    log = np.vectorize(mpmath.log)
    nhlog2pi = -0.5 * np.log(2 * np.pi)

    @staticmethod
    def forward(ctx, *args):
        """
        Args:
            args[0] = d; scalar (> 0)
            args[1] = kappa; (> 0) torch tensor of any shape

        Returns:
            logC = log C_d(kappa); torch tensor of the same shape as kappa
        """
        d = args[0]
        kappa = args[1]

        s = 0.5 * d - 1

        # log I_s(kappa)
        mp_kappa = mpmath.mpf(1.0) * kappa.detach().cpu().numpy()
        mp_logI = vMFLogPartition.log(vMFLogPartition.besseli(s, mp_kappa))
        logI = torch.from_numpy(np.array(mp_logI.tolist(), dtype=float)).to(kappa)

        if (logI != logI).sum().item() > 0:  # there is nan
            raise ValueError("NaN is detected from the output of log-besseli()")

        logC = d * vMFLogPartition.nhlog2pi + s * kappa.log() - logI

        # save for backward()
        ctx.s, ctx.mp_kappa, ctx.logI = s, mp_kappa, logI

        return logC

    @staticmethod
    def backward(ctx, *grad_output):
        s, mp_kappa, logI = ctx.s, ctx.mp_kappa, ctx.logI

        # log I_{s+1}(kappa)
        mp_logI2 = vMFLogPartition.log(vMFLogPartition.besseli(s + 1, mp_kappa))
        logI2 = torch.from_numpy(np.array(mp_logI2.tolist(), dtype=float)).to(logI)

        if (logI2 != logI2).sum().item() > 0:  # there is nan
            raise ValueError("NaN is detected from the output of log-besseli()")

        dlogC_dkappa = -(logI2 - logI).exp()

        return None, grad_output[0] * dlogC_dkappa


class VonMisesFisher(Distribution):
    """
    Von Mises-Fisher distribution on the unit sphere.

    The vMF distribution is a probability distribution on the (d-1)-dimensional
    unit sphere in R^d, parameterized by:
    - mu (loc): mean direction (unit vector)
    - kappa (concentration): concentration parameter (scalar > 0)

    The PDF is:
        p(x | mu, kappa) = C_d(kappa) * exp(kappa * mu^T x)

    where C_d(kappa) is the normalization constant.

    Args:
        loc: Mean direction, shape [..., d] (will be normalized to unit length)
        concentration: Concentration parameter, shape [...] (must be positive)
    """

    def __init__(self, loc, concentration, validate_args=None):
        """
        Initialize the von Mises-Fisher distribution.

        Args:
            loc: Mean direction tensor of shape [..., d]
            concentration: Concentration parameter of shape [...]
            validate_args: Whether to validate arguments
        """
        # Normalize loc to unit length
        self.loc = loc / loc.norm(dim=-1, keepdim=True).clamp(min=1e-10)
        self.concentration = concentration

        # Get dimension
        self.dim = self.loc.shape[-1]

        # Broadcast parameters
        if self.concentration.dim() < self.loc.dim() - 1:
            self.concentration = self.concentration.view(
                self.concentration.shape
                + (1,) * (self.loc.dim() - self.concentration.dim() - 1)
            )

        # Determine batch_shape and event_shape
        batch_shape = self.loc.shape[:-1]
        event_shape = torch.Size([self.dim])

        super(VonMisesFisher, self).__init__(
            batch_shape, event_shape, validate_args=validate_args
        )

    def expand(self, batch_shape, _instance=None):
        """
        Implements expanding the distribution to a new batch_shape.
        """
        new = self._get_checked_instance(VonMisesFisher, _instance)
        batch_shape = torch.Size(batch_shape)

        new.loc = self.loc.expand(batch_shape + (self.dim,))
        new.concentration = self.concentration.expand(batch_shape)
        new.dim = self.dim

        super(VonMisesFisher, new).__init__(
            batch_shape, self.event_shape, validate_args=False
        )
        new._validate_args = self._validate_args

        return new

    def sample(self, sample_shape=torch.Size()):
        """
        Sample from the von Mises-Fisher distribution.

        Uses the rejection sampling algorithm from vmf-lib.
        """
        with torch.no_grad():
            return self._sample_impl(sample_shape) # type: ignore

    def rsample(self, sample_shape=torch.Size()):
        """
        Reparameterizable sampling from vMF distribution.

        Note: This uses the same algorithm as sample() but allows gradients.
        The sampling process itself isn't truly differentiable, but we
        implement this to allow usage in variational inference contexts.
        """
        return self._sample_impl(sample_shape) # type: ignore

    def _sample_impl(self, sample_shape=torch.Size()):
        """
        Implementation of vMF sampling using rejection sampling.

        Based on the algorithm from vmf-lib/models.py
        """
        shape = self._extended_shape(sample_shape)
        N = int(np.prod(sample_shape)) if len(sample_shape) > 0 else 1

        if N == 0:
            return torch.empty(shape, device=self.loc.device, dtype=self.loc.dtype)

        d = self.dim

        # Flatten batch dimensions for easier processing
        loc_flat = self.loc.reshape(-1, d)

        # Handle concentration shape - squeeze out dimension if needed
        if self.concentration.dim() > 1:
            conc_flat = self.concentration.squeeze(-1).reshape(-1)
        else:
            conc_flat = self.concentration.reshape(-1)

        batch_size = loc_flat.shape[0]

        # Ensure concentrations match batch size
        if conc_flat.shape[0] != batch_size:
            raise ValueError(
                f"Concentration batch size {conc_flat.shape[0]} doesn't match loc batch size {batch_size}"
            )

        # Generate samples for each batch element
        all_samples = []

        for b in range(batch_size):
            mu_b = loc_flat[b]
            kappa_b = conc_flat[b].item()

            # Step-1: Sample uniform unit vectors in R^{d-1}
            v = torch.randn(N, d - 1, device=mu_b.device, dtype=mu_b.dtype)
            v = v / v.norm(dim=1, keepdim=True).clamp(min=1e-10)

            # Step-2: Sample v0 using rejection sampling
            kmr = np.sqrt(4 * kappa_b**2 + (d - 1) ** 2)
            bb = (kmr - 2 * kappa_b) / (d - 1)
            aa = (kmr + 2 * kappa_b + d - 1) / 4
            dd = (4 * aa * bb) / (1 + bb) - (d - 1) * np.log(d - 1)

            beta = torch.distributions.Beta(
                torch.tensor(0.5 * (d - 1), device=mu_b.device),
                torch.tensor(0.5 * (d - 1), device=mu_b.device),
            )
            uniform = torch.distributions.Uniform(
                torch.tensor(0.0, device=mu_b.device),
                torch.tensor(1.0, device=mu_b.device),
            )

            v0 = torch.tensor([], device=mu_b.device, dtype=mu_b.dtype)
            rsf = 10  # rejection sampling factor

            while len(v0) < N:
                n_needed = N - len(v0)
                eps = beta.sample([rsf * n_needed]).to(mu_b.device)
                uns = uniform.sample([rsf * n_needed]).to(mu_b.device)
                w0 = (1 - (1 + bb) * eps) / (1 - (1 - bb) * eps)
                t0 = (2 * aa * bb) / (1 - (1 - bb) * eps)
                det = (d - 1) * t0.log() - t0 + dd - uns.log()
                v0 = torch.cat([v0, w0[det >= 0]])
                if len(v0) > N:
                    v0 = v0[:N]
                    break

            v0 = v0.reshape(N, 1)

            # Step-3: Form x = [v0; sqrt(1-v0^2)*v]
            samples_b = torch.cat([v0, (1 - v0**2).sqrt() * v], dim=1)

            # Step-4: Householder transformation
            e1 = torch.zeros(d, 1, device=mu_b.device, dtype=mu_b.dtype)
            e1[0, 0] = 1.0
            e1mu = e1 - mu_b.unsqueeze(1)
            e1mu = e1mu / e1mu.norm(dim=0).clamp(min=1e-10)
            samples_b = samples_b - 2 * (samples_b @ e1mu) @ e1mu.t()

            all_samples.append(samples_b)

        # Stack samples: shape will be [batch_size, N, d]
        samples = torch.stack(all_samples, dim=0)

        # Reshape to match expected output shape
        # Expected shape: sample_shape + batch_shape + event_shape
        target_shape = sample_shape + self.batch_shape + self.event_shape
        samples = samples.permute(1, 0, 2)  # [N, batch_size, d]
        samples = samples.reshape(target_shape)

        return samples

    def log_prob(self, value):
        """
        Calculate the log probability density.

        log p(x | mu, kappa) = log C_d(kappa) + kappa * mu^T x
        """
        if self._validate_args:
            self._validate_sample(value)

        # Normalize value to unit sphere
        value_norm = value / value.norm(dim=-1, keepdim=True).clamp(min=1e-10)

        # Compute dot product: mu^T x
        dotp = (self.loc * value_norm).sum(dim=-1)

        # Compute log partition function
        logC = vMFLogPartition.apply(self.dim, self.concentration)

        # log p(x) = log C_d(kappa) + kappa * mu^T x
        log_prob = logC + self.concentration * dotp

        return log_prob

    @property
    def mean(self):
        """
        The mean direction of the vMF distribution.

        E[x] = A_d(kappa) * mu
        where A_d(kappa) = I_{d/2}(kappa) / I_{d/2-1}(kappa)

        For simplicity, we return the mode (mu) as an approximation.
        """
        # TODO: Implement the exact mean using the ratio of Bessel functions
        print("Warning: mean() currently returns mode (loc) as an approximation.")
        return self.loc

    @property
    def mode(self):
        """
        The mode of the vMF distribution is the mean direction mu.
        """
        return self.loc

    @property
    def variance(self):
        """
        Variance of the vMF distribution.

        This is not straightforward to compute as vMF is on a sphere.
        We return a placeholder based on concentration.
        """
        # Higher concentration = lower variance
        # This is a rough approximation
        return 1.0 / (self.concentration.unsqueeze(-1) + 1e-10)

    def entropy(self):
        """
        Calculate the entropy of the vMF distribution.

        H(x) = -log C_d(kappa) - kappa * A_d(kappa)
        where A_d(kappa) = I_{d/2}(kappa) / I_{d/2-1}(kappa)
        """
        logC = vMFLogPartition.apply(self.dim, self.concentration)

        # Approximate A_d(kappa) for entropy calculation
        # For large kappa: A_d(kappa) ≈ 1 - (d-1)/(2*kappa)
        # For small kappa: A_d(kappa) ≈ kappa/d

        # Use a rough approximation
        A_d = torch.tanh(self.concentration) * (
            1 - (self.dim - 1) / (2 * self.concentration + 1e-10)
        )

        entropy = -logC - self.concentration * A_d # type: ignore

        return entropy
