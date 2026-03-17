import torch
from torch import nn


class AttentionPooling(nn.Module):
    """Takes a sequence of shape (B, T, F). Uses a learnable
    query vector to compute multi head attention with the
    sequence. Uses the singular vector produced from the
    attention operation as a 'pooling' of the entire sequence
    """

    def __init__(self, embedding_dimension: int, num_heads: int):
        super().__init__()
        self.query_embedding = nn.Embedding(1, embedding_dimension)
        self.attention = nn.MultiheadAttention(
            embedding_dimension, num_heads, batch_first=True
        )

    def forward(self, x: torch.Tensor):
        """x: Tensor of shape (B, T, F)"""
        batches, timesteps = x.shape[: 2]
        query = self.query_embedding.forward(torch.tensor(0).to(x.device))
        query = query.unsqueeze(0).repeat(batches, timesteps, 1)
        x, _ = self.attention.forward(query, x, x)  # (B, T, F)
        # Every row of x is the same. Pick the first one
        return x[:, 0, :]  # (B, F)


class MeanPooling(nn.Module):

    def forward(self, x: torch.Tensor):
        """Computes the mean across the time dimension.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, F), where
                B is the batch size, T is the sequence length, and F
                is the feature dimension.

        Returns:
            torch.Tensor: Output tensor of shape (B, F), where the
                mean is computed across the time dimension (T).
        """
        return torch.mean(x, dim=1)


class Conv2DSubsampling(torch.nn.Module):
    """Downsamples the time and feature dimensions of a sequence
    of shape (B, T, F). T is the sequence dimension and F is
    the feature dimension. Output of the model will be a new
    Tensor (B, T', F') where T' and F' are the downsampled time
    and feature dimensions
    """

    def __init__(
        self,
        input_dimension: int,
        output_dimension: int,
        time_stride: int,
        feature_stride: int,
        dropout: float,
        activation: nn.Module | None = None
    ):
        """Args:
        input_dimension (int): Feature dimension F
        output_dimension (int): Output feature dimension F'
        time_stride (int): The stride of the time dimension
        feature_stride (int): The stride of the feature dimension
        dropout (float): Dropout for the convolutional layers
        activation (nn.Module | None): Activation function. Defaults to ReLU if None.

        """
        super().__init__()
        self.time_stride = time_stride
        self.feature_stride = feature_stride
        self.output_dimension = output_dimension
        activation = activation or nn.ReLU()

        # Decompose to get effective stride across two layers
        time_stride1, time_stride2 = self._get_factors(time_stride)
        feature_stride1, feature_stride2 = self._get_factors(feature_stride)

        # Create downsample layers
        conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=output_dimension,
            kernel_size=3,
            stride=(time_stride1, feature_stride1)
        )
        conv2 = nn.Conv2d(
            in_channels=output_dimension,
            out_channels=output_dimension,
            kernel_size=3,
            stride=(time_stride2, feature_stride2)
        )
        self.downsample = torch.nn.Sequential(
            conv1,
            activation,
            conv2,
            activation,
        )

        # After we downsample, we'll have a new tensor of shape
        # (B, C, T', F'), where C = output dimension. We'll then
        # reshape to get (B, T', C * F'). Our final linear layer will
        # convert from (C * F') back to the output dimension C
        feature_dimension1 = self._get_conv_output_dimension(
            input_dimension=input_dimension, kernel_size=3, stride=feature_stride1
        )
        feature_dimension2 = self._get_conv_output_dimension(
            input_dimension=feature_dimension1, kernel_size=3, stride=feature_stride2
        )
        conv_output_dim = output_dimension * feature_dimension2

        self.projection = nn.Sequential(
            nn.Linear(conv_output_dim, output_dimension), nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor):
        """Args:
            x (torch.Tensor): Input tensor - shape (B, T, F).

        Returns:
            torch.Tensor: Downsampled output - shape (B, T', F')
                where F' = output_dimension

        """
        B, T = x.shape[: 2]

        # Add a channel dimension
        x = x.unsqueeze(1)  # (B, 1, T, F)
        x = self.downsample.forward(x)  # (B, output_dimension, T, F')
        x = x.transpose(1, 2).contiguous()  # (B, T, output_dimension, F')
        x = x.view(B, T, -1)  # (B, T, output_dimension * F')
        x = self.projection.forward(x)  # (B, T, output_dimension)
        return x

    def get_sequence_dim_after_downsample(self, input_sequence_dimension: int):
        """Calculates the output sequence dimension after downsampling.

        Args:
            input_sequence_dimension (int): The input sequence length (T).

        Returns:
            int: The output sequence length (T') after downsampling.
        """
        time_stride1, time_stride2 = self._get_factors(self.time_stride)
        time_dimension1 = self._get_conv_output_dimension(
            input_dimension=input_sequence_dimension,
            kernel_size=3,
            stride=time_stride1,
            exact=True
        )
        time_dimension2 = self._get_conv_output_dimension(
            input_dimension=time_dimension1,
            kernel_size=3,
            stride=time_stride2,
            exact=True
        )
        return time_dimension2

    def _get_conv_output_dimension(
        self,
        input_dimension: int,
        kernel_size: int,
        stride: int,
        padding: int = 0,
        exact: bool = False
    ):
        """Calculates the output dimension after a convolution operation."""
        numerator = (input_dimension + 2 * padding - kernel_size)

        if exact and numerator % stride != 0:
            raise ValueError(
                f"Convolutional subsampling cannot evenly divide '{input_dimension}' \
                with stride: {stride}, kernel size: {kernel_size}, and padding: {padding}"
            )

        return numerator // stride + 1

    def _get_factors(self, n: int):
        """Returns the closest factors of an integer n

        Args:
            n (int): The number to be factored

        Returns:
            (int, int): Two integers a, b
            such that n = a * b

        """
        # The largest possible factor is sqrt(n)
        factor = int(n**0.5)

        # Keep moving down until we hit a factor
        while n % factor != 0:
            factor -= 1

        # Return the factor pair
        return max(factor, n // factor), min(factor, n // factor)
