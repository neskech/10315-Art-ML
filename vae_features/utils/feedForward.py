from typing import Literal, TypeAlias

import torch
from torch import nn

Norm: TypeAlias = Literal['layerNorm', 'batchNorm']


class FeedForward(nn.Module):
    """
    A feedforward neural network module with optional residual connections,
    normalization, activation, and dropout layers.
    """

    def __init__(
        self,
        layer_dimensions: list[int],
        dropout: float,
        use_residual: bool,
        activation: nn.Module | None = None,
        normalization: Norm = 'batchNorm',
        use_final_augmentations: bool = True,
        bias: bool = True
    ) -> None:
        """
        Initializes the FeedForward module.

        Args:
            layer_dimensions (list[int]): List of dimensions for each layer
            in the feedforward network.
            dropout (float): Dropout rate to apply after each layer.
            use_residual (bool): Whether to use residual connections.
            activation (nn.Module | None, optional): Activation function to
                use. Defaults to None.
            normalization (Norm, optional): Type of normalization to use
                ('layerNorm' or 'batchNorm'). Defaults to 'batchNorm'.
            use_final_augmentations (bool, optional): Whether to apply
                normalization, activation, and dropout to the final layer.
                Defaults to True.
            bias (bool, optional): Whether to include bias in linear layers.
                Defaults to True.
        """
        super().__init__()
        activation = activation or nn.ReLU()

        layers = []
        for i in range(len(layer_dimensions) - 1):
            dim1 = layer_dimensions[i]
            dim2 = layer_dimensions[i + 1]
            layers.append(nn.Linear(dim1, dim2, bias))

            if i < len(layer_dimensions) - 2 or use_final_augmentations:
                if normalization == 'batchNorm':
                    layers.append(nn.BatchNorm1d(dim2))
                else:
                    layers.append(nn.LayerNorm(dim2))

                layers.append(activation)
                layers.append(nn.Dropout(dropout))

        self.layers = torch.nn.Sequential(*layers)
        self.use_residual = use_residual

    def forward(self, x: torch.Tensor):
        """
        Applies the feedforward layers to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (B, F), where B is the batch size
                and F is the feature dimension.

        Returns:
            torch.Tensor: Output tensor after applying the feedforward layers.
        """
        residual = x
        x = self.layers(x)

        if self.use_residual:
            return x + residual

        return x


class BottleneckFeedForward(nn.Module):
    """
    A bottleneck feedforward neural network module with optional residual connections,
    normalization, activation, and dropout layers.
    """

    def __init__(
        self,
        base_dimension: int,
        bottleneck_dimension: int,
        dropout: float,
        use_residual: bool,
        activation: nn.Module | None = None,
        normalization: Norm = 'batchNorm',
        bias: bool = True
    ) -> None:
        """
        Initializes the BottleneckFeedForward module.

        Args:
            base_dimension (int): Dimension of the input and output features.
            bottleneck_dimension (int): Dimension of the bottleneck layer.
            dropout (float): Dropout rate to apply after the bottleneck layer.
            use_residual (bool): Whether to use residual connections.
            activation (nn.Module | None, optional): Activation function to
                use. Defaults to None.
            normalization (Norm, optional): Type of normalization to use
                ('layerNorm' or 'batchNorm'). Defaults to 'batchNorm'.
            bias (bool, optional): Whether to include bias in linear layers.
                Defaults to True.
        """
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(base_dimension, bottleneck_dimension, bias),
            activation, # type: ignore
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dimension, base_dimension, bias)
        )

        if normalization == 'batchNorm':
            self.normalization = nn.BatchNorm1d(base_dimension)
        else:
            self.normalization = nn.LayerNorm(base_dimension)

        self.use_residual = use_residual

    def forward(self, x: torch.Tensor):
        """
        Applies the bottleneck feedforward layers to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (B, F), where B is the batch size
                and F is the feature dimension.

        Returns:
            torch.Tensor: Output tensor after applying the bottleneck feedforward layers.
        """
        residual = x
        x = self.layers(x)

        if self.use_residual:
            return self.normalization(x + residual)

        return self.normalization(x)
