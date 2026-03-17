import math

import torch
from torch import nn


class PositionalEncoding1D(torch.nn.Module):
    """
    1D Positional Encoding using sinusoidal functions.

    This class generates sinusoidal positional encodings for 1D input tensors,
    allowing the model to encode sequential information.
    """

    def __init__(self, embedding_dimension: int, max_sequence_length: int = 512):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.max_sequence_length = max_sequence_length

        encoding = torch.zeros(max_sequence_length, embedding_dimension)
        position_indices = torch.arange(0, max_sequence_length).float()
        position_indices = position_indices.unsqueeze(1)  # (maxSeqLength, 1)

        # Divison term for cos and sine functions
        # This term creates a series of values that decrease geometrically,
        # used to generate varying frequencies for positional encodings
        division_term = torch.exp(
            torch.arange(0, embedding_dimension, 2).float() *
            (-math.log(10000.0) / embedding_dimension)
        )  # (embedDim / 2)

        # Multiplication between shape (maxSeqLength, 1) * (embedDim / 2)
        # expands the position tensor, so that we get new shapes of
        # (maxSeqLength, embedDim / 2) * (embedDim / 2)
        encoding[:, 0:: 2] = torch.sin(position_indices * division_term)
        encoding[:, 1:: 2] = torch.cos(position_indices * division_term)

        # Register the encoding as a buffer. This'll put the encoding
        # in the state dict whenever we save this module
        buffered_encoding = encoding.unsqueeze(0)  # (1, maxSeqLength, EmbedDim)
        self.register_buffer("encoding", buffered_encoding)

    def forward(self, x: torch.Tensor):
        """
        Adds the positional encoding to x

        Args:
            x (Tensor): Sequence of shape (B, T, E)

        Returns:
            Tensor: The sequence with the positional encoding
                - Shape (B, T, E)
        """
        T = x.shape[1]

        if T > self.max_sequence_length:
            raise ValueError(
                f"Input sequence length {T} exceeds maximum sequence length "
                f"{self.max_sequence_length}"
            )

        encoded = x + self.encoding[:, : T]  # type: ignore
        return encoded


class BucketedPositionalEncoding1D(torch.nn.Module):
    """
    A special form of positional encoding that encodes 'buckets'
    instead of individual tensors. Each 'bucket' in the sequence
    will get its own positional encoding. For example, given a bucket
    size of 10, each Tensor from [0, 9] will have the same encoding,
    and each tensor from [10, 19] will have a different encoding.
    """

    def __init__(self, embedding_dimension: int, bucket_size: int, max_buckets: int):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.bucket_size = bucket_size
        self.max_buckets = max_buckets
        self.max_sequence_length = bucket_size * max_buckets

        encoding = torch.zeros(self.max_sequence_length, embedding_dimension)

        # Creates an index tensor that buckets indices into sequences.
        # E.g. first 'bucket' indices are 0, next 'bucket' indices are 1, etc
        position_indices = torch.zeros(self.max_sequence_length).long() // bucket_size
        position_indices = position_indices.float().unsqueeze(1)

        # Divison term for cos and sine functions
        # This term creates a series of values that decrease geometrically,
        # used to generate varying frequencies for positional encodings
        division_term = torch.exp(
            torch.arange(0, embedding_dimension, 2).float() *
            (-math.log(10000.0) / embedding_dimension)
        )  # (embedDim / 2)

        # Multiplication between shape (maxSeqLength, 1) * (embedDim / 2)
        # expands the position tensor, so that we get new shapes of
        # (maxSeqLength, embedDim / 2) * (embedDim / 2)
        encoding[:, 0:: 2] = torch.sin(position_indices * division_term)
        encoding[:, 1:: 2] = torch.cos(position_indices * division_term)

        # Register the encoding as a buffer. This'll put the encoding
        # in the state dict whenever we save this module
        buffered_encoding = encoding.unsqueeze(0)  # (1, maxSeqLength, EmbedDim)
        self.register_buffer("encoding", buffered_encoding)

    def forward(self, x: torch.Tensor):
        """
        Adds the positional encoding to x

        Args:
            x (Tensor): Sequence of shape (B, T, E)

        Returns:
            Tensor: The sequence with the positional encoding
                - Shape (B, T, E)
        """
        T = x.shape[1]

        if T > self.max_sequence_length:
            raise ValueError(
                f"Input sequence length {T} exceeds maximum sequence length "
                f"{self.max_sequence_length}"
            )

        encoded = x + self.encoding[:, : T]  # type: ignore
        return encoded


class LearnedPositionalEmbedding1D(nn.Module):
    """
    Learned 1D positional embedding for input tensors.

    This class generates learned positional embeddings for 1D input tensors,
    allowing the model to encode sequential information.
    """

    def __init__(
        self, input_dimension: int, max_input_length: int = 512, dropout: float = 0.0
    ):
        super().__init__()
        self.max_input_length = max_input_length
        self.input_dimension = input_dimension
        self.embed = nn.Embedding(max_input_length, input_dimension)
        torch.nn.init.normal_(self.embed.weight, mean=0, std=0.02)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        """
            Input: x -> Torch.Tensor of shape
                (batch size, sequence length, dimension)
        """
        batch_size, sequence_length, dim = x.shape

        if sequence_length >= self.max_input_length:
            raise ValueError(
                f"Input sequence length {sequence_length} exceeds maximum input "
                f"length {self.max_input_length}"
            )
        if dim != self.input_dimension:
            raise ValueError(
                f"Input dimension {dim} does not match required input dimension "
                f"{self.input_dimension}"
            )

        indices = torch.arange(0, sequence_length)
        positional_encoding = self.embed.forward(indices)
        positional_encoding = positional_encoding.repeat((batch_size, 1, 1, 1))
        return self.dropout(x + positional_encoding)


# TODO: Change this to   https://github.com/s-chh/2D-Positional-Encoding-Vision-Transformer/blob/main/positional_encodings/pos_embed_sinusoidal.py


class LearnedPositionalEmbedding2D(nn.Module):
    """
    Learned 2D positional embedding for input tensors.

    This class generates learned positional embeddings for 2D input tensors,
    allowing the model to encode spatial information in two dimensions.
    """

    def __init__(self, input_dimension: int, max_input_length: int = 512):
        super().__init__()
        self.max_input_length = max_input_length
        self.input_dimension = input_dimension
        self.embed = nn.Embedding(max_input_length, input_dimension)

    def forward(self, x: torch.Tensor):
        """
        Adds the learned 2D positional encoding to the input tensor.

        Args:
            x (Tensor): Input tensor of shape (B, T, E), where B is the batch size,
                T is the sequence length, and E is the embedding dimension.

        Returns:
            Tensor: The input tensor with the learned 2D positional encoding added.
                Shape (B, T, E).
        """
        batch_size, timesteps, dim = x.shape

        if timesteps >= self.max_input_length:
            raise ValueError(
                f"Input sequence length {timesteps} exceeds maximum input "
                f"length {self.max_input_length}"
            )
        if dim != self.input_dimension:
            raise ValueError(
                f"Input dimension {dim} does not match required input "
                f"dimension {self.input_dimension}"
            )

        indices = torch.arange(0, timesteps)
        positional_encoding = self.embed.forward(indices)
        positional_encoding = positional_encoding.repeat((batch_size, 1, 1, 1))
        return x + positional_encoding
