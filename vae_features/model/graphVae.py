from torch import nn
import torch

from vae_features.utils.feedForward import FeedForward, Norm
from vae_features.utils.graphformer import GraphFormerDecoder, GraphFormerEncoder
from vae_features.utils.graphormerGraph import GraphFormerGraph
from vae_features.utils.jointEmbedding import JointEmbedding
from vae_features.utils.positionalEncoding import PositionalEncoding1D
from vae_features.utils.skeletonFormat import SkeletonFormat


class GraphVAE(nn.Module):
    def __init__(
        self,
        skeleton_format: SkeletonFormat,
        num_layers: int,
        joint_embedding_dimension: int,
        bone_embedding_dimension: int,
        decoder_joint_embedding_dimension: int,
        num_attention_heads: int,
        bottleneck_dimensions: int,
        bottleneck_activation: nn.Module,
        dropout: float,
    ) -> None:
        super().__init__()

        self.skeletonFormat = skeleton_format
        self.encoder_graph = GraphFormerGraph.from_skeleton_format(
            skeleton_format, num_virtual_nodes=1
        )
        self.virtual_node_index = self.skeletonFormat.get_joint_count()

        self.decoder_graph = GraphFormerGraph.from_skeleton_format(
            skeleton_format, num_virtual_nodes=0
        )
        positional_encoding = PositionalEncoding1D(joint_embedding_dimension)

        self.encoder = GraphFormerEncoder(
            num_layers=num_layers,
            node_dimension=3,
            edge_dimension=4,
            node_embedding_dimension=joint_embedding_dimension,
            edge_embedding_dimension=bone_embedding_dimension,
            num_attention_heads=num_attention_heads,
            bottleneck_dimension=bottleneck_dimensions,
            bottleneck_activation=bottleneck_activation,
            positional_encoding=positional_encoding,
            dropout=dropout,
            max_path_length=self.encoder_graph.num_nodes(),
            max_in_degree=self.encoder_graph.num_nodes(),
            max_out_degree=self.encoder_graph.num_nodes(),
        )

        self.decoder = GraphFormerDecoder(
            num_layers=num_layers,
            node_dimension=decoder_joint_embedding_dimension,
            edge_dimension=4,
            node_embedding_dimension=decoder_joint_embedding_dimension,
            edge_embedding_dimension=bone_embedding_dimension,
            num_attention_heads=num_attention_heads,
            bottleneck_dimension=bottleneck_dimensions,
            bottleneck_activation=bottleneck_activation,
            positional_encoding=positional_encoding,
            dropout=dropout,
            max_path_length=self.decoder_graph.num_nodes(),
            max_in_degree=self.decoder_graph.num_nodes(),
            max_out_degree=self.decoder_graph.num_nodes(),
        )

        self.joint_embedding = JointEmbedding(
            skeleton_format, embedding_dimension=decoder_joint_embedding_dimension
        )
        # TODO: Doesn't work with virtual node??
        self.edge_features = get_edge_features(self.skeletonFormat)

    def encode(self, joint_angles: torch.Tensor) -> torch.Tensor:
        """Encode into a latent vector

        Args:
            joint_angles (torch.Tensor): Shape (B, N, 3) where
                N is the number of joints

        Returns:
            torch.Tensor: Encoded latent
        """
        joint_encodings, _ = self.encoder.forward_unpadded(
            node_features=joint_angles,
            edge_features=self.edge_features,
            graph=self.encoder_graph,
            attention_mask=None,
        )
        latent = joint_encodings[self.virtual_node_index]
        return latent

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        
        decoded_sequence = self.decoder.forward_unpadded(
            node_features=self.joint_embedding.get_positional_embeddings(),
            edge_features=self.edge_features,

        )

    def encode_and_reconstruct(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(x)
        reconstruction = self.decode(latent)
        return latent, reconstruction


def get_edge_features(skeleton_format: SkeletonFormat):
    """
    Gets edge features, which are FIXED given a specified
    skeleton format. These features contain the normalized
    local offsets from parent to child joints, as well as a
    scalar for the length of that offset vector.

    Args:
        skeleton_format (SkeletonFormat): The format
            for which we derive the edge features

    Returns:
        Tensor: A feature Tensor of shape (num_edges, 4),
            where num_edges is the number of edges in the skeleton graph.
            Each row corresponds to the features for a single edge,
            in the order of edge indices as defined by the skeleton graph.
    """
    rest_pose = skeleton_format.get_rest_pose()  # (N, 3)

    if rest_pose is None:
        raise ValueError("Rest pose must be defined in the skeleton format")

    num_edges = skeleton_format.get_graph().num_edges()
    features = torch.zeros(num_edges, 4)

    for a, b in skeleton_format.get_graph().get_edges():
        a_vec = rest_pose[a]  # (3,)
        b_vec = rest_pose[b]  # (3,)
        bone_vector = b_vec - a_vec
        bone_length = bone_vector.norm()
        bone_vector = bone_vector / (bone_length + 1e-8)  # Avoid division by zero

        edge_index = skeleton_format.get_graph().get_edge_index((a, b))
        features[edge_index] = torch.cat([bone_vector, bone_length.unsqueeze(0)])

    return features
