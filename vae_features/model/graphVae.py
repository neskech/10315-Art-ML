from torch import nn
import torch

from vae_features.utils.angleFormat import euler_to_6d
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
        use_6d_rotation_format=False,
    ) -> None:
        super().__init__()
        self.use_6d_rotations = use_6d_rotation_format

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
            node_dimension=6 if self.use_6d_rotations else 3,
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

        if self.use_6d_rotations:
            self.decoder_head = nn.Linear(decoder_joint_embedding_dimension, 6)
        else:
            self.decoder_head = nn.Linear(decoder_joint_embedding_dimension, 3)

        self.joint_embedding = JointEmbedding(
            skeleton_format, embedding_dimension=decoder_joint_embedding_dimension
        )

        self.encoder_edge_features = get_edge_features(
            self.encoder_graph, self.skeletonFormat
        )
        self.decoder_edge_features = get_edge_features(
            self.decoder_graph, self.skeletonFormat
        )

    def encode(
        self, joint_angles: torch.Tensor, use_edge_features=True
    ) -> torch.Tensor:
        """Encode into a latent vector

        Args:
            joint_angles (torch.Tensor): Shape (B, N, 3) where
                N is the number of joints

        Returns:
            torch.Tensor: Encoded latent
        """
        if self.use_6d_rotations:
            joint_angles = euler_to_6d(joint_angles)

        joint_encodings, _ = self.encoder.forward_unpadded(
            node_features=joint_angles,
            edge_features=self.encoder_edge_features if use_edge_features else None,
            graph=self.encoder_graph,
            attention_mask=None,
        )
        latent = joint_encodings[self.virtual_node_index]
        latent = latent / (torch.norm(latent, dim=-1) + 1e-6)
        return latent

    def decode(self, latent: torch.Tensor, use_edge_features=True) -> torch.Tensor:
        batch_size = latent.shape[0]
        decoded_sequence, _, _ = self.decoder.forward_unpadded(
            node_features=self.joint_embedding.get_positional_embeddings(batch_size),
            encoder_outputs=latent,
            edge_features=self.decoder_edge_features if use_edge_features else None,
            graph=self.decoder_graph,
            self_attention_mask=None,
            cross_attention_mask=None,
        )
        joint_angles = self.decoder_head(decoded_sequence)
        return joint_angles

    def encode_and_reconstruct(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(x)
        reconstruction = self.decode(latent)
        return latent, reconstruction


def get_edge_features(graph: GraphFormerGraph, skeleton_format: SkeletonFormat):
    """
    Gets edge features, which are FIXED given a specified
    skeleton format. These features contain the normalized
    local offsets from parent to child joints, as well as a
    scalar for the length of that offset vector.

    Args:
        graph (GraphFormerGraph): Provided to get access to virtual nodes
        skeleton_format (SkeletonFormat): The format
            for which we derive the edge features

    Returns:
        Tensor: A feature Tensor of shape (num_edges, 4),
            where num_edges is the number of edges in the skeleton graph.
            Each row corresponds to the features for a single edge,
            in the order of edge indices as defined by the skeleton graph.
    """
    N = skeleton_format.get_joint_count()
    rest_pose = skeleton_format.get_rest_pose()  # (N, 3)

    if rest_pose is None:
        raise ValueError("Rest pose must be defined in the skeleton format")

    num_edges = graph.num_edges()
    features = torch.zeros(num_edges, 4)

    for a, b in skeleton_format.get_graph().get_edges():
        # If its a regular bone in the skeleton
        if a < N and b < N:
            a_vec = rest_pose[a]  # (3,)
            b_vec = rest_pose[b]  # (3,)
            bone_vector = b_vec - a_vec
            bone_length = bone_vector.norm()
            bone_vector = bone_vector / (bone_length + 1e-8)  # Avoid division by zero

            edge_index = skeleton_format.get_graph().get_edge_index((a, b))
            features[edge_index] = torch.cat([bone_vector, bone_length.unsqueeze(0)])
        else:
            # If its a virtual node edge
            features[edge_index] = torch.zeros(4)

    return features
