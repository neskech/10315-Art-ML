from torch import nn

from skeletonFormat import SkeletonFormat


class JointEmbedding(nn.Module):
    """
    Class that outputs joint embeddings to be fed into a
    transformer decoder. The embeddings are used
    for UVD joint position regression
    """

    def __init__(self, skeleton_format: SkeletonFormat, embedding_dimension: int):
        super().__init__()

        self.skeleton_format = skeleton_format

        num_joints = self.skeleton_format.get_joint_count()
        self.embedding_layer = nn.Embedding(num_joints, embedding_dimension)
        nn.init.xavier_uniform_(self.embedding_layer.weight)

    def get_positional_embeddings(self, batch_size: int):
        """
        Generates positional embeddings for a given skeleton format.

        Args:
            batch_size (int): The number of samples in the batch.
                Denoted as B.

        Returns:
            torch.Tensor: Batched positional embeddings of shape (B, N, E),
                where N is the number of joints and E is the embedding
                dimension.
        """
        joint_embedding = self.embedding_layer
        num_joints = self.skeleton_format.get_joint_count()

        # Embeddings for joint positions
        position = joint_embedding.weight[:num_joints]  # (N, E)

        batched_positions = position.unsqueeze(0).repeat(batch_size, 1, 1)  # (B, N, E)
        return batched_positions
