from torch import nn

from skeletonFormat import SkeletonFormat, SkeletonFormatID


class JointEmbedding(nn.Module):
    """
    Class that outputs joint embeddings to be fed into a
    transformer decoder. The first set of embeddings are used
    for UVD joint position regression. The second set of embeddings
    are used for the twist angles of each joint. The two sets are concatenated
    together for a total sequence length of 2N - 1, where N is the number of joints
    """

    def __init__(self, skeleton_formats: list[SkeletonFormat], embedding_dimension: int):
        super().__init__()

        self.skeleton_formats = skeleton_formats

        self.embedding_layers: dict[SkeletonFormatID, nn.Embedding] = {}
        for i, form in enumerate(skeleton_formats):
            num_joints = form.get_joint_count()
            layer = nn.Embedding(num_joints, embedding_dimension)
            nn.init.xavier_uniform_(layer.weight)
            self.register_module(f"Embedding layer #{i}", layer)

            self.embedding_layers[form.get_identifier()] = layer

        self.type_embeddings = nn.Embedding(2, embedding_dimension)

    def get_positional_embeddings(self, batch_size: int, skeleton_format: SkeletonFormat):
        """
        Generates positional embeddings for a given skeleton format.

        Args:
            batch_size (int): The number of samples in the batch.
                Denoted as B.
            skeleton_format (SkeletonFormat): The skeleton format object
                containing joint information.

        Returns:
            torch.Tensor: Batched positional embeddings of shape (B, N, E),
                where N is the number of joints and E is the embedding
                dimension.
        """
        joint_embedding = self.embedding_layers[skeleton_format.get_identifier()]
        num_joints = skeleton_format.get_joint_count()

        # Embeddings for joint positions
        position = joint_embedding.weight[: num_joints]  # (N, E)
        position = position + self.type_embeddings.weight[0]  # (N, E)

        batched_positions = position.unsqueeze(0).repeat(batch_size, 1, 1)  # (B, N, E)
        return batched_positions

    def get_twist_embeddings(self, batch_size: int, skeleton_format: SkeletonFormat):
        """
        Generates twist embeddings for a given skeleton format.

        Args:
            batch_size (int): The number of samples in the batch.
                Denoted as B.
            skeleton_format (SkeletonFormat): The skeleton format object
                containing joint information.

        Returns:
            torch.Tensor: Batched twist embeddings of shape (B, N - 1, E),
                where N is the number of joints and E is the embedding
                dimension. The root joint (index 0) does not have a twist angle,
                so we only take the embeddings for the remaining joints.
        """
        joint_embedding = self.embedding_layers[skeleton_format.get_identifier()]
        num_joints = skeleton_format.get_joint_count()

        # Embeddings for twist angles
        # The root joint (index 0) does not have a twist angle, so we only
        # take the embeddings for the remaining joints
        twist_angle = joint_embedding.weight[1: num_joints]  # (N - 1, E)
        twist_angle += self.type_embeddings.weight[1]  # (N - 1, E)

        batched_joint_angles = twist_angle.unsqueeze(0).repeat(
            batch_size, 1, 1
        )  # (B, N - 1, E)
        return batched_joint_angles


class MultiViewJointEmbedding(nn.Module):
    """
    Class that outputs joint embeddings to be fed into a
    transformer decoder. The first set of embeddings are used
    for 3D joint position regression. The second set of embeddings
    are used for the joint angles of each joint
    """

    def __init__(
        self, skeleton_formats: list[SkeletonFormat], embedding_dimension: int
    ) -> None:
        super().__init__()

        self.skeleton_formats = skeleton_formats

        self.embedding_layers: dict[SkeletonFormatID, nn.Embedding] = {}
        for i, form in enumerate(skeleton_formats):
            num_joints = form.get_joint_count()
            layer = nn.Embedding(num_joints, embedding_dimension)
            nn.init.xavier_uniform_(layer.weight)
            self.register_module(f"Embedding layer #{i}", layer)

            self.embedding_layers[form.get_identifier()] = layer

        self.type_embeddings = nn.Embedding(2, embedding_dimension)

    def get_positional_embeddings(self, batch_size: int, skeleton_format: SkeletonFormat):
        """
        Generates positional embeddings for a given skeleton format.

        Args:
            batch_size (int): The number of samples in the batch.
                Denoted as B.
            skeleton_format (SkeletonFormat): The skeleton format object
                containing joint information.

        Returns:
            torch.Tensor: Batched positional embeddings of shape (B, N, E),
                where N is the number of joints and E is the embedding
                dimension.
        """
        joint_embedding = self.embedding_layers[skeleton_format.get_identifier()]
        num_joints = skeleton_format.get_joint_count()

        # Embeddings for joint positions
        position = joint_embedding.weight[: num_joints]  # (N, E)
        position = position + self.type_embeddings.weight[0]  # (N, E)

        batched_positions = position.unsqueeze(0).repeat(batch_size, 1, 1)  # (B, N, E)
        return batched_positions

    def get_rotational_embeddings(self, batch_size: int, skeleton_format: SkeletonFormat):
        """
        Generates rotational embeddings for a given skeleton format.

        Args:
            batch_size (int): The number of samples in the batch.
                Denoted as B.
            skeleton_format (SkeletonFormat): The skeleton format object
                containing joint information.

        Returns:
            torch.Tensor: Batched rotational embeddings of shape (B, N, E),
                where N is the number of joints and E is the embedding
                dimension.
        """
        joint_embedding = self.embedding_layers[skeleton_format.get_identifier()]
        num_joints = skeleton_format.get_joint_count()

        # Embeddings for joint angles (NOT TWIST ANGLES)
        joint_angle = joint_embedding.weight[: num_joints]  # (N, E)
        joint_angle += self.type_embeddings.weight[1]  # (N, E)

        batched_joint_angles = joint_angle.unsqueeze(0).repeat(
            batch_size, 1, 1
        )  # (B, N, E)
        return batched_joint_angles
