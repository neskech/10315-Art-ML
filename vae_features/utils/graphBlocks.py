import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from graphormerGraph import GraphFormerGraph


class CentralityEncoding(nn.Module):
    """
    CentralityEncoding is a module that returns an encoding for
    each node in the graph based on its in-degree and out-degree.
    """

    def __init__(self, max_in_degree: int, max_out_degree: int, embedding_dimension: int):
        super().__init__()
        self.max_in_degree = max_in_degree
        self.max_out_degree = max_out_degree
        self.in_degree_embedding = nn.Embedding(max_in_degree + 1, embedding_dimension)
        self.out_degree_embedding = nn.Embedding(max_out_degree + 1, embedding_dimension)

    def forward(self, x: torch.Tensor, graphs: list[GraphFormerGraph]):
        """
        Args:
            x (Tensor): Padded node embeddings - Shape (B, T, Q)
                where T = max(T_i) = max sequence length and Q is the embedding dimension
        Returns:
            Tensor: The node embeddings with centrality encoding -
                Shape (B, T, Q)
        """
        B = x.data.shape[0]

        if B != len(graphs):
            raise ValueError(
                f"Number of graphs {len(graphs)} does not match batch size {B}"
            )

        in_degrees = [graph.get_in_degree_cache().to(x.device) for graph in graphs]
        out_degrees = [graph.get_out_degree_cache().to(x.device) for graph in graphs]

        padded_in_degrees = pad_sequence(in_degrees, batch_first=True)
        padded_out_degrees = pad_sequence(out_degrees, batch_first=True)

        padded_in_degrees = padded_in_degrees.clamp(max=self.max_in_degree)
        padded_out_degrees = padded_out_degrees.clamp(max=self.max_out_degree)

        in_degree_embeddings = self.in_degree_embedding.forward(padded_in_degrees)
        out_degree_embeddings = self.out_degree_embedding.forward(padded_out_degrees)

        return x + in_degree_embeddings + out_degree_embeddings

    def forward_unpadded(self, x: torch.Tensor, graph: GraphFormerGraph):
        """
        Args:
            x (torch.Tensor): Node embeddings - Shape (B, N, Q)
                where N is the number of nodes and Q is the embedding dimension.
            graph (GraphFormerGraph): A single graph object containing node
                and edge information.

        Returns:
            torch.Tensor: Node embeddings with centrality encoding - Shape (B, N, Q)
        """
        B = x.shape[0]

        in_degrees = graph.get_in_degree_cache().to(x.device)  # (N, )
        in_degrees = in_degrees.unsqueeze(0).repeat(B, 1)  # (B, N)

        out_degrees = graph.get_out_degree_cache().to(x.device)  # (N, )
        out_degrees = out_degrees.unsqueeze(0).repeat(B, 1)  # (B, N)

        in_degrees = in_degrees.clamp(max=self.max_in_degree)
        out_degrees = out_degrees.clamp(max=self.max_out_degree)

        # (B, N) for both
        in_degree_embeddings = self.in_degree_embedding.forward(in_degrees)
        out_degree_embeddings = self.out_degree_embedding.forward(out_degrees)

        return x + in_degree_embeddings + out_degree_embeddings


class SpatialEncoding(nn.Module):
    """
    SpatialEncoding is a module the returns an encoding
    for each pair of nodes in the graph. The encoding is based on the
    path length between the nodes.
    """

    def __init__(self, max_path_distance: int):
        super().__init__()
        # +1 because we reserve the last index for unreachable paths.
        # And then another +1 because there are N nodes in a path of length N-1,
        # and we need an encoding for each node in a path.
        self.max_path_distance = max_path_distance + 2
        self.distance_embedding = nn.Embedding(max_path_distance, 1)

    def forward(self, x: torch.Tensor, graphs: list[GraphFormerGraph]):
        """
        Args:
            x (torch.Tensor): Padded node embeddings - Shape (B, T, Q)
                where T = max(T_i) = max sequence length and Q is the embedding dimension
        Returns:
            Tensor: A matrix of distance encodings for each pairwise path
                between nodes - shape (B, T, T)
        """
        # T = The max sequence length
        B, T, _ = x.shape

        # Fill encoding with 'unreachable' encodings. (1,) -> (B, T, T)
        non_existant_path_encoding = self.distance_embedding.weight[-1]
        spatial_encoding = non_existant_path_encoding.view(1, 1, 1).repeat(B, T, T)
        spatial_encoding = spatial_encoding.to(x.device)

        for batch_idx, graph in enumerate(graphs):

            # (numPaths,) and (numPaths, 2) tensors
            path_distance_map, edge_index_map = graph.get_path_distance_cache()
            path_distance_map = path_distance_map.to(x.device)  # (numPaths,)
            edge_index_map = edge_index_map.to(x.device)  # (numPaths,

            # (numPaths, 1) and (numPaths,) tensors
            distance_embeddings = self.distance_embedding.weight[path_distance_map]
            distance_embeddings = distance_embeddings.squeeze(-1)

            node1_indices = edge_index_map[:, 0]  # (numPaths,)
            node2_indices = edge_index_map[:, 1]  # (numPaths,)
            spatial_encoding[batch_idx, node1_indices,
                             node2_indices] = distance_embeddings

        return spatial_encoding

    def forward_unpadded(self, x: torch.Tensor, graph: GraphFormerGraph):
        """
        Args:
            x (torch.Tensor): Node embeddings - Shape (B, N, Q)
                where N is the number of nodes and Q is the embedding dimension.
            graph (GraphFormerGraph): A single graph object containing node
                and edge information.

        Returns:
            torch.Tensor: A matrix of distance encodings for each pairwise path
                between nodes - shape (B, N, N)
        """

        B, T, _ = x.shape

        # Fill encoding with 'unreachable' encodings. (1,) -> (B, T, T)
        non_existant_path_encoding = self.distance_embedding.weight[-1]
        spatial_encoding = non_existant_path_encoding.view(1, 1, 1).repeat(B, T, T)
        spatial_encoding = spatial_encoding.to(x.device)

        # (numPaths,) and (numPaths, 2) tensors
        path_distance_map, edge_index_map = graph.get_path_distance_cache()
        path_distance_map = path_distance_map.to(x.device)  # (numPaths,)
        edge_index_map = edge_index_map.to(x.device)  # (numPaths,

        path_distance_map = path_distance_map.expand(B, -1)  # (B, numPaths)
        distance_embeddings = self.distance_embedding.weight[path_distance_map]
        distance_embeddings = distance_embeddings.squeeze(-1)  # (B, numPaths)

        node1_indices = edge_index_map[:, 0]  # (numPaths,)
        node2_indices = edge_index_map[:, 1]  # (numPaths,)
        spatial_encoding[:, node1_indices, node2_indices] = distance_embeddings

        return spatial_encoding


class EdgeEncoding(nn.Module):
    """
    EdgeEncoding is a module that returns an encoding for each pair of nodes in the graph.
    The encoding is based on the edges between each pair of nodes, and the corresponding
    features for each one of those edges.
    """

    def __init__(self, max_path_length: int, embedding_dimension: int):
        super().__init__()
        self.max_path_length = max_path_length
        self.embedding_dimension = embedding_dimension
        self.edge_weights = nn.Embedding(max_path_length, embedding_dimension)

    def _get_weight_indices(
        self,
        path_lengths: torch.Tensor,
        max_paths: int,
        device: torch.device,
        max_path_length: int | None = None
    ):
        """
        The edge encoding assigns a weight vector to each edge in a path. The edge's
        index in the path determines which weight vector to use. So, for example, the
        path [e_1, e_2, e_3] would be assigned the weight vectors [w_1, w_2, w_3].

        To do this operation, we need to collect a sequence of indices for each edge
        in the path. These indices will be used to index into the edge_weights
        embedding.

        Returns:
            Index tensor of shape (B, max_paths, max_path_length) where each entry
                is in the range [0, max_path_length], inclusive. This is because we
                reserve index 0 as being 'invalid'. This is necessary because the
                'max_path_length' dimension is padded to the maximum path length, and
                we want to ensure that invalid indices do not contribute to the dot
                product with the edge features.
        """
        B, _ = path_lengths.shape
        max_path_length = (
            self.max_path_length if max_path_length is None else max_path_length
        )

        # (B * max_paths, 1)
        counts = path_lengths.flatten().unsqueeze(1)
        # (1, max_path_length)
        ranges = torch.arange(1, max_path_length + 1)
        ranges = ranges.unsqueeze(0).to(device)

        # The mask zeros out the indices that are greater than the path length.
        # This is because we reserve index 0 as being 'invalid'. In the end, index
        # 0 will correspond to the 0 vector in the edge_weights embedding, which
        # will ensure that invalid indices do not contribute to the dot product with
        # the edge features.
        # (B * max_paths, max_path_length)
        mask = ranges <= counts

        expanded_ranges = ranges.expand(B * max_paths, -1)
        # (B * max_paths, max_path_length)
        weight_indices = expanded_ranges * mask
        # (B, max_paths, max_path_length)
        weight_indices = weight_indices.reshape(B, max_paths, max_path_length)
        weight_indices = weight_indices.long()

        return weight_indices

    def forward(
        self, edge_features: torch.Tensor, graphs: list[GraphFormerGraph], max_nodes: int
    ):
        """
        Args:
            edge_features (torch.Tensor): Edge features - Shape (B, numEdges', E),
                where numEdges' = max number of edges in any of the input graphs,
                and E is the embedding dimension of each feature
        Returns:
            Tensor: Shape (B, T, T) containing the scalar edge features,
                where T = max number of nodes in any of the input graphs.
        """
        B, _, E = edge_features.shape
        device = edge_features.device

        max_paths = max([g.get_path_distance_cache()[0].shape[0] for g in graphs])
        selected_edge_features = torch.zeros(B, max_paths, self.max_path_length,
                                             E).to(device)
        path_lengths = torch.zeros(B, max_paths).long().to(device)
        batched_index_to_node_pair = torch.zeros(B, max_paths, 2).long().to(device)

        for batch_idx, graph in enumerate(graphs):
            # (numPaths_i,) and (numPaths_i, 2)
            path_distance_map, index_to_node_pair = graph.get_path_distance_cache()
            path_distance_map = path_distance_map.to(device)
            index_to_node_pair = index_to_node_pair.to(device)

            num_paths = path_distance_map.shape[0]
            path_lengths[batch_idx, : num_paths] = path_distance_map
            batched_index_to_node_pair[batch_idx, : num_paths, :] = index_to_node_pair

            # (numPaths_i, max_path_length_i) and (numPaths_i, 2)
            path_cache, _ = graph.get_path_cache()
            path_cache = path_cache.to(device)
            num_paths, max_path_len = path_cache.shape
            selected_edge_features[
                batch_idx, : num_paths, : max_path_len, :] = edge_features[batch_idx,
                                                                           path_cache]

        # (B, max_paths, max_path_length)
        weight_indices = self._get_weight_indices(path_lengths, max_paths, device)

        # Due to padding, we denote invalid indices as '0' and
        # valid indices as being in the range [1...max_path_length].
        # In order for invalid indices to have a 0 dot product, we
        # make a new weight matrix that has index 0 as the 0 vector
        new_edge_weights = torch.cat([
            torch.zeros(1, self.embedding_dimension).to(edge_features.device),
            self.edge_weights.weight
        ])
        # (B, max_paths, max_path_length, E)
        weights = new_edge_weights[weight_indices]

        # (B, max_paths, max_path_length)
        dot_product = (selected_edge_features * weights).sum(dim=-1)

        # Prevent divide by 0
        path_lengths = path_lengths.clamp(min=1)
        # (B, max_paths)
        mean = dot_product.sum(dim=-1) / path_lengths

        # (B, max_nodes, max_nodes)
        edge_encoding = torch.zeros(B, max_nodes, max_nodes).to(device)

        # Get indices
        # (B, max_paths)
        batch_indices = torch.arange(B).view(B, 1).expand(B, max_paths)
        batch_indices = batch_indices.to(device)
        node1_indices = batched_index_to_node_pair[:, :, 0]  # (B, max_paths)
        node2_indices = batched_index_to_node_pair[:, :, 1]  # (B, max_paths)

        # Flatten all the index tensors and mean
        batch_flat = batch_indices.reshape(-1)
        node1_flat = node1_indices.reshape(-1)
        node2_flat = node2_indices.reshape(-1)
        mean_flat = mean.reshape(-1)

        edge_encoding[batch_flat, node1_flat, node2_flat] = mean_flat

        return edge_encoding

    def forward_unpadded(
        self, edge_features: torch.Tensor, graph: GraphFormerGraph, max_nodes: int
    ):
        """
        Args:
            edge_features (torch.Tensor): Edge features - Shape (B, numEdges', E),
                where numEdges' = max number of edges in the input graph,
                and E is the embedding dimension of each feature.
            graph (GraphFormerGraph): A single graph object containing node
                and edge information.
            max_nodes (int): Maximum number of nodes in the graph.

        Returns:
            torch.Tensor: Shape (B, N, N) containing the scalar edge features,
                where N = max number of nodes in the input graph.
        """
        B = edge_features.shape[0]
        device = edge_features.device

        # (numPaths,) and (numPaths, 2)
        path_distance_map, index_to_node_pair = graph.get_path_distance_cache()
        path_distance_map = path_distance_map.to(device)
        index_to_node_pair = index_to_node_pair.to(device)

        # (numPaths, max_path_length)
        path_cache, _ = graph.get_path_cache()
        path_cache = path_cache.to(device)

        # (B, numPaths)
        path_lengths = path_distance_map.unsqueeze(0).expand(B, -1)
        # (B, numPaths, 2)
        index_to_node_pair = index_to_node_pair.unsqueeze(0).expand(B, -1, -1)

        # (B, numPaths, max_path_length, E)
        # The edge features for each batch may not be the same, but the indices
        # used to index into them are
        selected_edge_features = edge_features[:, path_cache]

        num_paths = path_cache.shape[0]

        # (B, max_paths, max_path_length)
        max_path_length = path_cache.shape[1]
        weight_indices = self._get_weight_indices(
            path_lengths, num_paths, device, max_path_length
        )

        # Due to padding, we denote invalid indices as '0' and
        # valid indices as being in the range [1...max_path_length].
        # In order for invalid indices to have a 0 dot product, we
        # make a new weight matrix that has index 0 as the 0 vector
        new_edge_weights = torch.cat([
            torch.zeros(1, self.embedding_dimension).to(edge_features.device),
            self.edge_weights.weight
        ])
        # (B, max_paths, max_path_length, E)
        weights = new_edge_weights[weight_indices]

        # (B, max_paths, max_path_length)
        dot_product = (selected_edge_features * weights).sum(dim=-1)

        # (B, max_paths)
        path_lengths = path_lengths.clamp(min=1)
        mean = dot_product.sum(dim=-1) / path_lengths

        # (B, max_nodes, max_nodes)
        edge_encoding = torch.zeros(B, max_nodes, max_nodes).to(device)

        # Get indices
        # (B, max_paths)
        batch_indices = torch.arange(B).view(B, 1).expand(B, num_paths)
        batch_indices = batch_indices.to(device)
        node1_indices = index_to_node_pair[:, :, 0]  # (B, max_paths)
        node2_indices = index_to_node_pair[:, :, 1]  # (B, max_paths)

        # Flatten all the index tensors and mean
        batch_flat = batch_indices.reshape(-1)
        node1_flat = node1_indices.reshape(-1)
        node2_flat = node2_indices.reshape(-1)
        mean_flat = mean.reshape(-1)

        edge_encoding[batch_flat, node1_flat, node2_flat] = mean_flat

        return edge_encoding


class MultiHeadGraphAttention(nn.Module):
    """
    MultiHeadGraphAttention is a module that computes multi-head self-attention
    with additional spatial, centrality, and edge encodings for graph-based data.
    """

    def __init__(
        self,
        embedding_dimension: int,
        num_heads: int,
        attention_dropout: float = 0.0,
        dropout: float = 0.0
    ):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.num_heads = num_heads
        self.key_dimension = embedding_dimension // num_heads
        self.value_dimension = embedding_dimension // num_heads

        self.query_proj = nn.Linear(embedding_dimension, num_heads * self.key_dimension)
        self.key_proj = nn.Linear(embedding_dimension, num_heads * self.key_dimension)
        self.value_proj = nn.Linear(embedding_dimension, num_heads * self.value_dimension)

        nn.init.normal_(
            self.query_proj.weight,
            mean=0,
            std=np.sqrt(2.0 / (embedding_dimension + self.key_dimension))
        )
        nn.init.normal_(
            self.key_proj.weight,
            mean=0,
            std=np.sqrt(2.0 / (embedding_dimension + self.key_dimension))
        )
        nn.init.normal_(
            self.value_proj.weight,
            mean=0,
            std=np.sqrt(2.0 / (embedding_dimension + self.value_dimension))
        )

        self.final_projection = nn.Linear(
            num_heads * self.value_dimension, embedding_dimension
        )
        torch.nn.init.normal_(self.final_projection.weight)

        self.attention_dropout = nn.Dropout(attention_dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        spatial_encoding: torch.Tensor,
        edge_encoding: torch.Tensor | None,
        key_padding_mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Computes multi head attention, but with the additional information
        of path lengths (spatial encoding) and edge features (edge encoding).
        Note that this class can only be used with self attention. We assume
        that the key, query, and value matrices are all the same tensor.

        If the model is in training mode, it uses the optimized forward pass.
        Otherwise, it uses the unoptimized forward pass and returns attention weights.

        Args:
            query (torch.Tensor): Query vectors - Shape (B, T, D)
            key (torch.Tensor): Key vectors - Shape (B, T, D)
            value (torch.Tensor): Value vectors - Shape (B, T, D)
            spatial_encoding (torch.Tensor): Spatial encoding - Shape (B, T, T)
            edge_encoding (Optional[torch.Tensor]): Edge encoding - Shape (B, T, T)
            key_padding_mask (Optional[torch.Tensor], optional): Padding mask for the
                keys. Bool tensor of shape (B, T). Defaults to None.
            attention_mask (Optional[torch.Tensor], optional): Attention mask.
                Bool tensor of shape (B, T, T). Defaults to None.
        Returns:
            (Tensor, Optional[Tensor]):

               1) Attention output - Shape (B, T, D)
               2) Attention weights - Shape (T, T) or None if not returned
        """

        if self.training:
            return self.optimized_forward(
                query,
                key,
                value,
                spatial_encoding,
                edge_encoding,
                key_padding_mask,
                attention_mask
            )

        return self.unoptimized_forward(
            query,
            key,
            value,
            spatial_encoding,
            edge_encoding,
            key_padding_mask,
            attention_mask
        )

    def optimized_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        spatial_encoding: torch.Tensor,
        edge_encoding: torch.Tensor | None,
        key_padding_mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None
    ):
        """Optimized forward pass for multi-head graph attention.

        See forward() for details.
        """
        B, T, _ = query.shape

        # 1. Project and reshape q, k, v
        q = self.query_proj(query).view(B, T, self.num_heads,
                                        self.key_dimension).transpose(1, 2)
        k = self.key_proj(key).view(B, T, self.num_heads,
                                    self.key_dimension).transpose(1, 2)
        v = self.value_proj(value).view(B, T, self.num_heads,
                                        self.value_dimension).transpose(1, 2)

        # 2. Construct the attention bias mask
        # This combines all masks and encodings
        # that are added to the pre-softmax attention scores
        attn_bias = spatial_encoding
        if edge_encoding is not None:
            attn_bias = attn_bias + edge_encoding

        # Unsqueeze to make it broadcastable with the (B, num_heads, T, T)
        # attention matrix
        attn_bias = attn_bias.unsqueeze(1)

        if attention_mask is not None:
            if not attention_mask.dtype == torch.bool:
                raise ValueError('Attention mask must be bool tensor')
            attention_mask = attention_mask.to(query.device)

            # F.scaled_dot_product_attention expects a bool mask where True means IGNORE.
            # The current mask needs to be broadcastable to (B, num_heads, T, T)
            attn_bias = attn_bias.masked_fill(attention_mask.unsqueeze(1), float("-inf"))

        if key_padding_mask is not None:
            if not key_padding_mask.dtype == torch.bool:
                raise ValueError('Padding mask must be bool tensor')
            key_padding_mask = key_padding_mask.to(query.device)

            # Transform into shape (B, 1, T, T) so it can be broadcasted
            # to (B, num_heads, T, T)
            key_padding_mask = key_padding_mask.view(B, 1, 1, T)
            key_padding_mask = key_padding_mask.expand(-1, -1, T, -1)
            attn_bias = attn_bias.masked_fill(key_padding_mask, float('-inf'))

        # 3. Use the fused attention function
        # Note: `attn_mask` in this function is the combined bias/mask tensor
        output = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_bias,
            dropout_p=self.attention_dropout.p if self.training else 0.0,
            is_causal=False  # Assuming this is not causal attention
        )

        # 4. Final projection
        output = output.transpose(1, 2).contiguous().view(B, T, -1)
        output: torch.Tensor = self.dropout(self.final_projection(output))

        return output, None  # attn_weights are not returned by the fused implementation

    def unoptimized_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        spatial_encoding: torch.Tensor,
        edge_encoding: torch.Tensor | None,
        key_padding_mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None
    ):
        """Unoptimized forward pass for multi-head graph attention.

        See forward() for details.
        """
        # Get the batch size and sequence length
        B, T = query.shape[: 2]

        if key.shape[: 2] != (B, T) or value.shape[: 2] != (B, T):
            raise ValueError(
                f"Key and value tensors must have shape (B, T, D), but got key: "
                f"{key.shape} and value: {value.shape}"
            )

        # 1. Project and reshape q, k, v
        q = self.query_proj(query).view(B, T, self.num_heads,
                                        self.key_dimension).transpose(1, 2)
        k = self.key_proj(key).view(B, T, self.num_heads,
                                    self.key_dimension).transpose(1, 2)
        v = self.value_proj(value).view(B, T, self.num_heads,
                                        self.value_dimension).transpose(1, 2)

        # 2. Construct the attention bias mask
        # This combines all masks and encodings that are added to the
        # pre-softmax attention scores
        attn_bias = spatial_encoding
        if edge_encoding is not None:
            attn_bias = attn_bias + edge_encoding

        # Unsqueeze to make it broadcastable with the (B, num_heads, T, T)
        # attention matrix
        attn_bias = attn_bias.unsqueeze(1)

        if attention_mask is not None:
            if not attention_mask.dtype == torch.bool:
                raise ValueError('Padding mask must be bool tensor')
            attention_mask = attention_mask.to(query.device)

            # We expect a bool mask where True means IGNORE.
            # The current mask needs to be broadcastable to (B, num_heads, T, T)
            attn_bias = attn_bias.masked_fill(attention_mask.unsqueeze(1), float("-inf"))

        if key_padding_mask is not None:
            if not key_padding_mask.dtype == torch.bool:
                raise ValueError('Padding mask must be bool tensor')
            key_padding_mask = key_padding_mask.to(query.device)

            # Transform into shape (B, 1, T, T) so it can be broadcasted
            # to (B, num_heads, T, T)
            key_padding_mask = key_padding_mask.view(B, 1, 1, T)
            key_padding_mask = key_padding_mask.expand(-1, -1, T, -1)
            attn_bias = attn_bias.masked_fill(key_padding_mask, float('-inf'))

        # 3. Compute the attention scores
        attn = (q @ k.transpose(-2, -1)) / (q.shape[-1]**0.5)
        attn = attn + attn_bias
        attn = torch.softmax(attn, dim=-1)
        attn = self.attention_dropout(attn)
        output = attn @ v

        # 4. Compute the final output
        output = output.transpose(1, 2).contiguous().view(B, T, -1)
        output = self.dropout.forward(self.final_projection(output))

        attn_weights: torch.Tensor = attn.mean(dim=(0, 1))

        return output, attn_weights
