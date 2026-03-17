import torch
from torch import nn

from feedForward import BottleneckFeedForward
from graphBlocks import (
    CentralityEncoding,
    EdgeEncoding,
    MultiHeadGraphAttention,
    SpatialEncoding,
)
from graphormerGraph import GraphFormerGraph


class GraphFormerDecoderLayer(nn.Module):
    """
    The GraphFormerDecoderLayer implements a single layer of the GraphFormer decoder.
    It performs self-attention, cross-attention with encoder outputs, and a feedforward
    bottleneck operation, all with residual connections and layer normalization.
    """

    def __init__(
        self,
        embedding_dimension: int,
        num_attention_heads: int,
        bottleneck_dimension: int,
        bottleneck_activation: nn.Module,
        dropout: float
    ):
        super().__init__()

        self.self_attention = MultiHeadGraphAttention(
            embedding_dimension, num_attention_heads, dropout, dropout
        )

        self.cross_attention = nn.MultiheadAttention(
            embedding_dimension, num_attention_heads, dropout=dropout, batch_first=True
        )

        self.bottleneck = BottleneckFeedForward(
            embedding_dimension,
            bottleneck_dimension,
            dropout,
            use_residual=True,
            activation=bottleneck_activation,
            normalization='layerNorm'
        )

        self.normalization1 = torch.nn.LayerNorm(embedding_dimension)
        self.normalization2 = torch.nn.LayerNorm(embedding_dimension)
        self.normalization3 = torch.nn.LayerNorm(embedding_dimension)
        self.normalization4 = torch.nn.LayerNorm(embedding_dimension)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)
        self.dropout3 = torch.nn.Dropout(dropout)

    def forward(
        self,
        targets: torch.Tensor,
        encoder_outputs: torch.Tensor,
        spatial_encoding: torch.Tensor,
        edge_encoding: torch.Tensor | None,
        self_padding_mask: torch.Tensor | None = None,
        cross_padding_mask: torch.Tensor | None = None,
        self_attention_mask: torch.Tensor | None = None,
        cross_attention_mask: torch.Tensor | None = None
    ):
        """
        Performs a forward pass through the GraphFormerDecoderLayer.

        Args:
            targets (torch.Tensor): Input target features of shape
                (B, T, embedding_dimension), where T is the sequence length.
            encoder_outputs (torch.Tensor): Encoder output features of shape
                (B, T_enc, embedding_dimension).
            spatial_encoding (torch.Tensor): Spatial encoding tensor of shape
                (B, T, embedding_dimension).
            edge_encoding (Optional[torch.Tensor]): Optional edge encoding tensor
                of shape (B, T, embedding_dimension).
            self_padding_mask (Optional[torch.Tensor], optional): Mask for
                self-attention padding, shape (B, T).
            cross_padding_mask (Optional[torch.Tensor], optional): Mask for
                cross-attention padding, shape (B, T_enc).
            self_attention_mask (Optional[torch.Tensor], optional): Mask for
                self-attention, shape (B, T, T).
            cross_attention_mask (Optional[torch.Tensor], optional): Mask for
                cross-attention, shape (B, T, T_enc).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - Output tensor after decoding, shape (B, T, embedding_dimension).
            - Self-attention weights, shape (B, num_heads, T, T).
            - Cross-attention weights, shape (B, num_heads, T, T_enc).
        """
        x = targets
        x = self.normalization1(x)
        residual = x

        x, self_attention_weights = self.self_attention.forward(
            x,
            x,
            x,
            spatial_encoding,
            edge_encoding,
            key_padding_mask=self_padding_mask,
            attention_mask=self_attention_mask
        )
        x = self.dropout1(x)
        x += residual
        x = self.normalization2(x)
        residual = x

        x, cross_attention_weights = self.cross_attention.forward(
            x,
            encoder_outputs,
            encoder_outputs,
            key_padding_mask=cross_padding_mask,
            attn_mask=cross_attention_mask
        )
        x = self.dropout2(x)
        x += residual
        x = self.normalization3(x)
        x = self.bottleneck.forward(x)

        return x, self_attention_weights, cross_attention_weights


class GraphFormerDecoder(torch.nn.Module):
    """
    The GraphFormerDecoder implements a transformer-style decoder for
    graph-structured data. It supports both padded and unpadded batches
    of graphs, and incorporates spatial, centrality, and edge encodings
    to enhance the attention mechanism. The decoder is composed of
    multiple GraphFormerDecoderLayer modules, each performing
    self-attention, cross-attention with encoder outputs, and feedforward
    processing. This class is designed for use in graph-to-graph or
    graph-to-sequence tasks where rich relational information between
    nodes and edges is important.
    """

    def __init__(
        self,
        num_layers: int,
        node_dimension: int,
        edge_dimension: int,
        node_embedding_dimension: int,
        edge_embedding_dimension: int,
        num_attention_heads: int,
        bottleneck_dimension: int,
        bottleneck_activation: nn.Module,
        positional_encoding: nn.Module,
        dropout: float,
        max_path_length: int,
        max_in_degree: int,
        max_out_degree: int
    ):

        super().__init__()
        self.node_embedding_dimension = node_embedding_dimension
        self.edge_embedding_dimension = edge_embedding_dimension
        self.positional_encoding = positional_encoding

        if node_dimension == node_embedding_dimension:
            self.node_projection = nn.Identity()
        else:
            self.node_projection = nn.Linear(node_dimension, node_embedding_dimension)

        if edge_dimension == edge_embedding_dimension:
            self.edge_projection = nn.Identity()
        else:
            self.edge_projection = nn.Linear(edge_dimension, edge_embedding_dimension)

        self.spatial_encoding = SpatialEncoding(max_path_length)
        self.centrality_encoding = CentralityEncoding(
            max_in_degree, max_out_degree, self.node_embedding_dimension
        )
        self.edge_encoding = EdgeEncoding(max_path_length, self.edge_embedding_dimension)

        decoder_layers = []
        for _ in range(num_layers):
            decoder_layers.append(
                GraphFormerDecoderLayer(
                    node_embedding_dimension,
                    num_attention_heads,
                    bottleneck_dimension,
                    bottleneck_activation,
                    dropout
                )
            )

        self.decoder_layers = nn.ModuleList(decoder_layers)

    def forward(
        self,
        node_features: torch.Tensor,
        encoder_outputs: torch.Tensor,
        edge_features: torch.Tensor | None,
        graphs: list[GraphFormerGraph],
        self_padding_mask: torch.Tensor | None = None,
        cross_padding_mask: torch.Tensor | None = None,
        self_attention_mask: torch.Tensor | None = None,
        cross_attention_mask: torch.Tensor | None = None
    ):
        """
        Performs a forward pass through the GraphFormer decoder. This method
        supports batched graphs with padding.

        Args:
            node_features (torch.Tensor): Input node features of shape
                (B, T, node_feature_dim), where T is the max number of nodes
                in the batch.
            encoder_outputs (torch.Tensor): Encoder output features of shape
                (B, T_enc, node_embedding_dimension).
            edge_features (Optional[torch.Tensor]): Optional edge features of
                shape (B, num_edges, edge_feature_dim), where num_edges is the
                max number of edges in the batch.
            graphs (List[GraphFormerGraph]): List of graph objects
                corresponding to each sample in the batch.
            self_padding_mask (Optional[torch.Tensor], optional): Optional
                mask indicating padded nodes in the decoder, shape (B, T).
            cross_padding_mask (Optional[torch.Tensor], optional): Optional
                mask indicating padded nodes in the encoder, shape (B, T_enc).
            self_attention_mask (Optional[torch.Tensor], optional): Optional
                mask for self-attention, shape (B, T, T).
            cross_attention_mask (Optional[torch.Tensor], optional): Optional
                mask for cross-attention, shape (B, T, T_enc).
        Returns:
            Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
            - Output node representations after decoding, shape
              (B, T, node_embedding_dimension).
            - List of self-attention weights from each decoder layer.
            - List of cross-attention weights from each decoder layer.
        """
        node_features = self.node_projection(node_features)
        node_features = self.positional_encoding.forward(node_features)
        node_features = self.centrality_encoding.forward(node_features, graphs)

        spatial_encoding = self.spatial_encoding.forward(node_features, graphs)

        # Max nodes is the padded dimension of the node features
        # Node features is shape (B, max_nodes, node_embedding_dimension)
        max_nodes = node_features.shape[1]

        edge_encoding = None
        if edge_features is not None:
            edge_features = self.edge_projection.forward(edge_features)
            edge_encoding = self.edge_encoding.forward(edge_features, graphs, max_nodes)

        self_attention_weights = []
        cross_attention_weights = []
        x = node_features
        for layer in self.decoder_layers:
            x, self_attn_weights, cross_attn_weights = layer.forward(
                x,
                encoder_outputs,
                spatial_encoding,
                edge_encoding,
                self_padding_mask,
                cross_padding_mask,
                self_attention_mask,
                cross_attention_mask
            )
            self_attention_weights.append(self_attn_weights)
            cross_attention_weights.append(cross_attn_weights)

        return x, self_attention_weights, cross_attention_weights

    def forward_unpadded(
        self,
        node_features: torch.Tensor,
        encoder_outputs: torch.Tensor,
        edge_features: torch.Tensor | None,
        graph: GraphFormerGraph,
        self_attention_mask: torch.Tensor | None = None,
        cross_attention_mask: torch.Tensor | None = None
    ):
        """
        Performs a forward pass through the GraphFormer decoder without
        padding. This is used when all graphs in the batch have the same
        number of nodes and edges.

        Args:
            node_features (torch.Tensor): Input node features of shape
                (B, N, node_feature_dim), where N is the number of nodes.
            encoder_outputs (torch.Tensor): Encoder output features of shape
                (B, N_enc, node_embedding_dimension).
            edge_features (Optional[torch.Tensor]): Optional edge features of
                shape (B, num_edges, edge_feature_dim), where num_edges is the
                number of edges.
            graph (GraphFormerGraph): The graph object corresponding to the
                batch.
            self_attention_mask (Optional[torch.Tensor], optional): Optional
                mask for self-attention, shape (B, N, N).
            cross_attention_mask (Optional[torch.Tensor], optional): Optional
                mask for cross-attention, shape (B, N, N_enc).
        Returns:
            Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
            - Output node representations after decoding, shape
              (B, N, node_embedding_dimension).
            - List of self-attention weights from each decoder layer.
            - List of cross-attention weights from each decoder layer.
        """
        node_features = self.node_projection(node_features)
        node_features = self.positional_encoding.forward(node_features)
        node_features = self.centrality_encoding.forward_unpadded(node_features, graph)

        spatial_encoding = self.spatial_encoding.forward_unpadded(node_features, graph)

        # Node features is shape (B, N, node_embedding_dimension)
        max_nodes = node_features.shape[1]

        edge_encoding = None
        if edge_features is not None:
            edge_features = self.edge_projection.forward(edge_features)
            edge_encoding = self.edge_encoding.forward_unpadded(
                edge_features, graph, max_nodes
            )

        self_attention_weights = []
        cross_attention_weights = []
        x = node_features
        for layer in self.decoder_layers:
            x, self_attn_weights, cross_attn_weights = layer.forward(
                x,
                encoder_outputs,
                spatial_encoding,
                edge_encoding,
                self_padding_mask=None,
                cross_padding_mask=None,
                self_attention_mask=self_attention_mask,
                cross_attention_mask=cross_attention_mask
            )
            self_attention_weights.append(self_attn_weights)
            cross_attention_weights.append(cross_attn_weights)

        return x, self_attention_weights, cross_attention_weights


class GraphFormerEncoderLayer(nn.Module):
    """
    The GraphFormerEncoderLayer implements a single layer of the GraphFormer encoder.
    It performs self-attention and a feedforward bottleneck operation, both with
    residual connections and layer normalization.
    """

    def __init__(
        self,
        embedding_dimension: int,
        num_attention_heads: int,
        bottleneck_dimension: int,
        bottleneck_activation: nn.Module,
        dropout: float
    ):

        super().__init__()

        self.self_attention = MultiHeadGraphAttention(
            embedding_dimension, num_attention_heads, dropout, dropout
        )

        self.bottleneck = BottleneckFeedForward(
            embedding_dimension,
            bottleneck_dimension,
            dropout,
            use_residual=True,
            activation=bottleneck_activation,
            normalization='layerNorm'
        )

        self.normalization1 = torch.nn.LayerNorm(embedding_dimension)
        self.normalization2 = torch.nn.LayerNorm(embedding_dimension)
        self.normalization3 = torch.nn.LayerNorm(embedding_dimension)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        spatial_encoding: torch.Tensor,
        edge_encoding: torch.Tensor | None,
        padding_mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None
    ):
        """
        Performs a forward pass through the GraphFormerEncoderLayer.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, embedding_dimension),
                where B is the batch size, T is the sequence length.
            spatial_encoding (torch.Tensor): Spatial encoding tensor of shape
                (B, T, embedding_dimension).
            edge_encoding (Optional[torch.Tensor]): Optional edge encoding tensor
                of shape (B, T, embedding_dimension).
            padding_mask (Optional[torch.Tensor], optional): Mask for padding,
                shape (B, T).
            attention_mask (Optional[torch.Tensor], optional): Mask for attention,
                shape (B, T, T).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
            - Output tensor after encoding, shape (B, T, embedding_dimension).
            - Attention weights, shape (B, num_heads, T, T).
        """
        # Pre normalization
        x = self.normalization1(x)
        residual = x

        # Compute self attention
        x, attention_weights = self.self_attention.forward(
            x,
            x,
            x,
            spatial_encoding,
            edge_encoding,
            padding_mask,
            attention_mask
        )
        x = self.dropout(x)

        # Residual connection + norm + bottleneck
        x += residual
        x = self.normalization2(x)
        x = self.bottleneck.forward(x)

        return x, attention_weights


class GraphFormerEncoder(nn.Module):
    """
    The GraphFormerEncoder implements a transformer-style encoder for
    graph-structured data. It processes batches of graphs, supporting
    both padded and unpadded modes, and augments node features with
    spatial, centrality, and edge encodings. The encoder consists of
    multiple GraphFormerEncoderLayer modules, each applying self-attention
    and feedforward transformations. This class is suitable for learning
    node and graph representations that capture both local and global
    graph structure, making it useful for a variety of graph-based
    machine learning tasks.
    """

    def __init__(
        self,
        num_layers: int,
        node_dimension: int,
        edge_dimension: int,
        node_embedding_dimension: int,
        edge_embedding_dimension: int,
        num_attention_heads: int,
        bottleneck_dimension: int,
        bottleneck_activation: nn.Module,
        positional_encoding: nn.Module,
        dropout: float,
        max_path_length: int,
        max_in_degree: int,
        max_out_degree: int
    ):
        super().__init__()
        self.node_embedding_dimension = node_embedding_dimension
        self.edge_embedding_dimension = edge_embedding_dimension
        self.positional_encoding = positional_encoding

        if node_dimension == node_embedding_dimension:
            self.node_projection = nn.Identity()
        else:
            self.node_projection = nn.Linear(node_dimension, node_embedding_dimension)

        if edge_dimension == edge_embedding_dimension:
            self.edge_projection = nn.Identity()
        else:
            self.edge_projection = nn.Linear(edge_dimension, edge_embedding_dimension)

        self.spatial_encoding = SpatialEncoding(max_path_length)
        self.centrality_encoding = CentralityEncoding(
            max_in_degree, max_out_degree, self.node_embedding_dimension
        )
        self.edge_encoding = EdgeEncoding(max_path_length, self.edge_embedding_dimension)

        encoder_layers = []
        for _ in range(num_layers):
            encoder_layers.append(
                GraphFormerEncoderLayer(
                    node_embedding_dimension,
                    num_attention_heads,
                    bottleneck_dimension,
                    bottleneck_activation,
                    dropout
                )
            )
        self.encoder_layers = nn.ModuleList(encoder_layers)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor | None,
        graphs: list[GraphFormerGraph],
        padding_mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None
    ):
        """
        Performs a forward pass through the GraphFormer model. Note that T is
        the maximum number of nodes in the batch.

        Args:
            node_features (torch.Tensor): Input node features of shape
                (B, T, node_feature_dim).
            edge_features (Optional[torch.Tensor]): Optional edge features of
                shape (B, num_edges, edge_feature_dim) where num_edges is the
                number of edges in the graph.
            graphs (List[GraphFormerGraph]): List of graph objects
                corresponding to each sample in the batch.
            padding_mask (Optional[torch.Tensor], optional): Optional mask
                indicating padded nodes, shape (B, T).
            attention_mask (Optional[torch.Tensor], optional): Optional mask
                for attention mechanism, shape (B, T, T).

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]:
            - Output node representations after encoding, shape
              (B, T, node_embedding_dimension).
            - List of attention weights from each encoder layer.
        """

        node_features = self.node_projection(node_features)
        node_features = self.positional_encoding.forward(node_features)
        node_features = self.centrality_encoding.forward(node_features, graphs)

        spatial_encoding = self.spatial_encoding.forward(node_features, graphs)

        # Max nodes is the padded dimension of the node features
        # Node features is shape (B, max_nodes, node_embedding_dimension)
        max_nodes = node_features.shape[1]

        edge_encoding = None
        if edge_features is not None:
            edge_features = self.edge_projection.forward(edge_features)
            edge_encoding = self.edge_encoding.forward(edge_features, graphs, max_nodes)

        attention_weights = []
        x = node_features
        for layer in self.encoder_layers:
            x, weights = layer.forward(
                x,
                spatial_encoding,
                edge_encoding,
                padding_mask,
                attention_mask
            )
            attention_weights.append(weights)

        return x, attention_weights

    def forward_unpadded(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor | None,
        graph: GraphFormerGraph,
        attention_mask: torch.Tensor | None = None
    ):
        """
        Performs a forward pass through the GraphFormer encoder without
        padding. This is used when all graphs in the batch have the same
        number of nodes and edges, so no padding is required.

        Args:
            node_features (torch.Tensor): Input node features of shape
                (B, N, node_feature_dim), where N is the number of nodes.
            edge_features (Optional[torch.Tensor]): Optional edge features
                of shape (B, num_edges, edge_feature_dim), where num_edges
                is the number of edges.
            graph (GraphFormerGraph): The graph object corresponding to the
                batch.
            attention_mask (Optional[torch.Tensor], optional): Optional mask
                for attention mechanism, shape (B, N, N).
        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]:
            - Output node representations after encoding, shape
              (B, N, node_embedding_dimension).
            - List of attention weights from each encoder layer.
        """
        node_features = self.node_projection(node_features)
        node_features = self.positional_encoding.forward(node_features)
        node_features = self.centrality_encoding.forward_unpadded(node_features, graph)

        spatial_encoding = self.spatial_encoding.forward_unpadded(node_features, graph)

        # Node features is shape (B, N, node_embedding_dimension)
        max_nodes = node_features.shape[1]

        edge_encoding = None
        if edge_features is not None:
            edge_features = self.edge_projection.forward(edge_features)
            edge_encoding = self.edge_encoding.forward_unpadded(
                edge_features, graph, max_nodes
            )

        attention_weights = []
        x = node_features
        for layer in self.encoder_layers:
            x, weights = layer.forward(
                x,
                spatial_encoding,
                edge_encoding,
                padding_mask=None,
                attention_mask=attention_mask
            )
            attention_weights.append(weights)

        return x, attention_weights
