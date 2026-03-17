import itertools

import torch

from graph import Graph
from skeletonFormat import SkeletonFormat


class GraphFormerGraph(Graph[int]):
    """
    GraphFormerGraph is a graph data structure tailored for use with
    Graphormer-style models.

    It extends a generic Graph[int] class and provides additional utilities
    for caching graph properties and efficiently computing features required
    for graph-based neural networks.

    Features:
        - Caches shortest path distances between all node pairs as a matrix.
        - Caches edge indices along shortest paths for all node pairs.
        - Caches in-degree and out-degree for all nodes as tensors.
        - Supports adding virtual nodes connected to all existing nodes.
        - Can be constructed from a SkeletonFormat and augmented with virtual
          nodes.

    Note:
        This class assumes that node indices are integers and that the parent
        Graph class provides methods such as get_nodes(), get_children(),
        get_shortest_path(), get_edge_index(), add_node(), add_edge(), and
        append().
    """

    def __init__(self, undirected: bool = False) -> None:
        super().__init__(undirected)

    def _build_shortest_path_map(self):
        """Builds a map of shortest paths between all pairs of nodes.

        This method computes the shortest path for every pair of nodes in the graph
        and stores them in a dictionary. The keys are tuples of node indices, and the
        values are lists representing the shortest path from the first node to the second.
        """
        if hasattr(self, 'shortest_path_map'):
            return self.shortest_path_map

        shortest_path_map: dict[tuple[int, int], list[int] | None] = {}

        for a in self.get_nodes():
            for b in self.get_nodes():
                path = self.get_shortest_path(a, b)
                shortest_path_map[(a, b)] = path

        self.shortest_path_map = shortest_path_map

    def _build_path_distance_cache(self):
        """Builds a cache of path distances for all pairs of nodes
           with paths between them.
        """
        self._build_shortest_path_map()

        num_paths = len([
            path for path in self.shortest_path_map.values() if path is not None
        ])
        path_distance_map = torch.LongTensor(num_paths)

        # The pair of nodes each entry corresponds to is determined by the
        # iteration order of self.shortest_path_map.items()
        i = 0
        for _, path in self.shortest_path_map.items():
            if path is not None:
                path_distance_map[i] = _get_path_length(path)
                i += 1

        return path_distance_map

    def _build_path_cache(self):
        """Builds a cache of edge indices for all shortest paths in the graph."""
        self._build_shortest_path_map()

        max_path_length = max(
            _get_path_length(path)
            for path in self.shortest_path_map.values()
            if path is not None
        )
        edge_index_tensors = []

        for path in self.shortest_path_map.values():
            if path is None:
                continue

            index_tensor = torch.zeros(max_path_length).long()
            for i, (node1, node2) in enumerate(itertools.pairwise(path)):
                edge_index = self.get_edge_index((node1, node2))
                index_tensor[i] = edge_index

            edge_index_tensors.append(index_tensor)

        # (NumPaths, max_path_length)
        # The pair of nodes each row corresponds to is determined by the
        # iteration order of self.shortest_path_map.items()
        path_cache = torch.stack(edge_index_tensors, dim=0)
        return path_cache

    def _build_index_to_node_pair(self):
        """Builds a tensor that maps indices to pairs of nodes. The ordering of
           this tensor is the same as the key ordering in the shortest_path_map.
           Node pairs without paths between them will not be included in the
           tensor.
        """
        self._build_shortest_path_map()

        num_paths = len([
            path for path in self.shortest_path_map.values() if path is not None
        ])
        index_to_node_pair = torch.LongTensor(num_paths, 2)

        # The pair of nodes each row corresponds to is determined by the
        # iteration order of self.shortest_path_map.items()
        i = 0
        for (a, b), path in self.shortest_path_map.items():
            if path is not None:
                index_to_node_pair[i] = torch.tensor([a, b])
                i += 1

        return index_to_node_pair

    def _build_in_degree_cache(self):
        in_degree_map: dict[int, int] = {}

        for a in self.get_nodes():
            degree = 0

            for b, adjacents in self.adjacency_list.items():
                if a == b:
                    continue
                if a in adjacents:
                    degree += 1

            in_degree_map[a] = degree

        # Make a list where list[NodeIndex] = In degree of node
        in_degree_list = sorted(in_degree_map.items(), key=lambda x: x[0])
        in_degree_list = [degree for _, degree in in_degree_list]

        # We cache the tensor of indices
        in_degree_cache = torch.LongTensor(in_degree_list)
        return in_degree_cache

    def _build_out_degree_cache(self):
        out_degree_map: dict[int, int] = {}

        for a in self.get_nodes():
            out_degree_map[a] = len(self.get_children(a))

        # Make a list where list[NodeIndex] = Out degree of node
        out_degree_list = sorted(out_degree_map.items(), key=lambda x: x[0])
        out_degree_list = [degree for _, degree in out_degree_list]

        # We cache the tensor of indices
        out_degree_cache = torch.LongTensor(out_degree_list)
        return out_degree_cache

    def add_virtual_node(self, node: int):
        """Adds a virtual node to the graph and connects it to all existing nodes.

        Args:
            node (int): The index of the virtual node to be added.
        """
        self.add_node(node)
        for other in self.get_nodes():
            self.add_edge(node, other)
            self.add_edge(other, node)

    def clone(self):
        new_graph = GraphFormerGraph(self.undirected)
        new_graph.append(self)
        return new_graph

    def get_in_degree_cache(self):
        """The in degree cache. Stores the in degree
        of every node the graph

        Returns:
            Tensor: Shape (N,), where N is the number of nodes.
            Every entry in the tensor is the in degree for that node.
        """
        if not hasattr(self, 'in_degree_cache'):
            self.in_degree_cache = self._build_in_degree_cache()
        return self.in_degree_cache

    def get_out_degree_cache(self):
        """The out degree cache. Stores the out degree
        of every node the graph

        Returns:
            Tensor: Shape (N,), where N is the number of nodes.
            Every entry in the tensor is the out degree for that node.
        """
        if not hasattr(self, 'out_degree_cache'):
            self.out_degree_cache = self._build_out_degree_cache()
        return self.out_degree_cache

    def get_path_distance_cache(self):
        """The path distance cache. A tensor where each entry is the
        path distance between corresponding nodes in the graph.

        Returns:
            (Tensor, Tensor):

            path_distance_map: Shape (num_paths,) tensor where
                num_paths <= N^2, as it's likely not true that
                every pair of nodes has a path between them. To
                see what pair of nodes corresponds to which entry
                in the tensor, use the edge index tensor.

            index_to_node_pair: Shape (num_paths, 2) tensor where
                each row corresponds to a pair of nodes with a path
                between them. The first column is the index of the
                first node, the second column is the index of the
                second node. Each row in the tensor corresponds to
                the node pair in the path_distance_map tensor at
                the same index.
        """
        if not hasattr(self, 'path_distance_map'):
            self.path_distance_map = self._build_path_distance_cache()
        if not hasattr(self, 'index_to_node_pair'):
            self.index_to_node_pair = self._build_index_to_node_pair()
        return self.path_distance_map, self.index_to_node_pair

    def get_path_cache(self):
        """The edge cache. For every pair of nodes with a path between them,
        this output tensor will list the index of every edge along that path.

        Returns:
            (Tensor, Tensor):

            edge_index_path_map: Shape (num_paths, max_path_length). For every
                path, we store the index of every edge along that path. We
                have to pad the second dimension to the maximum path length.

            index_to_node_pair: Shape (num_paths, 2) tensor where each row
                corresponds to a pair of nodes with a path between them. The
                first column is the index of the first node, the second
                column is the index of the second node. Each row in the
                tensor corresponds to the node pair in the path_distance_map
                tensor at the same row.
        """
        if not hasattr(self, 'path_cache'):
            self.path_cache = self._build_path_cache()
        if not hasattr(self, 'index_to_node_pair'):
            self.index_to_node_pair = self._build_index_to_node_pair()
        return self.path_cache, self.index_to_node_pair

    @staticmethod
    def from_skeleton_format(skeleton_format: SkeletonFormat, num_virtual_nodes: int):
        """Creates a GraphFormerGraph from a SkeletonFormat and adds virtual nodes.

        Args:
            skeleton_format (SkeletonFormat): The skeleton format object containing
                the graph structure.
            num_virtual_nodes (int): The number of virtual nodes to add to the graph.

        Returns:
            GraphFormerGraph: A new GraphFormerGraph instance with the specified skeleton
                structure and virtual nodes.
        """
        graph = GraphFormerGraph()
        graph.append(skeleton_format.get_graph())

        N = graph.num_nodes()
        for _ in range(num_virtual_nodes):
            graph.add_virtual_node(N)
            N += 1

        return graph


def _get_path_length(path: list[int]) -> int:
    """Returns the length of a path, which is the number of edges in the path."""
    return len(path) - 1
