from __future__ import annotations

from collections import defaultdict, deque
from ctypes import ArgumentError
from typing import Callable, Generic, TypeVar

T = TypeVar('T')
E = TypeVar('E')


class Graph(Generic[T]):  # noqa: UP046
    """
    A generic graph implementation supporting directed and undirected graphs.

    Attributes:
        adjacency_list (defaultdict): Maps each node to its set of adjacent nodes.
        child_to_parent (dict): Maps each node to its set of parent nodes.
        edge_to_index (dict | None): Maps edges to their indices, used for edge indexing.
        undirected (bool): Indicates whether the graph is undirected.
    """

    def __init__(self, undirected: bool = False) -> None:
        self.adjacency_list: defaultdict[T, set[T]] = defaultdict(set)
        self.child_to_parent: dict[T, set[T]] = defaultdict(set)
        self.edge_to_index: dict[tuple[T, T], int] | None = None
        self.undirected = undirected

    def add_node(self, node: T):
        """Adds a node to the graph."""
        if node in self.adjacency_list:
            assert node in self.child_to_parent
            return

        self.adjacency_list[node] = set()
        self.child_to_parent[node] = set()

    def add_edge(self, node1: T, node2: T):
        """
        Adds an edge between two nodes in the graph.

        Args:
            node1 (T): The starting node of the edge.
            node2 (T): The ending node of the edge.

        Raises:
            ArgumentError: If either node does not exist in the graph.
        """
        if node1 not in self.adjacency_list:
            raise ArgumentError(f"Node '{node1}' does not exist")
        if node2 not in self.adjacency_list:
            raise ArgumentError(f"Node '{node2}' does not exist")

        self.adjacency_list[node1].add(node2)
        self.child_to_parent[node2].add(node1)

        if self.undirected:
            self.adjacency_list[node2].add(node1)
            self.child_to_parent[node1].add(node2)

        # Reset edge to index so that we have to rebuild it later on
        self.edge_to_index = None

    def delete_node(self, node: T):
        """
        Deletes a node and all its associated edges from the graph.

        Args:
            node (T): The node to be deleted.

        Raises:
            ArgumentError: If the node does not exist in the graph.
        """
        if node not in self.adjacency_list:
            raise ArgumentError(f"Node '{node}' does not exist")

        self.adjacency_list.pop(node)
        for _, adjacents in self.adjacency_list.items():
            adjacents.remove(node)

        assert node in self.child_to_parent
        self.child_to_parent.pop(node)
        for _, adjacents in self.child_to_parent.items():
            adjacents.remove(node)

    def delete_edge(self, node1: T, node2: T):
        """
        Deletes an edge between two nodes in the graph.

        Args:
            node1 (T): The starting node of the edge.
            node2 (T): The ending node of the edge.

        Raises:
            ArgumentError: If either node does not exist in the graph.
        """
        if node1 not in self.adjacency_list:
            raise ArgumentError(f"Node '{node1}' does not exist")

        if node2 not in self.adjacency_list:
            raise ArgumentError(f"Node '{node2}' does not exist")

        self.adjacency_list[node1].remove(node2)
        self.child_to_parent[node2].remove(node1)
        if self.undirected:
            self.adjacency_list[node2].remove(node1)
            self.child_to_parent[node1].remove(node2)

    def get_nodes(self):
        """Retrieves all nodes in the graph."""
        return list(self.adjacency_list.keys())

    def get_edges(self):
        """
        Retrieves all edges in the graph.

        Returns:
            list[tuple[T, T]]: A list of tuples representing the edges in the graph.
        """
        edges: list[tuple[T, T]] = []
        for node, adjacents in self.adjacency_list.items():
            for adjacent in adjacents:
                edges.append((node, adjacent))
        return edges

    def num_nodes(self):
        """Returns the number of nodes in the graph."""
        return len(self.adjacency_list)

    def num_edges(self):
        """Returns the number of edges in the graph."""
        return len(self.get_edges())

    def is_undirected(self):
        """Checks if the graph is undirected."""
        return self.undirected

    def get_children(self, node: T):
        """
        Retrieves the children (adjacent nodes) of a given node.

        Args:
            node (T): The node whose children are to be retrieved.

        Returns:
            set[T]: A set of child nodes connected to the given node.

        Raises:
            ArgumentError: If the node does not exist in the graph.
        """
        if node not in self.adjacency_list:
            raise ArgumentError(f"Node '{node}' does not exist")

        return self.adjacency_list[node]

    def get_parents(self, node: T):
        """
        Retrieves the parents (nodes pointing to) of a given node.

        Args:
            node (T): The node whose parents are to be retrieved.

        Returns:
            list[T]: A list of parent nodes connected to the given node.

        Raises:
            ArgumentError: If the node does not exist in the graph.
        """
        if node not in self.child_to_parent:
            raise ArgumentError(f"Node '{node}' does not exist")

        return list(self.child_to_parent[node])

    def get_edge_index(self, edge: tuple[T, T]):
        """
        Retrieves the index of a given edge in the graph.

        An edges index is defined by its index in a sorted list of all edges.

        Args:
            edge (tuple[T, T]): The edge whose index is to be retrieved.

        Returns:
            int: The index of the edge in the graph.

        Raises:
            KeyError: If the edge does not exist in the graph.
        """
        if self.edge_to_index is None:
            edges: list[tuple[T, T]] = []
            for node, adjacents in self.adjacency_list.items():
                for adj_node in adjacents:
                    edges.append((node, adj_node))

            edges.sort()
            self.edge_to_index = {edge: i for i, edge in enumerate(edges)}

        return self.edge_to_index[edge]

    def append(self, graph: Graph[T]):
        """
        Appends another graph to the current graph.

        Args:
            graph (Graph[T]): The graph to be appended.

        Raises:
            ValueError: If the undirected property of the two graphs does not match.
        """
        if self.is_undirected() != graph.is_undirected():
            raise ValueError(
                f"Cannot append graph with undirected="
                f"{graph.is_undirected()} to graph with undirected="
                f"{self.is_undirected()}"
            )

        for node in graph.get_nodes():
            self.add_node(node)
            for child in graph.get_children(node):
                self.add_node(child)
                self.add_edge(node, child)

    def clone(self):
        """
        Creates a deep copy of the graph.

        Returns:
            Graph[T]: A new graph instance that is a deep copy of the current graph.
        """
        new_graph = Graph[T](self.undirected)
        new_graph.append(self)
        return new_graph

    def bfs(self, node: T, fold_function: Callable[[T, E], E], initial_state: E):
        """
        Performs a breadth-first search (BFS) traversal starting from a given node.

        Args:
            node (T): The starting node for the BFS traversal.
            fold_function (Callable[[T, E], E]): A function that
                computes the new state based on the current node and state.
            initial_state (E): The initial state to start the traversal with.

        Yields:
            tuple[T, E]: A tuple containing the current node and its
                associated state during the traversal.
        """
        queue = deque([(node, initial_state)])
        while len(queue) > 0:
            current_node, state = queue.popleft()
            yield current_node, state
            for child in self.get_children(current_node):
                new_state = fold_function(child, state)
                queue.appendleft((child, new_state))

    def dfs(self, node: T, fold_function: Callable[[T, E], E], initial_state: E):
        """
        Performs a depth-first search (DFS) traversal starting from a given node.

        Args:
            node (T): The starting node for the DFS traversal.
            fold_function (Callable[[T, E], E]): A function that
                computes the new state based on the current node and state.
            initial_state (E): The initial state to start the traversal with.

        Yields:
            tuple[T, E]: A tuple containing the current node and its
                associated state during the traversal.
        """
        stack = [(node, initial_state)]
        while len(stack) > 0:
            current_node, state = stack.pop()
            yield current_node, state

            for child in self.get_children(current_node):
                new_state = fold_function(child, state)
                stack.append((child, new_state))

    def get_shortest_path(self, node1: T, node2: T):
        """
        Finds the shortest path between two nodes using BFS.

        Args:
            node1 (T): The starting node of the path.
            node2 (T): The ending node of the path.

        Returns:
            list[T] | None: A list of nodes representing the shortest path
                from node1 to node2, or None if no path exists.
        Raises:
            ArgumentError: If either node does not exist in the graph.
        """
        if node1 not in self.adjacency_list:
            raise ArgumentError(f"Node '{node1}' does not exist")
        if node2 not in self.adjacency_list:
            raise ArgumentError(f"Node '{node2}' does not exist")

        def fold_function(node: T, state: list[T]) -> list[T]:
            return state + [node]

        search = self.bfs(node1, fold_function, [node1])
        for node, path in search:
            if node == node2:
                return path

        return None

    def to_dict(self):
        """
        Converts the graph to a dictionary representation.

        Returns:
            dict: A dictionary containing the adjacency list and child-to-parent mappings.
        """
        return {
            'adjacency_list': {
                node: list(adj)
                for node, adj in self.adjacency_list.items()
            },
            'child_to_parent': {
                node: list(adj)
                for node, adj in self.child_to_parent.items()
            }
        }

    def from_dict(self, graph_dict: dict):
        """
        Populates the graph from a dictionary representation.

        Args:
            graph_dict (dict): A dictionary containing the adjacency
                list and child-to-parent mappings.
        """
        for node, adj in graph_dict['adjacency_list'].items():
            self.adjacency_list[node] = set(adj)

        for node, adj in graph_dict['child_to_parent'].items():
            self.child_to_parent[node] = set(adj)

    def __eq__(self, other: Graph[T]) -> bool:
        return self.adjacency_list == other.adjacency_list