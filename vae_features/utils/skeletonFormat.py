from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import TypeVar

import torch

from graph import Graph


@dataclass(frozen=True)
class SkeletonFormatID:
    """A class to represent a skeleton format identifier."""
    id: str


class SkeletonFormat:

    def __init__(
        self,
        joint_names: list[str],
        graph: Graph[int],
        rest_pose: torch.Tensor | None = None,
        identifier: SkeletonFormatID | None = None
    ):
        """Constructs a SkeletonFormat object from a graph of joint indices.

        Assumes that the joint names are sorted in topological order based
        on the graph structure.

        Args:
            joint_names (List[str]): List of joint names of length N
            Graph (Graph[int]): A graph containing the joint tree.
            Graph nodes are joint indices, and edges are parent-child
            relationships. The graph must be a tree
            rest_pose (torch.Tensor | None): An optional tensor of shape
                (N, 3) containing joint positions of the rest pose. If None,
                the rest pose is not defined.
        """
        # Verify that the joints are sorted in topological order
        index_to_joint_name = {idx: name for idx, name in enumerate(joint_names)}
        string_graph = _index_graph_to_string_graph(graph, index_to_joint_name)
        sorted_joint_names = _topologically_sort_joints(string_graph, joint_names)

        if sorted_joint_names != joint_names:
            raise ValueError(
                "Joint names must be sorted in topological order based on the "
                "graph structure"
            )

        self.joint_names = joint_names
        self._joint_name_to_index = {
            name: idx
            for idx, name in enumerate(self.joint_names)
        }
        self.graph = graph
        self.rest_pose = rest_pose

        if identifier is None:
            self.identifier = self._get_default_identifier()
        else:
            self.identifier = identifier

    @classmethod
    def from_joints_and_graph(
        cls,
        joint_names: list[str],
        graph: Graph[str],
        rest_pose: dict[str, torch.Tensor] | None = None,
        identifier: SkeletonFormatID | None = None
    ) -> SkeletonFormat:
        """Constructs a SkeletonFormat object from a graph of joint names.

        A skeleton format needs to define a strict ordering of joints. Each
        joint must be assigned a unique integer ID, which is used to index the
        joint dimension in tensors. The ordering of these indices should be
        topological - E.g. their order should be based on their distance from
        the root joint. If two joints have the same distance to the root, they
        should be sorted by their joint name.

        This function will automatically sort the joint names in topological
        order, taking care of all ordering dynamics for the user.

        Args:
            joint_names (List[str]): List of joint names
            Graph (Graph[str]): A graph containing the joint tree. Graph nodes
                are joint names, and edges are parent-child relationships. The
                graph must be a tree
            rest_pose (Dict[str, torch.Tensor] | None): A mapping from joint
                names to their rest pose positions. If None, the rest pose is
                not defined.
        """
        sorted_joint_names = _topologically_sort_joints(graph, joint_names)
        joint_name_to_index = {name: idx for idx, name in enumerate(sorted_joint_names)}
        index_graph = _string_graph_to_index_graph(graph, joint_name_to_index)

        rest_pose_tensor = None
        if rest_pose is not None:
            num_joints = len(sorted_joint_names)

            if len(rest_pose) != num_joints:
                raise ValueError(
                    f"Rest pose must have {num_joints} joints, but got {len(rest_pose)}"
                )

            rest_pose_tensor = torch.zeros(num_joints, 3)
            for joint_name, position in rest_pose.items():
                if joint_name not in joint_names:
                    raise ValueError(f"Joint '{joint_name}' not found in joint names")

                if position.shape != (3,):
                    raise ValueError(
                        f"Position for joint '{joint_name}' must be a 3D vector, but "
                        f"got {position.shape}"
                    )

                rest_pose_tensor[joint_name_to_index[joint_name]] = position

        return cls(
            joint_names=sorted_joint_names,
            graph=index_graph,
            rest_pose=rest_pose_tensor,
            identifier=identifier
        )

    @classmethod
    def from_dict(cls, save_dict: dict):
        """Creates a SkeletonFormat instance from a dictionary representation.

        Args:
            save_dict (dict): A dictionary containing the graph, joint names,
            and rest pose of the skeleton.

        Returns:
            SkeletonFormat: A SkeletonFormat instance created from the dictionary.
        """
        graph = Graph()
        graph.from_dict(save_dict['graph'])
        joint_names = save_dict['joint_names']
        rest_pose = torch.Tensor(save_dict['rest_pose'])

        instance = SkeletonFormat.from_joints_and_graph(joint_names, graph)
        instance.set_rest_pose(rest_pose)
        return instance

    def set_rest_pose(self, rest_pose: torch.Tensor):
        """Sets the rest pose of the skeleton format.

        Args:
            rest_pose (torch.Tensor): Tensor of shape (N, 3) containing joint positions
                of the rest pose.
        """
        if rest_pose.shape[0] != self.get_joint_count():
            raise ValueError(
                f"Rest pose must have {self.get_joint_count()} joints, but got "
                f"{rest_pose.shape[0]}"
            )

        self.rest_pose = rest_pose

    def _add_edges(self, edges: str):
        for edge_list in edges.split(','):

            edge_split = edge_list.split('-')
            for parent_joint, child_joint in itertools.pairwise(edge_split):

                missing_parent = parent_joint not in self.joint_names
                missing_child = child_joint not in self.joint_names
                if missing_parent or missing_child:
                    raise ValueError(
                        f"Joints '{parent_joint}' and/or '{child_joint}' do not exist"
                    )

                parent_idx = self._joint_name_to_index[parent_joint]
                child_idx = self._joint_name_to_index[child_joint]

                self.graph.add_node(parent_idx)
                self.graph.add_node(child_idx)
                self.graph.add_edge(parent_idx, child_idx)

                if len(self.graph.get_parents(child_idx)) > 1:
                    raise ValueError(f"Joint '{child_joint} has more than one parent")

    def _get_default_identifier(self):
        root_joint_name = self.index_to_joint_name(self.get_root_joint_index())
        num_joints = self.get_joint_count()
        id = f"J#={num_joints}_R={root_joint_name}"
        return SkeletonFormatID(id)

    def joint_name_to_index(self, joint_name: str):
        """Converts a joint name to its corresponding index."""
        if joint_name not in self.joint_names:
            raise ValueError(f"Joint '{joint_name}' does not exist")

        return self._joint_name_to_index[joint_name]

    def index_to_joint_name(self, joint_index: int):
        """Converts a joint index to its corresponding name."""
        if joint_index not in self.graph.get_nodes():
            raise ValueError(f"Joint index '{joint_index}' does not exist")

        return self.joint_names[joint_index]

    def get_unique_parent(self, joint_index: int):
        """Gets the unique parent joint index of the specified joint.

        The parent is unique because the skeleton graph is a tree.

        Args:
            joint_index (int): The index of the joint whose parent is to be retrieved.

        Returns:
            int | None: The index of the parent joint, or None if the joint is the root.
        """
        return _get_parent(joint_index, self.graph)

    def get_root_joint_index(self):
        """Gets the root joint index of the skeleton."""
        return _get_root_joint(self.graph)

    def get_leaf_joint_indices(self):
        """Gets the leaf joints of the skeleton.

        Leaf joints are joints that do not have any children.

        Returns:
            List[int]: A list of indices of the leaf joints.
        """
        leaves = []
        for joint_index in range(self.get_joint_count()):
            if len(self.graph.get_children(joint_index)) == 0:
                leaves.append(joint_index)
        return leaves

    def get_joint_names(self):
        """Gets the joint names of the skeleton."""
        return self.joint_names

    def get_joint_count(self):
        """Gets the number of joints in the skeleton."""
        return len(self.joint_names)

    def get_rest_pose(self):
        """Gets the rest pose of the skeleton."""
        return self.rest_pose

    def get_graph(self):
        """Gets the graph of the skeleton."""
        return self.graph

    def get_identifier(self):
        """Gets the identifier of the skeleton."""
        return self.identifier

    def set_identifier(self, identifier: str):
        """Sets the identifier of the skeleton."""
        self.identifier = SkeletonFormatID(identifier)

    def to_dict(self):
        """Converts the SkeletonFormat object to a dictionary representation.

        Returns:
            dict: A dictionary containing the graph, joint names, and
                rest pose of the skeleton.
        """
        index_to_joint_name = {idx: name for idx, name in enumerate(self.joint_names)}
        string_graph = _index_graph_to_string_graph(self.graph, index_to_joint_name)
        return {
            'graph': string_graph.to_dict(),
            'joint_names': self.joint_names,
            'rest_pose': None if self.rest_pose is None else self.rest_pose.tolist()
        }

    def __eq__(self, other: SkeletonFormat) -> bool:
        equality = self.joint_names == other.joint_names and self.graph == other.graph
        # assert (self.identifier == other.identifier) == equality, \
        #     "Identifiers must match if joint names and graph are equal"
        return equality


T = TypeVar('T')


def _get_parent(node: T, graph: Graph[T]):
    """Returns the parent joint index of the given joint index."""
    parents = graph.get_parents(node)
    if len(parents) == 0:
        return None

    assert len(parents) == 1, "Joint must have at most one parent"
    return parents[0]


def _get_root_joint(graph: Graph[T]):  # noqa: UP047
    """Returns the name of the root joint in the skeleton format."""
    # Pick a random joint and follow it to the top of the tree
    node = graph.get_nodes()[0]

    while _get_parent(node, graph) is not None:
        node = _get_parent(node, graph)
        assert node is not None

    return node


def _topologically_sort_joints(graph: Graph[str], joint_names: list[str]):
    """Sorts the joint names in a topological order based on the graph structure.

    A topological sort is a linear ordering of nodes based on their distance
    from the root joint. For nodes will have the same distance to the root,
    we sort them by their joint name.

    Returns:
        List[str]: A list of joint names sorted in topological order.
    """
    root = _get_root_joint(graph)
    joint_to_distance: dict[str, int] = {}

    def fold_function(node: str, state: int):
        return state + 1

    for node, distance in graph.bfs(root, fold_function, 0):
        joint_to_distance[node] = distance

    # If the distance is the same, sort by joint name.
    def sort_function(joint_name: str):
        distance = joint_to_distance[joint_name]
        return (distance, joint_name)

    return sorted(joint_names, key=sort_function)


def _string_graph_to_index_graph(
    string_graph: Graph[str], joint_name_to_index: dict[str, int]
):
    """
    Converts a string graph to an index graph based on the joint_name_to_index mapping.
    """
    index_graph = Graph[int]()
    for node in string_graph.get_nodes():
        index_node = joint_name_to_index[node]
        index_graph.add_node(index_node)
        for child in string_graph.get_children(node):
            index_child = joint_name_to_index[child]
            index_graph.add_node(index_child)
            index_graph.add_edge(index_node, index_child)
    return index_graph


def _index_graph_to_string_graph(
    index_graph: Graph[int], joint_index_to_name: dict[int, str]
):
    """
    Converts a string graph to an index graph based on the joint_index_to_name mapping.
    """
    string_graph = Graph[str]()
    for node in index_graph.get_nodes():
        string_node = joint_index_to_name[node]
        string_graph.add_node(string_node)
        for child in index_graph.get_children(node):
            string_child = joint_index_to_name[child]
            string_graph.add_node(string_child)
            string_graph.add_edge(string_node, string_child)
    return string_graph
