# coding=utf-8
"""
graph
Contains utility for graph
using networkx library
"""

import networkx as nx
import dijkstar
from typing import Any, Generator, Tuple, List, Sequence
from libs.plan.plan import Vertex
from libs.mesh.mesh import Edge


class Graph_nx:

    def __init__(self):
        self.graph = nx.Graph()

    def shallow_copy(self):
        """
        shallow copy of the graph
        :return:
        """
        graph_copy = nx.Graph()
        graph_copy.graph = self.graph.copy()

        return graph_copy

    def add_edge(self, i: Any, j: Any):
        """
        add edge to the graph, linking input nodes
        :return:
        """
        self.graph.add_edge(i, j)

    def add_node(self, n: Any):
        """
        add node to the graph
        :return:
        """
        self.graph.add_node(n)

    def remove_node(self, n: Any):
        """
        remove node to the graph
        :return:
        """
        self.graph.remove_node(n)

    def is_connected(self):
        """
        checks if the graph is connected
        :return:
        """
        return nx.is_connected(self.graph)

    def node_connected(self, n: Any):
        """
        checks if the given node is connected to the graph
        :return:
        """
        for node in self.graph.nodes():
            if n is not node and self.has_path(node, n):
                return True
        else:
            return False

    def has_path(self, i: Any, j: Any):
        """
        checks if the graph contains a path between given nodes
        :return:
        """
        return nx.has_path(self.graph, i, j)

    def nodes(self) -> Generator[Any, None, None]:
        """
        returns iterator on the graph nodes
        :return:
        """
        return self.graph.nodes()


class EdgeGraph:
    """
    Class Graph:
    function to build and deal with graph of mutable space edges,
    implementation for given graph libaries
    """

    # TODO : add same functionnalities with networkx library
    def __init__(self, graph_lib: str = 'Dijkstar'):
        self.graph_lib = graph_lib
        self.cost_function = None
        self.graph_struct = None

    def __repr__(self):
        output = 'Graph:\n'
        output += 'graph library :' + self.graph_lib + '\n'
        return output

    def init(self):
        """
        graph initialization
        :return:
        """
        if self.graph_lib == 'Dijkstar':
            self.graph_struct = dijkstar.Graph()
        else:
            raise ValueError('graph library does not exit')

    def add_edge_by_vert(self, vert1: Vertex, vert2: Vertex, cost: float):
        """
        add edge to the graph
        :return:
        """
        if self.graph_lib == 'Dijkstar':
            self.graph_struct.add_edge(vert1, vert2, {'cost': cost})
        else:
            raise ValueError('graph library does not exit')

    def add_edge(self, edge: Edge, cost: float):
        """
        add edge to the graph
        :return:
        """
        if self.graph_lib == 'Dijkstar':
            self.graph_struct.add_edge(edge.start, edge.end, {'cost': cost})
            self.graph_struct.add_edge(edge.end, edge.start, {'cost': cost})
        else:
            raise ValueError('graph library does not exit')

    def set_cost_function(self):
        """
        sets the graph cost function
        :return:
        """
        if self.graph_lib == 'Dijkstar':
            self.cost_function = lambda u, v, e, prev_e: e['cost']
        else:
            raise ValueError('graph library does not exit')

    def get_shortest_path(self,
                          edges1: Sequence[Edge],
                          edges2: Sequence[Edge]) -> Tuple[List[Vertex], float]:
        """
        get the shortest path between two edges sequences
        :return list of vertices on the path and cost of the path
        """
        # for each edge sequence we add a virtual node connected with cost 0 to all edges
        if self.graph_lib != 'Dijkstar':
            raise ValueError('graph library does not exit')
        for edge in edges1:
            self.graph_struct.add_edge("virtual1", edge.start, {'cost': 0})
            self.graph_struct.add_edge(edge.start, "virtual1", {'cost': 0})
        for edge in edges2:
            self.graph_struct.add_edge(edge.start, "virtual2", {'cost': 0})
            self.graph_struct.add_edge("virtual2", edge.start, {'cost': 0})
        search_tree_result = dijkstar.find_path(self.graph_struct, "virtual1", "virtual2",
                                                cost_func=self.cost_function)
        path = search_tree_result[0][1:-1]
        cost = search_tree_result[3]
        return path, cost