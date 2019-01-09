# coding=utf-8
"""
graph
Contains utility for graph
using networkx library
"""

import networkx as nx
import dijkstar
from typing import Any, Generator, Tuple
from libs.plan import Vertex
from libs.mesh import Edge

class Graph_nx:

    def __init__(self):
        self.graph = nx.Graph()

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
    #TODO : add same functionnalities with networkx library
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

    def get_shortest_path(self, edge1, edge2) -> Tuple['List[Vertex]', float]:
        """
        get the shortest path between two edges
        :return list of vertices on the path and cost of the path
        """
        if self.graph_lib == 'Dijkstar':
            search_tree_result = dijkstar.find_path(self.graph_struct, edge1.start, edge2.start,
                                                    cost_func=self.cost_function)
        else:
            raise ValueError('graph library does not exit')
        path = search_tree_result[0]
        cost = search_tree_result[3]
        return path, cost