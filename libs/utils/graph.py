# coding=utf-8
"""
graph
Contains utility for graph
using networkx library
"""

import logging
import networkx as nx
import dijkstar
from typing import Any, Generator, Tuple, List, Sequence, Union
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

    def __init__(self, graph_lib: str = "networkx"):
        assert graph_lib in ("Dijkstar", "networkx"), "unsupported graph lib"
        self.graph_lib = graph_lib
        graph_init = {
            "Dijkstar": dijkstar.Graph,
            "networkx": nx.Graph
        }
        self.graph_struct: Union[dijkstar.Graph, nx.Graph] = graph_init[graph_lib]()

    def __repr__(self):
        output = 'Graph:\n'
        output += 'graph library :' + self.graph_lib + '\n'
        return output

    def add_edge_by_vert(self, vert1: Vertex, vert2: Vertex, cost: float):
        """
        add edge to the graph
        :return:
        """
        if self.graph_lib == 'Dijkstar':
            self.graph_struct.add_edge(vert1, vert2, {'cost': cost})
            self.graph_struct.add_edge(vert2, vert1, {'cost': cost})
        elif self.graph_lib == "networkx":
            self.graph_struct.add_edge(vert1, vert2, cost=cost)
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
        elif self.graph_lib == "networkx":
            self.graph_struct.add_edge(edge.start, edge.end, cost=cost)
        else:
            raise ValueError('graph library does not exit')

    def get_shortest_path(self,
                          edges1: Sequence[Edge],
                          edges2: Sequence[Edge]) -> Tuple[List[Vertex], float]:
        """
        get the shortest path between two edges sequences
        :return: list of vertices on the path and cost of the path
        """
        if self.graph_lib == 'Dijkstar':
            # not nice algorithm like with networkx because we cannot remove nodes
            # so it would be hazardous to add virtual nodes
            path = None
            cost = None
            for edge1 in edges1:
                for edge2 in edges2:
                    search = dijkstar.find_path(self.graph_struct,
                                                edge1,
                                                edge2,
                                                cost_func=lambda u, v, e, prev_e: e['cost'])
                    if cost is None or cost > search[3]:
                        path = search[0]
                        cost = search[3]
            return path, cost
        if self.graph_lib == 'networkx':
            # for each edges sequence add a virtual node connected with cost 0 to all edges
            for edge in edges1:
                self.graph_struct.add_edge("virtual1", edge.start, cost=0)
                self.graph_struct.add_edge("virtual1", edge.end, cost=0)
            for edge in edges2:
                self.graph_struct.add_edge(edge.start, "virtual2", cost=0)
                self.graph_struct.add_edge(edge.end, "virtual2", cost=0)
            # compute shortest path between virtual nodes
            try:
                path = nx.shortest_path(self.graph_struct, "virtual1", "virtual2")[1:-1]
            except nx.exception.NetworkXNoPath:
                # TODO : for now, the only case where this exception is thrown is when a floor is
                # cut into several parts by a load bearing wall. This problem must be treated
                # possible treatment : by putting holes in the load bearing walls where they
                # can be crossed
                logging.warning('Graph_nx-no path found')
                return [], 0
            finally:
                # remove virtual nodes
                self.graph_struct.remove_node("virtual1")
                self.graph_struct.remove_node("virtual2")
            # compute cost
            if len(path) == 1:
                return path, 0
            cost = 0
            for i in range(len(path) - 1):
                cost += self.graph_struct[path[i]][path[i + 1]]["cost"]
            return path, cost
        raise NotImplementedError("Graph library not supported")
