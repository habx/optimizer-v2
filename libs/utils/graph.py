# coding=utf-8
"""
graph
Contains utility for graph
using networkx library
"""

import logging
import networkx as nx
import dijkstar
from typing import Any, Generator, Tuple, List, Union
from libs.plan.plan import Vertex
from libs.mesh.mesh import Edge


class GraphNx:
    """
    A graph Class
    Used to represent the connectivity of spaces in a plan
    """

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

    def connected_components(self) -> Generator[Any, Any, Any]:
        """
        Yields the connected components of the graph as sets of nodes
        """
        return nx.connected_components(self.graph)


class EdgeGraph:
    """
    Class Graph:
    function to build and deal with graph of mutable space edges,
    implementation for given graph libaries
    """

    graph_init = {
        "Dijkstar": dijkstar.Graph,
        "networkx": nx.Graph
    }

    def __init__(self, graph_lib: str = "networkx"):
        assert graph_lib in self.graph_init, "unsupported graph lib"
        self.graph_lib = graph_lib
        self.graph_struct: Union[dijkstar.Graph, nx.Graph] = self.graph_init[graph_lib]()

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
                          edges1: List[Edge],
                          edges2: List[Edge]) -> Tuple[List[Vertex], float]:
        """
        get the shortest path between two edges sequences.
        In order to get the shortest path, we create two virtual vertices connected
        at zero cost to each vertices of the edges of each list.
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
                                                edge1.start,
                                                edge2.start,
                                                cost_func=lambda u, v, e, prev_e: e['cost'])
                    if cost is None or cost > search[3]:
                        path = search[0]
                        cost = search[3]
            return path, cost

        if self.graph_lib == 'networkx':
            # for each edges sequence add a virtual node connected with cost 0 to all edges
            virtual_1 = Vertex(edges1[0].mesh)
            virtual_2 = Vertex(edges1[0].mesh)
            for edge in edges1:
                self.graph_struct.add_edge(virtual_1, edge.start, cost=0)
                self.graph_struct.add_edge(virtual_1, edge.end, cost=0)
            for edge in edges2:
                self.graph_struct.add_edge(edge.start, virtual_2, cost=0)
                self.graph_struct.add_edge(edge.end, virtual_2, cost=0)
            try:
                # compute shortest path between virtual nodes and remove first and last item
                # (corresponding to the virtual vertices)
                path = nx.shortest_path(self.graph_struct,
                                        virtual_1, virtual_2, weight='cost')[1:-1]

            except nx.exception.NetworkXNoPath:
                # TODO: for now, the only case where this exception is thrown should be when
                #       a floor is cut into several parts by a load bearing wall.
                #       This problem must be treated
                #       possible treatment : by putting holes in the load bearing walls at location
                #       where they can be crossed
                logging.info('GraphNx: no path found')
                return [], 0

            finally:
                # remove virtual nodes
                self.graph_struct.remove_node(virtual_1)
                self.graph_struct.remove_node(virtual_2)
                virtual_1.remove_from_mesh()
                virtual_2.remove_from_mesh()

            # compute cost
            cost = sum(self.graph_struct[path[i]][path[i + 1]]["cost"]
                       for i in range(len(path) - 1))

            return path, cost

        raise NotImplementedError("Graph library not supported")
