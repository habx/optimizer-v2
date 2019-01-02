# coding=utf-8
"""
graph
Contains utility for graph
using networkx library
"""

import networkx as nx
from typing import Any, Generator


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
