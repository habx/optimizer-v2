# coding=utf-8
"""
Test module for graph
"""

import libs.utils.graph as graph
from libs.mesh.mesh import Mesh

def test_shortest_path():
    """
    Test a simple
    :return:
    """
    perimeter = [(0, 0), (500, 0), (500, 500), (200, 500), (200, 200), (0, 200)]
    mesh = Mesh().from_boundary(perimeter)
    face = mesh.faces[0]
    edges = list(face.edges)
    g = graph.EdgeGraph()
    for edge in edges:
        g.add_edge(edge, edge.length)
    first_group = [edges[0], edges[5]]
    second_group = [edges[2], edges[3]]
    path, cost = g.get_shortest_path(first_group, second_group)
    assert len(path) == 2
    assert cost == 200

test_shortest_path()