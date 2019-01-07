# coding=utf-8
"""
Test module
"""
from libs.mesh import Vertex, Edge, Mesh
import libs.transformation as transformation


def test_compute_a_barycenter():
    """
    Test
    :return:
    """
    mesh = Mesh()
    vertex_1 = Vertex(mesh)
    vertex_2 = Vertex(mesh, 10, 10)
    vertex_3 = (transformation.get['barycenter']
                .config(vertex=vertex_2, coeff=0.5)
                .apply_to(vertex_1))

    assert vertex_3.coords == (5.0, 5.0)


def test_translate_a_vertex():
    """
    Test
    :return:
    """
    mesh = Mesh()
    vertex_1 = Vertex(mesh, 1.0, 1.0)
    vector = (3.0, 2.0)
    vertex_3 = (transformation.get['translation']
                .config(vector=vector)
                .apply_to(vertex_1))

    assert vertex_3.coords == (4.0, 3.0)


def test_project_a_vertex():
    """
    Test
    :return:
    """
    mesh = Mesh()
    vertex_2 = Vertex(mesh, 1.0, 1.0)
    vertex_1 = Vertex(mesh, 3.0, 3.0)
    vertex_3 = Vertex(mesh, 2.0, 0.0)
    vector = (0.0, 1.0)
    next_edge = Edge(mesh, vertex_2, None, None)
    edge = Edge(mesh, vertex_1, next_edge, None)
    vertex_4 = (transformation.get['projection']
                .config(vector=vector, edge=edge)
                .apply_to(vertex_3))

    assert vertex_4.coords == (2.0, 2.0)
