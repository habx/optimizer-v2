# coding=utf-8
"""
Transformation module
A transformation creates a vertex from another vertex, according to parameters
Examples :
• orthogonal projection,
• parallel projection,
• translation etc.
"""

import copy
from typing import Callable, Dict, Optional, TYPE_CHECKING
import logging

from libs.utils.custom_types import Vector2d
from libs.utils.geometry import (
    barycenter,
    move_point,
    same_half_plane
)

if TYPE_CHECKING:
    from libs.mesh import Vertex, Edge


class Transformation:
    """
    Transformation class
    """
    def __init__(self, name: str, action: Callable, params: Optional[Dict] = None):
        self.name = name
        self._params = params or {}
        self.action = action

    def config(self, **params):
        """
        Sets the transformation parameters
        :param params:
        :return:
        """
        self._params = params
        return self

    def apply_to(self, vertex: 'Vertex') -> Optional['Vertex']:
        """
        Executes the transformation
        :param vertex:
        :return: a new vertex or None (if the transformation cannot be performed)
        """

        new_vertex = self.action(vertex, **self._params)
        if new_vertex is None:
            return None

        vertex.add_child(new_vertex)
        new_vertex.parent = vertex
        new_vertex.transformation = copy.copy(self)

        return new_vertex


# Standard actions :
def _barycenter_action(source_vertex: 'Vertex', vertex: 'Vertex', coeff: float) -> 'Vertex':
    """
    Computes the barycenter
    :param source_vertex
    :param vertex
    :param coeff
    :return:
    """
    from libs.mesh import Vertex

    if coeff is None or coeff > 1 or coeff < 0:
        raise ValueError('A barycenter coefficient' +
                         'should have a value between 0 and 1: {0}'.format(coeff))

    x, y = barycenter(source_vertex.coords, vertex.coords, coeff)

    return Vertex(x, y)


def _translation_action(source_vertex: 'Vertex', vector: Vector2d) -> 'Vertex':
    """
    Translates a vertex according to a vector
    :param source_vertex:
    :param vector:
    :return: Vertex
    """
    from libs.mesh import Vertex

    return Vertex(*move_point(source_vertex.coords, vector))


def _projection_action(source_vertex: 'Vertex',
                       vector: Vector2d, edge: 'Edge') -> Optional['Vertex']:
    """
    Projects a vertex unto an edge according to a vector.
    Will return None if the projection point is outside the edge
    :param source_vertex:
    :param vector:
    :param edge:
    :return:
    """
    from libs.mesh import Vertex
    # check if the edge is facing the correct direction
    # per convention we can only project on an opposite facing edge
    # the edge also cannot be perpendicular to the projection vector
    if same_half_plane(edge.normal, vector):
        return None

    # we use shapely to compute the intersection point
    # create an extended shapely lineString from the edge
    sp_edge = edge.as_sp_extended
    # create a line to the edge at the vertex position
    sp_line = source_vertex.sp_line(vector)
    temp_intersection_point = sp_edge.intersection(sp_line)

    # we are only interested in a clean intersection
    # meaning only one Point
    if temp_intersection_point.geom_type != 'Point':
        return None

    # return a new vertex at the intersection point
    return Vertex(*temp_intersection_point.coords[0])


# transformations catalogue


get = {
    'barycenter': Transformation('barycenter', _barycenter_action),
    'translation': Transformation('translation', _translation_action),
    'projection': Transformation('projection', _projection_action)
}


if __name__ == '__main__':

    def compute_a_barycenter():
        """
        Test
        :return:
        """
        vertex_1 = Vertex()
        vertex_2 = Vertex(10, 10)
        vertex_3 = (get['barycenter']
                    .config(vertex=vertex_2, coeff=0.5)
                    .apply_to(vertex_1))

        print(vertex_3)

    compute_a_barycenter()

    def translate_a_vertex():
        """
        Test
        :return:
        """
        vertex_1 = Vertex(1.0, 1.0)
        vector = (3.0, 2.0)
        vertex_3 = (get['translation']
                    .config(vector=vector)
                    .apply_to(vertex_1))

        print(vertex_3)

    translate_a_vertex()

    def project_a_vertex():
        """
        Test
        :return:
        """
        vertex_2 = Vertex(1.0, 1.0)
        vertex_1 = Vertex(3.0, 3.0)
        vertex_3 = Vertex(2.0, 0.0)
        vector = (0.0, 1.0)
        next_edge = Edge(vertex_2, None, None)
        edge = Edge(vertex_1, next_edge, None)
        vertex_4 = (get['projection']
                    .config(vector=vector, edge=edge)
                    .apply_to(vertex_3))

        print(vertex_4)

    project_a_vertex()
