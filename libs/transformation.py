# coding=utf-8
"""
Transformation module
A transformation creates a vertex from another vertex, according to parameters
Examples :
• orthogonal projection,
• parallel projection,
• translation etc.
"""

from typing import Callable, Dict, Optional, TYPE_CHECKING

from libs.utils.custom_types import Vector2d, Coords2d
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
        self.params = params or {}
        self.action = action

    def config(self, **params):
        """
        Sets the transformation parameters
        :param params:
        :return:
        """
        self.params = params
        return self

    def apply_to(self, vertex: 'Vertex') -> Optional['Vertex']:
        """
        Executes the transformation
        :param vertex:
        :return: a new vertex or None (if the transformation cannot be performed)
        """
        from libs.mesh import Vertex

        new_point = self.action(vertex, **self.params)
        if new_point is None:
            return None

        new_vertex = Vertex(vertex.mesh, *new_point)

        return new_vertex


# Standard actions :
# Note : actions must return a coordinates Tuple

def _barycenter_action(source_vertex: 'Vertex', vertex: 'Vertex', coeff: float) -> Coords2d:
    """
    Computes the barycenter
    :param source_vertex
    :param vertex
    :param coeff
    :return: a coordinates Tuple
    """
    if coeff is None or coeff > 1 or coeff < 0:
        raise ValueError('A barycenter coefficient' +
                         'should have a value between 0 and 1: {0}'.format(coeff))

    return barycenter(source_vertex.coords, vertex.coords, coeff)


def _translation_action(source_vertex: 'Vertex',
                        vector: Vector2d,
                        coeff: Optional[float] = 1.0) -> Coords2d:
    """
    Translates a vertex according to a vector
    :param source_vertex:
    :param vector:
    :return: a coordinates Tuple
    """
    return move_point(source_vertex.coords, vector, coeff)


def _projection_action(source_vertex: 'Vertex',
                       vector: Vector2d, edge: 'Edge') -> Optional[Coords2d]:
    """
    Projects a vertex unto an edge according to a vector.
    Will return None if the projection point is outside the edge
    :param source_vertex:
    :param vector:
    :param edge:
    :return: a coordinates Tuple
    """
    # check if the edge is facing the correct direction
    # per convention we can only project on an opposite facing edge
    # the edge also cannot be perpendicular to the projection vector
    if same_half_plane(edge.normal, vector):
        return None

    # we use shapely to compute the intersection point
    # create an extended shapely lineString from the edge
    sp_edge = edge.as_sp_extended
    # create a line to the edge at the vertex position
    sp_line = source_vertex.sp_half_line(vector)
    temp_intersection_point = sp_edge.intersection(sp_line)

    # we are only interested in a clean intersection
    # meaning only one Point
    if temp_intersection_point.geom_type != 'Point':
        return None

    # return a new vertex at the intersection point
    return temp_intersection_point.coords[0]

# transformations catalogue


get = {
    'barycenter': Transformation('barycenter', _barycenter_action),
    'translation': Transformation('translation', _translation_action),
    'projection': Transformation('projection', _projection_action)
}
