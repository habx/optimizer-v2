# coding=utf-8
"""
Mesh module
"""

import math
from typing import Optional, Tuple, List, Sequence, Any, Generator
from random import randint
import matplotlib.pyplot as plt
import shapely as sp
from shapely.geometry.polygon import Polygon
from shapely.geometry import Point, LineString, LinearRing
import numpy as np

# model arbitrary parameters
# arbitrary value for the length of the line : it should be long enough to approximate infinity
# TODO all our arbitrary values should be in a different module or properties of the Mesh class
LINE_LENGTH = 500000
ANGLE_EPSILON = 1  # value to check if an angle has a specific value
COORD_EPSILON = 1  # coordinates precision for snapping purposes
MIN_ANGLE = 10  # min. acceptable angle in grid
COORD_DECIMAL = 2  # number of decimal of the points coordinates

# Type definition
# TODO : all our custom types should be in a separate module
Vector2d = Coords2d = Tuple[float, float]

# MODULE FUNCTIONS


def previous(item: Any, list_items: Sequence) -> Any:
    """
    Returns the previous item in a list
    :param item:
    :param list_items:
    :return: item
    """
    i_x = list_items.index(item)
    i_x = len(list_items) if i_x == 0 else i_x
    return list_items[i_x - 1]


def magnitude(vector: Vector2d) -> float:
    """
    Returns the magnitude of a vectorector
    :param vector: vector as a tuple
    :return: float
    """
    mag = np.sqrt(vector[0]**2 + vector[1]**2)
    return mag


def ccw_angle(vector_1: Vector2d, vector_2: Optional[Vector2d] = None) -> float:
    """
    Calculates the counter clockwise angle (in degrees) between vector_1 and vector_2
    if only one vector is given returns the ccw angle of the vector between [O, 360[
    :param vector_1: tuple
    :param vector_2: tuple
    :return: float, angle in deg
    """
    ang1 = np.arctan2(*vector_1[::-1])
    ang2 = ang1 if vector_2 is None else np.arctan2(*vector_2[::-1])
    ang1 = 0 if vector_2 is None else ang1
    ang = np.rad2deg((ang2 - ang1) % (2 * np.pi))
    # WARNING : we round the angle to prevent floating point error
    return np.round(ang) % 360.0


def angle_pseudo_equal(angle: float, value: float) -> bool:
    """
    Verify if an angle is very close to a specific value
    :param angle: float
    :param value: float
    :return: boolean
    """
    return value + ANGLE_EPSILON > angle > value - ANGLE_EPSILON


def nearest_point(point: Point, perimeter: LinearRing) -> Point:
    """
    Returns the first nearest point of a perimeter from a specific point
    :param point: shapely point
    :param perimeter:  shapely linearRing
    :return: shapely point
    """
    dist = perimeter.project(point)
    nearest_p = perimeter.interpolate(dist)
    return nearest_p


def move_point(point: Coords2d, vector: Vector2d, coeff: Optional[float] = 1.0) -> Coords2d:
    """
    Utility function to translate a point according to a vector direction
    :param point:
    :param vector:
    :param coeff:
    :return: a coordinates tuple
    """
    _x = point[0] + vector[0] * coeff
    _y = point[1] + vector[1] * coeff
    return _x, _y


def scale_line(line_string: LineString, ratio: float) -> LineString:
    """
    Returns a slightly longer lineString
    :param line_string:
    :param ratio:
    :return: LineString:
    """
    # noinspection PyUnresolvedReferences
    return sp.affinity.scale(line_string, ratio, ratio)


def unit_vector(angle: float) -> Vector2d:
    """
    Returns a unit vector oriented according to angle
    :param angle: float : an angle in degrees
    :return: a vector tuple
    """
    # convert angle to range [-pi, pi]
    angle %= 360
    angle = angle - 360 * np.sign(angle) if np.abs(angle) > 180 else angle
    rad = angle * np.pi / 180
    return np.cos(rad), np.sin(rad)


def normalized_vector(vector: Vector2d) -> Vector2d:
    """
    Returns a vector of same direction but of length 1
    :param vector:
    :return:
    """
    vector_length = magnitude(vector)
    coord_x = vector[0] / vector_length
    coord_y = vector[1] / vector_length

    return coord_x, coord_y


def sp_line_from_vertex(vertex: 'Vertex', vector: Vector2d,
                        length: float = LINE_LENGTH) -> LineString:
    """
    Returns a shapely LineString starting from the vertex and following the vector and of length Li
    :param vertex: startpoint of the LineString
    :param vector: direction of the lineString
    :param length: float length of the lineString
    :return:
    """
    start_point = vertex.coords
    end_point = start_point[0] + vector[0] * length, start_point[1] + vector[1] * length
    return LineString([start_point, end_point])


def distance(vertex_1: 'Vertex', vertex_2: 'Vertex') -> float:
    """
    computes the length of a segment between two vertices
    :param vertex_1: vertex
    :param vertex_2: vertex
    :return: float
    """
    vector = vertex_1.x - vertex_2.x, vertex_1.y - vertex_2.y
    return magnitude(vector)


def barycenter(point_1: Coords2d, point_2: Coords2d, coeff: float) -> Coords2d:
    """
    Calculates the barycenter of two points
    :param point_1: coordinates tuple
    :param point_2: coordinates tuple
    :param coeff: float, barycenter coefficient
    :return: coordinates tuple
    """
    x_coord = point_1[0] + (point_2[0] - point_1[0]) * coeff
    y_coord = point_1[1] + (point_2[1] - point_1[1]) * coeff
    return x_coord, y_coord


def random_color() -> str:
    """
    Convenient function for matplotlib
    :return: string
    """
    matplotlib_colors = ('b', 'g', 'r', 'c', 'm', 'y', 'k')
    return matplotlib_colors[randint(0, 6)]


def add_random_noise(coords: Coords2d, maximum: float = 1.0) -> Coords2d:
    """
    Adds random error to coordinates to test snapping
    :param coords:
    :param maximum: max absolute value of noise
    :return:
    """
    x_coord = coords[0] + random_unit() * maximum
    y_coord = coords[1] + random_unit() * maximum
    return x_coord, y_coord


def random_unit(dec: int = 100) -> float:
    """
    Returns a random float between -1 and +1
    :param dec: precision
    :return:
    """
    return float((randint(0, 2 * dec + 1) - dec)/dec)


def make_arrow(point: Coords2d, vector: Vector2d, normal: Vector2d) -> LineString:
    """
    Returns a polyline of the form o a semi-arrow translated from original edge
    Used to plot half-edge for debugging purposes
    :param point: starting point of the edge
    :param vector:
    :param normal:
    :return: LineString
    """
    distance_to_point = 10
    arrow_angle = 20
    arrow_head_length = 10
    vector_unit = normalized_vector(vector)

    start_point = move_point(point, normal, distance_to_point)
    end_point = move_point(start_point, vector)

    # reduce size of the arrow for better clarity
    if magnitude(vector) > 2 * distance_to_point:
        end_point = move_point(end_point, vector_unit, -1 * distance_to_point)
        start_point = move_point(start_point, vector_unit, distance_to_point)

    relative_angle = ccw_angle(vector) + 180 - arrow_angle
    arrow_head_vector = unit_vector(relative_angle)
    arrow_head_point = move_point(end_point, arrow_head_vector, arrow_head_length)
    arrow = LineString([start_point, end_point, arrow_head_point])

    return arrow


def plot_polygon(_ax, data1: Sequence[float], data2: Sequence[float],
                 options: Sequence[str] = ('vertices', 'edges', 'fill')):
    """
    Simple convenience function to plot a mesh face with matplotlib
    :param _ax:
    :param data1:
    :param data2:
    :param options: tuple of strings indicating what to show on plot
    :return:
    """
    if 'fill' in options:
        _ax.fill(data1, data2, alpha=0.3)
    if 'edges' in options:
        _ax.plot(data1, data2, 'k')
    if 'vertices' in options:
        _ax.plot(data1, data2, 'ro')

    return _ax


# MODULE CLASSES

class Vertex:
    """
    Vertex class
    """
    def __init__(self, x: float = 0, y: float = 0, edge: 'Edge' = None):
        """
        A simple Vertex class with barycentric capability
        By default sets the vertex to the origin (0, 0)
        :param x: float, x-axis coordinates
        :param y: float, y-axis coordinates
        :param edge: one edge starting from the vertex
        """
        self._x = float(np.around(float(x), decimals=COORD_DECIMAL))
        self._y = float(np.around(float(y), decimals=COORD_DECIMAL))
        self._edge = edge
        # attributes used to store barycenter data
        self._parents = None  # a tuple (vertex 1, vertex 2)
        self._coeff = None  # barycenter coefficient

    def __repr__(self):
        return 'vertex: ({x}, {y}) - {i}'.format(x=self.x, y=self.y, i=id(self))

    @property
    def x(self) -> float:
        """
        property
        :return: the x coordinate
        """
        return self._x

    @x.setter
    def x(self, value: float):
        """
        property
        Sets the x coordinate
        """
        self._x = float(value)

    @property
    def y(self) -> float:
        """
        property
        :return: the y coordinate
        """
        return self._y

    @y.setter
    def y(self, value: float):
        """
        property
        Sets the y coordinate
        """
        self._y = float(value)

    @property
    def edge(self):
        """
        property
        :return: the edge of the vertex
        """
        return self._edge

    @edge.setter
    def edge(self, value: 'Edge'):
        """
        property
        Sets the edge the vertex starts
        """
        self._edge = value

    @property
    def parents(self) -> Tuple:
        """
        property
        :return: the parents vertices of a barycentric vertex as a tuple
        """
        return self._parents

    @parents.setter
    def parents(self, value: Tuple):
        """
        property
        Sets the parents of a barycentric vertex
        """
        self._parents = value

    @property
    def coeff(self) -> float:
        """
        property
        :return: the parents vertices of a barycentric vertex as a tuple
        """
        return self._coeff

    @coeff.setter
    def coeff(self, value):
        """
        property
        Sets the parents of a barycentric vertex
        """
        self._parents = value

    @property
    def coords(self):
        """
        Returns the coordinates of the vertex in the form of a tuple
        :return:
        """
        return self.x, self.y

    @property
    def as_sp(self):
        """
        Returns a shapely Point data structure
        :return: shapely Point
        """
        return Point(self.x, self.y)

    @property
    def previous(self):
        """
        Returns the previous vertex according to edge flow
        :return: vertex
        """
        return self.edge.previous.start

    @property
    def next(self):
        """
        Returns the next vertex according to edge flow
        :return: vertex
        """
        return self.edge.next.start

    @property
    def edges(self) -> Generator['Edge', None, None]:
        """
        Returns all edges starting from the vertex
        :return: generator
        """
        yield self.edge
        edge = self.edge.previous.pair
        while edge is not self.edge:
            yield edge
            edge = edge.previous.pair

    def get_mesh(self) -> 'Mesh':
        """
        Returns the mesh containing the vertex
        :return: mesh
        """
        if self.edge.face is None:
            return self.edge.pair.face.get_mesh()

        return self.edge.face.get_mesh()

    def barycenter(self, vertex_1: 'Vertex', vertex_2: 'Vertex', coeff: float) -> 'Vertex':
        """
        Sets the coordinates to the barycenter of the given vertices
        and stores the information
        :param vertex_1:
        :param vertex_2:
        :param coeff: barycentric coefficient (starting from first vertex)
        :return: self
        """
        if coeff == 0:
            return vertex_1

        if coeff == 1:
            return vertex_2

        if coeff > 1 or coeff < 0:
            raise Exception('A barycenter coefficient should have a value between 0 and 1')

        self.parents = vertex_1, vertex_2
        self.coeff = coeff
        self.x, self.y = barycenter(vertex_1.coords, vertex_2.coords, coeff)

        return self

    def clean(self) -> Optional['Vertex']:
        """
        Removes an unneeded vertex. It is a vertex that is used only by one edge.
        :return: self or None
        """
        edges = list(self.edges)
        nb_edges = len(edges)
        if nb_edges > 2:
            print('Cannot clean a vertex used by more than one edge')
            return self
        if nb_edges == 2:
            if edges[0].previous.pair is not edges[1]:
                raise ValueError('Vertex is malformed and cannot be cleaned:{0}'.format(self))
            if self.edge.previous.next_is_aligned:
                self.edge.previous.next = self.edge.next
                self.edge.pair.next = self.edge.pair.next.next
                return None
        return self

    def is_equal(self, other: 'Vertex') -> bool:
        """
        Pseudo equality operator in order
        to avoid near misses due to floating point precision
        :param other:
        :return:
        """
        return distance(self, other) < COORD_EPSILON

    def nearest_point(self, face: 'Face') -> 'Vertex':
        """
        Returns the nearest point on the perimeter of the given face, to the vertex
        :param face:
        :return: vertex
        """
        point = nearest_point(self.as_sp, face.as_sp_linear_ring())
        return Vertex(*point.coords[0])

    def distance_to(self, other: 'Vertex') -> float:
        """
        Returns the distance between the vertex and another
        :param other: vertex
        :return: float
        """
        return distance(self, other)

    def snap_to(self, *others: 'Vertex') -> 'Vertex':
        """
        Used to snap a vertex to another one that is closed.
        The function returns the first vertex from the argument list
        that is localized inside the approximation radius (given by the pseudo equality is_equal)
        or self if no vertex is close enough
        example : vertex_to_snap = vertex_to_snap.snap_to(v1, v2, v3)
        :param others: one or many vertices
        :return: a vertex
        """
        for other in others:
            # case we try to snap a vertex to itself
            if self is other:
                return self
            if self.is_equal(other):
                # ensure that the reference to the vertex are still valid
                if self.edge is not None:
                    for edge in list(self.edges):
                        edge.start = other
                return other
        return self

    def snap_to_edge(self, *edges: 'Edge') -> Optional['Edge']:
        """
        Snaps a vertex to an edge if the vertex is close enough to the edge
        we split the edge and insert the vertex
        We snap to the first edge close enough
        :param edges: edges to check
        :return: the edge if a snap happened, None otherwise
        """
        for edge in edges:
            dist = edge.as_sp.distance(self.as_sp)
            if dist < COORD_EPSILON:
                return edge.split(self).next
        return None

    def get_edge(self, edges_list: Sequence['Edge']) -> Optional['Edge']:
        """
        Returns the first edge starting with this vertex from
        a list of edges
        :param edges_list:
        :return:
        """
        for edge in edges_list:
            if edge.start is self:
                return edge
        return None


class Edge:
    """
    Half Edge class
    """
    def __init__(self, start: Optional[Vertex], next_edge: Optional['Edge'],
                 face: Optional['Face'], pair: Optional['Edge'] = None):
        """
        A half edge data structure implementation.
        By convention our half edge structure is based on a CCW rotation.

        :param start: Vertex starting point for the edge
        :param pair: twin edge of opposite face
        :param face: the face that the edge belongs to, can be set to None for external edges
        :param next_edge: the next edge
        """
        self._start = start
        self._next = next_edge
        self._face = face
        # always add a pair Edge, because an edge should always have a pair edge
        self._pair = pair if pair else Edge(None, None, None, self)

    def __repr__(self):
        output = 'Edge: '
        output += 'id:{0}'.format(id(self))
        output += ' [({x1}, {y1}), ({x2},{y2})]'.format(x1=self.start.x, y1=self.start.y,
                                                        x2=self.end.x, y2=self.end.y)
        output += ' - f:{0}'.format(id(self.face))
        output += ' - n:{0}'.format(id(self.next))

        return output

    @property
    def start(self) -> Vertex:
        """
        property
        :return: the starting vertex of the edge
        """
        return self._start

    @start.setter
    def start(self, value: Vertex):
        """
        property
        Sets the starting vertex of the edge
        """
        self._start = value

    @property
    def pair(self) -> 'Edge':
        """
        property
        :return: the pair Edge of the edge
        """
        return self._pair

    @pair.setter
    def pair(self, value: 'Edge'):
        """
        property
        Sets the pair Edge of the edge
        """
        self._pair = value
        # a pair should always be reciprocal
        # note: we cannot use the setter because it will induce an infinite loop
        value._pair = self

    @property
    def next(self) -> 'Edge':
        """
        property
        :return: the next Edge of the edge
        """
        return self._next

    @next.setter
    def next(self, value: 'Edge'):
        """
        property
        Sets the next Edge of the edge
        """
        self._next = value

    @property
    def face(self) -> 'Face':
        """
        property
        :return: the face of the edge
        """
        return self._face

    @face.setter
    def face(self, value: Optional['Face']):
        """
        property
        Sets the face of the edge
        """
        self._face = value  # should be None for a boundary edge

    def get_mesh(self) -> 'Mesh':
        """
        Returns the mesh containing the edge
        :return: mesh
        """
        return self.face.get_mesh()

    @property
    def next_angle(self) -> float:
        """
        returns the counter clockwise angle between the next edge and this one
        :return: angle in degree
        """
        return ccw_angle(self.next.vector, self.opposite_vector)

    @property
    def previous_angle(self) -> float:
        """
        returns the counter clockwise angle in degrees between the edge and the previous one
        :return: angle in degree
        """
        return ccw_angle(self.vector, self.previous.opposite_vector)

    @property
    def next_is_outward(self) -> bool:
        """
        Specifies it the next edge is outward or inward
        :return: boolean
        """
        return self.next_angle > 180

    @property
    def next_is_aligned(self) -> bool:
        """
        Indicates if the next edge is approximately aligned with this one,
        using a pseudo equality on the angle
        :return: boolean
        """
        angle = angle_pseudo_equal(self.next_angle, 180)
        return angle

    @property
    def next_is_ortho(self) -> bool:
        """
        Indicates if the next edge i
        :return:
        """
        return angle_pseudo_equal(self.next_angle, 90)

    @property
    def length(self) -> float:
        """
        Calculates the length of the edge
        :return: float the length of the edge
        """
        return distance(self.start, self.end)

    @property
    def opposite_vector(self) -> Vector2d:
        """
        Convenient function to calculate the opposite vector of the edge
        :return: tuple containing x, y values
        """
        return self.start.x - self.end.x, self.start.y - self.end.y

    @property
    def vector(self) -> Vector2d:
        """
        Convenient function to calculate the direction vector of the edge
        :return: tuple containing x, y values
        """
        return self.end.x - self.start.x, self.end.y - self.start.y

    @property
    def unit_vector(self) -> Vector2d:
        """
        Returns a unit vector with the same direction as the edge
        :return: vector of length 1
        """
        return normalized_vector(self.vector)

    @property
    def end(self) -> Optional[Vertex]:
        """
        Returns the vertex at the end of the edge, if the edge has no next edge : return None
        :return: vertex
        """
        if self.next is None:
            return None
        return self.next.start

    @property
    def previous(self) -> 'Edge':
        """
        Returns the previous edge by looping through the whole face.
        Will fail if the edge is not a member of a proper formed face.
        :return: edge
        """
        for edge in self.siblings:
            if edge.next is None:
                # Note : we could actually allow this but I think this better for debugging purposes
                raise Exception('The face is badly formed : one of the edge has not a next edge')
            if edge.next is self:
                return edge
        raise Exception('Not previous edge found !')

    @property
    def ccw(self) -> 'Edge':
        """
        Returns the next edge starting from the same vertex as the edge.
        In counter clockwise order.
        :return: edge
        """
        return self.previous.pair

    @property
    def cw(self) -> 'Edge':
        """
        Returns the next edge starting from the same vertex as the edge.
        In clockwise order
        :return: edge
        """
        return self.pair.next

    @property
    def normal(self) -> Vector2d:
        """
        A CCW normal of the edge of length 1
        :return: a tuple containing x, y values
        """
        x, y = -self.vector[1], self.vector[0]
        length = math.sqrt(x**2 + y**2)
        return x/length, y/length

    def sp_ortho_line(self, vertex) -> LineString:
        """
        Returns a shapely LineString othogonal to the edge starting from the vertex
        :param vertex:
        :return: shapely LineString
        """
        return sp_line_from_vertex(vertex, self.normal)

    @property
    def as_sp(self) -> LineString:
        """
        The edge as shapely LineString
        :return: shapely LineString
        """
        return LineString([self.start.as_sp, self.end.as_sp])

    @property
    def as_sp_extended(self) -> LineString:
        """
        Returns a slightly longer lineString (from both edges) in order to prevent near miss
        due to floating point precision
        :return:
        """
        ratio = 2 * COORD_EPSILON / self.length + 1
        return scale_line(self.as_sp, ratio)

    @property
    def as_sp_dilated(self) -> Polygon:
        """
        Returns a dilated lineString to ensure snapping
        in order to prevent near miss due to floating point precision
        :return: a shapely Polygon
        """
        return self.as_sp.buffer(COORD_EPSILON, 1)

    @property
    def siblings(self) -> Generator['Edge', None, None]:
        """
        Returns the siblings of the edge
        :return: generator yielding each edge in the loop
        """
        yield self
        edge = self.next
        seen = []  # in order to detect infinite loop we stored each yielded edge
        while edge is not self:
            if edge in seen:
                print('INFINITE LOOP duplicate edge:{0}'.format(edge))
                # self.get_mesh().check()
                raise Exception('Infinite loop starting from edge:{0}'.format(self))
            seen.append(edge)
            yield edge
            edge = edge.next

    def is_linked_to_face(self, face: 'Face') -> bool:
        """
        Indicates if an edge is still linked to its face
        :param face:
        :return: boolean
        """
        if face.edge is self:
            return True
        for edge in self.siblings:
            if edge is face.edge:
                return True
        return False

    def replace_by(self, other: 'Edge') -> 'Edge':
        """
        Replaces an edge by another one
        :param other:
        :return:
        """
        if self.end is not other.end or self.start is not other.start:
            raise Exception('Cannot replace an edge by another with different extremity vertices')
        self.previous.next = other
        return self

    def remove(self):
        """
        Removes the edge from the face. The coresponding faces are merged
        1. remove the face from the mesh
        2. stitch the extremities of the remove edge
        3. attribute correct face to orphan edges
        :return: mesh
        """
        other_face = self.pair.face

        # attribute new face to orphan
        # note if the edge is a boundary edge we attribute the edge face
        # to the pair edge siblings, not the other way around
        if self.face is not None:
            self.face.remove_from_mesh()
            for edge in self.siblings:
                edge.face = other_face
        else:
            for edge in self.pair.siblings:
                edge.face = self.face

        # stitch the edge extremities
        self.pair.previous.next = self.next
        self.previous.next = self.pair.next

        # preserve the vertices start edge reference
        # for both the edge and its pair edge
        if self.start.edge is self:
            self.start.edge = self.pair.next
        if self.pair.start.edge is self.pair:
            self.pair.start.edge = self.next

        # clean eventually useless vertices
        self.start.clean()
        self.end.clean()

    def intersect(self, vector: Vector2d) -> 'Edge':
        """
        Finds the opposite point of the edge end on the face boundary
        according to a vector direction.
        A vertex is created at the corresponding point by splitting the edge.
        Returns the edge created at the intersection vertex, before the intersection.
        :param vector:
        :return: the cut edge
        """

        # create a line to the edge at the vertex position
        sp_line = sp_line_from_vertex(self.end, vector)

        # find the intersection
        closest_edge = None
        intersection_point = None
        smallest_distance = None

        # iterate trough every edge of the face but the cut edge
        for edge in self.siblings:
            # do not check for the first edge or the second one
            if edge is self:
                continue
            if edge is self.next:
                continue

            # looks for an intersection
            sp_edge = edge.as_sp_extended
            if sp_edge.intersects(sp_line):
                temp_intersection_point = sp_edge.intersection(sp_line)
                # if the orthogonal line intersects more than the point
                # (eg: overflows a whole edge) we do not cut
                if temp_intersection_point.geom_type != 'Point':
                    continue
                new_distance = sp_line.project(temp_intersection_point)
                if smallest_distance is None or new_distance < smallest_distance:
                    smallest_distance = new_distance
                    closest_edge = edge
                    intersection_point = temp_intersection_point

        # check if we have found an intersection point
        if smallest_distance is None:
            print('ERROR : could not cut the edge:{0}'.format(self))
            return self

        # create the intersection vertex
        intersection_vertex = Vertex(*intersection_point.coords[0])

        # split the intersected edge
        closest_edge = closest_edge.split(intersection_vertex)

        return closest_edge

    def link(self, other: 'Edge') -> 'Edge':
        """
        Creates an edge between two edges.
        We link the end vertices of each edge
        Note : we cannot linked two edges that are already linked
        (that are the next edge of each other)
        :param other: the edge to link to
        :return: the created edge
        """
        # check if the edges are already linked
        if other.next is self or self.next is other:
            print('WARNING: cannot link two edges ' +
                  ' that are already linked:{0}-{1}'.format(self, other))
            return self

        # create the new edge and its pair
        new_edge = Edge(self.end, other.next, self.face)
        new_edge.pair.start = other.end
        new_edge.pair.next = self.next

        # create a new face
        new_face = Face(new_edge.pair, self.get_mesh())
        new_edge.pair.face = new_face

        # modify initial edges next edges to follow the cut
        self.next = new_edge
        other.next = new_edge.pair

        # assign all the edges from one side of the cut to the new face
        for edge in new_edge.pair.siblings:
            edge.face = new_face

        return new_edge

    def cut(self, vertex: Vertex, angle: float = 90.0,
            traverse: str = 'absolute', vector: Optional[Vector2d] = None,
            initial: Optional['Edge'] = None) -> 'Edge':
        # TODO : add a max cut length (do not cut if the new edge exceeds the max_length)
        """
        Will cut a face from the edge at the given vertex
        following the given angle or the given vector
        :param vertex: where to cut the edge
        :param angle : indicates the angle to cut the face from the vertex
        :param traverse: String indicating if we must keep cutting the next face and how.
        Possible values : 'absolute', 'relative'
        :param vector: a vector indicating the absolute direction fo the cut,
        if present we ignore the angle parameter
        :param initial : initial edge reference to prevent looping for recursive cuts
        :return: self
        """

        # do not cut an edge on the boundary
        if self.face is None:
            return self

        # in case of a recursive cut check if we have looped on initial edge
        if self is initial:
            print('No cut is possible at this vertex:{0}'.format(self.start))
            return self

        first_edge = self

        # initialize reference for recursive cuts
        initial = initial if initial is not None else self

        # a relative angle or a vector can be provided as arguments of the method
        if vector is not None:
            angle = ccw_angle(self.vector, vector)

        # snap vertex if they are very close to the end or the start of the edge
        vertex = vertex.snap_to(self.start, self.end)

        # check for extremity cases
        # 1. the vertex is the start of the edge
        if vertex is self.start:
            angle_is_inside = angle < self.previous_angle - MIN_ANGLE

            if not angle_is_inside:
                if vector is not None:
                    return self.ccw.cut(vertex, vector=vector, initial=initial)

                print('WARNING: cannot cut according to angle:{0}'.format(angle))
                return self

            first_edge = self.previous
        # 2. the vertex is the end of the edge
        elif vertex is self.end:
            angle_is_inside = angle > 180.0 - self.next_angle + MIN_ANGLE

            if not angle_is_inside:
                if vector is not None:
                    return self.cw.cut(vertex, vector=vector, initial=initial)

                print('WARNING: cannot cut according to angle:{0}'.format(angle))
                return self
        # 3. the vertex is in the middle of the edge:
        # split the first edge
        else:
            first_edge = first_edge.split(vertex)

        # create a line to the edge at the vertex position
        line_vector = vector if vector is not None else unit_vector(ccw_angle(self.vector) + angle)
        closest_edge = first_edge.intersect(line_vector)
        intersection_vertex = closest_edge.end

        # assign a correct edge to the initial face
        # (to ensure that its edge is still included in the face)
        first_edge.face.edge = first_edge

        # link the two edges
        first_edge.link(closest_edge)

        # cut the next edge if traverse option is set
        if traverse == 'absolute':
            return closest_edge.pair.cut(intersection_vertex, vector=line_vector)

        if traverse == 'relative':
            return closest_edge.pair.cut(intersection_vertex, angle, traverse='relative')

        return self

    def cut_at_barycenter_ortho(self, coeff: float) -> 'Edge':
        """
        Cuts an edge orthogonally from the edge at the given barycentric position
        :param coeff: float
        :return: self
        """
        return self.cut_at_barycenter(coeff)

    def cut_at_barycenter(self, coeff: float, angle: float = 90.0) -> 'Edge':
        """
        Cuts an edge according to the provided angle (90° by default)
        and at the barycentric position
        :param coeff:
        :param angle:
        :return:
        """
        vertex = Vertex().barycenter(self.start, self.end, coeff)
        return self.cut(vertex, angle)

    def split(self, vertex: 'Vertex') -> 'Edge':
        """
        Splits the edge at a specific vertex
        :param vertex: a vertex object where we should split
        :return: the edge ending with the vertex
        """
        # check for vertices proximity and snap if needed
        vertex = vertex.snap_to(self.start, self.end)

        # check extremity cases : if the vertex is one of the extremities of the edge do nothing
        if vertex is self.start:
            return self.previous
        if vertex is self.end:
            return self

        # define edges
        edge = self
        next_edge = self.next
        edge_pair = self.pair
        edge_pair_previous = self.pair.previous

        # create a new edge and assign it to vertex if needed
        new_edge = Edge(vertex, next_edge, edge.face)
        new_edge.pair.face = edge_pair.face
        new_edge.pair.start = edge_pair.start
        vertex.edge = vertex.edge if vertex.edge is not None else new_edge

        # insure correct vertex reference
        if edge_pair.start.edge is edge_pair:
            edge_pair.start.edge = new_edge.pair

        # change the current edge destinations and starting point
        edge_pair_previous.next = new_edge.pair
        edge_pair.start = vertex
        edge.next = new_edge
        new_edge.pair.next = edge_pair

        return self

    def split_barycenter(self, coeff: float) -> 'Edge':
        """
        Splits the edge at the provided barycentric position. A vertex is created.
        :param coeff: float
        :return: self
        """
        vertex = Vertex().barycenter(self.start, self.end, coeff)
        return self.split(vertex)

    def plot_half_edge(self, ax, color: str = 'black'):
        """
        Plots a semi-arrow to indicate half-edge for debugging purposes
        :param ax:
        :param color:
        :return:
        """
        arrow = make_arrow(self.start.coords, self.vector, self.normal)
        x_coords, y_coords = zip(*arrow.coords)
        ax.plot(x_coords, y_coords, 'k', color=color)

    def plot_normal(self, ax, color: str = 'black'):
        """
        Plots the normal vector of the edge for debugging purposes
        :param ax:
        :param color:
        :return:
        """
        start_point = barycenter(self.start.coords, self.end.coords, 0.5)
        arrow = self.normal
        # noinspection PyCompatibility
        ax.quiver(*start_point, *arrow, color=color)


class Face:
    """
    Face Class
    """
    def __init__(self, edge: Optional[Edge], enclosing_mesh: Optional['Mesh'] = None):
        # any edge of the face
        self._edge = edge
        self._mesh = enclosing_mesh
        # add new face to mesh
        if enclosing_mesh is not None:
            enclosing_mesh.add_face(self)

    @property
    def edge(self) -> Edge:
        """
        property
        :return: the edge of the face
        """
        return self._edge

    @edge.setter
    def edge(self, value: Edge):
        """
        property
        Sets the edge of the face
        """
        self._edge = value

    @property
    def mesh(self) -> 'Mesh':
        """
        property
        :return: the mesh of the face
        """
        return self._mesh

    @mesh.setter
    def mesh(self, value: 'Mesh'):
        """
        property
        Sets the mesh of the face
        """
        self._mesh = value

    def __repr__(self):
        output = 'Face: ['
        for edge in self.edges():
            output += '({0}, {1})'.format(*edge.start.coords)
        return output + ']'

    def edges(self, from_edge: Optional[Edge] = None) -> Generator[Edge, None, None]:
        """
        Loops trough all the edges belonging to a face.
        We start at the edge stored in the face and follow each edge next until
        a full loop has been accomplished.
        NB: if the edges of the face do not form a proper loop
        the method will fail or loop for ever
        :param from_edge: from which edge of the face the loop starts
        :return: a generator
        """
        edge = from_edge or self.edge
        return edge.siblings

    @property
    def vertices(self) -> Generator[Vertex, None, None]:
        """
        Lists the vertices on the edges of the face
        :return: Generator
        """
        return (edge.start for edge in self.edges())

    @property
    def as_sp(self) -> Polygon:
        """
        Returns a shapely Polygon corresponding to the face geometry
        :return: Polygon
        """
        list_vertices = [vertex.coords for vertex in self.vertices]
        list_vertices.append(list_vertices[0])
        return Polygon(list_vertices)

    def as_sp_linear_ring(self) -> LinearRing:
        """
        Returns a shapely LinearRing corresponding to the face perimeter
        :return: LinearRing
        """
        list_vertices = [vertex.coords for vertex in self.vertices]
        return LinearRing(list_vertices)

    @property
    def as_sp_dilated(self) -> Polygon:
        """
        Returns a dilated Polygon corresponding to the face with a small buffer
        This is useful to prevent floating point precision errors.
        :return: Polygon
        """
        return self.as_sp.buffer(COORD_EPSILON, 1)

    @property
    def area(self) -> float:
        """
        Calculates and returns the area of the face
        We use the shapely module to compute the area
        :return: float
        """
        return self.as_sp.area

    @property
    def length(self) -> float:
        """
        Calculates the perimeter length of the face
        We use the shapely module to compute the perimeter
        :return: float
        """
        return self.as_sp.length

    def get_mesh(self) -> 'Mesh':
        """
        Returns the mesh the face belongs to
        :return: mesh
        """
        return self.mesh

    def add_to_mesh(self, mesh: 'Mesh') -> 'Face':
        """
        Adds the face to a mesh
        :param mesh:
        :return: face
        """
        mesh.add_face(self)
        return self

    def remove_from_mesh(self):
        """
        Removes the face reference from the mesh
        :return: None
        """
        mesh = self.get_mesh()
        if mesh is None:
            raise ValueError('Face has no mesh to remove it from: {0}'.format(self))
        mesh.remove_face(self)

    def get_edge(self, vertex: Vertex) -> Optional[Edge]:
        """
        Retrieves the half edge of the face starting with the given vertex.
        Returns None if no edge is found.
        :param vertex:
        :return: edge
        """
        for edge in self.edges():
            if edge.start is vertex:
                return edge
        return None

    def contains(self, other: 'Face') -> bool:
        """
        Indicates if the face contains another face.
        We use a dilated face in order to prevent floating point decimal errors
        :param other:
        :return:
        """
        return self.as_sp_dilated.contains(other.as_sp)

    def add_exterior(self, other: 'Face') -> 'Face':
        """
        Assigns to the external edge the provided face.
        Useful before inserting a face inside another
        :param other: face
        :return: self
        """
        for edge in self.edge.pair.siblings:
            edge.face = other

        return self

    def _insert_enclosed_face(self, face: 'Face') -> 'Face':
        """
        Insert a fully enclosed face inside a containing face
        Note: this method should always be called from insert_face
        1. select a vertex from the face
        2. find the nearest point on the perimeter of the containing face
        3. split the edge on the nearest point
        4. create the edge between the two point
        5. assign pair faces
        :param face:
        :return:
        """
        # create a fixed list of the enclosing face edges for ulterior navigation
        self_edges = list(self.edges())

        # find the vertex of the face closest to the enclosing face
        # TODO : only use the main orientation of the mesh boundary

        min_distance = None
        best_vertex = None
        best_near_vertex = None
        for vertex in face.vertices:
            near_vertex = vertex.nearest_point(self)
            distance_to_vertex = vertex.distance_to(near_vertex)
            if min_distance is None or distance_to_vertex < min_distance:
                best_vertex = vertex
                best_near_vertex = near_vertex
                min_distance = distance_to_vertex

        # create a new edge linking the vertex of the face to the enclosing face
        edge_shared = best_near_vertex.snap_to_edge(*self_edges)
        new_edge = Edge(best_near_vertex, best_vertex.edge.previous.pair, self)
        new_edge.pair.face = self
        new_edge.pair.start = best_vertex
        best_near_vertex.edge = new_edge
        new_edge.pair.next = edge_shared
        edge_shared.previous.next = new_edge
        best_vertex.edge.pair.next = new_edge.pair

        return self

    def _insert_touching_face(self, shared_vertices_edges: Sequence[Tuple[Edge, Edge]]) -> 'Face':
        """
        Inserts a face inside another when the inserted face has one or several touching points
        with the container face. A "stitching" algorithm is used.

        WARNING : Because the inserted face touches the container face,
        this can lead to the creation of several new faces.
        Because of the way the algorithm is coded it can even lead to the
        disappearance of the initial container face from the mesh.
        In order to enable the user to preserve references to the initial container face
        (for example if other faces need to be inserted) the biggest face is returned.

        ex: new_container_face = container_face.insert(face_to_insert)

        :param shared_vertices_edges:
        :return: the largest faces created inside self
        """

        touching_edge, edge, new_face = None, None, None
        all_faces = [self]

        # connect the edges together
        for shared in shared_vertices_edges:
            touching_edge, edge = shared
            previous_edge = edge.previous
            previous_touching_edge = touching_edge.previous

            # connect the correct edges
            previous_touching_edge.next = previous_edge.pair
            edge.pair.next = touching_edge

            # insure proper face edge reference
            self.edge = touching_edge

            # backward check for isolation
            # first check for 2-edged face
            if previous_edge.pair.next.next is previous_edge.pair:
                previous_edge.pair = previous_edge.pair.next.pair

            # else check for new face creation
            elif not previous_edge.pair.is_linked_to_face(self):
                new_face = Face(previous_edge.pair, self.mesh)
                all_faces.append(new_face)
                for orphan_edge in previous_edge.pair.siblings:
                    orphan_edge.face = new_face

        # forward check : at the end of the loop check forward for isolation
        if edge.pair.next.next is edge.pair:
            edge.pair = touching_edge.pair
            # remove face from edge
            self.remove_from_mesh()  # Note : this is problematic for face absolute reference
            all_faces.pop(0)  # remove self from the list of all the faces

        # return the biggest face
        biggest_new_face = max(all_faces, key=lambda face: face.area)
        return biggest_new_face

    def insert_face(self, face: 'Face') -> 'Face':
        """
        Inserts the face if it fits inside the receiving face

        """
        if not self.contains(face):
            raise ValueError("Cannot insert a face that is not" +
                             " entirely contained in the receiving face !:{0}".format(face))

        # add the new face to the mesh
        self.mesh.add_face(face)

        # add all pair edges to face
        face.add_exterior(self)

        # create a fixed list of the face edges for ulterior navigation
        self_edges = list(self.edges())

        # snap face vertices to edges of the container face
        # for performance purpose we store the snapped vertices and the corresponding edge
        shared_edges = []
        face_vertices = list(face.vertices)
        for vertex in face_vertices:
            edge_shared = vertex.snap_to_edge(*self_edges)
            if edge_shared is not None:
                edge = vertex.edge
                shared_edges.append((edge_shared, edge))
                # after a split: update list of edges
                self_edges = list(self.edges())

        nb_shared_vertices = len(shared_edges)

        # different use cases
        # case 1 : enclosed face
        if nb_shared_vertices == 0:
            return self._insert_enclosed_face(face)
        # case 2 : same face
        if nb_shared_vertices == len(self_edges):
            print('The inserted face is equal to the container face, no need to do anything')
            return self
        # case 3 : touching face
        return self._insert_touching_face(shared_edges)


class Mesh:
    """
    Mesh Class
    """
    def __init__(self, faces: Optional[List[Face]] = None, external_edge: Optional[Edge] = None):
        self._faces = faces
        self._external_edge = external_edge

    def __repr__(self):
        output = 'Mesh:\n'
        for face in self.faces:
            output += face.__repr__() + '\n'
        return output + '-'*24

    @property
    def faces(self) -> List[Face]:
        """
        property
        :return: the faces of the mesh
        """
        return self._faces

    @faces.setter
    def faces(self, value: List[Face]):
        """
        property
        Sets the faces of the mesh
        """
        self._faces = value

    @property
    def boundary_edge(self) -> Edge:
        """
        property
        :return: one of the external edge of the mesh
        """
        return self._external_edge

    @boundary_edge.setter
    def boundary_edge(self, value: Edge):
        """
        property
        Sets the external edge of the mesh
        """
        if value.face is not None:
            raise ValueError('An external edge cannot have a face: {0}'.format(value))
        self._external_edge = value

    def boundary_edges(self):
        """
        Generator to retrieve all the external edges of the mesh
        :return: generator
        """
        if self.boundary_edge is None:
            raise ValueError('An external edge must be specified for this mesh !')

        return self.boundary_edge.siblings

    def add_face(self, face: Face) -> 'Mesh':
        """
        Adds a face to the mesh
        :param face:
        :return: self
        """
        face.mesh = self
        if self.faces is None:
            self.faces = [face]
        else:
            self.faces.append(face)
        return self

    def remove_face(self, face: Face):
        """
        Removes from the mesh the face
        :param face:
        :return: self
        """
        if face not in self.faces:
            raise ValueError('Cannot remove the face that' +
                             ' is not already in the mesh, {0}'.format(face))
        self.faces.remove(face)

    def check(self) -> bool:
        """
        Checks if the mesh is correctly formed.
        It's a best effort. Only a few strong consistencies are checked.
        :return: boolean
        """
        is_valid = True
        print('------ checking Mesh :', end=" ")
        for face in self.faces:
            # print('Checking:', face)
            for edge in face.edges():
                if edge is None:
                    is_valid = False
                    print('\n!!! Edge is None for:', face)
                if edge.face is not face:
                    is_valid = False
                    print('\nwrong face in edge:', edge, edge.face)
                if edge.pair and edge.pair.pair is not edge:
                    is_valid = False
                    print('\nwrong pair attribution:', edge, edge.pair)
                if edge.start.edge is None:
                    is_valid = False
                    print('\n!!! vertex has no edge:', edge.start)
                if edge.start.edge.start is not edge.start:
                    is_valid = False
                    print('\nwrong edge attribution in:', edge.start)
                if edge.next.next is edge:
                    is_valid = False
                    print('\n2-edges face found:', face)
        print('OK' if is_valid else '!! NOT OK', ' ---------------')
        return is_valid

    # noinspection PyCompatibility
    def plot(self, ax=None, options=('fill', 'edges', 'half-edges', 'vertices')):
        """
        Plots a mesh using matplotlib library.
        A few options can be used:
        • 'fill' : add color to each face
        • 'edges' : outline each edge
        • 'normals' : display each half-edge normal vector. Useful for debugging.
        • 'half-edge': display arrows for each oriented half-edge. Useful for debugging.
        :param ax:
        :param options:
        :return: ax
        """

        if ax is None:
            _, ax = plt.subplots()

        ax.set_aspect('equal')

        for face in self.faces:
            color = random_color()
            x, y = face.as_sp.exterior.xy
            plot_polygon(ax, x, y, options)
            for edge in face.edges():
                # display edges normal vector
                if 'normals' in options:
                    edge.plot_normal(ax, color)
                # display half edges vector
                if 'half-edges' in options:
                    edge.plot_half_edge(ax, color)

        if 'boundary-edges' in options:
            color = random_color()
            for edge in self.boundary_edges():
                edge.plot_half_edge(ax, color)

        return ax

    def from_boundary(self, boundary: Sequence[Tuple[float, float]]) -> 'Mesh':
        """
        Creates a mesh object with one face from a list of points. We use a CCW rotation.
        Each point is a tuple of x, y coordinates
        Note : the boundary has to be in CCW order and the boundary can not cross itself.
        :param boundary: list of coordinates tuples
        :return: a Mesh object
        """
        # check if the perimeter respects the ccw rotation
        # we use shapely LinearRing object
        sp_perimeter = LinearRing(boundary)
        if not sp_perimeter.is_ccw:
            raise ValueError('The perimeter is not ccw:{0}'.format(boundary))
        if not sp_perimeter.is_simple:
            raise ValueError('The perimeter crosses itself:{0}'.format(boundary))

        initial_face = Face(None, self)
        initial_vertex = Vertex(boundary[0][0], boundary[0][1])
        initial_edge = Edge(initial_vertex, None, initial_face)

        self.boundary_edge = initial_edge.pair
        initial_face.edge = initial_edge
        initial_vertex.edge = initial_edge

        next_edge = initial_edge

        # we traverse the perimeter backward
        for i, point in enumerate(boundary[::-1]):
            # for the last item we loop on the initial edge
            if i == len(boundary) - 1:
                initial_edge.next = next_edge
                next_edge.pair.next = initial_edge.pair
                initial_edge.pair.start = next_edge.start
                break
            # create a new vertex
            vertex = Vertex(point[0], point[1])
            # create a new edge starting from this vertex
            current_edge = Edge(vertex, next_edge, initial_face)
            current_edge.pair.start = next_edge.start
            next_edge.pair.next = current_edge.pair
            # add the edge to the vertex
            vertex.edge = current_edge
            next_edge = current_edge

        return self
