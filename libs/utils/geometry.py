# coding=utf-8
"""
geometry module
Contains utility functions for computational geometry
TODO : we should structure this with a point class and a vector class
"""

from typing import Optional, Any, Sequence, Dict, Tuple, List
import numpy as np
import shapely as sp
from shapely.geometry import Point, LineString, LinearRing, Polygon
from random import randint
import math

from libs.utils.custom_types import Vector2d, Coords2d, ListCoords2d, FourCoords2d

COORD_DECIMAL = 4  # number of decimal of the points coordinates
ANGLE_EPSILON = 1.0  # value to check if an angle has a specific value
MIN_ANGLE = 5.0


def truncate(value: float, decimals: int = COORD_DECIMAL) -> float:
    """
    Rounds a value to the specified precision
    :param value:
    :param decimals:
    :return:
    """
    return float(np.around(float(value), decimals=decimals))


def magnitude(vector: Vector2d) -> float:
    """
    Returns the magnitude of a vector
    :param vector: vector as a tuple
    :return: float
    """
    mag = np.sqrt(vector[0] ** 2 + vector[1] ** 2)
    return mag


def direction_vector(point_1: Coords2d, point_2: Coords2d) -> Vector2d:
    """
    Convenient function to calculate the direction vector between two points
    :return: tuple containing x, y values
    """
    output_vector = point_2[0] - point_1[0], point_2[1] - point_1[1]
    return unit(output_vector)


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


def point_dict_to_tuple(point_as_dict: Dict[str, str]) -> Coords2d:
    """
    Transform a point as a dict into a point as a tuple
    ex: {'x': 2.0, 'y': 3.2 } -> (2.0, 3.2)
    :param point_as_dict:
    :return: tuple of floats
    """
    return point_as_dict['x'], point_as_dict['y']


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
    return truncate(np.cos(rad)), truncate(np.sin(rad))


def normal_vector(vector: Vector2d) -> Vector2d:
    """
    A CCW normal of the edge of length 1
    :return: a tuple containing x, y values
    """
    vector = -vector[1], vector[0]
    return unit(vector)


def unit(vector: Vector2d) -> Vector2d:
    """
    Returns a vector of same direction but of length 1.
    Note: to prevent division per zero, if the vector is equal to (0, 0) return (0,0)
    :param vector:
    :return:
    """
    vector_length = magnitude(vector)

    if vector_length == 0:
        return 0, 0

    coord_x = vector[0] / vector_length
    coord_y = vector[1] / vector_length

    return coord_x, coord_y


def opposite_vector(vector: Vector2d) -> Vector2d:
    """
    Returns the opposite vector
    :param vector:
    :return:
    """
    return -vector[0], -vector[1]


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


def random_unit(dec: int = 100) -> float:
    """
    Returns a random float between -1 and +1
    :param dec: precision
    :return:
    """
    return float((randint(0, 2 * dec + 1) - dec) / dec)


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


def same_half_plane(vector_1: Vector2d, vector_2: Vector2d) -> bool:
    """
    Returns True if the vectors are facing the same direction
    (meaning they point to the same half-plane)
    :param vector_1:
    :param vector_2:
    :return:
    """
    return dot_product(vector_1, vector_2) >= 0


def dot_product(vector_1: Vector2d, vector_2: Vector2d):
    """
    Returns the dot product of the two vectors
    :param vector_1:
    :param vector_2:
    :return:
    """
    return vector_1[0] * vector_2[0] + vector_1[1] * vector_2[1]


def cross_product(vector_1: Vector2d, vector_2: Vector2d) -> float:
    """
    Returns the cross product of the two vectors
    :param vector_1:
    :param vector_2:
    :return:
    """
    return vector_1[0] * vector_2[1] - vector_1[1] * vector_2[0]


def pseudo_equal(value: float, other: float, epsilon: float) -> bool:
    """
    Verify if an value is very close to a specific value, according to an epsilon float
    (returns value == other ± epsilon)
    :param value: float
    :param other: float
    :param epsilon: float
    :return: boolean
    """
    return other + epsilon > value > other - epsilon


def distance(point_1: Coords2d, point_2: Coords2d) -> float:
    """
    Returns the distance between two points
    :param point_1:
    :param point_2:
    :return:
    """
    vector = point_2[0] - point_1[0], point_2[1] - point_1[1]
    return magnitude(vector)


def rectangle(reference_point: Coords2d,
              orientation_vector: Vector2d,
              width: float,
              height: float,
              offset: float = 0) -> [Coords2d]:
    """
    Returns the perimeter points of a rectangle
                    WIDTH
                +--------------+
                |              |
                |              | HEIGHT
                | VECTOR       |
      <-offset->| *--->        |
    [P]---------+--------------+
  REF POINT

    :param reference_point:
    :param orientation_vector:
    :param width:
    :param height:
    :param offset:
    :return: a list of 4 coordinates points
    """
    orientation_vector = unit(orientation_vector)
    point = move_point(reference_point, orientation_vector, coeff=offset)
    output = [point]
    output += [move_point(output[0], orientation_vector, coeff=width)]
    output += [move_point(output[1], normal_vector(orientation_vector), coeff=height)]
    output += [move_point(output[2], orientation_vector, coeff=-width)]

    return output


def parallel(vector: Vector2d, other: Vector2d) -> bool:
    """
    Returns True if the vector are parallel
    :param vector:
    :param other:
    :return:
    """
    return pseudo_equal(ccw_angle(vector, opposite_vector(other)), 180.0, ANGLE_EPSILON)


def lines_intersection(line_1: Tuple[Coords2d, Vector2d],
                       line_2: Tuple[Coords2d, Vector2d]) -> Optional[Coords2d]:
    """
    Returns the intersection between two lines defined by a point and a directional vector.
    Will return None if the lines are parallel.
    :param line_1: a tuple of a point and a vector defining the first line
    :param line_2: a tuple of a point and a vector defining the second line
    :return:
    """
    a, u = line_1
    b, v = line_2
    assert v != (0, 0) and u != (0, 0), "Geometry: Line intersection : the vectors must no be null"
    d = (v[0] * u[1] - u[0] * v[1])
    # if d == 0: no or infinite number of solutions because the vectors are co-linears
    if d == 0:
        return None
    t = 1 / d * (u[1] * (a[0] - b[0]) - u[0] * (a[1] - b[1]))
    return b[0] + t * v[0], b[1] + t * v[1]


def project_point_on_segment(point: Coords2d,
                             vector: Coords2d,
                             segment: Tuple[Coords2d, Coords2d],
                             no_direction: bool = False,
                             epsilon: float = 1.0) -> Optional[Coords2d]:
    """
    Computes the projection of a point along a specified vector unto a specified segment.
    Returns None if there is no intersection.
    :param point:
    :param vector:
    :param segment:
    :param no_direction: if True, the projection does not need to be in the direction of the
    vector
    :param epsilon: the distance allowed to the extremities of the segment
    :return: an optional point
    """
    a = point
    u = vector
    b = segment[0]
    v = segment[1][0] - b[0], segment[1][1] - b[1]
    d = (v[0] * u[1] - u[0] * v[1])
    if d == 0:
        return None
    abx = a[0] - b[0]
    aby = a[1] - b[1]
    t = 1 / d * (u[1] * abx - u[0] * aby)
    len_segment = distance(*segment)
    relative_epsilon = epsilon / len_segment
    if t > 1 + relative_epsilon or t < -relative_epsilon:
        return None
    p = 1 / d * (v[1] * abx - v[0] * aby)
    if p < -relative_epsilon and not no_direction:
        return None
    return b[0] + t * v[0], b[1] + t * v[1]


def min_section(perimeter: List[Coords2d]) -> float:
    """
    Returns the minimum section of the perimeter.
    Note : this is a simplification of the algorithm needed to correctly implement
           ingress constraints on corridors, which should compute
           a min cross section width along a given access path.
    :param perimeter:
    :return:
    """
    depth = math.inf
    # assert sp.geometry.Polygon(perimeter).is_valid, "The specified polygon must be valid"
    n = len(perimeter)
    assert n > 2, "The perimeter must have at least 3 points"

    for i, point in enumerate(perimeter):
        # check projection distance with each segment (other than the next and previous one)
        k = (i + 1) % n
        previous_point = perimeter[(i - 1) % n]
        next_point = perimeter[k]
        previous_vector = previous_point[0] - point[0], previous_point[1] - point[1]
        next_vector = next_point[0] - point[0], next_point[1] - point[1]

        while k != (i - 1) % n:
            seg = (perimeter[k], perimeter[(k + 1) % n])
            vector = normal_vector((seg[0][0] - seg[1][0], seg[0][1] - seg[1][1]))
            k = (k + 1) % n
            if ccw_angle(next_vector, vector) >= ccw_angle(next_vector, previous_vector):
                continue
            projected_point = project_point_on_segment(point, vector, seg)
            if not projected_point:
                continue
            projected_depth = distance(point, projected_point)
            if projected_depth < depth:
                depth = projected_depth

    return depth


def rotate(polygon: ListCoords2d, ref_point: Coords2d, angle: float) -> ListCoords2d:
    """
    Return a rotated version of the polygon around a reference point
    :param polygon:
    :param ref_point:
    :param angle: in degrees, ccw
    :return:
    """
    ref_x, ref_y = ref_point
    # convert to radians
    angle = angle * math.pi / 180
    # rotate each point
    return tuple([(math.cos(angle) * (x - ref_x) - math.sin(angle) * (y - ref_y) + ref_x,
                   math.sin(angle) * (x - ref_x) + math.cos(angle) * (y - ref_y) + ref_y)
                  for x, y in polygon])


def minimum_rotated_rectangle(polygon: ListCoords2d) -> FourCoords2d:
    """
    Return the smallest bounding box of the polygon
    :param polygon:
    :return:
    """
    return Polygon(polygon).minimum_rotated_rectangle.exterior.coords[:-1]


def polygons_collision(poly_1: ListCoords2d,
                       poly_2: ListCoords2d,
                       tolerance:float=0) -> bool:
    """
    Return true if polygons are colliding, given the input tolerance
    :param poly_1:
    :param poly_2:
    :param tolerance:
    :return:
    """
    return Polygon(poly_1).buffer(-tolerance).intersects(Polygon(poly_2))

def polygon_border_collision(polygon: ListCoords2d,
                             border: ListCoords2d,
                             tolerance:float=0) -> bool:
    """
    Return true if polygon collides border, given the input tolerance
    :param polygon:
    :param border:
    :param tolerance:
    :return:
    """
    return Polygon(polygon).buffer(-tolerance).intersects(LinearRing(border))

def polygon_linestring_collision(polygon: ListCoords2d,
                                 linestring: ListCoords2d,
                                 tolerance:float=0) -> bool:
    """
    Return true if polygon collides linestring, given the input tolerance
    :param polygon:
    :param linestring:
    :param tolerance:
    :return:
    """
    return Polygon(polygon).buffer(-tolerance).intersects(LineString(linestring))