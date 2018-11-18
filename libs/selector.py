# coding=utf-8
"""
Selector module

A selector takes a space or an edge (and other optional arguments) and return a Generator of edges :
• selector are made from edge queries
• an edge query is a function that take a space or a face as argument and yield edges
• a query can be created from predicates

Note : isomorphism is used to apply the same selector to a space or an edge.
It assumes both classes have the self.edges method implemented

The module exports a catalog containing various selectors

example :
import selector

for edge in selector.catalog['boundary'].yield_from(my_space_or_face):
    do something with each edge (like a mutation for example)

"""

from typing import Sequence, Union, Generator, Callable, Any, Optional, TYPE_CHECKING

from libs.utils.catalog import Catalog
from libs.utils.geometry import ccw_angle

from libs.mesh import MIN_ANGLE, ANGLE_EPSILON

if TYPE_CHECKING:
    from libs.mesh import Edge, Face
    from libs.plan import Space
    from libs.seeder import Seed

EdgeQuery = Callable[[Union['Space', 'face'], Any], Generator['Edge', None, None]]
EdgeQueryFactory = Callable[..., EdgeQuery]
Predicate = Callable[['Edge'], bool]

catalog = Catalog('selectors')


class Selector:
    """
    Returns an iterator on a given mesh face
    """
    def __init__(self,
                 query: EdgeQuery,
                 predicates: Optional[Sequence[Predicate]] = None,
                 name: str = ''):
        self.name = name or query.__name__
        self.query = query
        self.predicates = predicates or []

    def __repr__(self):
        return 'Selector: {0}'.format(self.name)

    def yield_from(self,
                   space_or_face: Union['Space', 'Edge'],
                   *args) -> Generator[Edge, None, None]:
        """
        Runs the selector
        :param space_or_face:
        :return:
        """
        for edge in self.query(space_or_face, *args):
            filtered = True
            for predicate in self.predicates:
                filtered = filtered and predicate(edge)
                if not filtered:
                    break
            if filtered:
                yield edge


# Selector Catalog


# queries


def space_boundary(space: 'Space') -> Generator['Edge', None, None]:
    """
    Returns any face around the seed space
    :param space:
    :return:
    """
    for edge in list(space.edges):
        if edge.pair.face and edge.pair.face.space is not space:
            yield edge.pair.face


def seed_component_boundary(space: 'Space', seed: 'Seed') -> Generator['Edge', None, None]:
    """
    Returns the edge of the space adjacent to a face close to the component of the seed
    :param space:
    :param seed:
    :return:
    """
    component = seed.components[0]  # per convention we use the first component of the seed
    for edge in component.edges:
        face = edge.pair.face
        if face is None or face.space is space:
            continue
        # find a shared edge with the space
        for face_edge in face.edges:
            if face_edge.pair.space is space:
                break
        else:
            continue

        yield face_edge.pair


def boundary(face_or_space: Union['Face', 'Space']) -> Generator['Edge', None, None]:
    """
    Returns the edges of the face
    :param face_or_space:
    :return:
    """
    yield from face_or_space.edges

# predicates


def not_space_boundary(edge: Edge) -> bool:
    """
    Predicate
    Returns True if the edge is not a space boundary
    :param edge:
    :return:
    """
    return not edge.is_space_boundary


# predicates factories


def is_not(predicate: Predicate) -> Predicate:
    """
    Returns the opposite predicate
    :return:
    """
    def _predicate(edge: Edge) -> bool:
        return not predicate(edge)

    return _predicate


def edge_angle(min_angle: Optional[float] = None,
               max_angle: Optional[float] = None,
               previous: bool = False) -> Predicate:
    """
    Predicate factory
    Returns a predicate indicating if an edge angle with its next edge
    is comprised between the two provided values. Only one of the value can be provided
    :param min_angle:
    :param max_angle:
    :param previous : whether to check for the angle to the next edge or to the previous edge
    :return: boolean
    """
    if min_angle is None and max_angle is None:
        raise ValueError('A min or a max angle must be provided to the angle predicate factory')

    def _predicate(edge: Edge) -> bool:
        _angle = edge.next_angle if not previous else edge.previous_angle
        if min_angle is not None and max_angle is not None:
            return min_angle <= _angle <= max_angle
        if min_angle is not None:
            return _angle >= min_angle
        if max_angle is not None:
            return _angle <= max_angle
        return True

    return _predicate


def edge_length(min_length: float = None, max_length: float = None) -> Predicate:
    """
    Predicate factory
    Returns a predicate indicating if an edge has a length inferior or equal to the provided max
    length, or a superior length to the provided min_length or both
    :param min_length:
    :param max_length
    :return:
    """
    def _predicate(edge: Edge) -> bool:
        if min_length is not None and max_length is not None:
            return min_length <= edge.length <= max_length
        if min_length is not None:
            return edge.length >= min_length
        if max_length is not None:
            return edge.length <= max_length

    return _predicate


def is_linear(*category_names: str) -> Predicate:
    """
    Predicate Factory
    :param category_names:
    :return: a predicate
    """

    def _predicate(edge: Edge) -> bool:
        return edge.linear and edge.linear.category.name in category_names

    return _predicate


def touches_linear(*category_names: str, position: str = 'before') -> Predicate:
    """
    Predicate factory
    Returns a predicate indicating if an edge is between two linears of the provided category
    :param category_names: tuple of linear category names
    :param position : where should the edge be : before, after, between
    :return:
    """

    position_valid_values = 'before', 'after', 'between'

    if position not in position_valid_values:
        raise ValueError('Wrong position value in predicate factory touches_linear:' +
                         ' {0}'.format(position))

    def _predicate(edge: Edge) -> bool:
        if position == 'before':
            return edge.next.linear and edge.next.linear.category.name in category_names
        if position == 'after':
            return edge.next.linear and edge.next.linear.category.name in category_names
        if position == 'between':
            if edge.previous.linear and edge.previous.linear.category.name in category_names:
                if edge.next.linear and edge.next.linear.category.name in category_names:
                    return True
            return False

    return _predicate


def close_to_linear(*category_names: str, min_distance: float = 50.0) -> Predicate:
    """
    Predicate factory
    Returns a predicate indicating if the edge is closer than
    the provided minimum distance to a linear of the provided category
    :param min_distance:
    :param category_names
    :return: function
    """
    def _predicate(edge: Edge):
        linear_edges = []
        for sibling in edge.siblings:
            if sibling.linear and sibling.linear.category.name in category_names:
                linear_edges.append(sibling)

        if not linear_edges:
            return False

        for linear_edge in linear_edges:
            start_point_dist, end_point_dist = None, None
            if ccw_angle(edge.vector, linear_edge.opposite_vector) >= 90.0 - MIN_ANGLE:
                continue
            start_point_line = linear_edge.start.sp_line(linear_edge.normal)
            end_point_line = linear_edge.end.sp_line(linear_edge.normal)
            start_point_intersection = start_point_line.intersection(edge.as_sp)
            end_point_intersection = end_point_line.intersection(edge.as_sp)
            if (not start_point_intersection.is_empty
                    and start_point_intersection.geom_type == 'Point'):
                start_point_dist = start_point_intersection.distance(linear_edge.start.as_sp)
            if (not end_point_intersection.is_empty
                    and end_point_intersection.geom_type == 'Point'):
                end_point_dist = end_point_intersection.distance(linear_edge.end.as_sp)
            if not start_point_dist and not end_point_dist:
                continue
            dist_to_edge = min(d for d in (start_point_dist, end_point_dist) if d is not None)
            if dist_to_edge <= min_distance:
                return True

        return False

    return _predicate


catalog.add(Selector(boundary, 'boundary'))

catalog.add(Selector(seed_component_boundary, 'seed_component_boundary'))

catalog.add(Selector(boundary,
                     [edge_angle(180.0 + ANGLE_EPSILON, 270.0 - ANGLE_EPSILON, previous=True)],
                     'previous_angle_salient_non_ortho'))

catalog.add(Selector(boundary,
                     [edge_angle(180.0 + ANGLE_EPSILON, 270.0 - ANGLE_EPSILON)],
                     'next_angle_salient_non_ortho'))

catalog.add(Selector(boundary,
                     [edge_angle(90.0 + ANGLE_EPSILON, 180.0 - ANGLE_EPSILON)],
                     'next_angle_convex_non_ortho'))

catalog.add(Selector(boundary,
                     [edge_angle(90.0 + ANGLE_EPSILON, 180.0 - ANGLE_EPSILON, previous=True)],
                     'previous_angle_convex_non_ortho'))

catalog.add(Selector(boundary,
                     [close_to_linear('window', 'doorWindow', min_distance=90.0),
                      not_space_boundary],
                     'close_to_window'))

catalog.add(Selector(boundary,
                     [touches_linear('window', 'doorWindow', position='between')],
                     'touches_window'))

catalog.add(Selector(boundary,
                     [edge_angle(180.0 - ANGLE_EPSILON, 180.0 + ANGLE_EPSILON),
                      is_not(touches_linear('window', 'doorWindow')),
                      is_not(is_linear('window', 'doorWindow'))],
                     'aligned_edge'))
