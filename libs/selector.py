# coding=utf-8
"""
Selector module

A selector takes a space (and other optional arguments) and return a Generator of edges :
• selector are made from edge queries
• an edge query is a function that take a space as argument and yield edges
• a query can be created from predicates.
A predicate takes an edge and a space and returns a boolean.

The module exports a catalog containing various selectors

example :
import selector

for edge in selector.catalog['boundary'].yield_from(space):
    do something with each edge (like a mutation for example)

"""
from typing import Sequence, Generator, Callable, Any, Optional, TYPE_CHECKING

from libs.utils.geometry import ccw_angle, opposite_vector, pseudo_equal, barycenter, distance
from libs.mesh import MIN_ANGLE

if TYPE_CHECKING:
    from libs.mesh import Edge
    from libs.plan import Space
    from libs.seed import Seeder

EdgeQuery = Callable[['Space', Any], Generator['Edge', bool, None]]
EdgeQueryFactory = Callable[..., EdgeQuery]
Predicate = Callable[['Edge', 'Space'], bool]
PredicateFactory = Callable[..., Predicate]

EPSILON = 1.0
ANGLE_EPSILON = 5.0


class Selector:
    """
    Returns an iterator on a given plan space
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
                   space: 'Space',
                   *args) -> Generator['Edge', bool, None]:
        """
        Runs the selector and returns a generator
        :param space:
        :return:
        """
        for edge in self.query(space, *args):
            filtered = True
            for predicate in self.predicates:
                filtered = filtered and predicate(edge, space)
                if not filtered:
                    break
            if filtered:
                yield edge


class SelectorFactory:
    """
    Selector factory class
    Note :
    """

    def __init__(self, edge_query_factory: EdgeQueryFactory,
                 predicates_factories: Optional[Sequence[PredicateFactory]] = None,
                 name: str = ''):
        self.edge_query_factory = edge_query_factory
        self.predicate_factories = predicates_factories or []
        self.name = name or edge_query_factory.__name__

    def __call__(self,
                 query_args: Sequence,
                 predicates_args: Optional[Sequence[Sequence]] = None) -> Selector:
        predicates_args = predicates_args or []
        if predicates_args and len(self.predicate_factories) != len(predicates_args):
            raise ValueError('Arguments must be provided for each predicate factory: ' +
                             '{0}'.format(predicates_args))
        name = self.name
        edge_query = self.edge_query_factory(*query_args)
        predicates = []
        for i in range(len(self.predicate_factories)):
            predicates_args = predicates_args[i] if i < len(predicates_args) else []
            predicates.append(self.predicate_factories[i](*predicates_args))
        return Selector(edge_query, predicates, name=name)


# Selector Catalog


# Queries
def boundary(space: 'Space', *_) -> Generator['Edge', bool, None]:
    """
    Returns the edges of the face
    :param space:
    :return:
    """
    if space.edge:
        yield from space.edges


def boundary_unique(space: 'Space', *_) -> Generator['Edge', bool, None]:
    """
    Returns the edge reference face
    :param space:
    :return:
    """
    if space.edge:
        yield space.edge


def fixed_space_boundary(space: 'Space', *_) -> Generator['Edge', bool, None]:
    """
    Returns any face around the seed space. The list is fixed and won't be affected by
    changes to the space edges
    :param space:
    :return:
    """
    for edge in list(space.edges):
        yield edge


def other_seed_space_edge(space: 'Space', seeder: 'Seeder', *_) -> Generator['Edge', bool, None]:
    """
    Returns the edges of the seed space pointing to another seed space but not to a face
    attached to one of its component
    and not pointing to another seed
    """
    assert seeder, "The associated seed object must be provided"

    for edge in space.edges:
        face = edge.pair.face
        if face is None:
            continue
        other_space = space.plan.get_space_of_face(face)
        if not other_space:
            continue
        if other_space.category.name != "seed":
            continue
        seed = seeder.get_seed_from_space(other_space)
        if seed and seed.face_has_component(face):
            continue
        yield edge


def seed_component_boundary(space: 'Space', seeder: 'Seeder', *_) -> Generator['Edge', bool, None]:
    """
    Returns the edge of the space adjacent to a face adjacent to the component of the seed
    :param space:
    :param seeder:
    :return:
    """
    assert seeder, "The associated seeder object must be provided"

    seed = seeder.get_seed_from_space(space)
    component = seed.components[0]  # per convention we use the first component of the seed
    for edge in component.edges:
        face = edge.pair.face
        if (face is None or space.has_face(face)
                or space.plan.get_space_of_face(face).category.name != 'empty'):
            continue
        # find a shared edge with the space
        for face_edge in face.edges:
            if space.has_edge(face_edge.pair):
                break
        else:
            continue

        yield face_edge.pair


def boundary_unique_longest(space: 'Space', *_) -> Generator['Edge', bool, None]:
    """
    Returns the longest edge of the space that is not on the plan boundary
    :param space:
    :return:
    """
    space_edges_adjacent_to_seed = [
        edge for edge in space.edges if
        not edge.is_mesh_boundary
        and space.plan.get_space_of_edge(edge.pair).category.name == "seed"
    ]
    if space_edges_adjacent_to_seed:
        edge = max(space_edges_adjacent_to_seed, key=lambda edge: edge.length)
        yield edge
    else:
        return


def seed_duct(space: 'Space', *_) -> Generator['Edge', bool, None]:
    """
    Returns the edge that can be seeded for a duct
    """
    if not space.category or space.category.name != 'duct':
        raise ValueError('You should provide a duct to the query seed_duct!')

    # case n°1 : duct is along a boundary, we only set two seed point
    edge_along_plan = None
    for edge in space.edges:
        if edge.pair.face is None:
            edge_along_plan = edge
            break

    if edge_along_plan:
        yield edge_along_plan.next_ortho().pair
        yield edge_along_plan.previous_ortho().pair
    else:
        for edge in space.edges:
            if edge.next_ortho() is edge.next:
                yield edge.pair


# Query factories

def farthest_edges_barycenter(coeff: float = 0) -> EdgeQuery:
    """
    Returns the two farthest edges of the space according to their barycenter
    :param coeff:
    :return:
    """

    def _query(space: 'Space', *_) -> Generator['Edge', bool, None]:
        """
        Returns for a given space the two edges that are most far from one another,
        based on their middle
        :return:
        """
        kept_edges = []
        d_max = 0
        seen = []

        for edge in space.edges:
            seen.append(edge)
            d_max_edge = 0
            edge_far = None
            for edge_sibling in space.siblings(edge):
                # to prevent to compute n**2 distances (but only 1/2*(n-1)**2)
                if edge_sibling in seen:
                    continue
                point_1 = barycenter(edge.start.coords, edge.end.coords, coeff)
                point_2 = barycenter(edge_sibling.start.coords, edge_sibling.end.coords, coeff)
                d_tmp = distance(point_1, point_2)
                if d_tmp > d_max_edge:
                    d_max_edge = d_tmp
                    edge_far = edge_sibling
            if d_max_edge > d_max:
                kept_edges = [edge, edge_far]
                d_max = d_max_edge

        for edge in kept_edges:
            yield edge

    return _query


def oriented_edges(direction: str, epsilon: float = 35.0) -> EdgeQuery:
    """
    EdgeQuery factory
    Returns an edge query that yields edges facing the direction or the normal
    of the reference edge of a face or a space
    (with epsilon error on angle)
    :param direction:
    :param epsilon:
    :return: an EdgeQuery
    """
    if direction not in ('horizontal', 'vertical'):
        raise ValueError('A direction can only be horizontal or vertical: {0}'.format(direction))

    def _selector(space: 'Space', *_) -> Generator['Edge', bool, None]:

        if not space.edge:
            return

        vectors = ((space.edge.unit_vector, opposite_vector(space.edge.unit_vector))
                   if direction == 'horizontal' else (space.edge.normal,))

        for vector in vectors:
            edges_list = [edge for edge in space.exterior_edges
                          if pseudo_equal(ccw_angle(edge.normal, vector), 180.0, epsilon)]
            for edge in edges_list:
                yield edge

    return _selector


# predicates


def not_space_boundary(edge: 'Edge', space: 'Space') -> bool:
    """
    Predicate
    Returns True if the edge is not a space boundary
    :param edge:
    :param space:
    :return:
    """
    return not space.is_boundary(edge)


def adjacent_to_other_space(edge: 'Edge', space: 'Space') -> bool:
    """
        Predicate
        Returns True if the edge is adjacent to another space
        :param edge:
        :param space:
        :return:
        """
    space_pair = space.plan.get_space_of_face(edge.pair.face)
    return space_pair is not space and space.mutable


def not_adjacent_to_seed(edge: 'Edge', space: 'Space') -> bool:
    """
    return True if the edge.pair does not belong to a space of category seed
    :param edge:
    :param space:
    :return:
    """
    space_pair = space.plan.get_space_of_face(edge.pair.face)
    return space_pair is None or space_pair.category.name != 'seed'


def adjacent_empty_space(edge: 'Edge', space: 'Space') -> bool:
    """
    Return True if the edge pair belongs to a space of the 'empty' category
    :param edge:
    :param space:
    :return:
    """
    space_pair = space.plan.get_space_of_face(edge.pair.face)
    return space_pair and space_pair.category.name == 'empty'


def corner_stone(edge: 'Edge', space: 'Space') -> bool:
    """
    Returns True if the removal of the edge's face from the space
    will cut it in several spaces or is the only face
    """
    face = edge.pair.face

    if not face:
        return False

    other_space = space.plan.get_space_of_face(face)

    if not other_space:
        return False

    # case 1 : the only face of the space
    if len(other_space._faces_id) == 1:
        return True

    # case 2 : fully enclosing face
    face_edges = list(face.edges)
    for edge in other_space.exterior_edges:
        if edge not in face_edges:
            break
        face_edges.remove(edge)
    else:
        return False

    # case 4 : standard case
    forbidden_edges = list(face.edges)
    other_space.change_reference_edges(forbidden_edges)
    adjacent_faces = list(other_space.adjacent_faces(face))

    if len(adjacent_faces) == 1:
        return False

    remaining_faces = adjacent_faces[:]

    # temporarily remove the face_id from the other_space
    other_space.remove_face_id(face)

    # we must check to see if we split the other_space by removing the face
    # for each adjacent face inside the other_space check if they are still connected
    while remaining_faces:

        adjacent_face = remaining_faces[0]
        connected_faces = [adjacent_face]

        for connected_face in other_space.connected_faces(adjacent_face):
            # try to reach the other adjacent faces
            if connected_face in remaining_faces:
                remaining_faces.remove(connected_face)
            connected_faces.append(connected_face)

        remaining_faces.remove(adjacent_face)

        if len(remaining_faces) != 0:
            other_space.add_face_id(face)
            return True
        else:
            break

    other_space.add_face_id(face)
    return False


# predicate factories


def factorize(*predicates: Predicate) -> Sequence[PredicateFactory]:
    """
    Returns the predicate (synthetic sugar)
    :param predicates:
    :return: predicateFactory
    """
    return [lambda: predicate for predicate in predicates]


def is_not(predicate: Predicate) -> Predicate:
    """
    Returns the opposite predicate
    :return:
    """

    def _predicate(edge: 'Edge', space: 'Space') -> bool:
        return not predicate(edge, space)

    return _predicate


def is_previous(predicate: Predicate) -> Predicate:
    """
    Applies the predicate to the previous edge
    :param predicate:
    :return:
    """

    def _predicate(edge: 'Edge', space: 'Space') -> bool:
        return predicate(edge.previous, space)

    return _predicate


def is_next(predicate: Predicate) -> Predicate:
    """
    Applies the predicate to the next edge
    :param predicate:
    :return:
    """

    def _predicate(edge: 'Edge', space: 'Space') -> bool:
        return predicate(edge.next, space)

    return _predicate


def edge_angle(min_angle: Optional[float] = None,
               max_angle: Optional[float] = None,
               previous: bool = False,
               space_boundary: bool = False) -> Predicate:
    """
    Predicate factory
    Returns a predicate indicating if an edge angle with its next edge
    is comprised between the two provided values. Only one of the value can be provided
    :param min_angle:
    :param max_angle:
    :param previous : whether to check for the angle to the next edge or to the previous edge
    :param space_boundary: whether to check for angle between to space siblings
    :return: boolean
    """
    if min_angle is None and max_angle is None:
        raise ValueError('A min or a max angle must be provided to the angle predicate factory')

    def _predicate(edge: 'Edge', space: 'Space') -> bool:

        if space_boundary:
            if not space.is_boundary(edge):
                return False
            _angle = space.next_angle(edge) if not previous else space.previous_angle(edge)
        _angle = edge.next_angle if not previous else edge.previous_angle
        if min_angle is not None and max_angle is not None:
            if min_angle != max_angle:
                return min_angle + ANGLE_EPSILON <= _angle <= max_angle - ANGLE_EPSILON
            else:
                return min_angle - ANGLE_EPSILON <= _angle <= max_angle + ANGLE_EPSILON
        if min_angle is not None:
            return _angle >= min_angle + ANGLE_EPSILON
        if max_angle is not None:
            return _angle <= max_angle - ANGLE_EPSILON
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

    def _predicate(edge: 'Edge', _: 'Space') -> bool:
        if min_length is not None and max_length is not None:
            return min_length <= edge.length <= max_length
        if min_length is not None:
            return edge.length >= min_length
        if max_length is not None:
            return edge.length <= max_length

    return _predicate


def aligned_edges_length(min_length: float = None, max_length: float = None) -> Predicate:
    """
    Predicate factory
    Returns a predicate indicating if an edge an all following aligned edges have a total length
    inferior or equal to the provided max length, or a superior length to the provided min_length
    or both
    :param min_length:
    :param max_length
    :return:
    """

    def _predicate(edge: 'Edge', _: 'Space') -> bool:

        length = 0
        for aligned_edge in edge.aligned_siblings:
            length += aligned_edge.length

        if min_length is not None and max_length is not None:
            return min_length <= length <= max_length
        if min_length is not None:
            return length >= min_length
        if max_length is not None:
            return length <= max_length

    return _predicate


def is_linear(*category_names: str) -> Predicate:
    """
    Predicate Factory
    :param category_names:
    :return: a predicate
    """

    def _predicate(edge: 'Edge', space: 'Space') -> bool:
        linear = space.plan.get_linear(edge)
        return linear and linear.category.name in category_names

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

    def _predicate(edge: 'Edge', space: 'Space') -> bool:
        if position == 'before':
            next_linear = space.plan.get_linear(edge.next)
            return next_linear and next_linear.category.name in category_names
        if position == 'after':
            previous_linear = space.plan.get_linear(edge.previous)
            return previous_linear and previous_linear.category.name in category_names
        if position == 'between':
            next_linear = space.plan.get_linear(edge.next)
            previous_linear = space.plan.get_linear(edge.previous)
            if previous_linear and previous_linear.category.name in category_names:
                if next_linear and next_linear.category.name in category_names:
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

    def _predicate(edge: 'Edge', space: 'Space'):
        linear_edges = []
        for sibling in edge.siblings:
            linear = space.plan.get_linear(sibling)
            if linear and linear.category.name in category_names:
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


def space_area(min_area: float = None, max_area: float = None) -> Predicate:
    """
    Predicate factory
    Returns a predicate indicating if an edge belongs to a space with area in a range
    TODO : this is bad because we run this for every edge, it would be better as a query
    :param min_area:
    :param max_area
    :return:
    """

    def _predicate(_: 'Edge', space: 'Space') -> bool:
        if min_area is not None and max_area is not None:
            return min_area <= space.area <= max_area
        if min_area is not None:
            return space.area >= min_area
        if max_area is not None:
            return space.area <= max_area

    return _predicate


# Catalog Selectors

SELECTORS = {
    "space_boundary": Selector(boundary),

    "seed_component_boundary": Selector(seed_component_boundary),

    "previous_angle_salient_non_ortho": Selector(
        boundary,
        [
            edge_angle(180.0, 270.0, previous=True),
            aligned_edges_length(min_length=150.0),
            is_previous(aligned_edges_length(min_length=150.0))
        ]
    ),

    "next_angle_salient_non_ortho": Selector(
        boundary,
        [
            edge_angle(180.0, 270.0),
            aligned_edges_length(min_length=150.0),
            is_next(aligned_edges_length(min_length=150.0))
        ]
    ),

    "next_angle_convex_non_ortho": Selector(
        boundary,
        [
            edge_angle(90.0, 180.0),
            aligned_edges_length(min_length=150.0),
            is_next(aligned_edges_length(min_length=150.0))
        ]
    ),

    "previous_angle_convex_non_ortho": Selector(
        boundary,
        [
            edge_angle(90.0, 180.0, previous=True),
            aligned_edges_length(min_length=100.0),
            is_previous(aligned_edges_length(min_length=150.0))
        ]
    ),

    "previous_angle_salient_ortho": Selector(
        boundary,
        [
            edge_angle(270.0, 270.0, previous=True, space_boundary=True)
        ]
    ),
    "next_angle_salient_ortho": Selector(
        boundary,
        [
            edge_angle(270.0, 270.0, space_boundary=True)
        ]
    ),

    "close_to_window": Selector(
        boundary,
        [
            close_to_linear('window', 'doorWindow', min_distance=90.0),
            not_space_boundary
        ]
    ),

    "between_windows": Selector(
        boundary,
        [
            touches_linear('window', 'doorWindow', position='between')
        ]
    ),

    "between_edges_between_windows": Selector(
        boundary,
        [
            is_previous(touches_linear('window', 'doorWindow', position='after')),
            is_next(touches_linear('window', 'doorWindow'))
        ]
    ),

    "aligned_edges": Selector(
        boundary,
        [
            edge_angle(180.0, 180.0),
            is_not(touches_linear('window', 'doorWindow', 'frontDoor')),
            is_not(is_linear('window', 'doorWindow', 'frontDoor'))
        ]
    ),

    "edge_min_150": Selector(
        boundary,
        [
            edge_length(min_length=150.0)
        ]
    ),

    "edge_min_300": Selector(
        boundary,
        [
            edge_length(min_length=300.0)
        ]
    ),

    "edge_min_500": Selector(
        boundary,
        [
            edge_length(min_length=500.0)
        ]
    ),

    "boundary_other_empty_space": Selector(
        fixed_space_boundary, [
            adjacent_empty_space
        ]
    ),

    "seed_duct": Selector(seed_duct),

    "seed_empty_furthest_couple": Selector(farthest_edges_barycenter()),

    "farthest_couple_start_space_area_min_100000": Selector(
        farthest_edges_barycenter(),
        [
            space_area(min_area=100000)
        ]
    ),

    "farthest_couple_middle_space_area_min_100000": Selector(
        farthest_edges_barycenter(0.5),
        [
            space_area(min_area=100000)
        ]
    ),

    "area_max_100000": Selector(
        boundary_unique,
        [
            space_area(max_area=100000)
        ]
    ),

    "fuse_small_cell": Selector(
        boundary_unique_longest,
        [
            space_area(max_area=30000)
        ]
    ),

    "other_seed_space": Selector(
        other_seed_space_edge,
        [
            adjacent_to_other_space,
            is_not(corner_stone)
        ]
    ),
    "corner_stone": Selector(
        boundary,
        [
            corner_stone
        ]
    ),
    "single_edge": Selector(boundary_unique)
}


SELECTOR_FACTORIES = {
    "oriented_edges": SelectorFactory(oriented_edges, factorize(adjacent_empty_space)),
    "edges_length": SelectorFactory(lambda: boundary, [edge_length]),
}
