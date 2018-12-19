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
from libs.utils.geometry import ccw_angle, opposite_vector, pseudo_equal

from libs.mesh import MIN_ANGLE

if TYPE_CHECKING:
    from libs.mesh import Edge, Face
    from libs.plan import Space, SeedSpace
    from libs.seed import Seed

EdgeQuery = Callable[[Union['Space', 'face'], Any], Generator['Edge', bool, None]]
EdgeQueryFactory = Callable[..., EdgeQuery]
Predicate = Callable[['Edge'], bool]
PredicateFactory = Callable[..., Predicate]

SELECTORS = Catalog('selectors')

EPSILON = 1.0
ANGLE_EPSILON = 5.0


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
                   *args) -> Generator['Edge', bool, None]:
        """
        Runs the selector and returns a generator
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


def fixed_space_boundary(space: 'Space', *_) -> Generator['Edge', bool, None]:
    """
    Returns any face around the seed space. The list is fixed and won't be affected by
    changes to the space edges
    :param space:
    :return:
    """
    for edge in space.edges:
        if edge.pair.face and edge.pair.face.space is not space:
            yield edge


def safe_boundary_edge(space: 'SeedSpace', *_) -> Generator['Edge', bool, None]:
    """
    Returns the edges not pointing to the initial face of seed space
    """
    for edge in space.edges:
        space = edge.pair.space
        face = edge.pair.face
        if not space or space.category.name != 'seed':
            yield edge
            continue
        if space.face_component(face):
            continue
        yield edge


def seed_component_boundary(space: 'Space', seed: 'Seed', *_) -> Generator['Edge', bool, None]:
    """
    Returns the edge of the space adjacent to a face close to the component of the seed
    :param space:
    :param seed:
    :return:
    """
    component = seed.components[0]  # per convention we use the first component of the seed
    for edge in component.edges:
        face = edge.pair.face
        if face is None or face.space is space or face.space.category.name == 'seed':
            continue
        # find a shared edge with the space
        for face_edge in face.edges:
            if face_edge.pair.space is space:
                break
        else:
            continue

        yield face_edge.pair


def boundary(face_or_space: Union['Face', 'Space'], *_) -> Generator['Edge', bool, None]:
    """
    Returns the edges of the face
    :param face_or_space:
    :return:
    """
    yield from face_or_space.edges


def boundary_unique(face_or_space: Union['Face', 'Space'], *_) -> Generator['Edge', bool, None]:
    """
    Returns the edge reference face
    :param face_or_space:
    :return:
    """
    yield face_or_space.edge


def boundary_unique_longest(face_or_space: Union['Face', 'Space'], *_) -> Generator['Edge', bool, None]:
    """
    Returns the longest edge of the face_or_space that is not on the plan boundary
    :param face_or_space:
    :return:
    """

    ref_edge = face_or_space.edge
    if ref_edge:
        longest_edge = ref_edge
        length = ref_edge.length
        for edge in ref_edge.space_siblings:
            if not edge.is_mesh_boundary and edge.pair:
                if longest_edge.is_mesh_boundary:
                    longest_edge = edge
                elif pseudo_equal(edge.length, length, EPSILON) and edge.pair.space.area > longest_edge.pair.space.area:
                    longest_edge = edge
                elif edge.length > length:
                    length = edge.length
                    longest_edge = edge
        yield longest_edge


def homogeneous(space: 'Space', *_) -> Generator['Edge', bool, None]:
    """
    Returns the edge such that when edge.pair.face is added to the current space, area/perimeter is max
    compared to other adds
    #TODO : we should avoid using shapely for polygon fusion - try merge and reverse
    :param space:
    :return:
    """

    list_homogeneneous = []
    ref_edge = space.edge
    biggest_shape_factor = 0
    edge_homogeneous_growth = None
    if ref_edge and ref_edge.pair and ref_edge.pair.face:
        union_sp_poly = space.as_sp.union(ref_edge.pair.face.as_sp)
        biggest_shape_factor = (union_sp_poly.area / union_sp_poly.length)
        edge_homogeneous_growth = ref_edge
    for edge in ref_edge.space_siblings:
        if edge.pair and edge.pair.face:
            union_sp_poly = space.as_sp.union(edge.pair.face.as_sp)
            current_shape_factor = (union_sp_poly.area / union_sp_poly.length)
            if current_shape_factor > biggest_shape_factor:
                biggest_shape_factor = current_shape_factor
                edge_homogeneous_growth = edge
    if edge_homogeneous_growth:
        list_homogeneneous.append(edge_homogeneous_growth)

    for edge in list_homogeneneous:
        yield edge


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


def mutable() -> Predicate:
    """
    Predicate factory
    Returns a predicate indicating if an edge belongs to a mutable space
    :return:
    """

    def _predicate(edge: 'Edge') -> bool:
        if not edge.space.mutable:
            return False
        if edge.pair:
            if not edge.pair.space.mutable:
                return False
        return True

    return _predicate


def space_area(min_area: float = None, max_area: float = None) -> Predicate:
    """
    Predicate factory
    Returns a predicate indicating if an edge belongs to a space with area in a range
    :param min_area:
    :param max_area
    :return:
    """

    def _predicate(edge: 'Edge') -> bool:
        if min_area is not None and max_area is not None:
            return min_area <= edge.space.area <= max_area
        if min_area is not None:
            return edge.space.area >= min_area
        if max_area is not None:
            return edge.space.area <= max_area

    return _predicate


def seed_empty_furthest_couple(space: 'Space', *_) -> Generator['Edge', bool, None]:
    """
    Returns for a given space the two edges that are most far from one another, basde on their start vertex
    :return:
    """
    if not space.category or space.category.name != 'empty' or space.as_sp is None:
        raise ValueError('You should provide an empty component, with a reference edge, to the query seed_empty!')
    if space.as_sp.geom_type != 'Polygon':
        raise ValueError('The space on which action is led should be a polygon!')
    kept_edges = []
    d_max = 0

    for edge in space.edges:
        d_max_edge = 0
        edge_far = None
        for edge_sibling in edge.space_siblings:
            edge_start_sp = edge.start.as_sp
            d_tmp = edge_sibling.start.as_sp.distance(edge_start_sp)
            if d_tmp > d_max_edge:
                d_max_edge = d_tmp
                edge_far = edge_sibling
        if d_max_edge > d_max:
            kept_edges = [edge, edge_far]
            d_max = d_max_edge

    for edge in kept_edges:
        yield edge


def seed_empty_furthest_couple_middle(space: 'Space', *_) -> Generator['Edge', bool, None]:
    """
    Returns for a given space the two edges that are most far from one another, based on their middle
    :return:
    """
    if not space.category or space.category.name != 'empty' or space.as_sp is None:
        raise ValueError('You should provide an empty component, with a reference edge, to the query seed_empty!')
    if space.as_sp.geom_type != 'Polygon':
        raise ValueError('The space on which action is led should be a polygon!')
    kept_edges = []
    d_max = 0

    for edge in space.edges:
        d_max_edge = 0
        edge_far = None
        for edge_sibling in edge.space_siblings:
            edge_start_middle_sp = edge.barycenter(0.5).as_sp
            d_tmp = edge_sibling.barycenter(0.5).as_sp.distance(edge_start_middle_sp)
            if d_tmp > d_max_edge:
                d_max_edge = d_tmp
                edge_far = edge_sibling
        if d_max_edge > d_max:
            kept_edges = [edge, edge_far]
            d_max = d_max_edge

    for edge in kept_edges:
        yield edge


def corner_stone(edge: 'Edge') -> bool:
    """
    Returns True if the removal of the edge's face from the space
    will cut it in several spaces
    """

    def _get_adjacent_faces(_face: 'Face') -> Generator['Face', None, None]:
        """
            Recursive function to retrieve all the faces of the space
            :param _face:
            :return:
            """
        for _edge in _face.edges:
            # if the edge is a boundary of the space do not propagate
            if _edge.is_space_boundary:
                continue
            new_face = _edge.pair.face
            if new_face and new_face.space is removed_face.space and new_face not in seen:
                seen.append(new_face)
                yield new_face
                yield from _get_adjacent_faces(new_face)

    removed_face = edge.pair.face

    if removed_face is None or removed_face.space is None:
        return False

    adjacent_faces = set([_edge.pair.face for _edge in removed_face.edges
                          if _edge.pair.face and (_edge.pair.space is removed_face.space)])

    # only face in the space
    if len(adjacent_faces) < 2:
        return False

    found_isolated_face = False

    for face in adjacent_faces:
        seen = [removed_face, face]
        for other_face in _get_adjacent_faces(face):
            if other_face in adjacent_faces:
                break
        else:
            found_isolated_face = True
            break

    return found_isolated_face


# Query factories


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

    def _selector(space_or_face: Union['Space', 'Face'], *_) -> Generator['Edge', bool, None]:
        vectors = ((space_or_face.edge.unit_vector, opposite_vector(space_or_face.edge.unit_vector))
                   if direction == 'horizontal' else
                   (space_or_face.edge.normal,))

        for vector in vectors:
            edges_list = [edge for edge in space_or_face.edges
                          if pseudo_equal(ccw_angle(edge.normal, vector), 180.0, epsilon)]
            for edge in edges_list:
                face = edge.pair.face
                if face is None:
                    continue
                yield edge

    return _selector


# predicates


def not_space_boundary(edge: 'Edge') -> bool:
    """
    Predicate
    Returns True if the edge is not a space boundary
    :param edge:
    :return:
    """
    return not edge.is_space_boundary


def adjacent_to_other_space(edge: 'Edge') -> bool:
    """
        Predicate
        Returns True if the edge is adjacent to another space
        :param edge:
        :return:
        """
    return edge.pair.face and edge.pair.face.space is not edge.space and edge.space.mutable


def not_adjacent_to_seed(edge: 'Edge') -> bool:
    """
    return True if the edge.pair does not belong to a space of category seed
    :param edge:
    :return:
    """
    return edge.pair.space is None or edge.pair.space.category.name != 'seed'


def adjacent_empty_space(edge: 'Edge') -> bool:
    """
    Return True if the edge pair belongs to a space of the 'empty' category
    :param edge:
    :return:
    """
    return edge.pair.space and edge.pair.space.category.name == 'empty'


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

    def _predicate(edge: 'Edge') -> bool:
        return not predicate(edge)

    return _predicate


def is_previous(predicate: Predicate) -> Predicate:
    """
    Applies the predicate to the previous edge
    :param predicate:
    :return:
    """

    def _predicate(edge: 'Edge') -> bool:
        return predicate(edge.previous)

    return _predicate


def is_next(predicate: Predicate) -> Predicate:
    """
    Applies the predicate to the next edge
    :param predicate:
    :return:
    """

    def _predicate(edge: 'Edge') -> bool:
        return predicate(edge.next)

    return _predicate


def edge_angle(min_angle: Optional[float] = None,
               max_angle: Optional[float] = None,
               previous: bool = False,
               space: bool = False) -> Predicate:
    """
    Predicate factory
    Returns a predicate indicating if an edge angle with its next edge
    is comprised between the two provided values. Only one of the value can be provided
    :param min_angle:
    :param max_angle:
    :param previous : whether to check for the angle to the next edge or to the previous edge
    :param space: whether to check for angle between to space siblings
    :return: boolean
    """
    if min_angle is None and max_angle is None:
        raise ValueError('A min or a max angle must be provided to the angle predicate factory')

    def _predicate(edge: 'Edge') -> bool:

        if space:
            if not edge.is_space_boundary:
                return False
            _angle = edge.space_next_angle if not previous else edge.space_previous_angle
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

    def _predicate(edge: 'Edge') -> bool:
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

    def _predicate(edge: 'Edge') -> bool:

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

    def _predicate(edge: 'Edge') -> bool:
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

    def _predicate(edge: 'Edge') -> bool:
        if position == 'before':
            return edge.next.linear and edge.next.linear.category.name in category_names
        if position == 'after':
            return edge.previous.linear and edge.previous.linear.category.name in category_names
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

    def _predicate(edge: 'Edge'):
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


# Catalog Selectors


SELECTORS.add(

    Selector(boundary, 'boundary'),

    Selector(seed_component_boundary, 'seed_component_boundary'),

    Selector(boundary, [edge_angle(180.0, 270.0, previous=True),
                        aligned_edges_length(min_length=150.0),
                        is_previous(aligned_edges_length(min_length=150.0))],
             'previous_angle_salient_non_ortho'),

    Selector(boundary, [edge_angle(180.0, 270.0),
                        aligned_edges_length(min_length=150.0),
                        is_next(aligned_edges_length(min_length=150.0))],
             'next_angle_salient_non_ortho'),

    Selector(boundary, [edge_angle(90.0, 180.0),
                        aligned_edges_length(min_length=150.0),
                        is_next(aligned_edges_length(min_length=150.0))],
             'next_angle_convex_non_ortho'),

    Selector(boundary, [edge_angle(90.0, 180.0, previous=True),
                        aligned_edges_length(min_length=100.0),
                        is_previous(aligned_edges_length(min_length=150.0))],
             'previous_angle_convex_non_ortho'),

    Selector(boundary, [edge_angle(270.0, 270.0, previous=True, space=True)],
             'previous_angle_salient_ortho'),

    Selector(boundary, [edge_angle(270.0, 270.0, space=True)],
             'next_angle_salient_ortho'),

    Selector(boundary, [close_to_linear('window', 'doorWindow', min_distance=90.0),
                        not_space_boundary], 'close_to_window'),

    Selector(boundary, [touches_linear('window', 'doorWindow', position='between')],
             'between_windows'),

    Selector(boundary, [is_previous(touches_linear('window', 'doorWindow', position='after')),
                        is_next(touches_linear('window', 'doorWindow'))],
             'between_edges_between_windows'),

    Selector(boundary, [edge_angle(180.0, 180.0),
                        is_not(touches_linear('window', 'doorWindow', 'frontDoor')),
                        is_not(is_linear('window', 'doorWindow', 'frontDoor'))], 'aligned_edges'),

    Selector(boundary, [edge_length(min_length=150.0)], 'edge_min_150'),

    Selector(boundary, [edge_length(min_length=300.0)], 'edge_min_300'),

    Selector(boundary, [edge_length(min_length=500.0)], 'edge_min_500'),

    Selector(fixed_space_boundary, (adjacent_empty_space,), 'boundary_other_empty_space'),

    Selector(seed_component_boundary, name='surround_seed_component'),

    Selector(seed_duct, name='seed_duct'),

    Selector(seed_empty_furthest_couple, name='seed_empty_furthest_couple'),

    Selector(seed_empty_furthest_couple, [space_area(min_area=100000)],
             name='seed_empty_furthest_couple_space_area_min_100000'),

    Selector(seed_empty_furthest_couple_middle, [space_area(min_area=100000)],
             name='seed_empty_furthest_couple_middle_space_area_min_100000'),

    Selector(boundary_unique, [space_area(max_area=100000)], name='area_max=100000'),

    Selector(boundary_unique_longest, [space_area(max_area=30000), mutable()], name='fuse_small_cell'),

    Selector(homogeneous, (adjacent_empty_space,), name='homogeneous'),

    Selector(safe_boundary_edge, (adjacent_to_other_space, is_not(corner_stone)),
             'other_space_boundary')
)

# Catalog Selector factories


SELECTORS.add_factory(

    SelectorFactory(oriented_edges, factorize(adjacent_empty_space, ), name='oriented_edges')
)
