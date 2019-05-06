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
import math
import logging

from libs.utils.geometry import (
    ccw_angle,
    opposite_vector,
    pseudo_equal,
    barycenter,
    distance,
    parallel
)

if TYPE_CHECKING:
    from libs.mesh.mesh import Edge
    from libs.plan.plan import Space, Plan
    from libs.modelers.seed import Seeder

EdgeQuery = Callable[['Space', Any], Generator['Edge', bool, None]]
EdgeQueryFactory = Callable[..., EdgeQuery]
Predicate = Callable[['Edge', 'Space'], bool]
PredicateFactory = Callable[..., Predicate]

EPSILON = 1.0
ANGLE_EPSILON = 1.0
MIN_ANGLE = 5.0


class Selector:
    """
    Returns an iterator on a given plan space
    """

    __slots__ = 'name', 'query', 'predicates'

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
def space_boundary(space: 'Space', *_) -> Generator['Edge', bool, None]:
    """
    Returns the edges of the space
    :param space:
    :return:
    """
    if space.edge:
        yield from space.edges


def space_external_boundary(space: 'Space', *_) -> Generator['Edge', bool, None]:
    """
    Returns the external edges of the space
    :param space:
    :return:
    """
    yield from space.exterior_edges


def touching_space_boundary(space: 'Space', *_) -> Generator['Edge', bool, None]:
    """
    Returns the edges touching an edge of the space
    :param space:
    :param _:
    :return:
    """
    for edge in space.edges:
        yield edge.next


def boundary_faces(space: 'Space', *_) -> Generator['Edge', bool, None]:
    """
    Returns the edges of the face
    :param space:
    :return:
    """
    for face in space.faces:
        yield from face.edges


def boundary_faces_fixed(space: 'Space', *_) -> Generator['Edge', bool, None]:
    """
    Returns the edges of the face in a fixed list
    :param space:
    :return:
    """
    for face in space.faces:
        for edge in list(face.edges):
            yield edge


def boundary_unique(space: 'Space', *_) -> Generator['Edge', bool, None]:
    """
    Returns the edge reference of the space
    :param space:
    :return:
    """
    if space.edge:
        yield space.edge


def boundary_faces_biggest(space: 'Space', *_) -> Generator['Edge', bool, None]:
    """
    Returns the biggest edge of each face
    :param space:
    :return:
    """
    for face in space.faces:
        if face.edge:
            output = max((e for e in face.edges if space.has_edge(e.pair)),
                         key=lambda e: e.length, default=None)
            if output is not None:
                yield output


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


def cuts_linear(space: 'Space', *_) -> Generator['Edge', bool, None]:
    """
    Returns the edge cutting a linear
    :param space:
    :return:
    """
    plan = space.plan
    for edge in space.edges:
        linear = plan.get_linear(edge)
        if linear and linear.has_edge(space.next_edge(edge)):
            yield edge.next


def homogeneous(space: 'Space', *_) -> Generator['Edge', bool, None]:
    """
    Returns among all edges on the space border the one such as when the pair
     face is added to space, the size ratio defined as depth/width is closer to one
    """

    biggest_shape_factor = None
    edge_homogeneous_growth = None

    for edge in space.edges:
        if edge.pair and edge.pair.face and space.plan.get_space_of_edge(
                edge.pair).category.name == 'empty':
            face_added = edge.pair.face
            space_contact = space.plan.get_space_of_face(face_added)
            if space_contact.corner_stone(face_added):
                continue
            space_contact.remove_face(face_added)
            space.add_face(face_added)
            size_ratio = space.size.depth / space.size.width
            current_shape_factor = max(size_ratio, 1 / size_ratio)
            space.remove_face(face_added)
            space_contact.add_face(face_added)
            if biggest_shape_factor is None or current_shape_factor <= biggest_shape_factor:
                biggest_shape_factor = current_shape_factor
                edge_homogeneous_growth = edge

    if edge_homogeneous_growth:
        yield edge_homogeneous_growth


def best_aspect_ratio(space: 'Space', *_) -> Generator['Edge', bool, None]:
    """
    Returns among all edges on the space border the one such as when the pair
    face is added to space, the ratio perimeter/area is smallest
    """

    biggest_shape_factor = math.inf
    edge_homogeneous_growth = None

    for edge in space.edges:
        if edge.pair.face:
            face_added = edge.pair.face
            space_added = space.plan.get_space_of_face(face_added)
            if space_added.category.name != 'empty':
                continue
            current_shape_factor = space.aspect_ratio([face_added])
            if current_shape_factor < biggest_shape_factor:
                biggest_shape_factor = current_shape_factor
                edge_homogeneous_growth = edge

    if edge_homogeneous_growth:
        yield edge_homogeneous_growth


def improved_aspect_ratio(space: 'Space', *_) -> Generator['Edge', bool, None]:
    """
    Returns among all edges on the space border the one such as when the pair
    face is added to space, the ratio perimeter/area is reduced
    """

    best_shape_factor = space.perimeter**2 / space.area
    edge_homogeneous_growth = None

    for edge in space.edges:
        if not edge.pair.face:
            continue
        face_added = edge.pair.face
        space_added = space.plan.get_space_of_face(face_added)
        if space_added.category.name != 'empty':
            continue
        current_shape_factor = space.aspect_ratio([face_added])
        if current_shape_factor <= best_shape_factor:
            best_shape_factor = current_shape_factor
            edge_homogeneous_growth = edge

    if edge_homogeneous_growth:
        yield edge_homogeneous_growth


def seed_duct(space: 'Space', *_) -> Generator['Edge', bool, None]:
    """
    Returns the edge that can be seeded for a duct
    """
    if not space.category or space.category.name != 'duct':
        raise ValueError('You should provide a duct to the query seed_duct!')

    edge_along_plan = None
    for edge in space.edges:
        if edge.pair.face is None:
            edge_along_plan = edge
            break

    if edge_along_plan:
        yield edge_along_plan.next_ortho().pair
        yield edge_along_plan.previous_ortho().pair
        yield edge_along_plan.next_ortho().next_ortho().pair
    else:
        for edge in space.edges:
            if edge.next_ortho() is edge.next:
                yield edge.pair


def adjacent_to_rectangular_duct(space: 'Space', *_) -> Generator['Edge', bool, None]:
    """
    Returns the pair edge of each duct that has a rectangular form
    :param space:
    :param _:
    :return:
    """
    plan = space.plan
    space.bounding_box()
    for duct in plan.get_spaces("duct"):
        box = duct.bounding_box()
        is_rectangular = math.fabs(box[0] * box[1] - duct.area) < EPSILON
        if is_rectangular:
            yield from (edge.pair for edge in duct.edges if space.has_edge(edge.pair))


def one_edge_adjacent_to_rectangular_duct(space: 'Space', *_) -> Generator['Edge', bool, None]:
    """
    Returns the pair edge of each duct that has a rectangular form
    :param space:
    :param _:
    :return:
    """
    plan = space.plan
    for duct in plan.get_spaces("duct"):
        for edge in duct.edges:
            if space.has_edge(edge.pair):
                yield edge.pair
                break


def vertical_edge(space: 'Space', *_) -> Generator['Edge', bool, None]:
    """
    returns edges in vertical direction from space reference edge
    """

    if not space.edge:
        return

    vectors = (space.edge.normal, opposite_vector(space.edge.normal))

    for vector in vectors:
        edges_list = [edge for edge in space.edges
                      if pseudo_equal(ccw_angle(edge.normal, vector), 180.0, 35)]
        for edge in edges_list:
            yield edge


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
    min_edge_length = 40.0  # the minimum length of the edge to be considered

    if direction not in ('horizontal', 'vertical'):
        raise ValueError('A direction can only be horizontal or vertical: {0}'.format(direction))

    go_left = {}

    def _selector(space: 'Space',
                  seeder: Optional['Seeder'] = None) -> Generator['Edge', bool, None]:

        reference_edge = (seeder.get_seed_from_space(space).components[0].edge
                          if seeder else space.edge)

        if not space.is_boundary(reference_edge):
            raise ValueError("Selector: The edge must be a boundary of the space")

        if not reference_edge:
            return

        if direction == "horizontal":
            angle = ccw_angle(reference_edge.unit_vector
                              if go_left.get(space.id, False) else reference_edge.opposite_vector)
            edges_list = [edge for edge in space.siblings(reference_edge)
                          if pseudo_equal(ccw_angle(edge.normal), angle, epsilon)
                          and edge.length >= min_edge_length]

            if not edges_list:
                return
            # we alternate for each space : left and right to ensure a symmetric propagation
            # go_left is memoized
            if go_left.get(space.id, False):
                edges_list = edges_list[::-1]
                go_left[space.id] = False
            else:
                go_left[space.id] = True

            # we only return the first edge found
            yield edges_list[0]
        else:
            angle = ccw_angle(reference_edge.pair.normal)
            edges_list = [edge for edge in space.siblings(reference_edge)
                          if pseudo_equal(ccw_angle(edge.normal), angle, epsilon)
                          and edge.length >= min_edge_length]
            # only yield if all edges pair point to an empty space
            for e in edges_list:
                other_space = space.plan.get_space_of_edge(e.pair)
                if not other_space or other_space.category.name != "empty":
                    break
            else:
                yield from (e for e in edges_list)

    return _selector


def min_depth(depth: float, min_length: float = 10) -> EdgeQuery:
    """
    Returns an edge from a space that has a depth inferior to the specified value.
    Note : checks the adjacent face to return the edge between to thin faces
    :param depth: the minimum depth of the edge
    :param min_length: the minimum length of the edge
    :return: an EdgeQuery
    """

    def _selector(space: 'Space', *_) -> Generator['Edge', bool, None]:

        if not space.edge:
            return

        for face in space.faces:
            for edge in face.edges:
                if space.is_boundary(edge) or edge.length < min_length:
                    continue
                if edge.depth < depth:
                    yield edge

    return _selector


def tight_lines(depth: float, min_line_length: float = 20) -> EdgeQuery:
    """
    Returns a query that returns the edge of a line close to another line.
    The line is chosen to enable the best grid after its removal, according to the following rules:
    • we cannot pick a line that ends on a T junction (meaning its removal will create a non
    rectangular face)
    • if three lines are close together we pick the middle one
    • if there are only two lines we pick the shortest one
    :param depth:
    :param min_line_length:
    :return: an EdgeQuery
    """

    def _selector(space: 'Space', *_) -> Generator['Edge', bool, None]:

        if not space.edge:
            return

        output = None

        for edge in min_depth(depth)(space):
            edges = list(_parallel_edges(edge, space, depth, min_line_length))
            # specific case of triangle faces, otherwise edges should always have >= 2 elems
            if len(edges) <= 1:
                continue
            _edges = list(filter(_filter_lines(space), edges))
            if len(_edges) == 0:
                continue
            elif len(_edges) == 1:
                output = _edges[0]
            elif len(_edges) >= 2:
                # we calculate the length of the line that touches a border per increment of 20
                borders = [int(_border_length(edge, space) / 20) * 20 for edge in _edges]
                lengths = [int(_line_length(edge) / 20) * 20 for edge in _edges]
                logging.debug("Selector: removing tight lines")
                for criteria in (borders, lengths):
                    order = [sum((c > b + EPSILON) for b in criteria) for c in criteria]
                    if order.count(0) == 1:
                        i = order.index(0)
                        output = _edges[i]
                        break
                    if len(_edges) > 1 and order.count(len(_edges) - 1) == 1:
                        i = order.index(len(_edges) - 1)
                        del _edges[i]
                        del borders[i]
                        del lengths[i]
                else:
                    if len(_edges) == 3:
                        output = _edges[1]
                    elif len(_edges) == 2:
                        output = _edges[0] if edges.index(_edges[0]) == 1 else _edges[1]
                    else:
                        output = _edges[0]

            if output:
                yield output

    return _selector


def _tight_length(edge: 'Edge', depth: float) -> float:
    """
    Returns the length of the line where each face has a depth < depth
    :param edge:
    :param depth
    :return:
    """
    return sum(map(lambda e: e.length, filter(lambda e: e.depth < depth, edge.line)))


def _line_length(edge: 'Edge') -> float:
    """
    Returns the length of a line of an edge
    :param edge:
    :return:
    """
    return sum(map(lambda e: e.length, edge.line))


def _border_length(edge: 'Edge', space: 'Space') -> float:
    """
    Returns the length of the line where the edge are on the boundary of the space
    :param edge:
    :param space:
    :return:
    """
    plan = space.plan

    def _duct_loadbearingwall_length(_edge: 'Edge') -> float:
        if space.is_internal(_edge):
            return 0.0
        _edge_to_check = _edge if space.is_outside(_edge) else _edge.pair
        _space_to_check = plan.get_space_of_edge(_edge_to_check)
        if _space_to_check and _space_to_check.category.name in ("duct", "loadBearingWall"):
            return _edge_to_check.length
        return 0.0

    return sum(map(_duct_loadbearingwall_length, edge.line))


def _parallel_edges(edge: 'Edge', space: 'Space',
                    depth: float, min_length: float = 20.0) -> ['Edge']:
    """
    Returns up to three parallel close edges of minimum line length
          left
    *-------------->
    <--------------*

         middle
    *-------------->
    <--------------*

         right
    *------------->
    <-------------*
    :param edge
    :param depth
    :param min_length
    :return:
    """
    middle = None

    # look to the left
    left = _parallel(edge, space, depth, min_length)
    if left:
        middle = edge
    else:
        left = edge

    # look to the right
    right = _parallel(edge.pair, space, depth, min_length)
    if not right:
        if middle:
            new_left = _parallel(left, space, depth, min_length)
            if new_left:
                return [new_left, left, middle]
            return [left, middle]
        else:
            return [edge]
    else:
        if middle:
            right = right.pair
            return left, middle, right
        else:
            middle = right.pair
            new_right = _parallel(middle.pair, space, depth, min_length)
            if new_right:
                return [left, right, new_right]
            else:
                return [edge, right]


def _parallel(edge: 'Edge', space: 'Space',
              dist: float, min_length: float = 20) -> Optional['Edge']:
    """
    Returns an edge parallel to the left at a distance smaller than the specified dist
    and a line length superior at the min_length specified
    :param edge:
    :param space:
    :param dist:
    :param min_length
    :return: the parallel edge
    """
    for _edge in edge.siblings:
        if (_edge is not edge and edge.face is not None
                and not space.is_boundary(_edge)
                and parallel(_edge.pair.vector, edge.vector)
                and _edge.max_distance(edge, parallel=True) < dist
                and _line_length(_edge) > min_length):
            return _edge.pair
    return None


def _line_cuts_angle(line: ['Edge'], space: 'Space') -> bool:
    """
    return True if the line cuts an angle in half
    Its removal will create an angle > 180°.
    We check the length of the edges of the angle. An angle whose edges have a length inferior
    to min_length is acceptable.

    :param line:
    :param space
    :return:
    """
    min_length = 10
    if len(line) == 0:
        return False

    for i, edge in enumerate(line):
        # we skip the last edge
        if i == len(line) - 1:
            continue
        if not space.is_internal(edge):
            continue
        edges = list(edge.end.edges)
        if len(edges) != 4:
            continue
        edges = list(filter(lambda e: e not in (line[i], line[i + 1],
                                                line[i].pair, line[i + 1].pair), edges))
        if (edges[0].length > min_length and edges[1].length > min_length and
                not pseudo_equal(ccw_angle(edges[0].vector, edges[1].vector), 180, MIN_ANGLE)):
            return True

    return False


def _line_is_between_windows(line: ['Edge'], space: 'Space') -> bool:
    """
    Returns True if the line contains the only edge between two windows
    :param line:
    :param space:
    :return:
    """
    plan = space.plan

    end_edge = line[len(line) - 1]
    start_edge = line[0]

    j = len(line) - 1
    while not space.is_internal(end_edge) and j > 0:
        j -= 1
        end_edge = line[j]

    i = 0
    while not space.is_internal(start_edge) and i < j:
        i += 1
        start_edge = line[i]

    for edge in (start_edge.pair, end_edge):
        found_linear = False
        # forward check
        for _edge in edge.siblings:
            linear = plan.get_linear(_edge)
            if linear and linear.category.name in ("window", "doorWindow"):
                found_linear = True
                break
        # backward check
        for _edge in edge.pair.reverse_siblings:
            linear = plan.get_linear(_edge)
            if linear and linear.category.name in ("window", "doorWindow"):
                if found_linear:
                    return True

    return False


def _line_touches_duct_or_loadbearingwall(line: ['Edge'], space: 'Space') -> bool:
    """
    Returns True if the line touches a duct
    The objective is to prevent ducts from being isolated inside a face
    :param line:
    :param space:
    :return:
    """
    plan = space.plan

    for edge in line:
        if space.is_internal(edge):
            continue
        _edge_to_check = edge if space.is_outside(edge) else edge.pair
        _space_to_check = plan.get_space_of_edge(_edge_to_check)
        if _space_to_check and _space_to_check.category.name in ("duct", "loadBearingWall"):
            return True
    return False


def _filter_lines(space: 'Space') -> callable([['Edge'], bool]):
    """
    Filter function
    :param space:
    :return:
    """

    def _filter(edge: 'Edge') -> bool:
        line = edge.line
        return (not _line_is_between_windows(line, space)
                # and not _line_touches_duct_or_loadbearingwall(line, space)
                and not _line_cuts_angle(line, space))

    return _filter


def specific_category(*category_names: str) -> EdgeQuery:
    """
    Returns the boundary edge of th
    :param category_names:
    :return:
    """
    def _query(space: 'Space', *_) -> Generator['Edge', bool, None]:
        plan = space.plan
        if space and space.category.name in category_names:
            for e in space.exterior_edges:
                other = plan.get_space_of_edge(e.pair)
                if other and other.mutable:
                    yield e
    return _query


# predicates


def t_edge(edge: 'Edge', space: 'Space') -> bool:
    """
    Returns True if the edge removal will create a non rectangular face.
    :param edge:
    :param space:
    :return:
    """
    if space.is_boundary(edge) or space.is_boundary(edge.next):
        return False

    output = (edge.next.pair.next.pair.next.pair is edge
              and not edge.next.pair.next_is_aligned)

    return output


def only_face(edge: 'Edge', space: 'Space') -> bool:
    """
    Returns True if the face of the edge is the only face of the space
    :param edge:
    :param space:
    :return:
    """
    assert space.has_edge(edge), "The edge should belong to the space"

    if space.number_of_faces <= 1:
        return True


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


def adjacent_to_empty_space(edge: 'Edge', space: 'Space') -> bool:
    """
    Predicate
    Returns True if edge pair belongs to an empty space
    :param edge:
    :param space:
    :return:
    """
    space_pair = space.plan.get_space_of_face(edge.pair.face)
    val = space_pair is not space and space_pair is not None and space_pair.category.name is 'empty'
    return val


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
    face = edge.face

    if not face or not space:
        return False

    return space.corner_stone(face)


def not_aligned_edges(space: 'Space', *_) -> Generator['Edge', bool, None]:
    """
    Returns edges not aligned with their previous or next space sibling
    """
    for edge in space.edges:
        if is_not_aligned(edge, space) or is_not_aligned(edge, space, previous=True):
            yield edge


def is_not_aligned(edge: 'Edge', space: 'Space', previous: bool = False):
    """
    Returns True if edge is not aligned with the next one (or previous one if previous==True)
    """
    if previous:
        return not space.next_is_aligned(edge)

    return not space.next_is_aligned(space.previous_edge(edge))


def is_after_corner(edge: 'Edge', space: 'Space', corner_angle_min: float = 60,
                    corner_angle_max: float = 110):
    """
    Returns True if edge is after a corner of the space
    """
    return corner_angle_max >= ccw_angle(space.previous_edge(edge).vector,
                                         edge.vector) >= corner_angle_min


def h_edge(edge: 'Edge', space: 'Space') -> bool:
    """
    an edge that is the middle of an H shape (linking two aligned edges on each side)
    :param edge:
    :param space:
    :return:
    """
    if space.is_boundary(edge):
        return False

    for _edge in (edge, edge.pair):
        if space.is_boundary(_edge.next):
            return False
        if not (_edge.next.pair.next_is_aligned and _edge.next.pair.next.pair.next is _edge.pair):
            return False

    return True


def corner_face(edge: 'Edge', space: 'Space') -> bool:
    """
    Returns True if the edge is before a corner of the space
    :param edge:
    :param space:
    :return:
    """
    min_corner_angle = 50  # Arbitrary TODO: parametrize this
    if not space.is_internal(edge) and space.is_internal(edge.next):
        return False

    return edge.next_angle > 180 + min_corner_angle


def wrong_direction(edge: 'Edge', space: 'Space') -> bool:
    """
    Returns True if the edge is internal of the space and not along one of its direction
    :param edge:
    :param space:
    :return:
    """
    if not space.directions:
        return False

    if not space.is_internal(edge):
        return False

    vector = edge.unit_vector
    delta = map(lambda d: pseudo_equal(ccw_angle(vector, d), 180, ANGLE_EPSILON), space.directions)
    if max(list(delta)) == 0:
        return True

    return False


def is_mutable(_: 'Edge', space: 'Space') -> bool:
    """
    Returns True if the edge pair space is mutable
    :param _:
    :param space:
    :return:
    """
    return space and space.mutable


def has_needed_linear(edge: 'Edge', space: 'Space') -> bool:
    """
    Returns True if the edge face has an immutable component
    :param edge:
    :param space:
    :return:
    """
    face = edge.face

    if not space.category.needed_linears or not face:
        return False

    for _edge in edge.face.edges:
        linear = space.plan.get_linear_from_edge(_edge)
        if linear and linear.category in space.category.needed_linears:
            return True
    return False


def only_adjacent_to_immutable(edge: 'Edge', space: 'Space') -> bool:
    """
    Returns True if the edge face has an immutable component
    :param edge:
    :param space:
    :return:
    """
    face = edge.face

    if not space.category.needed_spaces or not face:
        return False

    # check if the face of the edge is adjacent to a needed space
    for _edge in face.edges:
        if space.is_boundary(_edge):
            other = space.plan.get_space_of_edge(_edge.pair)
            if other and other.category in space.category.needed_spaces:
                break
    else:
        return False

    # check if another face maintains the needed adjacency
    for _edge in space.edges:
        if _edge.face is face:
            continue
        _other = space.plan.get_space_of_edge(_edge.pair)
        if not _other:
            continue
        if _other.category is other.category:
            return False

    return True


def adjacent_to_external_space(edge: 'Edge', space: 'Space') -> bool:
    """
    Returns True if the edge pair is on the boundary of the mesh or belongs to an
    external space
    :param edge:
    :param space:
    :return:
    """
    if edge.pair.face is None:
        return True
    other = space.plan.get_space_of_edge(edge.pair)
    return not other or other.category.external


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


def _or(*predicates: Predicate) -> Predicate:
    """
    Returns a predicate that returns True if at least one of the specified predicates is True
    :param predicates:
    :return:
    """

    def _predicate(edge: 'Edge', space: 'Space') -> bool:
        for predicate in predicates:
            if predicate(edge, space):
                return True
        return False

    return _predicate


def pair(predicate: Predicate) -> Predicate:
    """
    Applies the predicate to the pair of the edge
    :param predicate:
    :return:
    """
    def _predicate(edge: 'Edge', space: 'Space') -> bool:
        pair_edge = edge.pair
        pair_space = space.plan.get_space_of_edge(pair_edge)
        return predicate(pair_edge, pair_space)

    return _predicate


def previous_has(predicate: Predicate) -> Predicate:
    """
    Applies the predicate to the previous edge
    :param predicate:
    :return:
    """

    def _predicate(edge: 'Edge', space: 'Space') -> bool:
        return predicate(edge.previous, space)

    return _predicate


def next_has(predicate: Predicate) -> Predicate:
    """
    Applies the predicate to the next edge
    :param predicate:
    :return:
    """

    def _predicate(edge: 'Edge', space: 'Space') -> bool:
        return predicate(edge.next, space)

    return _predicate


def space_previous_has(predicate: Predicate) -> Predicate:
    """
    Applies the predicate to the previous edge
    :param predicate:
    :return:
    """

    def _predicate(edge: 'Edge', space: 'Space') -> bool:
        return predicate(space.previous_edge(edge), space)

    return _predicate


def space_next_has(predicate: Predicate) -> Predicate:
    """
    Applies the predicate to the next edge
    :param predicate:
    :return:
    """

    def _predicate(edge: 'Edge', space: 'Space') -> bool:
        return predicate(space.next_edge(edge), space)

    return _predicate


def edge_angle(min_angle: Optional[float] = None,
               max_angle: Optional[float] = None,
               previous: bool = False,
               on_boundary: bool = False) -> Predicate:
    """
    Predicate factory
    Returns a predicate indicating if an edge angle with its next edge
    is comprised between the two provided values. Only one of the value can be provided
    :param min_angle:
    :param max_angle:
    :param previous : whether to check for the angle to the next edge or to the previous edge
    :param on_boundary: whether to check for angle between to space siblings
    :return: boolean
    """
    if min_angle is None and max_angle is None:
        raise ValueError('A min or a max angle must be provided to the angle predicate factory')

    def _predicate(edge: 'Edge', space: 'Space') -> bool:

        # compute the angle
        if on_boundary:
            if not space.is_boundary(edge):
                return False
            _angle = space.next_angle(edge) if not previous else space.previous_angle(edge)
        else:
            _angle = edge.next_angle if not previous else edge.previous_angle

        # check the angle value
        if min_angle is not None and max_angle is not None:
            if min_angle != max_angle:
                return min_angle + ANGLE_EPSILON <= _angle <= max_angle - ANGLE_EPSILON
            else:
                # we check for pseudo equality
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

    def _predicate(edge: 'Edge', _: 'Space') -> bool:
        if min_length is not None and max_length is not None:
            return min_length <= edge.length <= max_length
        if min_length is not None:
            return edge.length >= min_length
        if max_length is not None:
            return edge.length <= max_length

    return _predicate


def space_aligned_edges_length(min_length: float = None, max_length: float = None) -> Predicate:
    """
    Predicate factory
    Returns a predicate indicating if an edge an all following aligned edges have a total length
    inferior or equal to the provided max length, or a superior length to the provided min_length
    or both
    :param min_length:
    :param max_length
    :return:
    """

    def _predicate(edge: 'Edge', space: 'Space') -> bool:

        # the edge has to be on the boundary of the space
        if not space.is_boundary(edge):
            return False

        length = 0
        for aligned_edge in space.aligned_siblings(edge):
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
    Returns a predicate indicating if an edge is on, before, after
    or between two linears of the provided category
    :param category_names: tuple of linear category names. If no category is specified will
                           consider every linear.
    :param position : where should the edge be : before, after, between, on
    :return:
    """

    position_valid_values = 'before', 'after', 'between', 'on'

    if position not in position_valid_values:
        raise ValueError('Wrong position value in predicate factory touches_linear:' +
                         ' {0}'.format(position))

    def _predicate(edge: 'Edge', space: 'Space') -> bool:
        # check if the edge belongs to a linear
        linear = space.plan.get_linear(edge)
        is_on_linear = linear and (not category_names or linear.category.name in category_names)

        if position == 'on':
            return is_on_linear

        if is_on_linear:
            return False

        if position == 'before':
            next_linear = space.plan.get_linear(edge.next)
            return (next_linear and
                    (not category_names or next_linear.category.name in category_names))

        if position == 'after':
            previous_linear = space.plan.get_linear(edge.previous)
            return (previous_linear and
                    (not category_names or previous_linear.category.name in category_names))

        if position == 'between':
            next_linear = space.plan.get_linear(edge.next)
            previous_linear = space.plan.get_linear(edge.previous)
            if ((previous_linear and (not category_names
                                      or previous_linear.category.name in category_names))
                and (next_linear and (not category_names
                                      or next_linear.category.name in category_names))):
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

    def _predicate(edge: 'Edge', space: 'Space') -> bool:
        linear_edges = []
        for sibling in edge.siblings:
            linear = space.plan.get_linear(sibling)
            if linear and linear.category.name in category_names:
                linear_edges.append(sibling)

        if not linear_edges:
            return False

        for linear_edge in linear_edges:
            max_distance = linear_edge.max_distance(edge, parallel=True)
            if max_distance is not None and max_distance <= min_distance:
                return True

        return False

    return _predicate


def close_to_apartment_boundary(min_distance: float = 90.0, min_length: float = 20.0) -> Predicate:
    """
    Predicate factory
    Returns a predicate indicating if the edge is closer than
    the provided minimum distance to the mesh boundary
    :param min_distance:
    :param min_length
    :return: function
    """

    def _predicate(edge: 'Edge', space: 'Space') -> bool:
        plan: 'Plan' = space.plan

        # per convention if an edge is on the apartment boundary it cannot be too close
        if plan.is_external(edge.pair):
            return False

        # we must not take into account the external edge that is connected to the edge
        # in the case of non orthogonal cuts
        external_edges = [e for e in edge.siblings if plan.is_external(e.pair)
                          and e is not edge.next and e.next is not edge]

        if not external_edges:
            return False

        for external_edge in external_edges:
            if external_edge.length < min_length:
                continue
            max_distance = external_edge.max_distance(edge, parallel=True)
            if max_distance is not None and max_distance < min_distance:
                return True

        return False

    return _predicate


def adjacent_to_space(*category_names: str) -> Predicate:
    """
    Predicate factory
    Returns a predicate that returns True if the edge.pair belongs to a space of the
    specified category name
    :param category_names: "exterior" means the edge is adjacent to the outside of the apartment
    :return: a predicate
    """

    def _predicate(edge: 'Edge', space: 'Space') -> bool:
        if not space.is_boundary(edge):
            return False
        space = space.plan.get_space_of_edge(edge.pair)
        return (space is not None and space.category.name in category_names or
                "exterior" in category_names and space is None)

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


def cell_with_component(has_component: bool = False) -> Predicate:
    """
    Predicate factory
    Returns a predicate indicating if a space has components
    :param has_component:
    :return:
    """

    def _predicate(_: 'Edge', space: 'Space') -> bool:
        return len(space.components_category_associated()) == 0 if not has_component else True

    return _predicate


def has_space_pair() -> Predicate:
    """
    Predicate factory
    Returns a predicate indicating if a space has a pair through a given edge
    TODO: using a factory here makes no sense...
    :return:
    """

    def _predicate(edge: 'Edge', space: 'Space') -> bool:
        if not edge.pair:
            return False
        else:
            if space.plan.get_space_of_edge(edge.pair) is None:
                return False
        return True

    return _predicate


def face_proportion(max_proportion: float = 0.1) -> Predicate:
    """
    Predicate factory
    Returns a predicate indicating if (surface of edge.pair face)/(surface of edge space) and
    (surface of edge.pair face)/(surface of edge.pair space) are lower than max_proportion
    """

    def _predicate(edge: 'Edge', space: 'Space') -> bool:
        if not edge.face or not edge.pair.face:
            return False
        _space_area = space.plan.get_space_of_edge(edge).area
        face_pair_area = edge.pair.face.area
        space_pair_area = space.plan.get_space_of_edge(edge.pair).area

        if (face_pair_area / _space_area > max_proportion
                or face_pair_area / space_pair_area > max_proportion):
            return False
        return True

    return _predicate


def face_without_component() -> Predicate:
    """
    Predicate factory
    Returns a predicate indicating if edge pair face has non mutable components
    TODO: using a factory here makes no sense...
    """

    from libs.plan.category import SPACE_CATEGORIES
    from libs.plan.plan import Space

    def _predicate(edge: 'Edge', space: 'Space') -> bool:
        if not edge.pair.face:
            return False

        # creates a temporary space with edge.pair.face as only space
        tmp_space = Space(space.plan, space.floor, edge.pair,
                          SPACE_CATEGORIES[space.category.name])
        tmp_space.add_face(edge.pair.face)

        has_component = False
        if tmp_space.components_category_associated():
            has_component = True

        tmp_space.remove()

        return not has_component

    return _predicate


def face_area(max_area: Optional[float] = None, min_area: Optional[float] = None) -> Predicate:
    """
    Predicate factory
    Returns a predicate indicating if the edge face area is superior to a specified minimum
    and inferior to a specified maximum
    :param max_area
    :param min_area
    """
    def _predicate(edge: 'Edge', _: 'Space') -> bool:
        if not edge.face:
            return False
        _max_area = max_area or math.inf
        _min_area = min_area or 0
        return _max_area >= edge.face.area >= _min_area

    return _predicate


# Catalog Selectors

SELECTORS = {
    "space_boundary": Selector(space_boundary),

    "polish": Selector(
        space_boundary,
        [
            pair(is_mutable),
            face_proportion(0.3),
            face_without_component(),
            is_not(pair(corner_stone))

        ]
    ),

    "seed_component_boundary": Selector(seed_component_boundary),

    "previous_angle_salient": Selector(
        space_boundary,
        [
            is_not(adjacent_to_space("duct")),
            edge_angle(180.0, 360.0, previous=True),
            space_aligned_edges_length(min_length=20.0),
            previous_has(space_aligned_edges_length(min_length=20.0))
        ]
    ),

    "next_angle_salient": Selector(
        space_boundary,
        [
            is_not(adjacent_to_space("duct")),
            edge_angle(180.0, 360.0),
            space_aligned_edges_length(min_length=20.0),
            next_has(space_aligned_edges_length(min_length=20.0))
        ]
    ),

    "previous_angle_salient_non_ortho": Selector(
        space_boundary,
        [
            is_not(adjacent_to_space("duct")),
            edge_angle(180.0, 360.0, previous=True),
            is_not(edge_angle(270, 270, previous=True)),
            space_aligned_edges_length(min_length=150.0),
            previous_has(space_aligned_edges_length(min_length=150.0))
        ]
    ),

    "next_angle_salient_non_ortho": Selector(
        space_boundary,
        [
            is_not(adjacent_to_space("duct")),
            edge_angle(180.0, 360.0),
            is_not(edge_angle(270, 270)),
            space_aligned_edges_length(min_length=150.0),
            next_has(space_aligned_edges_length(min_length=150.0))
        ]
    ),

    "next_angle_convex": Selector(
        space_boundary,
        [
            edge_angle(90.0, 180.0),
            is_not(adjacent_to_space("duct")),
            space_aligned_edges_length(min_length=150.0),
            next_has(space_aligned_edges_length(min_length=150.0))
        ]
    ),

    "previous_angle_convex": Selector(
        space_boundary,
        [
            edge_angle(90.0, 180.0, previous=True),
            space_aligned_edges_length(min_length=150.0),
            previous_has(space_aligned_edges_length(min_length=150.0)),
            is_not(adjacent_to_space("duct"))
        ]
    ),

    "previous_angle_salient_ortho": Selector(
        space_boundary,
        [
            edge_angle(270.0, 270.0, previous=True, on_boundary=True),
            is_not(adjacent_to_space("duct"))
        ]
    ),
    "next_angle_salient_ortho": Selector(
        space_boundary,
        [
            edge_angle(270.0, 270.0, on_boundary=True),
            is_not(adjacent_to_space("duct"))
        ]
    ),

    "close_to_window": Selector(
        boundary_faces,
        [
            not_space_boundary,
            close_to_linear('window', 'doorWindow', min_distance=150.0)
        ]
    ),

    "close_to_front_door": Selector(
        boundary_faces,
        [
            not_space_boundary,
            close_to_linear('frontDoor', min_distance=80.0)
        ]
    ),

    "cuts_linear": Selector(cuts_linear, []),

    "between_windows": Selector(
        space_boundary,
        [
            touches_linear('window', 'doorWindow', position='between')
        ]
    ),

    "between_edges_between_windows": Selector(
        space_boundary,
        [
            previous_has(touches_linear('window', 'doorWindow', position='after')),
            next_has(touches_linear('window', 'doorWindow'))
        ]
    ),

    "front_door": Selector(space_boundary, [touches_linear("frontDoor", position="on")]),

    "before_front_door": Selector(space_boundary, [touches_linear("frontDoor"),
                                                   edge_length(min_length=10)]),

    "after_front_door": Selector(space_boundary, [touches_linear("frontDoor", position="after"),
                                                  edge_length(min_length=10)]),

    "starting_step": Selector(space_boundary, [touches_linear("startingStep", position="on")]),

    "before_starting_step": Selector(space_boundary, [touches_linear("startingStep"),
                                                      edge_length(min_length=10)]),

    "after_starting_step": Selector(space_boundary, [touches_linear("startingStep",
                                                                    position="after"),
                                                     edge_length(min_length=10)]),

    "before_window": Selector(
        space_boundary,
        [
            touches_linear("window", "doorWindow"),
            edge_length(min_length=50)
        ]
    ),

    "after_window": Selector(
        space_boundary,
        [
            touches_linear("window", "doorWindow", position="after"),
            edge_length(min_length=50)
        ]
    ),

    "aligned_edges": Selector(
        space_boundary,
        [
            edge_angle(180.0, 180.0),
            is_not(touches_linear('window', 'doorWindow', 'frontDoor')),
            is_not(is_linear('window', 'doorWindow', 'frontDoor'))
        ]
    ),

    "all_aligned_edges": Selector(
        boundary_faces_fixed,
        [
            edge_angle(180 - 15, 180 + 15),
            is_not(touches_linear('window', 'doorWindow', 'frontDoor')),
            is_not(is_linear('window', 'doorWindow', 'frontDoor')),
            is_not(adjacent_to_space('duct')),
            next_has(is_not(adjacent_to_space('duct')))
        ]
    ),

    "adjacent_to_load_bearing_wall": Selector(
        space_boundary,
        [
            adjacent_to_space("loadBearingWall")
        ]
    ),

    "edge_min_120": Selector(
        space_boundary,
        [
            edge_length(min_length=120.0)
        ]
    ),

    "edge_min_150": Selector(
        space_boundary,
        [
            edge_length(min_length=150.0)
        ]
    ),

    "edge_min_300": Selector(
        space_boundary,
        [
            edge_length(min_length=300.0)
        ]
    ),

    "edge_min_500": Selector(
        space_boundary,
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

    "farthest_couple_middle_space_area_min_50000": Selector(
        farthest_edges_barycenter(0.5),
        [
            space_area(min_area=50000)
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

    "homogeneous": Selector(homogeneous, name='homogeneous'),

    "best_aspect_ratio": Selector(best_aspect_ratio, name='homogeneous_aspect_ratio'),

    "improved_aspect_ratio":  Selector(improved_aspect_ratio, name='improved_aspect_ratio'),

    "fuse_very_small_cell_mutable": Selector(
        boundary_unique_longest,
        [
            space_area(max_area=10000)
        ]
    ),

    "fuse_small_cell_without_components": Selector(
        boundary_unique_longest,
        [
            space_area(max_area=15000),
            cell_with_component()
        ]
    ),

    "other_seed_space": Selector(
        other_seed_space_edge,
        [
            adjacent_to_other_space,
            is_not(pair(corner_stone))
        ]
    ),

    "corner_stone": Selector(
        space_boundary,
        [
            pair(corner_stone)
        ]
    ),
    "single_edge": Selector(boundary_unique),

    "swap_aligned": Selector(

        other_seed_space_edge,
        [
            adjacent_to_other_space,
            is_not_aligned,
            is_not(pair(corner_stone))
        ]
    ),

    "duct_edge_min_10": Selector(
        space_boundary,
        [
            adjacent_to_space("duct"),
            edge_length(min_length=10)
        ]
    ),

    "duct_edge_min_120": Selector(space_boundary, [adjacent_to_space("duct"),
                                                   edge_length(min_length=120)]),

    "duct_edge_min_160": Selector(space_boundary, [adjacent_to_space("duct"),
                                                   edge_length(min_length=160)]),

    "duct_edge_min_80": Selector(space_boundary, [adjacent_to_space("duct"),
                                                  edge_length(min_length=80)]),

    "duct_edge_not_touching_wall": Selector(space_boundary,
                                            [adjacent_to_space("duct"),
                                             space_next_has(adjacent_to_space("duct")),
                                             space_previous_has(adjacent_to_space("duct")),
                                             edge_length(min_length=20)]),

    "corner_duct_first_edge": Selector(space_boundary,
                                       [adjacent_to_space("duct"),
                                        space_next_has(adjacent_to_space("duct")),
                                        space_next_has(
                                            space_next_has(adjacent_to_space("exterior"))),
                                        space_previous_has(adjacent_to_space("exterior")),
                                        space_aligned_edges_length(min_length=80)]),

    "corner_duct_second_edge": Selector(space_boundary,
                                        [adjacent_to_space("duct"),
                                         space_next_has(adjacent_to_space("exterior")),
                                         space_previous_has(adjacent_to_space("duct")),
                                         space_previous_has(
                                             space_previous_has(adjacent_to_space("exterior"))),
                                         space_aligned_edges_length(min_length=80)]),

    "corner_face": Selector(boundary_faces, [corner_face]),

    "one_edge_per_rectangular_duct": Selector(one_edge_adjacent_to_rectangular_duct),

    "window_doorWindow": Selector(space_boundary, [touches_linear("window", "doorWindow",
                                                                  position="on")]),

    "close_to_wall": Selector(boundary_faces, [close_to_apartment_boundary(90, 80)]),

    "h_edge": Selector(boundary_faces, [h_edge, edge_length(max_length=200)]),

    "previous_concave_non_ortho": Selector(space_boundary, [
        edge_angle(180.0 + MIN_ANGLE, 270.0 - MIN_ANGLE, on_boundary=True, previous=True),
        space_aligned_edges_length(min_length=50.0),
        previous_has(space_aligned_edges_length(min_length=50.0))
    ]),

    "next_concave_non_ortho": Selector(space_boundary, [
        edge_angle(180.0 + MIN_ANGLE, 270.0 - MIN_ANGLE, on_boundary=True),
        space_aligned_edges_length(min_length=50.0),
        next_has(space_aligned_edges_length(min_length=50.0))
    ]),

    "previous_convex_non_ortho": Selector(space_boundary, [
        edge_angle(90.0 + MIN_ANGLE, 180.0 - MIN_ANGLE, on_boundary=True, previous=True),
        space_aligned_edges_length(min_length=50.0),
        previous_has(space_aligned_edges_length(min_length=50.0))
    ]),

    "next_convex_non_ortho": Selector(space_boundary, [
        edge_angle(90.0 + MIN_ANGLE, 180.0 - MIN_ANGLE, on_boundary=True),
        space_aligned_edges_length(min_length=50.0),
        next_has(space_aligned_edges_length(min_length=50.0))
    ]),

    "adjacent_to_empty_space": Selector(space_boundary, [adjacent_to_space("empty")]),

    "wrong_direction": Selector(touching_space_boundary, [wrong_direction]),

    "add_aligned": Selector(
        space_boundary,
        [
            adjacent_to_empty_space,
            is_after_corner,
        ]
    ),

    "add_aligned_vertical": Selector(
        vertical_edge,
        [
            adjacent_to_empty_space,
            is_after_corner,
        ]
    ),

    "not_aligned_edges": Selector(

        not_aligned_edges,
        [
            has_space_pair(),
        ]
    ),

    "face_min_area": Selector(boundary_faces_biggest, [face_area(1000)]),

    "bedroom_small_faces": Selector(specific_category("bedroom"), [face_area(max_area=15000),
                                                                   is_not(only_face),
                                                                   is_not(corner_stone)]),

    "bedroom_small_faces_pair": Selector(specific_category("bedroom"),
                                         [pair(face_area(max_area=15000)),
                                          pair(is_not(only_face)),
                                          pair(is_not(corner_stone))]),

    "is_mutable": Selector(space_external_boundary, [is_mutable, pair(is_mutable),
                                                     is_not(has_needed_linear),
                                                     is_not(only_face),
                                                     is_not(only_adjacent_to_immutable),
                                                     is_not(corner_stone)]),

    "plan_boundary_no_linear": Selector(space_external_boundary,
                                        [adjacent_to_external_space,
                                         edge_length(min_length=60),
                                         is_not(touches_linear(position='on'))])
}

SELECTOR_FACTORIES = {
    "oriented_edges": SelectorFactory(oriented_edges, factorize(adjacent_empty_space)),
    "edges_length": SelectorFactory(lambda: space_boundary, [edge_length]),
    "min_depth": SelectorFactory(min_depth),
    "tight_lines": SelectorFactory(tight_lines),
    "category": SelectorFactory(specific_category)
}
