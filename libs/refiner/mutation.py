# coding=utf-8
"""
Genetic Algorithm Mutation module
A mutation is a function that takes an individual as input and modifies it in place

We use 4 different mutations for a given space and edge (the edge belonging to the space) :
    • add the single face of the edge pair to the space
    • add all aligned faces with the edge pair face to the space
    • remove the single face of the edge
    • remove all aligned faces with the edge face

The idea is to select spaces that have a higher fitness value with a higher probability.
If the space is larger than needed pick with a higher probability a mutation that will reduce its
size, or do the opposite.

Aligned edges mutation will transform the space towards a rectangular shape.
Single face mutation will create more diversity.

Note : a mutation should set the modified spaces flag of the individual

"""
import random
import math
import logging
import enum
from typing import TYPE_CHECKING, Optional, Callable, Tuple, List, Container, Dict

from libs.operators.selector import SELECTORS

if TYPE_CHECKING:
    from libs.plan.plan import Space
    from libs.mesh.mesh import Edge
    from libs.refiner.core import Individual


class Case(enum.Enum):
    """
    Enum of different spaces cases
    """
    SMALL = "small"
    BIG = "big"
    DEFAULT = "default"


MutationType = Callable[['Space'], List['Space']]
MutationProbability = float
MutationTuple = Tuple[Tuple[MutationType, Dict[Case, MutationProbability]], ...]


def composite(mutations_pbx: MutationTuple, ind: 'Individual') -> 'Individual':
    """
    A composite mutation
    Mutations
    :param ind:
    :param mutations_pbx:
    :return:
    """
    space = _random_space(ind)
    if not space:
        return ind

    item = ind.fitness.cache.get("space_to_item", None)[space.id]

    space_size = Case.DEFAULT
    if item:  # corridor space has no item in spec
        if space.cached_area() < item.min_size.area:
            space_size = Case.SMALL
        elif space.cached_area() > item.max_size.area:
            space_size = Case.BIG

    dice = random.random()
    accumulated_pb = 0
    for mutation_pbx in mutations_pbx:
        pb = mutation_pbx[1][space_size]
        accumulated_pb += pb
        if dice <= accumulated_pb:
            modified_spaces = mutation_pbx[0](space)
            ind.modified_spaces |= {s.id for s in modified_spaces}
            break
    else:
        logging.debug("Refiner: No mutation occurred")

    return ind


def mutate_aligned(space: 'Space') -> 'Individual':
    """
    Mutates the plan.
    1. select a random space
    2. select a random mutable edge
    3. mutate
    :param space:
    :return: a single element tuple containing the mutated individual
    """
    ind = space.plan
    modified_spaces = add_aligned_faces(space)
    ind.modified_spaces |= {s.id for s in modified_spaces}

    return ind


def mutate_simple(space: 'Space') -> 'Individual':
    """
    Mutates the plan.
    1. select a random space
    2. select a random mutable edge
    3. remove the edge face from the space and gives it to the pair space
    :param: space:
    :return: a single element tuple containing the mutated individual
    """
    ind = space.plan
    modified_spaces = remove_face(space)
    if __debug__ and len(modified_spaces) > 2:
        logging.warning("Refiner: Mutation: A space was split !! %s", modified_spaces[2])
    if __debug__ and space.number_of_faces == 0:
        logging.warning("Refiner: A space has no face left !!")
    ind.modified_spaces |= {s.id for s in modified_spaces}
    return ind


"""
PICKERS operators
"""


def _random_space(ind: 'Individual') -> Optional['Space']:
    """
    Returns a random mutable space of the plan
    :param ind:
    :return:
    """
    mutable_spaces = list(ind.mutable_spaces())
    if not mutable_spaces:
        logging.warning("Mutation: Random space, no mutable space was found !!")
        return None
    spaces_fitnesses = (math.fabs(ind.fitness.sp_wvalue[s.id]) for s in mutable_spaces)
    # we randomly select a space with higher weights for the spaces with lower fitnesses
    # Note : weighted fitnesses are negative hence the absolute value
    return random.choices(mutable_spaces, weights=spaces_fitnesses)[0]


def _random_removable_edge(space: 'Space') -> Optional['Edge']:
    """
    Returns a random edge of the space
    :param space:
    :return:
    """
    mutable_edges = list(SELECTORS["can_be_removed"].yield_from(space))
    if not mutable_edges:
        logging.debug("Mutation: Random edge, no edge was found !! %s", space)
        return None
    weights = [1.0/len(list(e.aligned_siblings)) for e in mutable_edges]
    return random.choices(mutable_edges, weights=weights, k=1)[0]


def _random_addable_edge(space: 'Space') -> Optional['Edge']:
    """
    Returns a random edge of the space
    :param space:
    :return:
    """
    mutable_edges = list(SELECTORS["can_be_added"].yield_from(space))
    if not mutable_edges:
        logging.debug("Mutation: Random edge, no edge was found !! %s", space)
        return None

    weights = [1.0/len(list(e.aligned_siblings)) for e in mutable_edges]
    return random.choices(mutable_edges, weights=weights, k=1)[0]


"""
MUTATIONS operators
"""


def add_aligned_faces(space: 'Space') -> List['Space']:
    """
    Adds to space all the faces of the aligned edges
    • checks if the edge is just after a corner
    • gather all the next aligned edges
    • for each edge add the corresponding faces
    :return:
    """
    edge = _random_addable_edge(space)
    if not edge:
        return []

    max_angle = 25.0  # the angle used to determine if two edges are aligned

    # retrieve all the aligned edges
    spaces_and_edges = map(lambda e: (space.plan.get_space_of_edge(e.pair), e.pair),
                           space.aligned_siblings(edge, max_angle))
    # group edges by spaces
    edges_by_spaces = {}
    for other, _edge in spaces_and_edges:
        if other is None:
            continue
        if other not in edges_by_spaces:
            edges_by_spaces[other] = {_edge}
        else:
            edges_by_spaces[other].add(_edge)

    modified_spaces = [space]
    for other in edges_by_spaces:
        if not other.mutable:
            continue
        # check if we are breaking the space if we remove the faces
        if other.corner_stone(*list(e.face for e in edges_by_spaces[other])):
            continue

        # remove all edges that have a needed linear for the space
        for e in edges_by_spaces[other].copy():
            if _has_needed_linear(e, other):
                edges_by_spaces[other].remove(e)

        # remove all edges that are the only ones whose face is adjacent to a needed space
        for e in edges_by_spaces[other].copy():
            if _adjacent_to_needed_space(e, other, edges_by_spaces[other]):
                edges_by_spaces[other].remove(e)

        faces_id = list(set(map(lambda e: e.face.id, edges_by_spaces[other])))
        space.add_face_id(*faces_id)
        other.remove_face_id(*faces_id)
        # set the reference edges of each spaces
        space.set_edges()
        other.set_edges()
        modified_spaces.append(other)

    return modified_spaces if len(modified_spaces) > 1 else []


def add_face(space: 'Space') -> List['Space']:
    """
    Adds the face of the edge pair to the space
    :param space:
    :return:
    """
    edge = _random_addable_edge(space)
    if not edge:
        return []

    plan = space.plan
    face = edge.pair.face
    other_space = plan.get_space_of_edge(edge.pair)
    if not other_space:
        return []

    logging.debug("Mutation: Adding face %s to space %s and removing it from space %s", face, space,
                  other_space)

    created_spaces = other_space.remove_face(face)
    space.add_face(face)
    return [space] + list(created_spaces)


def remove_aligned_faces(space: 'Space') -> List['Space']:
    """
    Removes all aligned faces with the edge from the space and add them to their
    corresponding pair spaces
    :param space:
    :return:
    """
    if space.number_of_faces <= 1:
        return []

    edge = _random_removable_edge(space)
    if not edge:
        return []

    plan = space.plan

    max_angle = 25.0  # the angle used to determine if two edges are aligned

    # check that we are not removing any important faces
    edges = space.aligned_siblings(edge, max_angle)
    if space.corner_stone(*list(e.face for e in edges)):
        return []

    # we must check also that the list of faces does not remove any needed linears or adjacent
    # spaces. We remove all the needed faces.
    for edge in edges[:]:
        if _has_needed_linear(edge, space):
            edges.remove(edge)

    for edge in edges[:]:
        if _adjacent_to_needed_space(edge, space, edges):
            edges.remove(edge)

    if not edges:
        return []

    modified_spaces = [space]

    faces_by_spaces = {}

    for edge in edges:
        other = plan.get_space_of_edge(edge.pair)
        if other is None:
            continue
        if other not in faces_by_spaces:
            faces_by_spaces[other] = {edge.face}
        else:
            faces_by_spaces[other].add(edge.face)

    for other in faces_by_spaces:
        if not other.mutable:
            continue
        faces_id = list(map(lambda f: f.id, faces_by_spaces[other]))
        other.add_face_id(*faces_id)
        space.remove_face_id(*faces_id)
        # set the reference edges of each spaces
        space.set_edges()
        other.set_edges()
        modified_spaces.append(other)

    return modified_spaces


def remove_face(space: 'Space') -> List['Space']:
    """
    remove the edge face: by removing to the specified space and adding it to the other space
    Eventually merge the second space with all other specified spaces
    Returns a list of space :
    • the merged space
    • the newly created spaces by the removal of the face
    The mutation is reversible : swap_face(edge.pair, swap_face(edge, spaces))
    :param space:
    :return: a list of space
    """
    edge = _random_removable_edge(space)
    if not edge:
        return []

    plan = space.plan
    face = edge.face
    other_space = plan.get_space_of_edge(edge.pair)
    if not other_space:
        return []

    logging.debug("Mutation: Removing face %s of space %s and adding it to space %s", face, space,
                  other_space)

    created_spaces = space.remove_face(face)
    other_space.add_face(face)

    return [other_space] + list(created_spaces)


"""
UTILITY Functions
"""


def _has_needed_linear(edge: 'Edge', space: 'Space') -> bool:
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


def _adjacent_to_needed_space(edge: 'Edge', space: 'Space',
                              removed_edges: Container['Edge']) -> bool:
    """
    Returns True if the edge face has an immutable component
    :param edge:
    :param space:
    :param removed_edges: other edges removed from the space
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
        if _edge.face is face or _edge in removed_edges:
            continue
        _other = space.plan.get_space_of_edge(_edge.pair)
        if not _other:
            continue
        if _other.category is other.category:
            return False

    return True


__all__ = ['mutate_simple', 'mutate_aligned']
