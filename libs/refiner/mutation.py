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
from collections import defaultdict

from libs.operators.selector import SELECTORS
from libs.plan.category import LINEAR_CATEGORIES, SPACE_CATEGORIES

if TYPE_CHECKING:
    from libs.plan.plan import Space
    from libs.mesh.mesh import Edge
    from libs.refiner.core import Individual

MIN_ADJACENCY_EDGE_LENGTH = 80.0


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
            logging.debug("Ind: %s, Space mutated %s - %s", ind, space, mutation_pbx[0])

            modified_spaces = mutation_pbx[0](space)
            ind.modified_spaces |= {s.id for s in modified_spaces}
            break
    else:
        logging.info("Refiner: No mutation occurred")

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


def _mutable_edges(space: 'Space') -> ['Edge']:
    """
    Returns the list of all the edges from the boundary of the space adjacent to another mutable
    space
    :param space:
    :return:
    """
    return list(SELECTORS["mutable_edges"].yield_from(space))


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
    weights = [1.0/len(list(space.aligned_siblings(e, 25.0))) for e in mutable_edges]
    return random.choices(mutable_edges, weights=weights, k=1)[0]
    # return random.choice(mutable_edges)


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

    weights = [1.0/len(list(space.aligned_siblings(e, 25.0))) for e in mutable_edges]
    return random.choices(mutable_edges, weights=weights, k=1)[0]
    # return random.choice(mutable_edges)

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

    mutable_edges = _mutable_edges(space)
    if not mutable_edges:
        return []
    edge = random.choice(mutable_edges)

    max_angle = 25.0  # the angle used to determine if two edges are aligned

    # retrieve all the aligned edges
    spaces_and_edges = map(lambda _e: (space.plan.get_space_of_edge(_e.pair), _e.pair),
                           space.aligned_siblings(edge, max_angle))
    # group edges by spaces
    edges_by_spaces = defaultdict(set)
    for other, _edge in spaces_and_edges:
        if other is None:
            continue
        edges_by_spaces[other].add(_edge)

    modified_spaces = [space]
    for other in edges_by_spaces:
        if not other.mutable:
            continue

        # remove all edges that have a needed linear for the space
        for e in edges_by_spaces[other].copy():
            if _has_needed_linear(e, other):
                edges_by_spaces[other].remove(e)

        # remove all edges that are the only ones whose face is adjacent to a needed space
        for e in edges_by_spaces[other].copy():
            if _adjacent_to_needed_space(e, other, edges_by_spaces[other]):
                edges_by_spaces[other].remove(e)

        if not edges_by_spaces[other]:
            continue

        # check if we are breaking the space if we remove the faces
        # Note : we must check after removing the edges linked to a needed linear or space
        faces = set(e.face for e in edges_by_spaces[other])
        if other.corner_stone(*faces, min_adjacency_length=MIN_ADJACENCY_EDGE_LENGTH):
            continue

        faces_id = [f.id for f in faces]
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
    mutable_edges = _mutable_edges(space)
    if not mutable_edges:
        return []
    edge = random.choice(mutable_edges)

    plan = space.plan
    face = edge.pair.face
    other_space = plan.get_space_of_edge(edge.pair)

    # check we can actually remove the face
    if not other_space or not other_space.mutable:
        return []

    if other_space.number_of_faces == 1:
        return []

    if _has_needed_linear(edge.pair, other_space):
        return []

    if _adjacent_to_needed_space(edge.pair, other_space, [edge.pair]):
        return []

    if other_space.corner_stone(face, min_adjacency_length=MIN_ADJACENCY_EDGE_LENGTH):
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

    mutable_edges = _mutable_edges(space)
    if not mutable_edges:
        return []
    edge = random.choice(mutable_edges)

    plan = space.plan

    max_angle = 25.0  # the angle used to determine if two edges are aligned

    # check that we are not removing any important faces
    edges = space.aligned_siblings(edge, max_angle)

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

    logging.debug("Removing aligned edges : %s from space %s", edges, space)

    modified_spaces = [space]
    faces_by_spaces = defaultdict(set)
    faces = set(list(e.face for e in edges))

    for edge in edges:
        other = plan.get_space_of_edge(edge.pair)
        if other is None or not other.mutable and edge.face in faces:
            faces.remove(edge.face)
            continue
        # NOTE : we must add back a face that was removed because one of its edge was not adjacent
        # to a mutable space but has another edge that is adjacent
        if edge.face not in faces:
            faces.add(edge.face)
        faces_by_spaces[other].add(edge.face)

    if not faces or space.corner_stone(*faces, min_adjacency_length=MIN_ADJACENCY_EDGE_LENGTH):
        return []

    for other in faces_by_spaces:
        if not other.mutable:
            continue
        # Note : a face can be adjacent to multiple other spaces. We must check that the face
        #        has not already been given to another space. This is why we only retain
        #        the intersection of the identified faces set with the remaining faces of the space
        faces_id = set(map(lambda f: f.id, faces_by_spaces[other])) & set(space._faces_id)
        other.add_face_id(*faces_id)
        space.remove_face_id(*faces_id)
        # set the reference edges of each spaces
        other.set_edges()
        modified_spaces.append(other)

    # Note : we must set the edges of the space once all the faces have been removed
    space.set_edges()

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
    if space.number_of_faces <= 1:
        return []

    mutable_edges = _mutable_edges(space)
    if not mutable_edges:
        return []
    edge = random.choice(mutable_edges)

    plan = space.plan
    face = edge.face
    other_space = plan.get_space_of_edge(edge.pair)
    if not other_space:
        return []

    if _has_needed_linear(edge, space):
        return []

    if _adjacent_to_needed_space(edge, space, [edge]):
        return []

    if space.corner_stone(face, min_adjacency_length=MIN_ADJACENCY_EDGE_LENGTH):
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
    TODO : number of linears (example : living or bedroom if small windows)
    :param edge:
    :param space:
    :return:
    """
    other_entrances = (SPACE_CATEGORIES["living"], SPACE_CATEGORIES["livingKitchen"])

    face = edge.face
    has_entrance = True

    if SPACE_CATEGORIES["entrance"] not in (s.category for s in space.plan.mutable_spaces()):
        has_entrance = False

    if ((not space.category.needed_linears or not face)
            and (has_entrance or space.category not in other_entrances)):
        return False

    needed_linears = list(space.category.needed_linears)
    if not has_entrance and space.category in other_entrances:
        needed_linears.append(LINEAR_CATEGORIES["frontDoor"])

    for _edge in edge.face.edges:
        linear = space.plan.get_linear_from_edge(_edge)
        if linear and linear.category in needed_linears:
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
