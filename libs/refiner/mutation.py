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

MIN_ADJACENCY_EDGE_LENGTH = None  # this is not working, default to None


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
        logging.debug("Refiner: Mutation: no space selected %s", ind)
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
        logging.debug("Refiner: Mutation: No mutation occurred")

    if not ind.modified_spaces:
        logging.debug("Refiner: Mutation: No mutation occurred")

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
        logging.debug("Refiner: Mutation: No mutable edges %s", space)
        return []
    edge = random.choice(mutable_edges)

    max_angle = 25.0  # the angle used to determine if two edges are aligned

    # retrieve all the aligned edges pairs and their space
    spaces_and_edges = map(lambda _e: (space.plan.get_space_of_edge(_e.pair), _e.pair),
                           space.aligned_siblings(edge, max_angle))
    # group edges by spaces
    edges_by_spaces = defaultdict(set)
    for other, _edge in spaces_and_edges:
        if other is None or not other.mutable:
            continue
        edges_by_spaces[other].add(_edge)

    modified_spaces = [space]
    for other in edges_by_spaces:

        # remove all edges that have a needed linear for the space
        for e in edges_by_spaces[other].copy():
            if _has_needed_linear(e, other, space):
                edges_by_spaces[other].remove(e)

        # remove all edges that are the only ones whose face is adjacent to a needed space
        for e in edges_by_spaces[other].copy():
            if _adjacent_to_needed_space(e, other, edges_by_spaces[other]):
                edges_by_spaces[other].remove(e)

        if not edges_by_spaces[other]:
            logging.debug("Refiner: Mutation: No more edges for space %s", other)
            continue

        # check if we are breaking the space if we remove the faces
        # Note : we must check after removing the edges linked to a needed linear or space
        faces = set(e.face for e in edges_by_spaces[other])
        if other.corner_stone(*faces, min_adjacency_length=MIN_ADJACENCY_EDGE_LENGTH):
            continue

        logging.debug("Mutation: Adding aligned edges %s to space %s from % s",
                      edges_by_spaces[other], space, other)

        faces_id = [f.id for f in faces]
        space.add_face_id(*faces_id)
        other.remove_face_id(*faces_id)
        # set the reference edges of each spaces
        other.set_edges()
        modified_spaces.append(other)

    # Note : we must set the edges of the space once all the faces have been removed
    space.set_edges()

    return modified_spaces if len(modified_spaces) > 1 else []


def add_face(space: 'Space') -> List['Space']:
    """
    Adds the face of the edge pair to the space
    :param space:
    :return:
    """
    mutable_edges = _mutable_edges(space)
    if not mutable_edges:
        logging.debug("Refiner: Mutation: No mutable edges %s", space)
        return []
    edge = random.choice(mutable_edges)

    plan = space.plan
    face = edge.pair.face
    other_space = plan.get_space_of_edge(edge.pair)

    # check we can actually remove the face
    if not other_space or not other_space.mutable:
        logging.debug("Refiner: Mutation: No other space %s", space)
        return []

    if other_space.number_of_faces == 1:
        return []

    if _has_needed_linear(edge.pair, other_space, space):
        return []

    if _adjacent_to_needed_space(edge.pair, other_space, [edge.pair]):
        return []

    if other_space.corner_stone(face, min_adjacency_length=MIN_ADJACENCY_EDGE_LENGTH):
        return []

    logging.debug("Refiner: Mutation: Adding face %s to space %s and removing it from space %s",
                  face, space, other_space)

    other_space.remove_face(face)
    space.add_face(face)
    return [space, other_space]


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
        logging.debug("Refiner: Mutation: No mutable edges %s", space)
        return []
    edge = random.choice(mutable_edges)

    plan = space.plan

    max_angle = 25.0  # the angle used to determine if two edges are aligned

    # check that we are not removing any important faces
    edges = space.aligned_siblings(edge, max_angle)

    logging.debug("refiner: Mutation: Removing aligned edges : %s from space %s", edges, space)

    faces_by_spaces = defaultdict(set)
    removed_faces = set()

    for edge in edges[:]:
        # we must check also that the list of faces does not remove any needed linears or adjacent
        # spaces. We remove all the needed faces.
        if _adjacent_to_needed_space(edge, space, edges):
            continue
        other = plan.get_space_of_edge(edge.pair)
        # we need to know what is the other space to check if the linear must be kept
        if _has_needed_linear(edge, space, other):
            continue
        if other is None or not other.mutable:
            continue
        removed_faces.add(edge.face)
        faces_by_spaces[other].add(edge.face)  # Note : the same face can be linked to 2+ spaces

    # check if we have at least one face to swap
    if not faces_by_spaces or space.corner_stone(*removed_faces):
        return []

    modified_spaces = [space]
    for other in faces_by_spaces:
        # Note : a face can be adjacent to multiple other spaces. We must check that the face
        #        has not already been given to another space. This is why we only retain
        #        the intersection of the identified faces set with the remaining faces of the space
        faces_id = set(map(lambda f: f.id,
                           faces_by_spaces[other])).intersection(space.faces_id)
        if not faces_id:
            continue
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
        logging.debug("Refiner: Mutation: No mutable edges %s", space)
        return []
    edge = random.choice(mutable_edges)

    plan = space.plan
    face = edge.face
    other_space = plan.get_space_of_edge(edge.pair)
    if not other_space:
        logging.debug("Refiner: Mutation: No other space %s", space)
        return []

    if _has_needed_linear(edge, space, other_space):
        return []

    if _adjacent_to_needed_space(edge, space, [edge]):
        return []

    if space.corner_stone(face, min_adjacency_length=MIN_ADJACENCY_EDGE_LENGTH):
        return []

    logging.debug("Refiner: Mutation: Removing face %s of space %s and adding it to space %s", face,
                  space, other_space)

    space.remove_face(face)
    other_space.add_face(face)

    return [other_space, space]


"""
UTILITY Functions
"""


def _has_needed_linear(edge: 'Edge', space: 'Space', other: Optional['Space'] = None) -> bool:
    """
    Returns True if the edge face has an immutable component
    TODO : number of linears (example : living or bedroom if small windows)
    :param edge:
    :param space:
    :param other:
    :return:
    """
    # some spaces cannot be given certain linears
    forbidden_linears = {
        SPACE_CATEGORIES["circulation"]: {LINEAR_CATEGORIES["doorWindow"],
                                          LINEAR_CATEGORIES["window"]},
        SPACE_CATEGORIES["toilet"]: {LINEAR_CATEGORIES["doorWindow"], LINEAR_CATEGORIES["window"]},
        SPACE_CATEGORIES["bathroom"]: {LINEAR_CATEGORIES["doorWindow"]}
    }

    face = edge.face

    front_door = next(space.plan.get_linears("frontDoor"))
    # if there is no front door we make the assumption we are on a level of a
    # multiple level apartment. NOTE : this will not work with triplex as we have no way
    # of knowing which starting step correspond to the stairs going to the entrance level
    if not front_door:
        front_door = next(space.plan.get_linears("startingStep"))

    if space.category is SPACE_CATEGORIES["entrance"]:
        needed_linears = space.category.needed_linears
        # Wen a plan has no entrance, we make the assumption that if a space has the frontDoor
        # then it is considered as an entrance and should therefore keep the frontDoor linear
    elif space.has_linear(front_door):
        needed_linears = set(space.category.needed_linears) | {LINEAR_CATEGORIES["frontDoor"]}
    else:
        needed_linears = space.category.needed_linears

    if not space.category.needed_linears or not face:
        return False

    for _edge in edge.face.edges:
        # only check boundary edges
        if space.is_internal(_edge):
            continue
        linear = space.plan.get_linear_from_edge(_edge)
        # keep only one needed linear
        if linear and linear.category in needed_linears:
            # if the linear belongs to the forbidden linears of the other space
            # consider that the current space needs this linear
            if other and linear.category in forbidden_linears.get(other.category, []):
                return True
            for other_edge in space.exterior_edges:
                other_linear = space.plan.get_linear_from_edge(other_edge)
                if (other_linear and other_linear is not linear
                        and other_linear.category is linear.category):
                    break
            else:
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
    for _edge in space.exterior_edges:
        if _edge.face is face or _edge in removed_edges:
            continue
        _other = space.plan.get_space_of_edge(_edge.pair)
        if not _other:
            continue
        if _other.category is other.category:
            return False

    return True


__all__ = ['add_face', 'add_aligned_faces', 'remove_face', 'remove_aligned_faces']
