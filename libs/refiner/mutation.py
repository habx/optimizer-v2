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
import logging
from typing import TYPE_CHECKING, Optional, Callable, Tuple, List

from libs.operators.selector import SELECTORS

if TYPE_CHECKING:
    from libs.plan.plan import Plan, Space
    from libs.mesh.mesh import Edge
    from libs.refiner.core import Individual

MutationType = Callable[['Individual'], 'Individual']
MutationProbability = float


def compose(mutations: List[Tuple[MutationType, MutationProbability]],
            ind: 'Individual') -> 'Individual':
    """
    Creates a mutation composed of different mutations
    :param mutations: a list of tuples of mutation and associated probability
                      ordered from rarest to most frequent.
                      ex: [(mutate_simple, 0.1), (mutate_aligned, 1.0)]
                          mutate_aligned will occur 90% of the time
    :param ind:
    :return: the mutated individual
    """
    # Note : Make sure the mutations are ordered from rarest to more frequent
    # per convention : only one mutation can occur so it is important to start from the
    # rarest
    dice = random.random()
    for mutation, pb in mutations:
        if dice <= pb:
            ind = mutation(ind)
            break
    else:
        logging.debug("Refiner: No mutation occurred")

    return ind


def mutate_aligned(ind: 'Individual') -> 'Individual':
    """
    Mutates the plan.
    1. select a random space
    2. select a random mutable edge
    3. mutate
    TODO : we must check that we are not removing an essential edge of the space !!
    :param ind:
    :return: a single element tuple containing the mutated individual
    """
    space = _random_space(ind)
    edge = _random_mutable_edge(space)
    if edge:
        modified_spaces = add_aligned_face(edge, space)
        ind.modified_spaces |= {s.id for s in modified_spaces}

    return ind


def mutate_simple(ind: 'Individual') -> 'Individual':
    """
    Mutates the plan.
    1. select a random space
    2. select a random mutable edge
    3. remove the edge face from the space and gives it to the pair space
    :param ind:
    :return: a single element tuple containing the mutated individual
    """
    space = _random_space(ind)
    edge = _random_mutable_edge(space)
    if edge:
        modified_spaces = remove_face(edge, space)
        if __debug__ and len(modified_spaces) > 2:
            logging.warning("Refiner: Mutation: A space was split !! %s", modified_spaces[2])
        if __debug__ and space.number_of_faces == 0:
            logging.warning("Refiner: A space has no face left !!")
        ind.modified_spaces |= {s.id for s in modified_spaces}

    return ind


"""
PICKERS operators
"""


def _random_space(plan: 'Plan') -> Optional['Space']:
    """
    Returns a random mutable space of the plan
    :param plan:
    :return:
    """
    mutable_spaces = list(plan.mutable_spaces())
    if not mutable_spaces:
        logging.warning("Mutation: Random space, no mutable space was found !!")
        return None
    return random.choice(mutable_spaces)


def _random_mutable_edge(space: 'Space') -> Optional['Edge']:
    """
    Returns a random edge of the space
    :param space:
    :return:
    """
    mutable_edges = list(SELECTORS["is_mutable"].yield_from(space))
    if not mutable_edges:
        logging.debug("Mutation: Random edge, no edge was found !! %s", space)
        return None
    return random.choice(mutable_edges)


"""
MUTATIONS operators
"""


def add_aligned_face(edge: 'Edge', space: 'Space') -> List['Space']:
    """
    Adds to space all the faces of the aligned edges
    • checks if the edge is just after a corner
    • gather all the next aligned edges
    • for each edge add the corresponding faces
    :return:
    """
    assert space.has_edge(edge), "Mutation: The edge must belong to the first space"

    max_angle = 25.0  # the angle used to determine if two edges are aligned

    # retrieve all the aligned edges
    edges_and_spaces = map(lambda e: (space.plan.get_space_of_edge(e.pair), e.pair),
                           space.aligned_siblings(edge, max_angle))
    # group faces by spaces
    faces_by_spaces = {}
    for _space, _edge in edges_and_spaces:
        if _space is None:
            continue
        if _space not in faces_by_spaces:
            faces_by_spaces[_space] = {_edge.face}
        else:
            faces_by_spaces[_space].add(_edge.face)

    modified_spaces = [space]
    for _space in faces_by_spaces:
        if not _space.mutable:
            continue
        # check if we are breaking the space if we remove the faces
        if _space.corner_stone(*faces_by_spaces[_space]):
            continue
        faces_id = list(map(lambda f: f.id, faces_by_spaces[_space]))
        space.add_face_id(*faces_id)
        _space.remove_face_id(*faces_id)
        # set the reference edges of each spaces
        space.set_edges()
        _space.set_edges()
        modified_spaces.append(_space)

    return modified_spaces if len(modified_spaces) > 1 else []


def remove_face(edge: 'Edge', space: 'Space') -> List['Space']:
    """
    remove the edge face: by removing to the specified space and adding it to the other space
    Eventually merge the second space with all other specified spaces
    Returns a list of space :
    • the merged space
    • the newly created spaces by the removal of the face
    The mutation is reversible : swap_face(edge.pair, swap_face(edge, spaces))
    :param edge:
    :param space:
    :return: a list of space
    """
    assert space.has_edge(edge), "Mutation: The edge must belong to the space"

    plan = space.plan
    face = edge.face
    other_space = plan.get_space_of_edge(edge.pair)
    if not other_space:
        return []

    logging.debug("Mutation: Swapping face %s of space %s to space %s", face, space, other_space)

    # only remove the face if it belongs to a space
    created_spaces = space.remove_face(face)
    other_space.add_face(face)

    return [other_space] + list(created_spaces)


__all__ = ['mutate_simple', 'mutate_aligned']
