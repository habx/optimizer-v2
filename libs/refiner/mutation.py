# coding=utf-8
"""
Genetic Algorithm Mutation module
A mutation is a function that takes an individual as input and modifies it in place
"""
import random
import logging
from typing import TYPE_CHECKING, Optional, Callable, Tuple, List

from libs.operators.mutation import MUTATIONS
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
        MUTATIONS["add_aligned_face"].apply_to(edge, space, store_initial_state=False)
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
        modified_spaces = MUTATIONS["remove_face"].apply_to(edge, space, store_initial_state=False)
        if __debug__ and len(modified_spaces) > 2:
            logging.warning("Refiner: Mutation: A space was split !! %s", modified_spaces[2])
        if __debug__ and space.number_of_faces == 0:
            logging.warning("Refiner: A space has no face left !!")
    return ind


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


__all__ = ['mutate_simple', 'mutate_aligned']
