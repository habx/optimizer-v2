# coding=utf-8
"""
Genetic Algorithm Mutation module
A mutation is a function that takes an individual as input and modifies it in place
"""
import random
import logging
from typing import TYPE_CHECKING, Optional

from libs.operators.mutation import MUTATIONS
from libs.operators.selector import SELECTORS

if TYPE_CHECKING:
    from libs.plan.plan import Plan, Space
    from libs.mesh.mesh import Edge
    from libs.refiner.core import Individual


def mutate_aligned(ind: 'Individual') -> 'Individual':
    """
    Mutates the plan.
    1. select a random space
    2. select a random mutable edge
    3. mutate the edge
    :param ind:
    :return: a single element tuple containing the mutated individual
    """
    space = _random_space(ind)
    edge = _random_edge(space)
    MUTATIONS["swap_aligned_face"].apply_to(edge, space)
    return ind


def mutate_simple(ind: 'Individual') -> 'Individual':
    """
    Mutates the plan.
    1. select a random space
    2. select a random mutable edge
    3. mutate the edge
    :param ind:
    :return: a single element tuple containing the mutated individual
    """
    space = _random_space(ind)
    edge = _random_edge(space)
    if edge:
        modified_spaces = MUTATIONS["remove_face"].apply_to(edge, space)
        if len(modified_spaces) > 2:
            logging.warning("Refiner: Mutation: A space was split !! %s", modified_spaces[2])
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


def _random_edge(space: 'Space') -> Optional['Edge']:
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
