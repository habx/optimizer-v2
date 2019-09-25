"""
Selection Module for Refiner
"""
import random
import logging

from typing import TYPE_CHECKING, List
from libs.refiner import space_nsga, nsga

if TYPE_CHECKING:
    from libs.refiner.core import Individual
    from libs.refiner.core import MutateFunc


def nsga_select(mutate_func: 'MutateFunc',
                ratio: float,
                pop: List['Individual'],
                k: int) -> List['Individual']:
    """
    Keep only the `ratio` best individuals.
    Replace the removed one by a random top individual mutated.
    NOTE: we must preserve diversity
    :param mutate_func:
    :param pop:
    :param k: number of individual of the total pop (can be different from the size of the specified
    population
    :param ratio:
    :return:
    """
    # note we need to reverse because fitness values are negative
    len_pop = len(pop)
    elite_size = int(len_pop * ratio / 2.0)
    pareto_pop = nsga.select_pareto_front(pop)
    pareto_pop.sort(key=lambda _i: _i.fitness.wvalue, reverse=True)
    logging.info("NSGA select : pareto front size {}".format(len(pareto_pop)))

    elite_pop = []
    fitness_values = []
    # preserve diversity by removing identical individual
    # per simplicity, we consider that an individual is identical to another i
    # if their fitnesses are the same
    offset = 0
    while len(elite_pop) < elite_size:
        while len(pareto_pop) > offset and pareto_pop[offset].fitness.wvalue in fitness_values:
            logging.debug("Found same individual ! %i", offset)
            offset += 1
        if len(pareto_pop) <= offset:
            break
        elite_pop.append(pareto_pop[offset])
        fitness_values.append(pareto_pop[offset].fitness.wvalue)

    if k <= len(elite_pop):
        return elite_pop[0:k]

    logging.info("NSGA select : elite pop size {}".format(len(elite_pop)))

    new_pop = []

    # try multiple mutations to increase diversity in the pool
    for i in range(k - len(elite_pop)):
        pb = random.random()
        mutations_num = 1
        if .5 < pb <= .8:
            mutations_num = 2
        elif pb > .8:
            mutations_num = 3
        ind = random.choice(elite_pop).clone()
        for j in range(mutations_num):
            ind = mutate_func(ind)
        new_pop.append(ind)

    return elite_pop + new_pop


def elite_select(mutate_func: 'MutateFunc',
                 ratio: float,
                 pop: List['Individual'],
                 k: int) -> List['Individual']:
    """
    Keep only the `ratio` best individuals.
    Replace the removed one by a random top individual mutated.
    NOTE: we must preserve diversity
    :param mutate_func:
    :param pop:
    :param k: number of individual of the total pop (can be different from the size of the specified
    population
    :param ratio:
    :return:
    """
    # note we need to reverse because fitness values are negative
    pop.sort(key=lambda _i: _i.fitness.wvalue, reverse=True)
    len_pop = len(pop)
    elite_size = int(len_pop*ratio)
    elite_pop = []
    fitness_values = []
    # preserve diversity by removing identical individual
    # per simplicity, we consider that an individual is identical to another i
    # if their fitnesses are the same
    offset = 0
    while len(elite_pop) < elite_size:
        while len(pop) > offset and pop[offset].fitness.wvalue in fitness_values:
            logging.debug("Found same individual ! %i", offset)
            offset += 1
        if len(pop) <= offset:
            break
        elite_pop.append(pop[offset])
        fitness_values.append(pop[offset].fitness.wvalue)

    if k <= len(elite_pop):
        return elite_pop[0:k]

    new_pop = []

    # try multiple mutations to increase diversity in the pool
    for i in range(k - len(elite_pop)):
        pb = random.random()
        mutations_num = 1
        if .5 < pb <= .8:
            mutations_num = 2
        elif pb > .8:
            mutations_num = 3
        ind = random.choice(elite_pop).clone()
        for j in range(mutations_num):
            ind = mutate_func(ind)
        new_pop.append(ind)

    return elite_pop + new_pop
