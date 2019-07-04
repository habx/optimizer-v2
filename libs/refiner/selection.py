"""
Selection Module for Refiner
"""
import random
import logging

from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from libs.refiner.core import Individual
    from libs.refiner.core import MutateFunc


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
    pop.sort(key=lambda i: i.fitness.wvalue, reverse=True)
    len_pop = len(pop)
    logging.info("LENGTH OF POP %i", len_pop)
    elite_size = int(len_pop*ratio)
    elite_pop = []
    fitness_values = []
    # preserve diversity
    offset = 0
    while len(elite_pop) < elite_size:
        while len(pop) > offset and pop[offset].fitness.wvalue in fitness_values:
            logging.info("Found same individual ! %i", offset)
            offset += 1
        if len(pop) <= offset:
            break
        elite_pop.append(pop[offset])
        fitness_values.append(pop[offset].fitness.wvalue)

    logging.info("Elite size: %s", len(elite_pop))

    if k <= len(elite_pop):
        return elite_pop[0:k]

    new_pop = []

    # try multiple mutations to increase diversity in the pool
    for i in range(k - len(elite_pop)):
        pb = random.random()
        mutations_num = 1
        if .3 < pb <= .6:
            mutations_num = 2
        elif pb > .6:
            mutations_num = 3
        ind = random.choice(elite_pop).clone()
        for j in range(mutations_num):
            ind = mutate_func(ind)
        new_pop.append(ind)

    return elite_pop + new_pop

