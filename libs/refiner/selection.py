"""
Selection Module for Refiner
"""
import random

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
    elite_size = int(len_pop*ratio)
    pop = pop[0:elite_size]
    if k <= elite_size:
        return pop[0:k]
    new_pop = [mutate_func(random.choice(pop).clone()) for _ in range(k - elite_size)]
    return pop + new_pop

