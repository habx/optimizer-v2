# coding=utf-8
"""
Genetic algorithm operator module : population
A population function is used to create the initial population from a given individual.
It is expected to take as argument a seed individual and a population length.
It must return a list of individuals.
"""
from typing import TYPE_CHECKING, List
from libs.refiner.core import mutateFunc
from libs.refiner.core import populateFunc

if TYPE_CHECKING:
    from libs.refiner.core import Individual


def clone(seed: 'Individual', length: int) -> List['Individual']:
    """
    Simple clones of an initial seed individual
    :param seed:
    :param length:
    :return:
    """
    return [seed.clone() for _ in range(length)]


def fc_mutate(mutation: mutateFunc) -> populateFunc:
    """
    *Factory*
    Creates a population function by mutating the seed individual with the specified mutation
    :param mutation: a mutation function to apply to the seed individual
    :return: a list of individual
    """
    def _population_func(seed: 'Individual', length: int) -> List['Individual']:
        return [mutation(seed.clone()) for _ in range(length)]

    return _population_func


__all__ = ['fc_mutate', 'clone']