# coding=utf-8
"""
Constraint module : describes the type of constraint that can be applied to a space or a linear
These constraints could be used by a genetic algorithm to compute a multi-objective cost function

A constraint computes a score

"""

from typing import Callable
from libs.plan import Space


class Constraint:
    """
    A type of constraint
    """
    def __init__(self, name: str, score: Callable, min_score: float = 0):
        self.name = name
        self.score = score
        self.min_score = min_score

    def __repr__(self):
        return 'Constraint: {0}'.format(self.name)

    @property
    def is_satisfied(self) -> bool:
        """
        Returns True if the constraint is satisfied
        :return:
        """
        return self.score() <= self.min_score


class SpaceConstraint(Constraint):
    """
    A space constraint
    A rule that can be applied to a Space instance
    Examples : 'square'
    """
    pass


class LinearConstraint(Constraint):
    """
    A type of linear constraint
    """
    pass


def is_square(space: Space) -> float:
    """
    Scores the area / perimeter ratio of a space. Will equal 100 if the space is a square.
    :param space:
    :return:
    """
    return space.area / ((space.perimeter / 4)**2) * 100


def few_corners(space: 'Space') -> float:
    """
    Scores a space by counting the number of corners
    :param space:
    :return:
    """
    number_of_corners = 0
    for edge in space.edges:
        if not edge.next_is_aligned:
            number_of_corners += 1

    return number_of_corners


space_constraints = {
    'square': SpaceConstraint('square', is_square, 100)
}
