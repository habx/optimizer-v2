# coding=utf-8
"""
Constraint module : describes the type of constraint that can be applied to a space or a linear
These constraints could be used by a genetic algorithm to compute a multi-objective cost function
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


space_constraints = {
    'square': SpaceConstraint('square', is_square, 100)
}
