# coding=utf-8
"""
Constraint module : describes the type of constraint that can be applied to a space or a linear
These constraints could be used by a genetic algorithm to compute a multi-objective cost function

A constraint computes a score

"""
import math
from typing import TYPE_CHECKING, Callable, Union, Dict, Optional, Any
import logging

from libs.utils.catalog import Catalog
from libs.utils.geometry import ccw_angle
from libs.size import Size

if TYPE_CHECKING:
    from libs.plan import Space, Linear

scoreFunction = Callable[[Union['Space', 'Linear']], float]
scoreFunctionFactory = Callable[..., scoreFunction]

CONSTRAINTS = Catalog('constraints')


class Constraint:
    """
    A type of constraint
    An imperative constraint must be verified. A mutation cannot be applied if it will
    break an imperative constraint.
    If a constraint is not imperative, it will be considered an objective constraint.
    In this case, a mutation will be applied if it will increase the score of the modified spaces
    according to the constraint score function.
    """
    def __init__(self,
                 name: str,
                 params: Dict[str, Any],
                 score_factory: scoreFunctionFactory,
                 imperative: bool = True):
        self.name = name
        self.params = params
        self.score_factory = score_factory  # the min the better
        self.imperative = imperative

    def __repr__(self):
        return 'Constraint: {0}'.format(self.name)

    def check(self,
              space_or_linear: Union['Space', 'Linear'],
              other: Optional['Constraint'] = None) -> bool:
        """
        Returns True if the constraint is satisfied. A constraint is deemed satisfied if
        its score is inferior or equal to zero
        :param space_or_linear:
        :param other: an other constraint
        :return:
        """
        if other is not None:
            return self.score(space_or_linear) <= 0 or other.score(space_or_linear) <= 0
        return self.score(space_or_linear) <= 0

    def set(self, **params) -> 'Constraint':
        """
        Sets the params of the constraint
        For example a maximum width
        :param params:
        :return:
        """
        for key, value in params.items():
            if key in self.params:
                self.params[key] = value
            else:
                raise ValueError('A parameter can only be modified not added: {0}'.format(key))

        return self

    def score(self, space_or_linear: Union['Space', 'Linear']) -> float:
        """
        Computes a score of how the space or the linear respects the constraint
        :param space_or_linear:
        :return: a score
        """
        return self.score_factory(self.params)(space_or_linear)


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


# score functions

def is_square(space: 'Space') -> float:
    """
    Scores the area / perimeter ratio of a space. Will equal 100 if the space is a square.
    :param space:
    :return:
    """
    return space.area / ((space.perimeter / 4)**2) * 100


# score functions factories

def few_corners(params: Dict) -> scoreFunction:
    """
    Scores a space by counting the number of corners
    :param params:
    :return:
    """
    min_corners = params['min_corners']

    def _score(space: 'Space') -> float:
        number_of_corners = 0
        for edge in space.edges:
            if ccw_angle(edge.vector, edge.space_next.vector) >= 20.0:
                number_of_corners += 1

        logging.debug(math.fabs(number_of_corners - min_corners))
        return math.fabs(number_of_corners - min_corners)

    return _score


def max_size(params: Dict) -> scoreFunction:
    """
    Checks if a space has a size inferior to the specified size
    :param params
    :return:
    """
    _max_size = params['max_size']

    def _score(space: 'Space') -> float:
        if space.size <= _max_size:
            return 0
        else:
            return _max_size.distance(space.size)

    return _score


few_corners_constraint = SpaceConstraint('few_corners', {'min_corners': 4}, few_corners,
                                         imperative=False)
CONSTRAINTS.add(few_corners_constraint)

max_size_constraint = SpaceConstraint('max_size', {'max_size': Size(100000, 1000, 1000)}, max_size)
CONSTRAINTS.add(max_size_constraint)

max_size_s_constraint = SpaceConstraint('max_size_s', {'max_size': Size(180000, 400, 350)},
                                        max_size)
CONSTRAINTS.add(max_size_s_constraint)

max_size_xs_constraint = SpaceConstraint('max_size_xs', {'max_size': Size(90000, 250, 300)},
                                         max_size)
CONSTRAINTS.add(max_size_xs_constraint)
