# coding=utf-8
"""
Constraint module : describes the type of constraint that can be applied to a space or a linear
These constraints could be used by a genetic algorithm to compute a multi-objective cost function

A constraint computes a score

"""
import math
from typing import TYPE_CHECKING, Callable, Union, Dict, Optional, Any

from libs.utils.geometry import ccw_angle
from libs.size import Size

if TYPE_CHECKING:
    from libs.plan import Space

scoreFunction = Callable[[Union['Space', 'Linear']], float]
scoreFunctionFactory = Callable[..., scoreFunction]


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
                 score_factory: scoreFunctionFactory,
                 params: Dict[str, Any], name: str = "",
                 imperative: bool = True):
        self.score_factory = score_factory  # the min the better
        self.params = params
        self.name = name
        self.imperative = imperative

    def __repr__(self):
        return 'Constraint: {0}'.format(self.name)

    def check(self,
              *spaces: 'Space',
              other_constraint: Optional['Constraint'] = None) -> bool:
        """
        Returns True if the constraint is satisfied. A constraint is deemed satisfied if
        its score is inferior or equal to zero
        :param spaces:
        :param other_constraint: an other constraint
        :return:
        """
        for space in spaces:
            # per convention if the params specify a category name, we only apply the constraint
            # to spaces of the specified category
            if ("category_name" in self.params
                    and space.category.name != self.params["category_name"]):
                continue
            if other_constraint is not None:
                if self.score(space) > 0 and other_constraint.score(space) > 0:
                    return False
            if self.score(space) > 0:
                return False
        return True

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

    def score(self, space: 'Space') -> float:
        """
        Computes a score of how the space respects the constraint
        :param space:
        :return: a score
        """
        return self.score_factory(self.params)(space)


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


# score functions factories

def square_shape(_: Dict) -> scoreFunction:
    """
    Scores the area / perimeter ratio of a space.
    :return:
    """
    def _score(space: 'Space') -> float:
        if (space is None or space.size is None or
                space.size.depth is None or space.size.width is None):
            return 0.0
        rectangle_area = space.size.depth * space.size.width
        return max((rectangle_area - space.area)/rectangle_area, 0.0)

    return _score


def few_corners(params: Dict) -> scoreFunction:
    """
    Scores a space by counting the number of corners.
    Note : we only count the exterior corners of the space
    We do not count the corners of internal holes.
    A corner is defined as an angle between two boundaries edge superior to 20.0
    :param params:
    :return:
    """
    min_corners = params['min_corners']
    corner_min_angle = 20.0

    def _score(space: 'Space') -> float:
        number_of_corners = 0
        for edge in space.exterior_edges:
            if ccw_angle(edge.vector, space.next_edge(edge).vector) >= corner_min_angle:
                number_of_corners += 1

        if number_of_corners == 0:
            return 0

        return math.fabs((number_of_corners - min_corners)/number_of_corners)

    return _score


def max_size(params: Dict) -> scoreFunction:
    """
    Checks if a space has a size inferior to the specified size
    :param params
    :return:
    """
    _max_size: Size = params['max_size']

    def _score(space: 'Space') -> float:
        if space.size <= _max_size:
            return 0
        else:
            return _max_size.distance(space.size)

    return _score


def min_size(params: Dict) -> scoreFunction:
    """
    Checks if a space has a size inferior to the specified size
    :param params
    :return:
    """
    _min_size: Size = params['min_size']

    def _score(space: 'Space') -> float:
        if space.size >= _min_size:
            return 0
        else:
            return _min_size.distance(space.size)

    return _score


# Imperative constraints
min_size_constraint = SpaceConstraint(min_size,
                                      {"min_size": Size(3900, 60, 60)},
                                      "min-size")

max_size_constraint = SpaceConstraint(max_size,
                                      {"max_size": Size(100000, 1000, 1000)},
                                      "max_size")

max_size_constraint_seed = SpaceConstraint(max_size,
                                           {"max_size": Size(100000, 1000, 1000),
                                            "category_name": "seed"},
                                           "max_size")

max_size_s_constraint_seed = SpaceConstraint(max_size,
                                             {"max_size": Size(180000, 400, 350),
                                              "category_name": "seed"},
                                             "max_size_s")

max_size_xs_constraint_seed = SpaceConstraint(max_size,
                                              {"max_size": Size(90000, 300, 300),
                                               "category_name": "seed"},
                                              "max_size_xs")

# objective constraints
square_shape = SpaceConstraint(square_shape,
                               {"max_ratio": 100.0},
                               "square_shape", imperative=False)

few_corners_constraint = SpaceConstraint(few_corners,
                                         {"min_corners": 4},
                                         "few_corners",
                                         imperative=False)


CONSTRAINTS = {
    "few_corners": few_corners_constraint,
    "min_size": min_size_constraint,
    "max_size": max_size_constraint,
    "max_size_seed": max_size_constraint_seed,
    "max_size_s_seed": max_size_s_constraint_seed,
    "max_size_xs_seed": max_size_xs_constraint_seed,
    "square_shape": square_shape
}
