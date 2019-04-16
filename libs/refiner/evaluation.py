# coding=utf-8
"""
Evaluation Module
Contains the function used to evaluate the fitness of an individual
An evaluation function should take an individual

We use a composition factory:
ex.
my_evaluation_func = compose([fc_score_area(spec), score_corner, score_bounding_box)

"""
import math
from typing import TYPE_CHECKING, Optional, Sequence, List, Callable
from libs.refiner.core import evaluateFunc

if TYPE_CHECKING:
    from libs.specification.specification import Specification
    from libs.refiner.core import Individual

scoreFunc = Callable[['Individual'], float]


def compose(funcs: List[scoreFunc]) -> evaluateFunc:
    """
    A factory to compose evaluation function from a list of score function
    :param funcs:
    :return:
    """
    def _evaluate_func(ind: 'Individual') -> Sequence[float]:
        return tuple(map(lambda x: x(ind), funcs))

    return _evaluate_func


def fc_score_area(spec: Optional['Specification'] = None) -> scoreFunc:
    """
    Score function factory
    Returns a score function that evaluates the proximity of the individual to a specification
    instance
    :param spec:
    :return:
    """
    def _score(ind: 'Individual') -> float:
        area_score = 0.0
        if spec is not None:
            for space in ind.spaces:
                if not space.category.mutable:
                    continue
                corresponding_items = (item for item in spec.items
                                       if item.category is space.category)
                space_score = min((math.fabs((item.required_area - space.cached_area())
                                             / item.required_area)
                                  for item in corresponding_items), default=100)
                area_score += space_score

        return area_score

    return _score


def score_corner(ind: 'Individual') -> float:
    """

    :param ind:
    :return:
    """
    min_corners = 4
    score = 0.0
    for space in ind.spaces:
        if not space.category.mutable or space.category.name == "living":
            continue
        score += (space.number_of_corners() - min_corners) / min_corners
    return score


def score_bounding_box(ind: 'Individual') -> float:
    """

    :param ind:
    :return:
    """
    score = 0.0
    for space in ind.spaces:
        if not space.category.mutable or space.category.name == "living":
            continue
        if space.cached_area() == 0:
            score += 1000
            continue
        box = space.bounding_box()
        box_area = box[0] * box[1]
        space_score = math.fabs((space.cached_area() - box_area) / box_area)
        score += space_score

    return score


def score_aspect_ratio(ind: 'Individual') -> float:
    """

    :param ind:
    :return:
    """
    score = 0.0
    min_aspect_ratio = 16
    for space in ind.spaces:
        if not space.category.mutable or space.category.name == "living":
            continue
        if space.cached_area() == 0:
            score += 1000
            continue
        space_score = math.fabs(space.perimeter**2/space.cached_area()/min_aspect_ratio - 1.0)
        score += space_score

    return score


__all__ = ['compose', 'score_aspect_ratio', 'score_bounding_box', 'fc_score_area', 'score_corner']
