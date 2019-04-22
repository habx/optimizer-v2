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
from typing import TYPE_CHECKING, Sequence, List, Callable, Dict
from libs.refiner.core import evaluateFunc

if TYPE_CHECKING:
    from libs.specification.specification import Specification, Item
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


def create_item_dict(_spec: 'Specification') -> Dict[int, 'Item']:
    """
    Creates a 1-to-1 dict between spaces and items of the specification
    :param _spec:
    :return:
    """
    output = {}
    spec_items = _spec.items[:]
    for sp in _spec.plan.spaces:
        if not sp.category.mutable:
            continue
        corresponding_items = list(filter(lambda i: i.category is sp.category, spec_items))
        best_item = min(corresponding_items,
                        key=lambda i: math.fabs(i.required_area - sp.cached_area()))
        assert best_item, "Score: Each space should have a corresponding item in the spec"
        output[sp.id] = best_item
        spec_items.remove(best_item)
    return output


def fc_score_area(spec: 'Specification') -> scoreFunc:
    """
    Score function factory
    Returns a score function that evaluates the proximity of the individual to a specification
    instance
    Note: The dictionary matching spaces to specification items is memoized in the factory
    :param spec:
    :return:
    """

    def _score(ind: 'Individual') -> float:
        # create the item_to_space dict
        area_score = 0.0
        if spec is not None:
            for space in ind.spaces:
                if not space.category.mutable:
                    continue
                item = space_to_item[space.id]
                space_area = space.cached_area()
                if space_area < item.min_size.area:
                    space_score = ((space_area - item.min_size.area)/space_area)**2
                elif space_area > item.max_size.area:
                    space_score = ((space_area - item.max_size.area)/space_area)**2
                else:
                    space_score = 0
                area_score += space_score

        return area_score

    space_to_item = create_item_dict(spec)
    return _score


def score_corner(ind: 'Individual') -> float:
    """

    :param ind:
    :return:
    """
    min_corners = 4
    score = 0.0
    num_space = 0
    for space in ind.spaces:
        if not space.category.mutable or space.category.name == "living":
            continue
        score += (space.number_of_corners() - min_corners) / min_corners
        num_space += 1
    return score / num_space


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
        area = space.cached_area()
        space_score = math.fabs((area - box_area) / area)
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


__all__ = ['compose', 'score_aspect_ratio', 'score_bounding_box', 'fc_score_area', 'score_corner',
           'create_item_dict']
