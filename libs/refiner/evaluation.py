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
import logging
from typing import TYPE_CHECKING, Sequence, List, Callable, Dict

from libs.utils.geometry import ccw_angle, pseudo_equal

if TYPE_CHECKING:
    from libs.specification.specification import Specification, Item
    from libs.refiner.core import Individual
    from libs.plan.plan import Space

scoreFunc = Callable[['Individual'], float]


def compose(funcs: List[scoreFunc], spec: 'Specification', ind: 'Individual') -> Sequence[float]:
    """
    A factory to compose evaluation function from a list of score function
    :param funcs:
    :param spec:
    :param ind:
    :return:
    """
    return tuple(map(lambda x: x(spec, ind), funcs))


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
        corresponding_items = list(filter(lambda i: i.category.name == sp.category.name,
                                          spec_items))
        best_item = min(corresponding_items,
                        key=lambda i: math.fabs(i.required_area - sp.cached_area()))
        assert best_item, "Score: Each space should have a corresponding item in the spec"
        output[sp.id] = best_item
        spec_items.remove(best_item)
    return output


def fc_score_area(spec: 'Specification', ind: 'Individual') -> float:
    """
    Returns a score that evaluates the proximity of the individual to a specification
    instance
    Note: The dictionary matching spaces to specification items is memoized in the factory
    :param spec:
    :param ind
    :return:
    """
    space_to_item = create_item_dict(spec)

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


def score_corner(_: 'Specification', ind: 'Individual') -> float:
    """
    :param _:
    :param ind:
    :return:
    """
    excluded_spaces = ()
    min_corners = 4
    score = 0.0
    num_space = 0
    for space in ind.spaces:
        if not space.category.mutable or space.category.name in excluded_spaces:
            continue
        score += (space.number_of_corners() - min_corners) / min_corners
        num_space += 1
    return score / num_space


def number_of_corners(space: 'Space') -> int:
    """
    Returns the number of "inside" corners. The corners of the boundary edges of a space
    that are adjacent to another mutable space.
    :param space:
    :return:
    """
    corner_min_angle = 20.0
    num_corners = 0
    internal_edges = []
    for e in space.exterior_edges:
        other = space.plan.get_space_of_edge(e)
        if not other or other.category.external:
            continue
        internal_edges.append(e)

    for edge in internal_edges:
        angle = ccw_angle(edge.opposite_vector, space.next_edge(edge).vector)
        if not pseudo_equal(angle, 180.0, corner_min_angle):
            num_corners += 1

    return num_corners


def score_bounding_box(_: 'Specification', ind: 'Individual') -> float:
    """
    :param _:
    :param ind:
    :return:
    """
    excluded_spaces = ()
    score = 0.0
    for space in ind.spaces:
        if not space.category.mutable or space.category.name in excluded_spaces:
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


def score_aspect_ratio(_: 'Specification', ind: 'Individual') -> float:
    """
    :param _
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


def check_area(plan, spec):
    """
    Compares the plan area with the specification objectives
    :param plan:
    :param spec:
    :return:
    """
    logging.info("Refiner: Checking Plan Surface: %s", plan.name)

    item_dict = create_item_dict(spec)
    for space in plan.mutable_spaces():
        area = round(space.cached_area()) / (100 ** 2)
        min_area = item_dict[space.id].min_size.area / (100 ** 2)
        max_area = item_dict[space.id].max_size.area / (100 ** 2)
        ok = min_area <= area <= max_area
        logging.info("  • Area {} : {} -> [{}, {}]: {}".format(space.category.name, area, min_area,
                                                               max_area, "✅" if ok else "❌"))


__all__ = ['compose', 'score_aspect_ratio', 'score_bounding_box', 'fc_score_area', 'score_corner',
           'create_item_dict', 'check_area']
