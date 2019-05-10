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
from typing import TYPE_CHECKING, Sequence, List, Callable, Dict, Optional

from libs.utils.geometry import ccw_angle, pseudo_equal, min_section

if TYPE_CHECKING:
    from libs.specification.specification import Specification, Item
    from libs.refiner.core import Individual
    from libs.plan.plan import Space, Plan

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


def create_item_dict(_spec: 'Specification') -> Dict[int, Optional['Item']]:
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
        # Note : corridors have no corresponding spec item
        best_item = min(corresponding_items,
                        key=lambda i: math.fabs(i.required_area - sp.cached_area()), default=None)
        if sp.category.name != "circulation":
            assert best_item, "Score: Each space should have a corresponding item in the spec"
        output[sp.id] = best_item
        if best_item is not None:
            spec_items.remove(best_item)
    return output


def score_area(spec: 'Specification', ind: 'Individual') -> float:
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
            if item is None:
                continue
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
    excluded_spaces = ("circulation",)
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


def check_area(plan: 'Plan', spec: 'Specification') -> None:
    """
    Compares the plan area with the specification objectives
    :param plan:
    :param spec:
    :return:
    """
    logging.info("Refiner: Checking Plan Surface: %s", plan.name)

    item_dict = create_item_dict(spec)
    for space in plan.mutable_spaces():
        if item_dict[space.id] is None:
            continue
        area = round(space.cached_area()) / (100 ** 2)
        min_area = item_dict[space.id].min_size.area / (100 ** 2)
        max_area = item_dict[space.id].max_size.area / (100 ** 2)
        ok = min_area <= area <= max_area
        logging.info("  • Area {} : {} -> [{}, {}]: {}".format(space.category.name, area, min_area,
                                                               max_area, "✅" if ok else "❌"))


def score_connectivity(_: 'Specification', ind: 'Individual') -> float:
    """
    Returns a score indicating whether the plan is connected
    :param _:
    :param ind:
    :return:
    """
    min_length = 90.0
    cost_per_unconnected_space = 1.0
    score = 0

    front_door = list(ind.get_linears("frontDoor"))[0]
    front_door_space = ind.get_space_of_edge(front_door.edge)
    connected_circulation_spaces = [front_door_space]

    circulation_spaces = list(ind.circulation_spaces())
    if front_door_space in circulation_spaces:
        circulation_spaces.remove(front_door_space)
    # for simplicity we estimate that at maximum a circulation space is connected
    # to the front door through 3 others spaces.
    for _ in range(3):
        for circulation_space in circulation_spaces[:]:
            for connected_space in connected_circulation_spaces:
                if circulation_space.adjacent_to(connected_space, 90.0):
                    # we place the newly found spaces at the beginning of the array
                    connected_circulation_spaces.insert(0, circulation_space)
                    circulation_spaces.remove(circulation_space)
                    break

    score += len(circulation_spaces) * cost_per_unconnected_space

    for space in ind.mutable_spaces():
        if space.category.circulation:
            continue

        found_adjacency = False
        for edge in space.exterior_edges:
            if edge.pair.face is None:
                continue
            other = ind.get_space_of_edge(edge.pair)
            if not other or not other.category.circulation:
                continue
            # forward check
            shared_length = edge.length
            for e in space.siblings(edge):
                if e is edge:
                    continue
                if other.has_edge(e.pair):
                    shared_length += e.length

            # backward check
            for e in other.siblings(edge.pair):
                if e is edge.pair:
                    continue
                if space.has_edge(e.pair):
                    shared_length += e.length

            if shared_length >= min_length:
                found_adjacency = True
                break

        score += cost_per_unconnected_space if not found_adjacency else 0

    return score


def score_corridor_width(_: 'Specification', ind: 'Individual') -> float:
    """
    Computes a score of the min width of each corridor
    :param _:
    :param ind:
    :return:
    """
    min_width = 90.0
    score = 0
    corridors = ind.get_spaces("circulation")
    for corridor in corridors:
        polygon = corridor.boundary_polygon()
        width = min_section(polygon)
        score += ((width - min_width)/min_width)**2

    return score


__all__ = ['compose', 'score_aspect_ratio', 'score_bounding_box', 'score_area', 'score_corner',
           'create_item_dict', 'check_area']
