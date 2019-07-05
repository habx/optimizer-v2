# coding=utf-8
"""
Evaluation Module
Contains the function used to evaluate the fitness of an individual
An evaluation function should take an individual

"""
import math
import logging
from typing import TYPE_CHECKING, List, Callable, Dict, Optional, Tuple

from libs.utils.geometry import ccw_angle, pseudo_equal, min_section
from libs.plan.category import SPACE_CATEGORIES

if TYPE_CHECKING:
    from libs.specification.specification import Specification, Item
    from libs.refiner.core import Individual
    from libs.plan.plan import Space, Floor
    from libs.space_planner.solution import Solution

# a score function returns a specific score for each space of the plan. It takes
# as arguments a specification object and an individual
scoreFunc = Callable[['Specification', 'Individual'], Dict[int, float]]


def compose(funcs: List[scoreFunc],
            spec: 'Specification', ind: 'Individual') -> Dict[int, Tuple[float, ...]]:
    """
    The main function to create an evaluation function from a list of score functions.
    The function is expected to be used with a Toolbox instance that will create the corresponding
    partial function by fixing the left parameters *funcs* and *specs*.
    :param funcs:
    :param spec:
    :param ind:
    :return:
    """
    current_fitness = ind.fitness.sp_values
    scores = [f(spec, ind) for f in funcs]
    spaces_id = [s.id for s in ind.mutable_spaces()]
    n_scores = len(funcs)

    for space_id in spaces_id:
        new_space_fitness = [None] * n_scores
        for i in range(n_scores):
            if space_id in scores[i]:
                new_space_fitness[i] = scores[i][space_id]
            else:
                new_space_fitness[i] = current_fitness[space_id][i]
        current_fitness[space_id] = tuple(new_space_fitness)

    ind.modified_spaces = set()  # reset the set of the modified spaces
    return current_fitness


def create_item_dict(_solution: 'Solution') -> Dict[int, Optional['Item']]:
    """
    Creates a 1-to-1 dict between spaces and items of the specification
    :param _solution:
    :return:
    """
    output = {}
    for space, item in _solution.space_item.items():
        if item.category.name == "circulation":
            output[space.id] = None
        else:
            output[space.id] = item
    return output


def _score_space_area(space_area: float, min_area: float, max_area: float) -> float:
    """
    Scores the space area
    :param space_area:
    :param min_area:
    :param max_area:
    :return:
    """
    sp_score = 0
    if space_area < min_area:
        # note: worse to be smaller
        sp_score = (((min_area - space_area) / min_area) ** 2) * 100 * 3.0
    elif space_area > max_area:
        sp_score = (((space_area - max_area) / max_area) ** 2) * 100.0
    return sp_score


def score_area(spec: 'Specification', ind: 'Individual') -> Dict[int, float]:
    """
    Returns a score that evaluates the proximity of the individual to a specification
    instance
    :param spec:
    :param ind
    :return:
    """
    min_areas = {
        SPACE_CATEGORIES["bedroom"]: 90000.0,
        SPACE_CATEGORIES["bathroom"]: 30000.0,
        SPACE_CATEGORIES["toilet"]: 10000.0,
        "default": 10000.0
    }
    min_size_penalty = 1000.0

    space_to_item = ind.fitness.cache.get("space_to_item", None)

    area_score = {}
    if spec is None:
        logging.warning("Refiner: score area: not spec specified")
        return {}

    for space in ind.mutable_spaces():
        if space.id not in ind.modified_spaces:
            continue
        if space.category is SPACE_CATEGORIES["circulation"]:
            area_score[space.id] = 0.0
            continue
        item = space_to_item[space.id]
        space_area = space.cached_area()

        if space_area < min_areas.get(space.category, min_areas["default"]):
            area_score[space.id] = min_size_penalty
            continue

        area_score[space.id] = _score_space_area(space_area, item.min_size.area, item.max_size.area)

    return area_score


def score_corner(_: 'Specification', ind: 'Individual') -> Dict[int, float]:
    """
    :param _:
    :param ind:
    :return:
    """
    excluded_spaces = ()
    score = {}
    for space in ind.mutable_spaces():
        if space.id not in ind.modified_spaces:
            continue
        if space.category in excluded_spaces:
            score[space.id] = 0.0
            continue
        # score[space.id] = number_of_corners(space)
        score[space.id] = max(space.number_of_corners() - 4.0, 0)
        if space.has_holes:
            score[space.id] += 8.0  # arbitrary penalty (corners count double in a hole ;-))
    return score


def number_of_corners(space: 'Space') -> int:
    """
    Returns the number of "inside" corners. The corners of the boundary edges of a space
    that are not adjacent to an external space.
    :param space:
    :return:
    """
    if not space.edge:
        logging.warning("Refiner: Space with no edge found : %s", space)
        return 0

    corner_min_angle = 20.0
    num_corners = 0
    previous_corner = False
    for e in space.exterior_edges:
        other = space.plan.get_space_of_edge(e.pair)
        if not other or other.category.external:
            if previous_corner:
                num_corners -= 1
                previous_corner = False
            continue
        angle = ccw_angle(e.opposite_vector, space.next_edge(e).vector)
        if not pseudo_equal(angle, 180.0, corner_min_angle):
            num_corners += 1
            previous_corner = True
        else:
            previous_corner = False
    # loop back to initial edge
    other = space.plan.get_space_of_edge(space.edge.pair)
    if not other or other.category.external:
        if previous_corner:
            num_corners -= 1

    return num_corners


def score_bounding_box(_: 'Specification', ind: 'Individual') -> Dict[int, float]:
    """
    Computes a score as the difference between the area of a space and its bounding box
    :param _:
    :param ind:
    :return:
    """
    ratios = {
        SPACE_CATEGORIES["circulation"]: 5.0,
        SPACE_CATEGORIES["livingKitchen"]: 0.5,
        SPACE_CATEGORIES["living"]: 0.5,
        "default": 1.0
    }
    score = {}
    for space in ind.mutable_spaces():
        if space.id not in ind.modified_spaces:
            continue
        if space.cached_area() == 0:
            score[space.id] = 100
            continue
        box = space.bounding_box()
        box_area = box[0] * box[1]
        area = space.cached_area()
        space_score = ((area - box_area) / area)**2 * 100.0
        score[space.id] = space_score * ratios.get(space.category, ratios["default"])

    return score


def score_perimeter_area_ratio(_: 'Specification', ind: 'Individual') -> Dict[int, float]:
    """
    :param _
    :param ind:
    :return:
    """
    excluded_spaces = ()
    score = {}
    min_aspect_ratio = 16
    for space in ind.mutable_spaces():
        if space.category is SPACE_CATEGORIES["circulation"]:
            box = space.bounding_box()
            width = min(box)
            score[space.id] = _score_space_area(width, 90.0, 110.0)
            continue
        if space.id not in ind.modified_spaces:
            continue
        if space.category in excluded_spaces:
            score[space.id] = 0
            continue
        if space.cached_area() == 0:
            score[space.id] = 100
            continue
        space_score = math.fabs(space.perimeter**2/space.cached_area()/min_aspect_ratio - 1.0)
        score[space.id] = space_score

    return score


def score_width_depth_ratio(_: 'Specification', ind: 'Individual') -> Dict[int, float]:
    """

    :param _:
    :param ind:
    :return:
    """
    ratios = {
        "bedroom": 1.2,
        "toilet": 1.7,
        "bathroom": 1.2,
        "entrance": 1.7,
        "default": 1.5
    }
    circulation_min_width = 80.0
    circulation_max_width = 110.0
    score = {}
    for space in ind.mutable_spaces():
        if space.id not in ind.modified_spaces:
            continue
        box = space.bounding_box()
        # different rule for circulation we look for a specific width
        if space.category is SPACE_CATEGORIES["circulation"]:
            if circulation_min_width <= min(box) <= circulation_max_width:
                score[space.id] = 0
            elif circulation_min_width > min(box):
                # we add a ten times penalty because a narrow corridor can not be tolerated
                score[space.id] = ((circulation_min_width - min(box))/circulation_min_width) * 10
            elif circulation_max_width < min(box):
                score[space.id] = ((min(box) - circulation_max_width) / circulation_max_width)
            continue
        if min(box) == 0.:
            logging.warning("Refiner: Evaluation: Width Depth Ratio a space is empty %s", space)
            continue
        space_ratio = max(box)/min(box)
        ratio = ratios.get(space.category.name, ratios["default"])
        if space_ratio >= ratio:
            score[space.id] = (space_ratio - ratio)**2
        else:
            score[space.id] = 0

    return score


def score_connectivity(_: 'Specification', ind: 'Individual') -> Dict[int, float]:
    """
    Returns a score indicating whether the plan is connected
    :param _:
    :param ind:
    :return:
    """
    min_length = 80.0
    penalty = 1.0  # per unconnected space
    score = {}

    front_door = list(ind.get_linears("frontDoor"))[0]
    starting_steps = list(ind.get_linears("startingStep")) if ind.has_multiple_floors else []
    for floor in ind.floors.values():
        if floor is not front_door.floor:
            starting_step = [ss for ss in starting_steps if ss.floor is floor][0]
            root_space = ind.get_space_of_edge(starting_step.edge)
        else:
            root_space = ind.get_space_of_edge(front_door.edge)

        floor_score = _score_floor_connectivity(ind, floor, root_space, min_length, penalty)
        score.update(floor_score)

    return score


def _score_floor_connectivity(ind: 'Individual',
                              floor: 'Floor',
                              root_space: 'Space',
                              min_length: float = 80.0,
                              penalty: float = 1.0) -> Dict[int, float]:
    """
    Scores the connectivity of a given floor
    :return:
    """
    score = {}
    connected_spaces = [root_space]
    score[root_space.id] = 0.0

    circulation_spaces = [s for s in ind.circulation_spaces() if s.floor is floor]
    if root_space in circulation_spaces:
        circulation_spaces.remove(root_space)
    unconnected_spaces = circulation_spaces[:]
    for connected_space in connected_spaces:
        for unconnected_space in unconnected_spaces[:]:
            if unconnected_space.adjacent_to(connected_space, min_length):
                # we place the newly found spaces at the beginning of the array
                connected_spaces.append(unconnected_space)
                unconnected_spaces.remove(unconnected_space)
                score[unconnected_space.id] = 0.0
        if not unconnected_spaces:
            break

    for unconnected_space in unconnected_spaces:
        score[unconnected_space.id] = penalty

    circulation_spaces.append(root_space)

    for space in ind.mutable_spaces():
        # only check the connectivity of the space of the floor
        if space.floor is not floor:
            continue
        # no need to check the root space or the circulation space
        if space.category.circulation or space is root_space:
            continue
        # check if a circulation is adjacent to the space
        for circulation_space in circulation_spaces:
            if circulation_space.adjacent_to(space, min_length):
                score[space.id] = 0.0
                break
        else:
            score[space.id] = penalty

    return score


def score_circulation_width(_: 'Specification', ind: 'Individual') -> Dict[int, float]:
    """
    Computes a score of the min width of each corridor
    :param _:
    :param ind:
    :return:
    """
    min_width = 90.0
    score = {}
    circulation_categories = (
        SPACE_CATEGORIES["entrance"],
        SPACE_CATEGORIES["circulation"],
        # SPACE_CATEGORIES["livingKitchen"],
        # SPACE_CATEGORIES["living"]
    )

    for space in ind.mutable_spaces():
        if space.id not in ind.modified_spaces:
            continue
        if not space.edge:
            score[space.id] = 0.0
            continue
        if space.category not in circulation_categories:
            score[space.id] = 0.0
            continue
        polygon = space.boundary_polygon()
        width = min_section(polygon)
        if width > min_width:
            score[space.id] = 0
        else:
            score[space.id] = (min_width - width)/min_width*100.0

    return score


"""
UTILITY Function
"""


def check(ind: 'Individual', solution: 'Solution') -> None:
    """
    Compares the plan area with the specification objectives
    :param ind:
    :param solution:
    :return:
    """
    logging.info("Refiner: Checking Plan : %s", ind.name)

    item_dict = create_item_dict(solution)
    for space in ind.mutable_spaces():
        if space.id not in item_dict:
            continue
        item = item_dict[space.id]
        area = round(space.cached_area()) / (100 ** 2)
        min_area = item_dict[space.id].min_size.area / (100 ** 2) if item else "x"
        max_area = item_dict[space.id].max_size.area / (100 ** 2) if item else "x"
        ok = min_area <= area <= max_area if item else "x"
        msg = " • {} = {}: {} -> [{}, {}]: {} | {} - {}".format(space.id,
                                                                space.category.name, area, min_area,
                                                                max_area, "✅" if ok else "❌",
                                                                ind.fitness.sp_values[space.id],
                                                                ind.fitness.sp_wvalue[space.id])
        logging.info(msg)


__all__ = ['compose', 'score_perimeter_area_ratio', 'score_bounding_box', 'score_area',
           'score_corner', 'create_item_dict', 'check']
