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
from libs.space_planner.circulation import Circulator
from libs.plan.category import SPACE_CATEGORIES

if TYPE_CHECKING:
    from libs.specification.specification import Specification, Item
    from libs.refiner.core import Individual
    from libs.plan.plan import Space, Floor
    from libs.mesh.mesh import Edge

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
    mutable_spaces_id = {space.id for space in spec.plan.mutable_spaces()}
    output = {k: tuple(d.get(k, None) for d in (f(spec, ind) for f in funcs))
              for k in mutable_spaces_id}
    return output


def create_item_dict(_spec: 'Specification') -> Dict[int, Optional['Item']]:
    """
    Creates a 1-to-1 dict between spaces and items of the specification
    :param _spec:
    :return:
    """
    output = {}
    spec_items = _spec.items[:]
    for sp in _spec.plan.mutable_spaces():
        corresponding_items = list(filter(lambda i: i.category is sp.category, spec_items))
        # Note : corridors have no corresponding spec item
        best_item = min(corresponding_items,
                        key=lambda i: math.fabs(i.required_area - sp.cached_area()), default=None)
        if sp.category is not SPACE_CATEGORIES["circulation"]:
            assert best_item, "Score: Each space should have a corresponding item in the spec"
        output[sp.id] = best_item
        if best_item is not None:
            spec_items.remove(best_item)
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
        sp_score = (((min_area - space_area) / min_area) ** 2) * 100
    elif space_area > max_area:
        sp_score = (((space_area - max_area) / max_area) ** 2) * 100
    return sp_score


def score_area(spec: 'Specification', ind: 'Individual') -> Dict[int, float]:
    """
    Returns a score that evaluates the proximity of the individual to a specification
    instance
    :param spec:
    :param ind
    :return:
    """
    with_circulation = False
    min_bedroom_area = 90000
    min_size_penalty = 500

    # circulation space have no target areas in the spec
    excluded_spaces = (SPACE_CATEGORIES["circulation"],)
    space_to_item = ind.fitness.cache.get("space_to_item", None)

    # Create a corridor if needed and check which spaces are modified
    removed_areas = {}

    if with_circulation:
        circulator = Circulator(plan=ind, spec=spec)
        circulator.connect(space_to_item, _score_space_area)
        edge_paths: Dict[int, List[List['Edge']]] = circulator.paths['edge']
        edge_direction = circulator.directions

        modified_spaces = {}
        for level in ind.levels:
            for line in edge_paths[level]:
                for edge in line:
                    _edge = edge if edge_direction[level][edge] > 0 else edge.pair
                    space = ind.get_space_of_edge(_edge)
                    if not space:
                        logging.warning("Refiner: Path Edge has no space :%s", _edge)
                        continue
                    if space in modified_spaces:
                        modified_spaces[space].append(_edge)
                    else:
                        modified_spaces[space] = [_edge]

        # compute the removed area for each modified space
        # Note : the area of two rotated rectangle of same depth and sharing a corner is equal to :
        #         tan(90.0 - angle / 2 * pi / 180.0) * depth**2

        depth = 90.0
        for space in modified_spaces:
            previous_e = modified_spaces[space][0]
            removed_area = previous_e.length * depth
            for e in modified_spaces[space][1:]:
                if e is not space.next_edge(previous_e):
                    removed_area += e.length * depth
                    previous_e = e
                    continue
                shared_area = 0
                angle = space.next_angle(previous_e)
                if not pseudo_equal(angle, 180.0, 5.0):  # for performance purpose
                    shared_area = math.tan(math.pi/180 * (90.0 - angle / 2)) * depth**2
                removed_area += e.length * depth - shared_area
            removed_areas[space] = removed_area

    area_score = {}
    if spec is None:
        logging.warning("Refiner: score area: not spec specified")
        return {}

    for space in ind.mutable_spaces():
        if space.id not in ind.modified_spaces:
            continue
        if space.category in excluded_spaces:
            area_score[space.id] = _score_space_area(space.cached_area(), 10000, 10000) / 10.0
            continue
        item = space_to_item[space.id]
        space_area = space.cached_area()

        if with_circulation:
            space_area -= (removed_areas[space] if space in removed_areas else 0)

        if space.category is SPACE_CATEGORIES["bedroom"] and space_area < min_bedroom_area:
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
        score[space.id] = number_of_corners(space)
        if space.has_holes:
            score[space.id] += 4.0
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
    excluded_spaces = (SPACE_CATEGORIES["circulation"],)
    score = {}
    for space in ind.mutable_spaces():
        if space.id not in ind.modified_spaces:
            continue
        if space.category in excluded_spaces:
            score[space.id] = 0.0
            continue
        if space.cached_area() == 0:
            score[space.id] = 100
            continue
        box = space.bounding_box()
        box_area = box[0] * box[1]
        area = space.cached_area()
        space_score = ((area - box_area) / area)**2 * 100.0
        score[space.id] = space_score

    return score


def score_perimeter_area_ratio(_: 'Specification', ind: 'Individual') -> Dict[int, float]:
    """
    :param _
    :param ind:
    :return:
    """
    excluded_spaces = (SPACE_CATEGORIES["circulation"],)
    score = {}
    min_aspect_ratio = 16
    for space in ind.mutable_spaces():
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
    excluded_spaces = (SPACE_CATEGORIES["circulation"],)
    ratios = {
        "bedroom": 1.2,
        "toilet": 1.7,
        "bathroom": 1.2,
        "entrance": 1.7,
        "default": 1.5
    }
    score = {}
    for space in ind.mutable_spaces():
        if space.id not in ind.modified_spaces:
            continue
        if space.category in excluded_spaces:
            score[space.id] = 0
            continue
        box = space.bounding_box()
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
        if not space.edge:
            score[space.id] = 0.0
            continue
        if space.category not in circulation_categories:
            score[space.id] = 0.0
            continue
        if space.id not in ind.modified_spaces:
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


def check(ind: 'Individual', spec: 'Specification') -> None:
    """
    Compares the plan area with the specification objectives
    :param ind:
    :param spec:
    :return:
    """
    logging.info("Refiner: Checking Plan : %s", ind.name)

    item_dict = create_item_dict(spec)
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
