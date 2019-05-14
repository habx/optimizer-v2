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
from typing import TYPE_CHECKING, List, Callable, Dict, Optional, Tuple

from libs.utils.geometry import ccw_angle, pseudo_equal, min_section

if TYPE_CHECKING:
    from libs.specification.specification import Specification, Item
    from libs.refiner.core import Individual
    from libs.plan.plan import Space

# a score function returns a specific score for each space of the plan. It takes
# as arguments a specification object and an individual
scoreFunc = Callable[['Specification', 'Individual'], Dict[int, float]]


def compose(funcs: List[scoreFunc],
            spec: 'Specification', ind: 'Individual') -> Dict[int, Tuple[float, ...]]:
    """
    A factory to compose evaluation function from a list of score function
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


def score_area(spec: 'Specification', ind: 'Individual') -> Dict[int, float]:
    """
    Returns a score that evaluates the proximity of the individual to a specification
    instance
    :param spec:
    :param ind
    :return:
    """
    excluded_spaces = ("circulation",)  # circulation space have no target areas in the spec
    space_to_item = ind.fitness.cache.get("space_to_item", None)

    if space_to_item is None:
        logging.debug("Refiner: Score: computing space_to_item dictionary")
        space_to_item = create_item_dict(spec)
        ind.fitness.cache["space_to_item"] = space_to_item

    # create the item_to_space dict
    area_score = {}
    if spec is None:
        logging.warning("Refiner: score area: not spec specified")
        return {}

    for space in ind.mutable_spaces():
        if space.id not in ind.modified_spaces:
            continue
        if space.category.name in excluded_spaces:
            area_score[space.id] = 0.0
            continue
        item = space_to_item[space.id]
        space_area = space.cached_area()
        if space_area < item.min_size.area:
            space_score = math.fabs((space_area - item.min_size.area)/space_area)*100
        elif space_area > item.max_size.area:
            space_score = math.fabs((space_area - item.max_size.area)/space_area)*100
        else:
            space_score = 0
        area_score[space.id] = space_score

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
        if space.category.name in excluded_spaces:
            score[space.id] = 0.0
            continue
        score[space.id] = number_of_corners(space)
    return score


def number_of_corners(space: 'Space') -> int:
    """
    Returns the number of "inside" corners. The corners of the boundary edges of a space
    that are not adjacent to an external space.
    :param space:
    :return:
    """
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
    excluded_spaces = ("circulation",)
    score = {}
    for space in ind.mutable_spaces():
        if space.id not in ind.modified_spaces:
            continue
        if space.category.name in excluded_spaces:
            score[space.id] = 0.0
            continue
        if space.cached_area() == 0:
            score[space.id] = 100
            continue
        box = space.bounding_box()
        box_area = box[0] * box[1]
        area = space.cached_area()
        space_score = math.fabs((area - box_area) / area) * 100.0
        score[space.id] = space_score

    return score


def score_aspect_ratio(_: 'Specification', ind: 'Individual') -> Dict[int, float]:
    """
    :param _
    :param ind:
    :return:
    """
    excluded_spaces = ("living", "livingKitchen", "circulation")
    score = {}
    min_aspect_ratio = 16
    for space in ind.mutable_spaces():
        if space.id not in ind.modified_spaces:
            continue
        if space.category.name in excluded_spaces:
            score[space.id] = 0
            continue
        if space.cached_area() == 0:
            score[space.id] = 100
            continue
        space_score = math.fabs(space.perimeter**2/space.cached_area()/min_aspect_ratio - 1.0)
        score[space.id] = space_score

    return score


def score_connectivity(_: 'Specification', ind: 'Individual') -> Dict[int, float]:
    """
    Returns a score indicating whether the plan is connected
    :param _:
    :param ind:
    :return:
    """
    min_length = 90.0
    cost_per_unconnected_space = 1.0
    score = {}

    front_door = list(ind.get_linears("frontDoor"))[0]
    front_door_space = ind.get_space_of_edge(front_door.edge)
    connected_circulation_spaces = [front_door_space]
    score[front_door_space.id] = 0.0

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
                    score[circulation_space.id] = 0.0
                    break

    for unconnected_space in circulation_spaces:
        score[unconnected_space.id] = cost_per_unconnected_space

    for space in ind.mutable_spaces():
        if space.category.circulation or space is front_door_space:
            continue

        found_adjacency = False
        for edge in space.exterior_edges:
            if edge.pair.face is None:
                continue
            other = ind.get_space_of_edge(edge.pair)
            if not other or other not in connected_circulation_spaces:
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

        score[space.id] = cost_per_unconnected_space if not found_adjacency else 0.0

    return score


def score_corridor_width(_: 'Specification', ind: 'Individual') -> Dict[int, float]:
    """
    Computes a score of the min width of each corridor
    :param _:
    :param ind:
    :return:
    """
    min_width = 90.0
    score = {}
    corridors = ind.get_spaces("circulation")
    for corridor in corridors:
        if corridor.id not in ind.modified_spaces:
            continue
        polygon = corridor.boundary_polygon()
        width = min_section(polygon)
        score[corridor.id] = ((width - min_width)/min_width)**2

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


__all__ = ['compose', 'score_aspect_ratio', 'score_bounding_box', 'score_area', 'score_corner',
           'create_item_dict', 'check']
