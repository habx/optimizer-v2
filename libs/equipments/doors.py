# coding=utf-8
"""
Door module
Puts doors in a plan

"""

import logging
from typing import List, Tuple
from shapely import geometry

from libs.plan.plan import Space, Plan, Edge, Linear, LINEAR_CATEGORIES
from libs.io.plot import plot_save

from libs.utils.geometry import (
    parallel,
    move_point,
    dot_product,
    ccw_angle
)

DOOR_WIDTH = 90
DOOR_WIDTH_TOLERANCE = 20
epsilon = 2


# TODO
# deal with intersection with other linears rather than doors only?
# DOOR_WIDTH_TOLERANCE should be set to a lower value, epsilon?
# more generic rule for openning inside/outside a room

def get_adjacent_circulation_spaces(space: 'Space') -> List['Space']:
    """
    get all circulation spaces adjacent to space with adjacent min adjacent length
    :param space:
    :return:
    """
    adjacent_spaces = [adj for adj in
                       space.adjacent_spaces(length=DOOR_WIDTH - DOOR_WIDTH_TOLERANCE)
                       if adj.category.circulation]
    return adjacent_spaces


###############################################

##### selection rules

def select_circulation_spaces(space: 'Space') -> List['Space']:
    """
    get all circulation spaces adjacent to space with adjacent min adjacent length
    :param space:
    :return:
    """
    return get_adjacent_circulation_spaces(space)


def select_preferential_circulation_space(space: 'Space') -> List['Space']:
    """
    get entrance if entrance is adjacent to space,
    else adjacent corridors
    else an adjacent circulation space if any
    :param space:
    :return:
    """
    adjacent_circulation_spaces = get_adjacent_circulation_spaces(space)
    if not adjacent_circulation_spaces:
        return []

    entrances = [sp for sp in adjacent_circulation_spaces if sp.category.name is 'entrance']
    if entrances:
        return entrances

    corridors = [sp for sp in adjacent_circulation_spaces if sp.category.name is 'circulation']
    if corridors:
        return corridors

    return [adjacent_circulation_spaces[0]]


def bathroom_proximity(space: 'Space') -> List['Space']:
    """
    selects circulation space adjacent to space and having maximum number of contact with bathrooms
    :param space:
    :return:
    """
    return room_proximity(space, "bathroom")


def bedroom_proximity(space: 'Space') -> List['Space']:
    """
    selects circulation space adjacent to space and having maximum number of contact with bedroom
    :param space:
    :return:
    """
    return room_proximity(space, "bedroom")


def room_proximity(space: 'Space', cat_name: str) -> List['Space']:
    """
    selects circulation space adjacent to space and having maximum number of contact with rooms of
    category name cat_name
    :param space:
    :param cat_name:
    :return:
    """

    def _get_nb_of_adjacent_cat(_space, _cat_name):
        return len([sp for sp in _space.adjacent_spaces() if sp.category.name is cat_name])

    adjacent_circulation_spaces = get_adjacent_circulation_spaces(space)
    if not adjacent_circulation_spaces:
        return []

    corridor = [sp for sp in adjacent_circulation_spaces if sp.category.name is 'circulation']
    entrance = [sp for sp in adjacent_circulation_spaces if sp.category.name is 'entrance']

    if not corridor and not entrance:
        return [adjacent_circulation_spaces[0]]
    if not corridor:
        return [entrance[0]]
    if not entrance:
        return [corridor[0]]

    if _get_nb_of_adjacent_cat(corridor[0], cat_name) >= _get_nb_of_adjacent_cat(entrance[0],
                                                                                 cat_name):
        return [corridor[0]]
    else:
        return [entrance[0]]


space_selection_rules = {
    "default_circulation": select_circulation_spaces,
    "default_non_circulation": select_preferential_circulation_space,
    "bedroom": bathroom_proximity,
    "bathroom": bedroom_proximity,
}


def place_doors(plan: 'Plan'):
    """
    Places the doors in the plan
    Process:
    -for each room, selection of the spaces the room has to open on
    -for each couple of space between which a door has to be set, selection of the optimal
    door position
    :param plan:
    :return:
    """

    def _open_space(_space: 'Space'):
        """
        place necessary doors on _space border
        :param _space:
        :return:
        """

        if _space.category.name == "entrance":
            return
        if _space.category.name == "circulation":
            return

        if _space.category.name in space_selection_rules:
            # rooms for which specific rules are designed
            list_openning_spaces = space_selection_rules[_space.category.name](_space)
        elif _space.category.circulation:
            list_openning_spaces = space_selection_rules["default_circulation"](_space)
        else:
            list_openning_spaces = space_selection_rules["default_non_circulation"](_space)

        for openning_space in list_openning_spaces:
            place_door_between_two_spaces(_space, openning_space)

    # treat mutable spaces in ascending area order - smallest spaces are are the most constrained
    mutable_spaces = sorted((sp for sp in plan.spaces if sp.mutable),
                            key=lambda x: x.area, reverse=True)

    for mutable_space in mutable_spaces:
        _open_space(mutable_space)


###############################################

##### door rules

def along_corner(contact_line: List['Edge'], space: 'Space', start: bool = True) -> bool:
    """
    checks that the door would open along a wall or not
    :param contact_line:
    :param space:
    :param start:
    :return:
    """
    sp = space.plan.get_space_of_edge(contact_line[0])
    if start:
        if ccw_angle(contact_line[0].vector, sp.previous_edge(contact_line[0]).vector) > 180:
            return True
        return False
    else:
        if ccw_angle(sp.next_edge(contact_line[-1]).vector, contact_line[-1].vector) > 180:
            return True
        return False


def door_space(contact_line: List['Edge'], space: 'Space', start: bool = True) -> bool:
    """
    checks the door can open on 90 deg without intersecting another door or a wall
    :param contact_line:
    :param space:
    :param start:
    :return:
    """

    def _get_linear_poly(_start_point: Tuple, _end_point: Tuple):
        linear_vect = [_end_point[0] - _start_point[0], _end_point[1] - _start_point[1]]
        linear_vect_ortho = [-linear_vect[1], linear_vect[0]]
        poly_points = [_start_point,
                       _end_point,
                       move_point(_end_point, linear_vect_ortho, 1),
                       move_point(_start_point, linear_vect_ortho, 1),
                       ]
        poly = geometry.Polygon([[p[0], p[1]] for p in poly_points])
        # return poly.buffer(-epsilon)
        #TODO : a buffer of poly would be more adapted
        return poly.centroid.buffer(DOOR_WIDTH / 3)
        # return poly

    door_vect = contact_line[0].unit_vector
    if start:
        start_point = contact_line[0].start.coords
        end_point = move_point(start_point, door_vect, DOOR_WIDTH)
    else:
        end_point = contact_line[-1].end.coords
        start_point = move_point(end_point, door_vect, -DOOR_WIDTH)
    door_poly = _get_linear_poly(start_point, end_point)
    door_poly_reverse = _get_linear_poly(end_point, start_point)

    sp_door = space.plan.get_space_of_edge(contact_line[0])
    sp_door_pair = space.plan.get_space_of_edge(contact_line[0].pair)

    if not sp_door_pair.as_sp.contains(door_poly_reverse):
        # not possible to access the door
        return False
    if not sp_door.as_sp.contains(door_poly):
        # the door cannot completely open in the reception space
        return False

    # checks that the door does not intersect another door
    other_doors = [linear for linear in sp_door.plan.linears if
                   linear.category.name is 'door' and sp_door.has_linear(linear)]
    for other_door in other_doors:
        linear_poly = _get_linear_poly(list(other_door.edges)[0].start.coords,
                                       list(other_door.edges)[-1].end.coords)
        if linear_poly.intersects(door_poly):
            # the door intersects another door
            return False
    return True


# TODO :
# def distant_from_linear(contact_line: List['Edge'], space: 'Space', start: bool = True) -> bool:
#     return True


def door_width(contact_line: List['Edge'], *_):
    """
    checks there is contact_line is long enough for a door to be placed
    :param contact_line:
    :param _:
    :return:
    """
    length = sum(e.length for e in contact_line)
    return length > DOOR_WIDTH - epsilon


door_position_rules = {
    # "imperative": [door_space, door_width],
    "imperative": [door_width, door_space],
    # "non_imperative": [along_corner, distant_from_linear],
    "non_imperative": [along_corner]
    # "non_imperative": []
}


def get_door_edges(contact_line: List['Edge'], start: bool = True) -> List['Edge']:
    """
    determines edges of contact_line that will be door linears
    The output list, door_edges, is a list of contiguous edges
    A door has width DOOR_WIDTH unless the length of contact_line is smaller
    :param contact_line:
    :param start:
    :return:
    """

    def _is_edge_of_point(_edge: 'Edge', _point: Tuple):
        """
        checks if point is on the segment defined by edge
        assumes _point belongs to the line defined by _edge
        :param _edge:
        :param _point:
        :return:
        """
        vect1 = (_point[0] - _edge.start.coords[0], _point[1] - _edge.start.coords[1])
        vect2 = (_point[0] - _edge.end.coords[0], _point[1] - _edge.end.coords[1])
        return dot_product(vect1, vect2) <= 0

    if not start:
        contact_line = [e.pair for e in contact_line]
        contact_line.reverse()

    # determines door edges
    if contact_line[0].length > DOOR_WIDTH - epsilon:  # deal with snapping
        end_edge = contact_line[0]
    else:
        end_door_point = move_point(contact_line[0].start.coords,
                                    contact_line[0].unit_vector,
                                    DOOR_WIDTH)
        end_edge = list(e for e in contact_line if _is_edge_of_point(e, end_door_point))[0]
    end_index = [i for i in range(len(contact_line)) if contact_line[i] is end_edge][0]
    door_edges = contact_line[:end_index + 1]

    # splits door_edges[-1] if needed, so as to get a proper door width
    end_split_coeff = (DOOR_WIDTH - end_edge.start.distance_to(
        contact_line[0].start)) / end_edge.length
    if not 1 >= end_split_coeff >= 0:
        end_split_coeff = 0 * (end_split_coeff < 0) + (end_split_coeff > 1)
    if end_edge.length > end_split_coeff * end_edge.length > 1:
        door_edges[-1] = end_edge.split_barycenter(end_split_coeff).previous

    if not start:
        door_edges = [e.pair for e in door_edges]
        door_edges.reverse()

    return door_edges


def get_door_position(space: 'Space', lines: List[List['Edge']]) -> Tuple:
    """
    gets the contact portion between both space where the door will stand, and whether the door
    is at the beginning or end of this portion
    :param space:
    :param lines:
    :return:
    """

    def _get_portion_score(_space: 'Space', _line: List['Edge'], _start: bool) -> int:
        """
        gets the score of the contact portion
        :param _space:
        :param _line:
        :param _start:
        :return:
        """
        score = 0
        for score_func in door_position_rules["imperative"]:
            if not score_func(_line, _space, _start):
                # if an imperative constraint is not satisfied, zero score
                return 0
        score += sum(
            score_func(_line, _space, _start) for score_func in
            door_position_rules["non_imperative"])
        return score

    def _kept_portion(_space: Space, _lines: List[List['Edge']], start: bool) -> Tuple:
        """
        selection of the best contact portion to place door
        :param _space:
        :param _lines:
        :param start:
        :return:
        """
        score = 0
        line = _lines[0]
        for _l, _line in enumerate(_lines):
            current_score = _get_portion_score(_space, _line, start)
            if current_score > score:
                line = _line
                score = current_score
        return line, score

    longest_line = sorted(lines, key=lambda x: sum(e.length for e in x))[-1]
    longest_length = sum(e.length for e in longest_line)
    if longest_length <= DOOR_WIDTH:
        # no optimal placement
        return longest_line, True

    sorted_lines = sorted(lines, key=lambda x: sum(e.length for e in x))
    line_start, score_start = _kept_portion(space, sorted_lines, start=True)
    line_end, score_end = _kept_portion(space, sorted_lines, start=False)

    if score_end == score_start == 0:
        # no optimal placement
        return longest_line, True

    if score_end > score_start:
        return line_end, False

    return line_start, True


def place_door_between_two_spaces(space: 'Space', circulation_space: 'Space'):
    """
    places a door between space and circulation_space
    process :
    gets the straight contact portions between both spaces
    :param space:
    :param circulation_space:
    :return:
    """

    # gets contact edges between both spaces
    contact_edges = [edge for edge in space.edges if edge.pair in circulation_space.edges]

    # reorders contact_edges
    start_index = 0
    for i, edge in enumerate(contact_edges):
        if not space.previous_edge(edge) in contact_edges:
            start_index = i
            break
    contact_edges = contact_edges[start_index:] + contact_edges[:start_index]

    # gets the longest contact straight portion between both spaces
    lines = [[contact_edges[0]]]
    for edge in contact_edges[1:]:
        if parallel(lines[-1][-1].vector, edge.vector) and edge.start is lines[-1][-1].end:
            lines[-1].append(edge)
        else:
            lines.append([edge])

    # door arbitrarily opens on the inside for toilets and bathroom
    # TODO : more generic rule should be applied
    inside = False if space.category.name in ["toilet", "bathroom"] else True
    if not inside:
        for l, line in enumerate(lines):
            lines[l] = [e.pair for e in reversed(line)]

    contact_line, start = get_door_position(space, lines)
    contact_length = contact_line[0].start.distance_to(contact_line[-1].end)

    if contact_length < DOOR_WIDTH:
        # the door is placed on the whole portion
        door_edges = contact_line
    else:
        door_edges = get_door_edges(contact_line[:], start=start)

    # set linear
    door = Linear(space.plan, space.floor, door_edges[0], LINEAR_CATEGORIES["door"])

    if len(door_edges) == 1:
        return
    for door_edge in door_edges[1:]:
        door.add_edge(door_edge)


def door_plot(plan: 'Plan', save: bool = True):
    """
    plots plan with door
    :param plan:
    :param save:
    :return:
    """
    ax = plan.plot(save=False)
    number_of_levels = plan.floor_count
    for level in range(number_of_levels):
        _ax = ax[level] if number_of_levels > 1 else ax
        for linear in plan.linears:
            if linear.category.name == "door":
                start_edge = list(linear.edges)[0]
                sp = plan.get_space_of_edge(start_edge)
                if not parallel(start_edge.vector, sp.previous_edge(start_edge).vector):
                    start_door_point = list(linear.edges)[0].start.coords
                    end_door_point = list(linear.edges)[-1].end.coords
                else:
                    start_door_point = list(linear.edges)[-1].end.coords
                    end_door_point = list(linear.edges)[0].start.coords

                door_vect = (end_door_point[0] - start_door_point[0],
                             end_door_point[1] - start_door_point[1])
                door_vect_ortho = start_edge.normal
                door_vect_ortho = tuple([DOOR_WIDTH * x for x in door_vect_ortho])

                pt_end = (start_door_point[0] + 0.5 * (door_vect[0] + door_vect_ortho[0]),
                          start_door_point[1] + 0.5 * (door_vect[1] + door_vect_ortho[1]))
                _ax.arrow(start_door_point[0], start_door_point[1],
                          pt_end[0] - start_door_point[0],
                          pt_end[1] - start_door_point[1])

    plot_save(save)


if __name__ == '__main__':
    import argparse
    from libs.modelers.grid import GRIDS
    from libs.modelers.seed import SEEDERS
    from libs.modelers.corridor import Corridor, CORRIDOR_BUILDING_RULES
    from libs.specification.specification import Specification

    # logging.getLogger().setLevel(logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--plan_index", help="choose plan index",
                        default=1)

    args = parser.parse_args()
    plan_index = int(args.plan_index)

    plan_name = None
    if plan_index < 10:
        plan_name = '00' + str(plan_index) + ".json"
    elif 10 <= plan_index < 100:
        plan_name = '0' + str(plan_index) + ".json"


    def get_plan(input_file: str = "001.json") -> Tuple['Plan', 'Specification']:

        import libs.io.reader as reader
        import libs.io.writer as writer
        from libs.space_planner.space_planner import SPACE_PLANNERS
        from libs.io.reader import DEFAULT_PLANS_OUTPUT_FOLDER

        folder = DEFAULT_PLANS_OUTPUT_FOLDER

        spec_file_name = input_file[:-5] + "_setup0"
        plan_file_name = input_file

        try:
            new_serialized_data = reader.get_plan_from_json(input_file)
            plan = Plan(input_file[:-5]).deserialize(new_serialized_data)
            spec_dict = reader.get_json_from_file(spec_file_name + ".json",
                                                  folder)
            spec = reader.create_specification_from_data(spec_dict, "new")
            spec.plan = plan
            return plan, spec

        except FileNotFoundError:
            plan = reader.create_plan_from_file(input_file)
            spec = reader.create_specification_from_file(input_file[:-5] + "_setup0" + ".json")

            GRIDS["002"].apply_to(plan)
            # GRIDS['optimal_finer_grid'].apply_to(plan)
            SEEDERS["directional_seeder"].apply_to(plan)
            spec.plan = plan

            space_planner = SPACE_PLANNERS["standard_space_planner"]
            best_solutions = space_planner.apply_to(spec, 3)

            new_spec = space_planner.spec

            if best_solutions:
                solution = best_solutions[0]
                plan = solution.plan
                new_spec.plan = plan
                writer.save_plan_as_json(plan.serialize(), plan_file_name)
                writer.save_as_json(new_spec.serialize(), folder, spec_file_name + ".json")
                return plan, new_spec
            else:
                logging.info("No solution for this plan")


    def main(input_file: str):

        out = get_plan(input_file)
        plan = out[0]
        spec = out[1]
        plan.name = input_file[:-5]

        corridor = Corridor(corridor_rules=CORRIDOR_BUILDING_RULES["no_cut"]["corridor_rules"],
                            growth_method=CORRIDOR_BUILDING_RULES["no_cut"]["growth_method"])
        corridor.apply_to(plan, spec=spec, show=False)

        print("ENTER DOOR PROCESS")
        bool_place_single_door = False
        if bool_place_single_door:
            cat1 = "bathroom"
            cat2 = "circulation"
            space1 = list(sp for sp in plan.spaces if
                          sp.category.name == cat1)[0]
            space2 = list(sp for sp in plan.spaces if
                          sp.category.name == cat2 and sp in space1.adjacent_spaces())[0]

            place_door_between_two_spaces(space1, space2)
        else:
            place_doors(plan)
        # plan.plot()
        door_plot(plan)


    plan_name = "003.json"
    main(input_file=plan_name)
