# coding=utf-8
"""
Door module
Puts doors in a plan

"""

import logging
from typing import List, Tuple

from libs.plan.plan import Space, Plan, Edge, Linear, LINEAR_CATEGORIES
from libs.io.plot import plot_save

from libs.utils.geometry import (
    parallel,
    move_point,
    dot_product,
    ccw_angle
)

DOOR_WIDTH = 90
epsilon = 2


# TODO
# -deal with intersecting doors
# =>starts with smallest rooms and when adding new doors check for existing linears for intersection check
# deal with door intersectnig other linears
# checks there is enough space to open the door
# some more rules
# if we have the choice, opens room on entrance rather than corridor except for water rooms :
# to be opened as close as possible to the doors


def place_doors(plan: 'Plan'):
    """
    Places the doors in the plan
    Process:
    -for circulation spaces,
        *place doors between the space and adjacent corridors if any
        *place a door between the space and the entrance if it is adjacent
        *else place a door with any adjacent circulation space
    -for non circulation spaces:
        *place a door between the space and a corridor if any
        *else place a door between the space and the entrance if it is adjacent
        *else place a door with any adjacent circulation space
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

        adjacent_circulation_spaces = [adj for adj in _space.adjacent_spaces() if
                                       adj.category.circulation]

        circulation_spaces = [sp for sp in adjacent_circulation_spaces if
                              sp.category.name in ["circulation"]]
        entrance_spaces = [sp for sp in adjacent_circulation_spaces if
                           sp.category.name in ["entrance"]]

        if _space.category.circulation:
            for entrance_space in entrance_spaces:
                place_door_between_two_spaces(_space, entrance_space)
            for circulation_space in circulation_spaces:
                place_door_between_two_spaces(_space, circulation_space)
            if not entrance_spaces and not circulation_spaces and adjacent_circulation_spaces:
                place_door_between_two_spaces(_space, adjacent_circulation_spaces[0])
        else:
            for circulation_space in circulation_spaces:
                place_door_between_two_spaces(_space, circulation_space)
                return
            for entrance_space in entrance_spaces:
                place_door_between_two_spaces(_space, entrance_space)
                return
            if adjacent_circulation_spaces:
                place_door_between_two_spaces(_space, adjacent_circulation_spaces[0])

    mutable_spaces = [sp for sp in plan.spaces if sp.mutable]
    for mutable_space in mutable_spaces:
        _open_space(mutable_space)


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
        return dot_product(vect1, vect2) < 0

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
    door_edges[-1] = end_edge.split_barycenter(end_split_coeff).previous

    if not start:
        door_edges = [e.pair for e in door_edges]
        door_edges.reverse()

    return door_edges


def place_door_between_two_spaces(space: 'Space', circulation_space: 'Space'):
    """
    places a door between space1 and space2

    :param space:
    :param circulation_space:
    :return:
    """

    def _check_corner(_contact_line: List['Edge'], start: bool = True) -> bool:
        """
        checks that the door would open along a wall
        :param _contact_line:
        :param start:
        :return:
        """
        sp = space.plan.get_space_of_edge(_contact_line[0])
        if start:
            if ccw_angle(_contact_line[0].vector, sp.previous_edge(_contact_line[0]).vector) > 180:
                return True
            return False
        else:
            if ccw_angle(sp.next_edge(_contact_line[-1]).vector, _contact_line[-1].vector):
                return True
            return False

    def _check_intersection(_contact_line: List['Edge'], start: bool = True) -> bool:
        """
        checks the door would not intersect with an existing linear when opening
        :param _contact_line:
        :param start:
        :return:
        """
        return True

    def _get_door_position(_lines: List[List['Edge']]) -> Tuple:
        """
        gets the contact portion between both space where the door will stand, and if
        :param _lines:
        :return:
        """
        longest_line = sorted(_lines, key=lambda x: sum(e.length for e in x))[-1]
        longest_length = sum(e.length for e in longest_line)
        if longest_length <= DOOR_WIDTH:
            return longest_line, True

        sorted_lines = (sorted(_lines, key=lambda x: sum(e.length for e in x), reverse=True))
        for l, line in enumerate(sorted_lines):
            line_length = sum(e.length for e in line)
            if line_length < DOOR_WIDTH:
                continue
            if _check_corner(line, start=True) and _check_intersection(line, start=True):
                return sorted_lines[l], True and _check_intersection(line, start=True)
            if _check_corner(line, start=False):
                return sorted_lines[l], False

        return sorted_lines[0], True

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

    inside = False if space.category.name in ["toilet", "bathroom"] else True
    if not inside:
        for l, line in enumerate(lines):
            lines[l] = [e.pair for e in reversed(line)]

    # contact_line = sorted(lines, key=lambda x: sum(e.length for e in x))[-1]
    contact_line, start = _get_door_position(lines)
    contact_length = contact_line[0].start.distance_to(contact_line[-1].end)

    if contact_length < DOOR_WIDTH:
        door_edges = contact_line
    else:
        door_edges = get_door_edges(contact_line[:], start=start)

    # if space.category.name in ["toilet", "bathroom"]:
    #    door_edges = [d_e.pair for d_e in reversed(door_edges)]

    # set linear
    door = Linear(space.plan, space.floor, door_edges[0], LINEAR_CATEGORIES["door"])
    if len(door_edges) == 1:
        return
    for door_edge in door_edges[1:]:
        door.add_edge(door_edge)


def plot(plan: 'Plan', save: bool = True):
    """
    plots plan with doors
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

        # TODO : à reprendre
        # * 61 : wrong corridor shape

        out = get_plan(input_file)
        plan = out[0]
        spec = out[1]
        plan.name = input_file[:-5]

        # corridor = Corridor(layer_width=25, nb_layer=5)

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
        plot(plan)


    plan_name = "062.json"
    main(input_file=plan_name)
