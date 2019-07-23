from typing import Optional, Dict, List, Type, Callable
from enum import Enum

import matplotlib.pyplot as plt

from libs.plan.plan import Plan, Space, Face, Edge
from libs.space_planner.solution import Solution
from libs.space_planner.circulation import Circulator, CostRules
from libs.specification.specification import Specification
from libs.plan.category import SPACE_CATEGORIES
from libs.io.plot import Plot
from libs.utils.geometry import (
    ccw_angle,
    pseudo_equal,
    parallel
)

GrowCorridor = Callable[['Corridor', List['Edge'], bool], 'Space']

EPSILON = 1
SMALL_CORRIDOR_AREA = 5000


class CorridorRules:
    def __init__(self,
                 width: float = 130,
                 penetration_length: float = 90):
        # used to get penetration with precise length
        self.width = width  # maximum width of the corridor
        self.penetration_length = penetration_length  # maximum penetration length, when needed


class Corridor:
    """
    Corridor Class

    Class to build circulation spaces between every isolated room space of the plan.

    If a plot is given to the corridor, it will use it to display in real time the changes occurring
    on the plan.
    """

    def __init__(self,
                 corridor_rules: CorridorRules,
                 growth_method: 'GrowCorridor',
                 circulation_cost_rules: Type[Enum] = CostRules,
                 plot: Optional['Plot'] = None):

        self.corridor_rules = corridor_rules
        self.circulation_cost_rules = circulation_cost_rules
        self.growth_method = growth_method
        self.plot = plot
        self.spec: Specification = None
        self.plan: Plan = None
        self.circulator: Circulator = None
        self.corner_data: Dict = None
        self.grouped_faces: Dict[int, List[List['Face']]] = None

    def _clear(self):
        self.plan = None
        self.spec = None
        self.circulator = None
        self.paths = []
        self.corner_data = {}
        self.grouped_faces = {}

    def apply_to(self, solution: 'Solution', show: bool = False):
        """
        Runs the corridor
        -creates a circulator and determines circulation paths in the plan
        -grows corridor spaces around those paths
        :param solution
        :param show: whether to display a real-time visualization of the corridor
        :return:
        """
        self._clear()
        self.spec = solution.spec
        self.plan = solution.spec.plan

        # store mutable spaces, for repair purpose
        initial_mutable_spaces = [sp for sp in self.plan.spaces if
                                  sp.mutable and not sp.category.name is "circulation"]
        dict_item = self._store_repair_info(initial_mutable_spaces, solution)

        # computes circulation paths and stores them
        self.circulator = Circulator(plan=solution.spec.plan, spec=solution.spec,
                                     cost_rules=self.circulation_cost_rules)

        self.circulator.connect()
        # self.circulator.plot()
        self._set_paths()

        # Real time plot updates
        if show:
            self._initialize_plot()

        # grow corridor spaces around circulation paths
        for path in self.paths:
            self.grow(path, show)

        # merging corridor spaces when needed
        self._merge_corridors()

        # space repair process : if some spaces have been cut by corridor growth
        final_mutable_spaces = [sp for sp in self.plan.spaces if
                                sp.mutable and not sp.category.name is "circulation"]
        self._repair_spaces(initial_mutable_spaces, final_mutable_spaces)

        # reconstruct dict of items - required if some space has disappeared in the process
        self._reconstruct_item_dict(solution, dict_item)

    def _store_repair_info(self, initial_mutable_spaces: List['Space'], solution: 'Solution'):
        """
        stores groups of faces that belong to non mutable spaces, for repair purpose
        :param initial_mutable_spaces:
        :param solution:
        :return:
        """
        grouped_faces = {level: [] for level in self.plan.levels}
        dict_item = {}
        for sp in initial_mutable_spaces:
            sp_faces = [f for f in sp.faces]
            grouped_faces[sp.floor.level].append(sp_faces)
            item = [solution.space_item[s] for s in initial_mutable_spaces if s.id is sp.id][0]
            dict_item[item] = sp_faces
        self.grouped_faces = grouped_faces
        return dict_item

    def _reconstruct_item_dict(self, solution: 'Solution', dict_item: 'Dict'):
        """
        some spaces may have disappeared in the corridor process
        for further purposes, reconstruct solution.space_item
        :param solution:
        :param dict_item:
        :return:
        """
        solution.space_item = {}
        for item in dict_item:
            group_faces_item = dict_item[item]
            for f in group_faces_item:
                space_f = self.plan.get_space_of_face(f)
                if space_f and space_f.category is not SPACE_CATEGORIES['circulation']:
                    solution.space_item[space_f] = item
                    break

    def _merge_corridors(self):
        """
        merges corridors spaces
        :return:
        """
        self._rectangular_merge()
        self._small_space_merge()
        self.plan.remove_null_spaces()

    def _small_space_merge(self):
        """
        merges small corridor spaces with adjacent corridor space (if any) that has maximal contact
        :return:
        """
        corridors = list(self.plan.get_spaces("circulation"))
        small_corridors = [corridor for corridor in corridors
                           if corridor.area < SMALL_CORRIDOR_AREA]

        merge = True if small_corridors else False
        while merge:
            merge = False
            for small_corridor in small_corridors:
                adjacent_corridors = (adj for adj in small_corridor.adjacent_spaces() if
                                      adj in corridors)
                if not adjacent_corridors:
                    continue

                # among adjacent corridors gets the one with maximal contact length
                contact_length = 0
                adjacent_corridor_selected = None
                for adjacent_corridor in adjacent_corridors:
                    current_contact_length = adjacent_corridor.contact_length(small_corridor)
                    if current_contact_length > contact_length:
                        contact_length = current_contact_length
                        adjacent_corridor_selected = adjacent_corridor
                if not adjacent_corridor_selected:
                    continue
                adjacent_corridor_selected.merge(small_corridor)
                small_corridors.remove(small_corridor)
                merge = True
                break

    def _rectangular_merge(self):
        """
        merges corridor spaces when the merge is a rectangle
        purpose : ease the refiner process
        :return:
        """

        corridors = list(self.plan.get_spaces("circulation"))

        merge = True if corridors else False
        while merge:
            merge = False
            for corridor in corridors:
                adjacent_corridors = (adj for adj in corridor.adjacent_spaces() if adj in corridors)
                for adjacent_corridor in adjacent_corridors:
                    if corridor.number_of_corners(adjacent_corridor) == 4:
                        corridor.merge(adjacent_corridor)
                        corridors.remove(adjacent_corridor)
                        merge = True
                        break
                if merge:
                    break

    def _repair_spaces(self, initial_mutable_spaces: List['Space'],
                       final_mutable_spaces: List['Space']):
        """
        if in the process a mutable space has been split by a corridor propagation,
        the split space is set back to its state before corridor propagation
        This may break a corridor into pieces.
        :param initial_mutable_spaces: list of spaces before corridor propagation
        :param final_mutable_spaces: list of faces after corridor propagation
        :return:
        """

        def _get_group_face(_level: int, _face: 'Face') -> List['Face']:
            # get the group of faces _face belongs to
            for group in self.grouped_faces[_level]:
                if _face in group:
                    return group

        if len(initial_mutable_spaces) == len(final_mutable_spaces):
            # no space has been split
            return

        merge = True
        grouped_faces = []
        for level in self.plan.levels:
            grouped_faces += self.grouped_faces[level]
        while merge:
            for grouped_face in grouped_faces:
                # check if the group is distributed within several spaces (other than corridor)
                # in which case we proceed to repair
                l = [sp for sp in self.plan.spaces if not set(grouped_face).isdisjoint(
                    list(sp.faces)) and not sp.category.name == "circulation"]
                if len(l) > 1:  # a space has been split by corridor propagation
                    repair_space = l[0]
                    repair_faces = _get_group_face(repair_space.floor.level, repair_space.face)
                    while len(list(repair_space.faces)) != len(repair_faces):
                        for face in repair_faces:
                            space_of_face = self.plan.get_space_of_face(face)
                            if space_of_face is repair_space:
                                continue
                            # adds face if adjacent to repair_space
                            if [e for e in face.edges if
                                e.pair.face and repair_space.has_face(e.pair.face)]:
                                repair_space.add_face(face)
                                # repair_faces.remove(face)
                                space_of_face.remove_face(face)
                                break
                    self.plan.remove_null_spaces()
                    final_mutable_spaces = [sp for sp in self.plan.spaces if
                                            sp.mutable and not sp.category.name is "circulation"]
                    break

            if len(initial_mutable_spaces) == len(final_mutable_spaces):
                merge = False

    def _set_paths(self):
        """
        Possibly adds edges at the beginning and end of the path to account for
        corridor penetration within the room.
        :param:
        :return:
        """

        for path_info in self.circulator.paths_info:
            current_path = [t[0] for t in path_info.edge_path]
            if not current_path:
                continue
            if path_info.departure_penetration:
                current_path = self._add_penetration_edges(current_path)
            if path_info.arrival_penetration:
                current_path = self._add_penetration_edges(current_path,
                                                           start=False)
            self._update_growing_direction(current_path)
            self.paths.append(current_path)

    def _add_penetration_edges(self, edge_list: List['Edge'], start: bool = True):
        """
        Possibly adds edges at the beginning and end of the path
        to account for corridor penetration within the room.
        A path has to penetrate a room if following conditions are satisfied
            -it extends on the room border, not inside the room space
            -it is not on the plan border
            -it is not along a load bearing wall
        When penetration conditions are satisfied, the penetration shall have a length equal to
        penetration_length
        :param List['Edge']: ordered list of vertices forming a circulation path
        :return:
        """

        penetration_length = self.corridor_rules.penetration_length

        def _add_edges(_edge_list: List['Edge']):
            """
            Adds edges to the list, at the beginning (if start) or end if
            penetration condition are satisfied until penetration length is reached
            :param _edge_list: ordered list of vertices forming a circulation path
            :return:
            """

            l = 0  # penetration length
            continue_penetration = True
            while l < penetration_length and continue_penetration:
                limit_edge = _edge_list[0].pair if start else _edge_list[-1]
                limit_vertex = limit_edge.end
                penetration_edges = [edge for edge in limit_vertex.edges if
                                     edge.face and edge.pair.face]
                for edge in penetration_edges:
                    if parallel(edge.vector,
                                limit_edge.vector):  # and _penetration_condition(edge):
                        penetration_edge = edge
                        if l + penetration_edge.length > penetration_length:
                            continue_penetration = False
                        else:
                            _edge_list = [penetration_edge.pair] + _edge_list if start \
                                else _edge_list + [penetration_edge]
                            l += penetration_edge.length
                        break
                else:
                    continue_penetration = False

            return _edge_list

        edge_list = _add_edges(edge_list)

        return edge_list

    def _update_growing_direction(self, path: List['Edge']):
        """
        Some edges of the circulation path can be split in the slicing process.
        The path then contains new edges for which growing direction information need to be set
        :param path:
        :return:
        """

        def _get_neighbor_direction(_edge, _path):
            # for _e in self.circulator.directions[level]:
            for _e in _path:
                if _e in self.circulator.directions[level] and parallel(_e.vector, _edge.vector):
                    return self.circulator.directions[level][_e]

        if self.plan.get_space_of_edge(path[0]):
            level = self.plan.get_space_of_edge(path[0]).floor.level
        else:
            level = self.plan.get_space_of_edge(path[0].pair).floor.level
        for e, edge in enumerate(path):
            if edge not in self.circulator.directions[level]:
                self.circulator.directions[level][edge] = _get_neighbor_direction(edge, path)

    def _update_path(self, path: List['Edge']):
        """
        Some edges of the circulation path can be split in the slicing process.
        The path then has to be updated with new resulting edges.
        For those new edges, a growing direction has to be set
        :param path:
        :return:
        """

        def _line_forward(_edge: 'Edge') -> List['Edge']:
            """
            returns edges aligned with e, contiguous, in forward direction
            :param _edge:
            :return:
            """
            output = []
            current = _edge
            while current:
                output.append(current)
                current = current.aligned_edge or current.continuous_edge
            return output[1:]

        def _repair_path(_start_edge: 'Edge', _end_edge: 'Edge'):
            """
            gets the list of edges aligned with _start_edge and ending with an edge linked to
            _end_edge
            :param _start_edge:
            :param _end_edge:
            :return:
            """
            line = _line_forward(_start_edge)
            _added_edges = []
            for _e in line:
                _added_edges.append(_e)
                if _e.end is _end_edge.start:
                    break
            else:
                assert _start_edge, "circulation path could not be repaired"

            return _added_edges

        if not path:
            return path

        repair = True
        while repair:
            for e, edge in enumerate(path[:-1]):
                if edge.end is not path[e + 1].start:
                    added_edges = _repair_path(edge, path[e + 1])
                    path[e + 1:e + 1] = added_edges
                    break
            else:
                repair = False

        self._update_growing_direction(path)

        return path

    def _corner_fill(self, show: bool = False):
        """
        Fills corridor corners
        -path corner edges are stored in self.corner_data
        -finds edges aligned with path corner edges and selects those that
        are along a corridor space
        -adds corridor space portions along those edges
        :param show:
        :return:
        """

        def _condition(_edge: 'Edge'):
            """
            condition that shall be verified for a corridor space to grow from edge _edge
            :param _edge:
            :return:
            """
            if not self.plan.get_space_of_edge(_edge):
                return False
            if not self.plan.get_space_of_edge(_edge).category is SPACE_CATEGORIES['circulation']:
                return False
            return True

        def _line_forward(_edge: 'Edge') -> List['Edge']:
            """
            returns edges aligned with e, contiguous, in forward direction
            :param _edge:
            :return:
            """
            output = []
            current = _edge
            while current:
                output.append(current)
                current = current.aligned_edge or current.continuous_edge
            return output[1:]

        for edge in self.corner_data:
            corner_edge = edge if self.corner_data[edge] > 0 else edge.pair
            if self.plan.get_space_of_edge(corner_edge):
                floor = self.plan.get_space_of_edge(corner_edge).floor
            else:
                continue

            line = []
            for line_edge in _line_forward(edge):
                # line_edge contains edges along wich a corridor space will grow to fill a corner
                if ((_condition(line_edge))
                        and sum([l_e.length for l_e in line]) < self.corridor_rules.width):
                    line.append(line_edge)
                else:
                    break
            for line_edge in line:
                support_edge = line_edge if self.corner_data[edge] > 0 else line_edge.pair
                corridor_space = Space(self.plan, floor, category=SPACE_CATEGORIES['circulation'])
                self.add_corridor_portion(support_edge, self.corridor_rules.width, corridor_space,
                                          show, corner=True)
                if not corridor_space:
                    break

    def grow(self, path: List['Edge'], show: bool = False) -> 'Corridor':
        """
        -Grows corridor spaces around the circulation path defined by path
        -Each straight corridor portion is treated separately, corners at this stage may not
        be properly treated
        -Fills corridor corners
        Merge built corridor spaces when they are adjacent
        :param path: ordered list of vertices forming a circulation path
        :param show:
        :return:
        """

        self._path_growth(path, show)

        self._corner_fill(show)

        self.plan.remove_null_spaces()

        return self

    def _path_growth(self, path: List['Edge'], show: bool = False):
        """
        Circulation path defined by path is decomposed into its straight portions
        A corridor space is grown around each portion
        :param path: ordered list of contiguous edges forming a circulation path
        :param show: ordered list of contiguous edges forming a circulation path
        :return:
        """

        edge_lines = self._get_straight_parts(path)
        for edge_line in edge_lines:
            self.growth_method(self, edge_line, show)

    @staticmethod
    def _get_straight_parts(path: List['Edge']) -> List[List['Edge']]:
        """
        decomposes the path into its straight sub-parts
        :param path: ordered list of contiguous edges forming a circulation path
        :return:
        """
        edge_lines = []
        edge_ini = path[0]
        l = []
        for e in path:
            if not parallel(edge_ini.vector, e.vector):
                edge_lines.append(l)
                edge_ini = e
                l = [e]
            else:
                l.append(e)
        edge_lines.append(l)
        return edge_lines

    def get_parallel_layers_edges(self, edge: 'Edge', width: 'float') -> 'List[Edge]':
        """
        Returns
        successive layer edges defined as
            -being parallel to edge
            -in vertical direction from edge
            -with distance to edge less than width
        :param edge:
        :param width:
        :return:
        """

        def _layer_condition(layer_edge: 'Edge'):
            if not layer_edge.face:
                return False
            if not self.plan.get_space_of_edge(layer_edge).category.mutable:
                # corridor shall not bite an on non mutable space
                return False
            if [l for l in layer_edge.face.edges if self.plan.get_linear(l)]:
                # corridor shall not contain a face that contains a linear
                return False
            if self.plan.get_linear(layer_edge.pair):
                return False
            return True

        layer_edges = []
        next_layer = True
        start_edge = edge
        if not _layer_condition(start_edge):
            return layer_edges
        while next_layer:
            for e in start_edge.face.edges:
                angle = ccw_angle(e.normal, start_edge.normal)
                angle = angle - 180 * (angle > 180 + EPSILON)
                dist_tmp = e.start.distance_to(edge.end)
                if (pseudo_equal(angle, 180, 10)
                        and dist_tmp < width + EPSILON
                        and _layer_condition(e)):
                    start_edge = e.pair
                    layer_edges.append(e)
                    if not start_edge.face:
                        next_layer = False
                    break
            else:
                next_layer = False
        return layer_edges

    def _space_totally_overlapped(self, level: int, face: 'Face') -> bool:
        # checks if the space that contained _face before corridor propagation
        # will be completely overlapped by corridors if _face is removed from it
        for group_face in self.grouped_faces[level]:
            if face in group_face:
                for f in group_face:
                    if f is face:
                        continue
                    if not self.plan.get_space_of_face(f).category.name == "circulation":
                        # not all faces of group_face are in corridors
                        return False
        return True

    def add_corridor_portion(self, edge: 'Edge', max_width: float, corridor_space: 'Space',
                             show: bool = False, corner: bool = False):
        """
        Builds a corridor space : starts with edge face and iteratively adds faces
        in normal direction to edge until the corridor space width reaches the objective
        :param edge:
        :param max_width:
        :param corridor_space:
        :param show:
        :param corner:
        :return:
        """

        layer_edges = self.get_parallel_layers_edges(edge, max_width)

        for layer_edge in layer_edges:
            sp = self.plan.get_space_of_edge(layer_edge)
            if not sp.category.name == "circulation":
                # ensures a corridor does not totally overlapp a space that was present before
                # corridor propagation
                # NB : a space sp_0 can be cut by corridor propagation, leading so sp_1 and sp_2
                # we authorize sp_1 or sp2 to be overlapped by a corridor, but not both
                if len(list(sp.faces)) == 1:
                    if self._space_totally_overlapped(sp.floor.level, sp.face):
                        break
                # if corner and sp.corner_stone(layer_edge.face):
                #    break
                corridor_space.add_face(layer_edge.face)
                sp.remove_face(layer_edge.face)
                if show:
                    self.plot.update([sp, corridor_space])

    def _initialize_plot(self, plot: Optional['Plot'] = None):
        """
        Creates a plot
        :return:
        """
        # if the corridor has already a plot : do nothing
        if self.plot:
            return

        if not plot:
            self.plot = Plot(self.plan)
            plt.ion()
            self.plot.draw(self.plan)
            plt.show()
            plt.pause(0.0001)
        else:
            self.plot = plot


# growth methods
def straight_path_growth_directionnal(corridor: 'Corridor', edge_line: List['Edge'],
                                      show: bool = False) -> 'Space':
    """
    Builds a corridor by growing a space around the line
    -get the growing direction of the line
    -grows the corridor on the growing side
    :param corridor:
    :param edge_line:
    :param show:
    :return:
    """
    plan = corridor.plan
    if plan.get_space_of_edge(edge_line[0]):
        floor = plan.get_space_of_edge(edge_line[0]).floor
    else:
        floor = plan.get_space_of_edge(edge_line[0].pair).floor

    level = floor.level

    corridor_space = Space(plan, floor, category=SPACE_CATEGORIES['circulation'])

    growing_direction = corridor.circulator.directions[level][edge_line[0]]
    for e, edge in enumerate(edge_line):
        support_edge = edge if growing_direction > 0 else edge.pair
        corridor.add_corridor_portion(support_edge, corridor.corridor_rules.width,
                                      corridor_space,
                                      show)
        if e == len(edge_line) - 1:
            # info stored for corner filling
            corridor.corner_data[edge] = growing_direction
    return corridor_space


# corridor rules
no_cut_rules = CorridorRules(width=140, penetration_length=130)

CORRIDOR_BUILDING_RULES = {
    "no_cut": {
        "corridor_rules": no_cut_rules,
        "growth_method": straight_path_growth_directionnal
    }
}

if __name__ == '__main__':
    import argparse
    import tools.cache

    # logging.getLogger().setLevel(logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--plan_index", help="choose plan index",
                        default=1)

    args = parser.parse_args()
    plan_index = int(args.plan_index)

    plan_name = None
    if plan_index < 10:
        plan_name = '00' + str(plan_index)  # + ".json"
    elif 10 <= plan_index < 100:
        plan_name = '0' + str(plan_index)  # + ".json"


    def main(plan_number: str):

        # corridor = Corridor(layer_width=25, nb_layer=5)

        solution = tools.cache.get_solution(plan_number, grid="002", seeder="directional_seeder",
                                            solution_number=0)

        corridor = Corridor(corridor_rules=CORRIDOR_BUILDING_RULES["no_cut"]["corridor_rules"],
                            growth_method=CORRIDOR_BUILDING_RULES["no_cut"]["growth_method"])
        corridor.apply_to(solution, show=False)

        plan = solution.spec.plan

        plan.check()
        plan.name = "corridor_" + plan.name
        plan.plot()


    plan_name = "060"
    main(plan_number=plan_name)
