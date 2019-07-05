import logging
from typing import Optional, Tuple, Dict, List, Type, Callable
from enum import Enum

import matplotlib.pyplot as plt
from functools import reduce

from libs.modelers.grid import GRIDS
from libs.modelers.seed import SEEDERS
from libs.plan.plan import Plan, Space, Face, Edge, Vertex
from libs.space_planner.circulation import Circulator, CostRules
from libs.specification.specification import Specification
from libs.plan.category import SPACE_CATEGORIES
from libs.io.plot import Plot
from libs.utils.geometry import (
    ccw_angle,
    pseudo_equal,
    move_point,
    parallel
)

GrowCorridor = Callable[['Corridor', List['Edge'], bool], 'Space']

EPSILON = 1


# TODO LIST:
# -deal with one vertex path


class CorridorRules:
    def __init__(self, layer_width: float = 110,
                 nb_layer: int = 2,
                 layer_cut: bool = False,
                 ortho_cut: bool = False,
                 width: float = 110,
                 penetration_length: float = 90,
                 penetration: bool = False,
                 recursive_cut_length: float = 400,
                 merging: bool = True):
        self.layer_width = layer_width  # width of a layer, when layer_cut is activated
        self.nb_layer = nb_layer  # number of layers cut around the circulation path
        self.layer_cut = layer_cut  # whether to cut layers along the circulation path or not
        self.ortho_cut = ortho_cut  # whether to the mesh orthogonally to end and start path,
        # used to get penetration with precise length
        self.width = width  # maximum width of the corridor
        self.penetration_length = penetration_length  # maximum penetration length, when needed
        self.penetration = penetration  # whether penetration is accounted for or not
        self.recursive_cut_length = recursive_cut_length  # param controling length of recursive cut
        self.merging = merging  # whether adjacent corridor spaces shall be merged or not


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

    def apply_to(self, plan: 'Plan', spec: 'Specification', show: bool = False):
        """
        Runs the corridor
        -creates a circulator and determines circulation paths in the plan
        -refines the mesh around those paths
        -grows corridor spaces around those paths
        :param plan:
        :param spec:
        :param show: whether to display a real-time visualization of the corridor
        :return:
        """
        self._clear()
        self.spec = spec
        self.plan = plan

        # store mutable spaces, for repair purpose
        initial_mutable_spaces = [sp for sp in self.plan.spaces if
                                  sp.mutable and not sp.category.name is "circulation"]
        # store groups of faces that belong to non mutable spaces, for repair purpose
        grouped_faces = {level: [] for level in self.plan.levels}
        for sp in initial_mutable_spaces:
            grouped_faces[sp.floor.level].append([f for f in sp.faces])
        self.grouped_faces = grouped_faces

        # computes circulation paths and stores them
        self.circulator = Circulator(plan=plan, spec=spec, cost_rules=self.circulation_cost_rules)
        self.circulator.connect()
        #self.circulator.plot()

        self._set_paths()

        # Real time plot updates
        if show:
            self._initialize_plot()

        # Refines the mesh around the circulation paths and grow corridor spaces around
        for path in self.paths:
            self.cut(path, show).grow(path, show)

        self.plan.remove_null_spaces()

        final_mutable_spaces = [sp for sp in self.plan.spaces if
                                sp.mutable and not sp.category.name is "circulation"]

        # merging corridor spaces to get the smallest set of rectangular corridors
        self._rectangular_merge()

        # space repair process : if some spaces have been cut by corridor growth
        self._repair_spaces(initial_mutable_spaces, final_mutable_spaces)

        # linear repair
        if self.corridor_rules.layer_cut or self.corridor_rules.ortho_cut:
            self.plan.mesh.watch()

    def _rectangular_merge(self):
        """
        merges corridor spaces when the merge is a rectangle
        purpose : ease the refiner process
        :return:
        """
        corridors = [sp for sp in self.plan.spaces if
                     sp.category is SPACE_CATEGORIES["circulation"]]
        merge = True if corridors else False
        while merge:
            merge = False
            for corridor in corridors:
                adjacent_corridors = [adj for adj in corridor.adjacent_spaces() if adj in corridors]
                for adjacent_corridor in adjacent_corridors:
                    if corridor.number_of_corners(adjacent_corridor) == 4:
                        corridor.merge(adjacent_corridor)
                        merge = True
                        break
                if merge:
                    corridors = [sp for sp in self.plan.spaces
                                 if sp.category is SPACE_CATEGORIES["circulation"]]
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

            if self.corridor_rules.penetration:
                if path_info.departure_penetration:
                    current_path = self._add_penetration_edges(current_path)
                if path_info.arrival_penetration:
                    current_path = self._add_penetration_edges(current_path,
                                                               start=False)
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
                            # splits penetration edge to get proper penetration length
                            coeff = (penetration_length - l) / penetration_edge.length
                            if penetration_edge.length * coeff < 2:
                                # snapping exception
                                pass
                            else:
                                # in penetration case, when an ortho_cut is performed, we proceed
                                # to edge split so as to get precise penetration length
                                if self.corridor_rules.ortho_cut:
                                    penetration_edge.split_barycenter(coeff=coeff)
                                    _edge_list = [penetration_edge.pair] + _edge_list if start \
                                        else _edge_list + [penetration_edge]
                            l = penetration_length
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

        self.plan.update_from_mesh()
        self.plan.simplify()

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

    def cut(self, path: List['Edge'], show: bool = False) -> 'Corridor':
        """
        Cuts layers in the mesh around the circulation path defined by path.
        For each edge of the path, cut self.nb_layer layers with spacing equal to self.layer_width
        :param path: ordered list of vertices forming a circulation path
        :param show:
        :return:
        """
        if show:
            self._initialize_plot()

        path = self._update_path(path)
        if not path:
            return self

        if self.corridor_rules.ortho_cut:
            # mesh cut, orthogonal to edge path
            self._ortho_slice(path[0], start=True, show=show)
            self._ortho_slice(path[-1], show=show)
            path = self._update_path(path)

        if self.corridor_rules.layer_cut:
            # layer slices parallel to path edges
            for edge in path:
                self._layer_slice(edge, show)

        return self

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

        def _condition(e: 'Edge'):
            if (self.plan.get_space_of_edge(e)
                    and self.plan.get_space_of_edge(e).category.name is "circulation"):
                return True
            return False

        def _line_forward(e: 'Edge') -> List['Edge']:
            """
            returns edges aligned with e, contiguous, in forward direction
            :param e:
            :return:
            """
            output = []
            current = e
            while current:
                output.append(current)
                current = current.aligned_edge or current.continuous_edge
            return output[1:]

        for edge in self.corner_data:

            if self.plan.get_space_of_edge(edge):
                floor = self.plan.get_space_of_edge(edge).floor
            else:
                floor = self.plan.get_space_of_edge(edge.pair).floor

            line = []
            for line_edge in _line_forward(edge):
                if ((_condition(line_edge) or _condition(line_edge.pair))
                        and sum([l_e.length for l_e in line]) < self.corridor_rules.width):
                    line.append(line_edge)
                else:
                    break
            for line_edge in line:
                corridor_space = Space(self.plan, floor, category=SPACE_CATEGORIES['circulation'])
                self.add_corridor_portion(line_edge, self.corner_data[edge]["ccw"], corridor_space,
                                          show)
                self.add_corridor_portion(line_edge.pair, self.corner_data[edge]["cw"],
                                          corridor_space,
                                          show)
                corner_edge = edge if self.corner_data[edge]["ccw"] > 0 else edge.pair
                space_corner = self.plan.get_space_of_edge(corner_edge)
                if not space_corner or not corridor_space:
                    continue
                if not space_corner.category.name == "circulation" and len(
                        list(space_corner.faces)) < 2:  # do not remove the space
                    continue
                if list([e for e in space_corner.edges if
                         e.pair.face and corridor_space.has_face(e.pair.face)]):
                    # merge if spaces are adjacent
                    corridor_space.merge(space_corner)

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
        if show:
            self._initialize_plot()

        path = self._update_path(path)

        self._path_growth(path, show)

        self._corner_fill(show)

        if self.corridor_rules.merging:
            self._corridor_merge()

        return self

    def _corridor_merge(self):
        """
        merges corridor spaces when they are adjacent
        :return:
        """

        def merging(spaces: List['Space']) -> bool:
            """
            iterative merge attempt of list elements with following elements in the list
            :param spaces:
            :return:
            """
            count = 0
            merge = False
            while count < len(spaces) - 1:
                for space in spaces[count + 1:]:
                    if spaces[count].adjacent_to(space):
                        spaces.remove(space)
                        spaces[count].merge(space)
                        merge = True
                        break
                else:
                    count += 1
            return merge

        corridor_spaces = [space for space in self.plan.spaces if
                           space.category.name is "circulation"]

        fusion = True
        while fusion:
            fusion = merging(corridor_spaces)

    def _path_growth(self, path: List['Edge'], show: bool = False):
        """
        Circulation path defined by path is decomposed into its straight portions
        A corridor space is grown around each portion
        :param path: ordered list of contiguous edges forming a circulation path
        :param show: ordered list of contiguous edges forming a circulation path
        :return:
        """

        edge_lines = self._get_straight_parts(path)
        corridor_spaces = []
        for edge_line in edge_lines:
            # created_space = self._straight_path_growth(edge_line, show)
            created_space = self.growth_method(self, edge_line, show)
            # created_space = self._straight_path_growth_directionnal(edge_line, show)
            adjacent_corridor_space = [sp for sp in corridor_spaces if
                                       sp.adjacent_to(created_space)]

            if adjacent_corridor_space and self.corridor_rules.merging:
                # if adjacent_corridor_space:
                adjacent_corridor_space[0].merge(created_space)
            else:
                corridor_spaces.append(created_space)

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

    def get_parallel_layers_edges(self, edge: 'Edge', width: 'float') -> Tuple['List', 'List']:
        """
        Returns
        *successive layer edges defined as
            -being parallel to edge
            -in vertical direction from edge
            -with distance to edge less than width
        *list of rounded distances from edge to the layer edges. NB : distances are rounded so that
        edges on a straight path have consistant distances to their respective layers
        :param edge:
        :param width:
        :return:
        """

        def _layer_condition(layer_edge: 'Edge'):
            if not layer_edge.face:
                return False
            if not self.plan.get_space_of_edge(layer_edge).category.mutable:
                return False
            if self.plan.get_linear(layer_edge):
                return False
            if self.plan.get_linear(layer_edge.pair):
                return False
            return True

        layer_edges = []
        next_layer = True
        dist = [0]
        start_edge = edge
        if not _layer_condition(start_edge):
            return dist, layer_edges
        while next_layer:
            for e in start_edge.face.edges:
                angle = ccw_angle(e.normal, start_edge.normal)
                angle = angle - 180 * (angle > 180 + EPSILON)

                dist_tmp = e.start.distance_to(edge.end)
                if (pseudo_equal(angle, 180, 10)
                        and dist_tmp < width + 2 * EPSILON
                        and _layer_condition(e)):
                    dist.append(int(round(dist_tmp / 5)) * 5 + 5)  # rounding
                    start_edge = e.pair
                    layer_edges.append(e)
                    if not start_edge.face:
                        next_layer = False
                    break
            else:
                next_layer = False
        return dist, layer_edges

    def add_corridor_portion(self, edge: 'Edge', width: float, corridor_space: 'Space',
                             show: bool = False):
        """
        Builds a corridor space : starts with edge face and iteratively adds faces
        in normal direction to edge until the corridor space width reaches the objective
        :param edge:
        :param width:
        :param corridor_space:
        :param show:
        :return:
        """

        def _space_totally_overlapped(_level: int, _face: 'Face') -> bool:
            # checks if the space that contained _face before corridor propagation
            # will be completely overlapped by corridors if _face is removed from it
            for group_face in self.grouped_faces[_level]:
                if _face in group_face:
                    for f in group_face:
                        if f is _face:
                            continue
                        if not self.plan.get_space_of_face(f).category.name == "circulation":
                            # not all faces of group_face are in corridors
                            return False
            return True

        layer_edges = self.get_parallel_layers_edges(edge, width)[1]

        for layer_edge in layer_edges:
            sp = self.plan.get_space_of_edge(layer_edge)
            if not sp.category.name == "circulation":
                # ensures a corridor does not totally overlapp a space that was present before
                # corridor propagation
                # NB : a space sp_0 can be cut by corridor propagation, leading so sp_1 and sp_2
                # we authorize sp_1 or sp2 to be overlapped by a corridor, but not both
                if len(list(sp.faces)) == 1:
                    if _space_totally_overlapped(sp.floor.level, sp.face):
                        break
                corridor_space.add_face(layer_edge.face)
                sp.remove_face(layer_edge.face)
                if show:
                    self.plot.update([sp, corridor_space])

    def _ortho_slice(self, edge: 'Edge', start: bool = False, show: bool = False):
        """
        cut mesh orthogonally to the edge, at its start (if start) or end
        :param edge:
        :param start:
        :param show:
        :return :
        """

        vertex = edge.start if start else edge.end
        # cuts on both sides
        self._recursive_cut(edge, vertex)

        if show:
            self.plot.update_faces([sp for sp in self.plan.spaces])

    def _layer_slice(self, edge: 'Edge', show: bool = False):
        """
        cut mesh in parallel to the edge, self.nb_cut layers are cut,
        spaced from self.layer_width apart
        :param edge:
        :param show:
        :return :
        """

        def _get_containing_face(vertex):
            # gets the face in which the vertex is lying
            # for better performance, try faces ordered by their distance to vertex
            faces = reduce(lambda a, b: a + b, [list(space.faces) for space in self.plan.spaces])
            faces = sorted(faces, key=lambda x: x.edge.start.distance_to(vertex))

            # gets the face in which the vertex is lying
            for face in faces:
                if face.as_sp.contains(vertex.as_sp):
                    return face
            return None

        def _cut_condition(e):
            # condition to satisfy for edge cutting
            if not e.face or not self.plan.get_space_of_edge(e).mutable:
                return False
            return True

        def _slice(start_point, coeff: float) -> bool:
            """
            Slicing process :
            -start_point is moved in edge normal direction with amplitude coeff
            -The resulting point: slice_point, is projected in the direction edge.vector
            -a recursive cut is applied from the projection point in the direction edge.vector
            :param start_point:
            :param coeff:
            :return:
            """
            slice_point = move_point(start_point, edge.normal,
                                     coeff * self.corridor_rules.layer_width)
            slice_vertex = Vertex(mesh=edge.mesh, x=slice_point[0], y=slice_point[1])
            face = _get_containing_face(slice_vertex)
            if face and self.plan.get_space_of_edge(face.edge).mutable:
                out = slice_vertex.project_point(face, edge.unit_vector)
                if not out:
                    slice_vertex.remove_from_mesh()
                    return False
                intersection_vertex = out[0]
                edge_to_cut = out[1]
                self._recursive_cut(edge_to_cut, intersection_vertex)
                if not intersection_vertex.edge:
                    # cleaning
                    intersection_vertex.remove_from_mesh()
                # cleaning
                slice_vertex.remove_from_mesh()
                if show:
                    self.plot.update_faces([sp for sp in self.plan.spaces])
            else:
                # cleaning
                slice_vertex.remove_from_mesh()
                return False
            return True

        def _slice_loop(e: 'Edge', ccw: bool = True):
            """
            successive mesh slices around edge e
            process :
            -starts from the edge middle (start_point), and moves orthogonally, ccw or cw
            -moves iteratively start_point with moving step = self.layer_width
            -cuts the mesh from the moving point, in parallel to e
            :param e:
            :param ccw:
            :return:
            """
            if not _cut_condition(e):
                return

            start_point = move_point([edge.start.x, edge.start.y], edge.unit_vector,
                                     edge.length / 2)
            sign = 1 if ccw else -1
            for s in range(1, abs(self.corridor_rules.nb_layer)):
                sl = _slice(start_point, sign * s)
                if not sl:
                    break

        _slice_loop(edge.pair, ccw=False)
        _slice_loop(edge, ccw=True)

    def _recursive_cut(self, edge: 'Edge', vertex: 'Vertex'):
        """
        Recursively cut the mesh from vertex, orthogonally to edge
        :param edge:
        :param vertex:
        :return:
        """
        space_ini = self.plan.get_space_of_edge(edge)

        # def _projects_on_linear(v: 'Vertex', direction_edge: 'Edge') -> bool:
        #     """
        #     returns true if v projection in a direction normal to edge cuts a linear
        #     :param v:
        #     :param direction_edge:
        #     :return:
        #     """
        #
        #     if not direction_edge.face.as_sp.intersects(v.as_sp):
        #         # TODO : this check should not be necessary - to be investigated
        #         return False
        #     out = v.project_point(direction_edge.face, direction_edge.normal)
        #
        #     if out:
        #         intersect_vertex_next = out[0]
        #         next_cut_edge = out[1]
        #         for e in self.plan.get_space_of_edge(next_cut_edge).edges:
        #             linear = self.plan.get_linear(e)
        #             if linear and (
        #                     linear.has_edge(next_cut_edge) or linear.has_edge(
        #                 next_cut_edge.next)):
        #                 intersect_vertex_next.remove_from_mesh()
        #                 return True
        #         intersect_vertex_next.remove_from_mesh()
        #     return False

        def _start_cut_conditions(e: 'Edge'):
            """
            :param e:
            :return:
            """
            if not e.face:
                return False

            # if _projects_on_linear(v, e):
            #    return False

            # if self.plan.get_linear(e):
            #    return False

            if self.plan.get_space_of_edge(e) and not self.plan.get_space_of_edge(e).mutable:
                return False

            return True

        def callback(new_edges: Optional[Tuple[Edge, Edge]]) -> bool:
            """
            Callback to insure space consistency
            Will stop the cut if it returns True
            :param new_edges: Tuple of the new edges created by the cut
            """
            start_edge, end_edge, new_face = new_edges
            sp = self.plan.get_space_of_edge(end_edge)
            # add the created face to the space
            if new_face is not None:
                sp.add_face_id(new_face.id)
            if not self.plan.get_space_of_edge(end_edge.pair):
                return True
            if self.plan.get_space_of_edge(end_edge.pair) and not self.plan.get_space_of_edge(
                    end_edge.pair).mutable:
                return True
            # if end_edge.pair and end_edge.pair.face:
            #    if _projects_on_linear(end_edge.pair.end, end_edge.pair):
            #        return True

            return False

        if (not space_ini
                # or not space_ini.mutable
                or not vertex.mesh):
            return

        if _start_cut_conditions(edge):
            edge.recursive_cut(vertex, max_length=self.corridor_rules.recursive_cut_length,
                               callback=callback)

        if not vertex.mesh:
            return

        if _start_cut_conditions(edge.pair):
            edge.pair.recursive_cut(vertex, max_length=self.corridor_rules.recursive_cut_length,
                                    callback=callback)

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
    if corridor.plan.get_space_of_edge(edge_line[0]):
        floor = corridor.plan.get_space_of_edge(edge_line[0]).floor
    else:
        floor = corridor.plan.get_space_of_edge(edge_line[0].pair).floor

    level = floor.level

    corridor_space = Space(corridor.plan, floor, category=SPACE_CATEGORIES['circulation'])

    growing_direction = corridor.circulator.directions[level][edge_line[0]]
    for e, edge in enumerate(edge_line):
        support_edge = edge if growing_direction > 0 else edge.pair
        corridor.add_corridor_portion(support_edge, corridor.corridor_rules.width,
                                      corridor_space,
                                      show)
        if e == len(edge_line) - 1:
            # info stored for corner filling
            width_ccw = corridor.corridor_rules.width if growing_direction > 0 else 0
            width_cw = corridor.corridor_rules.width if growing_direction < 0 else 0
            corridor.corner_data[edge] = {"cw": width_cw, "ccw": width_ccw}
    return corridor_space


def straight_path_growth_directionnal_no_cut(corridor: 'Corridor', edge_line: List['Edge'],
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

    if corridor.plan.get_space_of_edge(edge_line[0]):
        floor = corridor.plan.get_space_of_edge(edge_line[0]).floor
    else:
        floor = corridor.plan.get_space_of_edge(edge_line[0].pair).floor

    level = floor.level

    corridor_space = Space(corridor.plan, floor, category=SPACE_CATEGORIES['circulation'])

    growing_direction = corridor.circulator.directions[level][edge_line[0]]
    for e, edge in enumerate(edge_line):
        support_edge = edge if growing_direction > 0 else edge.pair
        corridor.add_corridor_portion(support_edge, corridor.corridor_rules.width,
                                      corridor_space,
                                      show)
        if e == len(edge_line) - 1:
            # info stored for corner filling
            width_ccw = corridor.corridor_rules.width if growing_direction > 0 else 0
            width_cw = corridor.corridor_rules.width if growing_direction < 0 else 0
            corridor.corner_data[edge] = {"cw": width_cw, "ccw": width_ccw}
    return corridor_space


def straight_path_growth(corridor: 'Corridor', edge_line: List['Edge'],
                         show: bool = False) -> 'Space':
    """
    Builds a corridor by growing a space around the line
    Strategy :
    -for each edge of edge_line identify the layers in vertical directions : cw and ccw
    -for each edge edge_i, deduce the maximum width_i the corridor can have on the portion
    around edge_i in cw(resp ccw) direction
    -the corridor in cw (resp ccw) direction has a maximum width equal to min_i(width_i)
    -grows the corridor alternatively on each side while
        *a layer is available
        *maximum corridor width has not been reached
    :param corridor:
    :param edge_line: straight circulation path
    :param show:
    :return: built corridor space
    """

    def _lists_intersection(l: List[List]) -> List:
        # returns a sorted list containing elements that are in every lists of l
        intersect = set(l[0])
        for s in l[1:]:
            intersect.intersection_update(s)
        intersect = list(intersect)
        intersect.sort()
        return intersect

    def _get_layers_width(ccw=True) -> List:
        # returns the list of layers' width on ccw (resp cw) side of the path
        portions_width = []
        for line_edge in edge_line:
            start_edge = line_edge if ccw else line_edge.pair
            portion_width = \
                corridor.get_parallel_layers_edges(start_edge, corridor.corridor_rules.width)[0]
            portions_width.append(portion_width)
        return _lists_intersection(portions_width)

    width_ccw_list = _get_layers_width()
    width_cw_list = _get_layers_width(ccw=False)

    # the corridor is grown on each side while
    #  -growth is possible and corridor width is
    #  -under self.corridor_rules["width"]

    width_ccw = width_ccw_list[-1]
    width_cw = width_cw_list[-1]
    count_ccw = len(width_ccw_list) - 1
    count_cw = len(width_cw_list) - 1
    while width_ccw + width_cw > corridor.corridor_rules.width + EPSILON:
        if width_ccw > width_cw:
            count_ccw -= 1
        else:
            count_cw -= 1
        width_ccw = width_ccw_list[count_ccw]
        width_cw = width_cw_list[count_cw]

    corridor_space = Space(corridor.plan, corridor.plan.floor,
                           category=SPACE_CATEGORIES['circulation'])
    for e, edge in enumerate(edge_line):
        corridor.add_corridor_portion(edge, width_ccw, corridor_space, show)
        corridor.add_corridor_portion(edge.pair, width_cw, corridor_space, show)
        if e == len(edge_line) - 1:
            # info stored for corner filling
            corridor.corner_data[edge] = {"cw": width_cw, "ccw": width_ccw}
    return corridor_space


# corridor rules
no_cut_rules = CorridorRules(penetration=True, layer_cut=False, ortho_cut=False, merging=False,
                             width=130, penetration_length=130)
coarse_cut_rules = CorridorRules(penetration=True, layer_cut=True, ortho_cut=True, merging=True)
fine_cut_rules = CorridorRules(penetration=True, layer_cut=True, ortho_cut=True, nb_layer=5,
                               layer_width=25,
                               merging=True)

CORRIDOR_BUILDING_RULES = {
    "no_cut": {
        "corridor_rules": no_cut_rules,
        "growth_method": straight_path_growth_directionnal
    },
    "coarse": {
        "corridor_rules": coarse_cut_rules,
        "growth_method": straight_path_growth_directionnal
    },
    "fine": {
        "corridor_rules": fine_cut_rules,
        "growth_method": straight_path_growth
    }
}

if __name__ == '__main__':
    import argparse

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

        plan.check()

        plan.name = "corridor_" + plan.name
        plan.plot()


    plan_name = "050.json"
    main(input_file=plan_name)
