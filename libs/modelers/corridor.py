import logging
from typing import Optional, Tuple, Dict, List

import matplotlib.pyplot as plt
from functools import reduce

from libs.modelers.grid import GRIDS
from libs.modelers.seed import SEEDERS
from libs.plan.plan import Plan, Space, Edge, Vertex
from libs.space_planner.circulation import Circulator, COST_RULES
from libs.plan.category import SPACE_CATEGORIES
from libs.io.plot import Plot
from libs.utils.geometry import (
    ccw_angle,
    pseudo_equal,
    move_point,
    direction_vector,
    parallel
)


# TODO LIST:
# -remove angle measure and add global epsilon
# -deal with corridor corners
# -deal with one vertex path
# -cut slices orthogonally to start and final edges if needed by the refiner
# -for now corridor is grown on each side of a path as long as growth is possible,
#  other growing strategies could be implemented :
#   *grow on bigger rooms,
#   *grow so as to minimize the number of corners...


class Corridor:
    """
    Corridor Class

    Class to build circulation spaces between every isolated room space of the plan.

    If a plot is given to the corridor, it will use it to display in real time the changes occurring
    on the plan.
    """

    def __init__(self,
                 corridor_rules: Dict = None,
                 circulation_cost_rules: Dict = COST_RULES,
                 plot: Optional['Plot'] = None
                 ):

        self.corridor_rules = corridor_rules
        self.circulation_cost_rules = circulation_cost_rules
        self.plot = plot
        self.plan: Plan = None
        self.circulator: Circulator = None
        self.growth_data: Dict = None

    def _clear(self):
        self.plan = None
        self.circulator = None
        self.paths = []
        self.growth_data = {}

    def apply_to(self, plan: 'Plan', show: bool = False):
        """
        Runs the corridor
        -creates a circulator and determines circulation paths in the plan
        -refines the mesh around those paths
        -grows corridor spaces around those paths
        :param plan:
        :param show: whether to display a real-time visualization of the corridor
        :return:
        """
        self._clear()
        self.plan = plan

        # computes circulation paths and stores them
        self.circulator = Circulator(plan=plan, cost_rules=self.circulation_cost_rules)
        self.circulator.connect()
        self.circulator.plot()

        for level in self.circulator.connecting_paths:
            vertex_paths = self.circulator.connecting_paths[level]
            for vertex_path in vertex_paths:
                if len(vertex_path) > 1:  # TODO : deal with one vertice path
                    vertex_path = self._add_penetration_vertices(vertex_path)
                    self.paths.append(vertex_path)

        # Real time plot updates
        if show:
            self._initialize_plot()

        # Refines the mesh around the circulation paths and grow corridor spaces around
        for path in self.paths:
            self.cut(path, show).grow(path, show)

    def _add_penetration_vertices(self, vertex_list: List['Vertex']):
        """
        Possibly adds vertices at the beginning and end of the path
        to account for corridor penetration within the room.
        A path has to penetrate a room if following conditions are satisfied
            -it extends on the room border, not inside the room space
            -it is not on the plan border
            -it is not along a load bearing wall
        When penetration conditions are satisfied, the penetration shall have a length equal to
        penetration_length
        :param List['Vertex']: ordered list of vertices forming a circulation path
        :return:
        """

        penetration_length = self.corridor_rules["penetration_length"]

        def _penetration_condition(edge: 'Edge') -> bool:
            """
            Determines if the edge is added to the path
            :param edge:
            :return:
            """
            if self.plan.get_space_of_edge(edge) is self.plan.get_space_of_edge(edge.pair):
                # edge not on space border
                return False
            if not edge.face or not edge.pair.face:
                # edge on the plan border
                return False
            cat = ["loadBearingWall", "duct"]
            if (self.plan.get_space_of_edge(edge).category.name in cat
                    or self.plan.get_space_of_edge(edge.pair).category.name in cat):
                # edge along a non mutable space among cat
                return False
            return True

        def _add_vertices(vert_list: List['Vertex'], start=True):
            """
            Adds vertices to the list, at the beginning (if start) or end if
            penetration condition are satisfied until penetration length is reached
            :param vert_list: ordered list of vertices forming a circulation path
            :param start:
            :return:
            """
            edge_list = self._get_edge_path(vert_list)

            l = 0  # penetration length
            continue_penetration = True
            while l < penetration_length and continue_penetration:
                limit_edge = edge_list[0].pair if start else edge_list[-1]
                limit_vertex = limit_edge.end
                penetration_edges = [edge for edge in limit_vertex.edges if
                                     edge.face and edge.pair.face]
                for edge in penetration_edges:
                    if parallel(edge.vector, limit_edge.vector) and _penetration_condition(edge):
                        penetration_edge = edge
                        if l + penetration_edge.length > penetration_length:
                            # splits penetration edge to get proper penetration length
                            coeff = (penetration_length - l) / penetration_edge.length
                            if penetration_edge.length * coeff < 2:
                                # snapping exception
                                added_vertex = penetration_edge.start
                            else:
                                penetration_edge.split_barycenter(coeff=coeff)
                                added_vertex = penetration_edge.end
                            l = penetration_length
                        else:
                            edge_list = [penetration_edge.pair] + edge_list if start \
                                else edge_list + [penetration_edge]
                            added_vertex = penetration_edge.end
                            l += penetration_edge.length
                        if added_vertex not in vert_list:
                            vert_list = [added_vertex] + vert_list if start else vert_list + [
                                added_vertex]
                        break
                else:
                    continue_penetration = False

            return vert_list

        vertex_list = _add_vertices(vertex_list)
        vertex_list = _add_vertices(vertex_list, start=False)

        self.plan.update_from_mesh()
        self.plan.simplify()

        return vertex_list

    @staticmethod
    def _get_edge_path(vertex_path: List['Vertex']) -> List['Edge']:
        """
        From a list of vertices forming a path, generates the list of contiguous edges following
        the path
        :param vertex_path: ordered list of vertices forming a circulation path
        :return: ordered list of contiguous edges forming a circulation path
        """
        edge_path = []

        for v, vertex in enumerate(vertex_path[:-1]):
            ref_dir = direction_vector((vertex.x, vertex.y),
                                       (vertex_path[v + 1].x, vertex_path[v + 1].y))
            edge = None
            vert_end = vertex
            while not edge or (vert_end is not vertex_path[v + 1]):
                for edge in vert_end.edges:
                    dir_e = edge.unit_vector
                    if 1 > ccw_angle(ref_dir, dir_e) > -1:
                        edge_path.append(edge)
                        vert_end = edge.end
                        break

        return edge_path

    def cut(self, path: List['Vertex'], show: bool = False) -> 'Corridor':
        """
        Cuts layers in the mesh around the circulation path defined by path.
        For each edge of the path, cut self.nb_layer layers with spacing equal to self.layer_width
        :param path: ordered list of vertices forming a circulation path
        :param show:
        :return:
        """
        if show:
            self._initialize_plot()

        edge_path = self._get_edge_path(path)

        # layer slices parallel to path edges
        for edge in edge_path:
            self._layer_slice(edge, show)

        edge_path = self._get_edge_path(path)

        # mesh cut, orthogonal to edge path
        for edge in edge_path:
            self._ortho_slice(edge, start=True, show=show)
        self._ortho_slice(edge_path[-1], show=show)

        # # mesh cut, orthogonal to start and final path edges
        # self._ortho_slice(edge_path[0], start=True, show=show)
        # self._ortho_slice(edge_path[-1], show=show)

        return self

    def _corner_fill(self, show: bool = False):

        def condition(e: 'Edge'):
            if (self.plan.get_space_of_edge(e)
                    and self.plan.get_space_of_edge(e).category.name is "circulation"):
                return True
            return False

        for edge in self.growth_data:
            line = []
            corridor_space = Space(self.plan, self.plan.floor,
                                   category=SPACE_CATEGORIES['circulation'])
            for e in edge.line_forward():
                if (condition(e) or condition(e.pair)):
                    line.append(e)
                else:
                    break
            for e in line:
                self._add_corridor_portion(e, self.growth_data[edge]["ccw"], corridor_space,
                                           show)
                self._add_corridor_portion(e.pair, self.growth_data[edge]["cw"], corridor_space,
                                           show)

    def grow(self, path: List['Vertex'], show: bool = False) -> 'Corridor':
        """
        Grows corridor spaces around the circulation space defined by path.
        Merge built corridor spaces when they are adjacent
        :param path: ordered list of vertices forming a circulation path
        :param show:
        :return:
        """
        if show:
            self._initialize_plot()

        edge_path = self._get_edge_path(path)

        self._path_growth(edge_path, show)

        self._corner_fill(show)

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

    def _path_growth(self, edge_path: List['Edge'], show: bool = False):
        """
        Circulation path defined by edge_path is decomposed into its straight portions
        A corridor space is grown around each portion
        :param edge_path: ordered list of contiguous edges forming a circulation path
        :param show: ordered list of contiguous edges forming a circulation path
        :return:
        """

        edge_lines = self._get_straight_parts(edge_path)
        corridor_spaces = []
        for edge_line in edge_lines:
            created_space = self._straight_path_growth(edge_line, show)
            adjacent_corridor_space = [sp for sp in corridor_spaces if
                                       sp.adjacent_to(created_space)]

            if adjacent_corridor_space:
                adjacent_corridor_space[0].merge(created_space)
            else:
                corridor_spaces.append(created_space)

    @staticmethod
    def _get_straight_parts(edge_path: List['Edge']) -> List[List['Edge']]:
        """
        decomposes the path into its straight sub-parts
        :param edge_path: ordered list of contiguous edges forming a circulation path
        :return:
        """
        edge_lines = []
        edge_ini = edge_path[0]
        l = []
        for e in edge_path:
            if not parallel(edge_ini.vector, e.vector):
                edge_lines.append(l)
                edge_ini = e
                l = [e]
            else:
                l.append(e)
        edge_lines.append(l)
        return edge_lines

    def _straight_path_growth(self, edge_line: List['Edge'], show: bool = False) -> 'Space':
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
        :param edge_line: straight circulation path
        :param show:
        :return: built corridor space
        """

        def _get_width(ccw=True) -> float:
            # returns the maximum expansion on ccw (resp cw) side of the path
            portions_width = []
            for e in edge_line:
                start_edge = e if ccw else e.pair
                portion_width = \
                    self._get_parallel_layers_edges(start_edge, self.corridor_rules["width"])[0]
                portions_width.append(portion_width)
            width = 0 if not portions_width else min(portions_width)
            return width

        width_ccw = _get_width()
        width_cw = _get_width(ccw=False)

        # the corridor is grown on each side while
        #  -growth is possible and corridor width is
        #  -under self.corridor_rules["width"]
        narrow_step = self.corridor_rules["layer_width"]
        while width_ccw + width_cw > self.corridor_rules["width"] + 1:
            if width_ccw + width_cw - self.corridor_rules["layer_width"] < self.corridor_rules[
                "width"]:
                narrow_step = width_ccw + width_cw - self.corridor_rules["width"]
            if width_ccw > width_cw:
                width_ccw -= narrow_step
            else:
                width_cw -= narrow_step

        corridor_space = Space(self.plan, self.plan.floor, category=SPACE_CATEGORIES['circulation'])
        for e, edge in enumerate(edge_line):
            self._add_corridor_portion(edge, width_ccw, corridor_space, show)
            self._add_corridor_portion(edge.pair, width_cw, corridor_space, show)
            if e == len(edge_line) - 1:
                self.growth_data[edge] = {"cw": width_cw, "ccw": width_ccw}
        return corridor_space

    def _get_parallel_layers_edges(self, edge: 'Edge', width: 'float') -> Tuple[float, 'List']:
        """
        Returns
        *successive layer edges defined as
            -being parallel to edge
            -in vertical direction from edge
            -with distance to edge less than width
        *distance from edge to the last layer edge
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
        dist = 0
        start_edge = edge
        if not _layer_condition(start_edge):
            return dist, layer_edges
        while next_layer:
            for e in start_edge.face.edges:
                angle = ccw_angle(e.normal, start_edge.normal)
                angle = angle - 180 * (angle > 180)

                dist_tmp = e.start.distance_to(edge.end)
                if (pseudo_equal(angle, 180, 10)
                        and dist_tmp < width + 2
                        and _layer_condition(e)):
                    dist = dist_tmp
                    start_edge = e.pair
                    layer_edges.append(e)
                    if not start_edge.face:
                        next_layer = False
                    break
            else:
                next_layer = False
        return dist, layer_edges

    def _add_corridor_portion(self, edge: 'Edge', width: float, corridor_space: 'Space',
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
        layer_edges = self._get_parallel_layers_edges(edge, width)[1]
        for layer_edge in layer_edges:
            sp = self.plan.get_space_of_edge(layer_edge)
            if not sp.category.name == "circulation":
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
                                     coeff * self.corridor_rules["layer_width"])
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
            for s in range(1, abs(self.corridor_rules["nb_layer"])):
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

        def _projects_on_linear(v: 'Vertex', direction_edge: 'Edge') -> bool:
            """
            returns true if v projection in a direction normal to edge cuts a linear
            :param v:
            :param direction_edge:
            :return:
            """

            if not direction_edge.face.as_sp.intersects(v.as_sp):
                # TODO : this check should not be necessary - to be investigated
                return False
            out = v.project_point(direction_edge.face, direction_edge.normal)

            if out:
                intersect_vertex_next = out[0]
                next_cut_edge = out[1]
                for e in self.plan.get_space_of_edge(next_cut_edge).edges:
                    linear = self.plan.get_linear(e)
                    if linear and (
                            linear.has_edge(next_cut_edge) or linear.has_edge(
                        next_cut_edge.next)):
                        intersect_vertex_next.remove_from_mesh()
                        return True
                intersect_vertex_next.remove_from_mesh()
            return False

        def _start_cut_conditions(v: 'Vertex', e: 'Edge'):
            """
            :param v:
            :param e:
            :return:
            """
            if not e.face:
                return False

            if _projects_on_linear(v, e):
                return False

            if self.plan.get_linear(e):
                return False

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
            if end_edge.pair and end_edge.pair.face:
                if _projects_on_linear(end_edge.pair.end, end_edge.pair):
                    return True

            return False

        if (not space_ini
                # or not space_ini.mutable
                or not vertex.mesh):
            return

        # if edge.face and not _projects_on_linear(vertex, edge) and not self.plan.get_linear(edge) \
        #        and self.plan.get_space_of_edge(edge).mutable:
        if _start_cut_conditions(vertex, edge):
            edge.recursive_cut(vertex, max_length=self.corridor_rules["recursive_cut_length"],
                               callback=callback)

        if not vertex.mesh:
            return

        # if edge.pair.face and not _projects_on_linear(vertex,
        #                                              edge.pair) and not self.plan.get_linear(
        #    edge.pair) and self.plan.get_space_of_edge(edge.pair).mutable:
        if _start_cut_conditions(vertex, edge.pair):
            edge.pair.recursive_cut(vertex, max_length=self.corridor_rules["recursive_cut_length"],
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


CORRIDOR_RULES = {
    "layer_width": 25,
    "nb_layer": 5,
    "recursive_cut_length": 400,
    "width": 100,
    "penetration_length": 90
}

if __name__ == '__main__':
    import argparse

    logging.getLogger().setLevel(logging.DEBUG)

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


    def get_plan(input_file: str = "001.json") -> 'Plan':

        import libs.io.reader as reader
        import libs.io.writer as writer
        from libs.space_planner.space_planner import SpacePlanner
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
            return plan

        except FileNotFoundError:
            plan = reader.create_plan_from_file(input_file)
            spec = reader.create_specification_from_file(input_file[:-5] + "_setup0" + ".json")

            GRIDS['optimal_grid'].apply_to(plan)
            SEEDERS["simple_seeder"].apply_to(plan)
            spec.plan = plan
            space_planner = SpacePlanner("test", spec)
            best_solutions = space_planner.solution_research()
            new_spec = space_planner.spec

            if best_solutions:
                solution = best_solutions[0]
                plan = solution.plan
                new_spec.plan = plan
                writer.save_plan_as_json(plan.serialize(), plan_file_name)
                writer.save_as_json(new_spec.serialize(), folder, spec_file_name + ".json")
                return plan
            else:
                logging.info("No solution for this plan")


    def main(input_file: str):

        # TODO : à reprendre
        # * 35, 40 : linear cut
        # * 46 : not enough contact with entrance
        # * 31 : wrong corridor shape

        plan = get_plan(input_file)
        plan.name = input_file[:-5]

        # corridor = Corridor(layer_width=25, nb_layer=5)
        corridor = Corridor(corridor_rules=CORRIDOR_RULES)
        corridor.apply_to(plan, show=False)

        plan.check()

        plan.name = "corridor_" + plan.name
        plan.plot()


    # plan_name = "059.json"
    main(input_file=plan_name)
