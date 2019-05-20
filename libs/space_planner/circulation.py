# coding=utf-8
"""
Circulation module

used to detect isolated rooms and generate a path to connect them

"""

import logging
import math

from typing import Dict, List, Tuple
from libs.plan.plan import Space, Plan, Vertex
from libs.mesh.mesh import Edge
from libs.specification.specification import Specification
from libs.io.plot import plot_save
from libs.utils.graph import GraphNx, EdgeGraph
from libs.plan.category import LINEAR_CATEGORIES
from libs.utils.geometry import parallel, opposite_vector, move_point


# TODO : deal with load bearing walls by defining locations where they can be crossed

class Circulator:
    """
    Circulator Class
    contains utilities to detect isolated rooms connect them to circulation spaces
    """

    def __init__(self, plan: Plan, spec: 'Specification', cost_rules: Dict = None):
        self.plan = plan
        self.path_calculator = PathCalculator(plan=self.plan, cost_rules=cost_rules)
        self.path_calculator.build()
        self.connectivity_graph = GraphNx()
        self.spec = spec

    def clear(self):
        self.reachable_edges = {space: [] for space in self.plan.spaces}
        self.connecting_paths = {level: [] for level in self.plan.list_level}
        self.connecting_edges = {level: [] for level in self.plan.list_level}
        self.growing_directions = {level: {} for level in self.plan.list_level}
        self.circulation_cost = 0

    def draw_path(self, set1: List['Edge'], set2: List['Edge'], level: int) -> Tuple[
        'List[Vertex]', float]:
        """
        Finds the shortest path between two sets of edges
        :return list of vertices on the path and cost of the path
        """
        graph = self.path_calculator.graph[level]
        path, cost = graph.get_shortest_path(set1,
                                             set2)

        return path, cost

    def multilevel_connection(self):
        """
        in multi-level case, adds a connection between spaces containing the stair at each level
        """
        number_of_floors = self.plan.floor_count
        space_connection_between_floors = []

        if number_of_floors > 1:
            for level in self.plan.list_level:
                for space in self.plan.spaces:
                    if (space.floor.level is level and "startingStep" in
                            space.components_category_associated()):
                        space_connection_between_floors.append(space)
                        break

        for i in range(number_of_floors - 1):
            self.connectivity_graph.add_edge(space_connection_between_floors[i],
                                             space_connection_between_floors[i + 1])

    def init_reachable_edges(self):
        """
        for each space, determines which edges can be the arrival of a circulation path
        linking this space
        :return:
        """

        def parallel_neighbor(e: 'Edge', sp: 'Space', next_edge: bool = True):
            neighbor_edge = sp.next_edge(e) if next_edge else sp.previous_edge(e)
            if parallel(e.vector, neighbor_edge.vector):
                return True
            return False

        def is_corner_edge(e: 'Edge', sp: 'Space'):

            if (not parallel_neighbor(e, sp) or
                    not parallel_neighbor(e, sp, next_edge=False)):
                return True
            sp_pair = self.plan.get_space_of_edge(e.pair)
            if sp_pair:
                if (not parallel_neighbor(e.pair, sp_pair) or
                        not parallel_neighbor(e.pair, sp_pair, next_edge=False)):
                    return True
            return False

        def is_adjacent_to_other_space(e: 'Edge'):
            sp_pair = self.plan.get_space_of_edge(e.pair)
            if not sp_pair or sp_pair.category.external:
                return False
            return True

        def get_reachable_edges(sp: 'Space'):
            reachable_edges = list(edge for edge in sp.edges if
                                   is_corner_edge(edge, sp) and is_adjacent_to_other_space(edge))
            return reachable_edges

        mutable_spaces = [space for space in self.plan.spaces if space.mutable]
        for space in mutable_spaces:
            self.reachable_edges[space] = get_reachable_edges(space)

    def init_connectivity_graph(self):
        """
        builds a connectivity graph of the plan, each circulation space is a node
        :return:
        """

        for space in self.plan.circulation_spaces():
            self.connectivity_graph.add_node(space)

        # builds connectivity graph for circulation spaces
        for space in self.plan.circulation_spaces():
            for other in self.plan.circulation_spaces():
                if other is not space and other.adjacent_to(space):
                    # if spaces are adjacent, they are connected in the graph
                    self.connectivity_graph.add_edge(space, other)

        self.multilevel_connection()

        self.set_circulation_path()

    def expand_connectivity_graph(self):
        """
        connects each non circulation space of the plan to a circulation space
        :return:
        """
        for space in self.plan.mutable_spaces():
            if space not in self.connectivity_graph.nodes():
                self.connectivity_graph.add_node(space)
                for other in self.plan.circulation_spaces():
                    if other is not space and other.adjacent_to(space):
                        self.connectivity_graph.add_edge(space, other)

        for node in list(self.connectivity_graph.nodes()):
            if not self.connectivity_graph.node_connected(node):
                connected_rooms = self.connect_space_to_circulation_graph(node)
                # connected_rooms contains the list of rooms connected by connecting node to a
                # circulation space
                for connected_room in connected_rooms:
                    if not self.connectivity_graph.node_connected(connected_room):
                        # connected_room is no longer isolated
                        self.connectivity_graph.add_edge(connected_room, node)

    def set_circulation_path(self):
        """
        ensures circulation spaces are all connected
        :return:
        """

        father_nodes = {}

        for room in self.plan.mutable_spaces():
            if room.category.name is 'entrance':
                father_nodes[room.floor.level] = room
                break
        else:
            for room in self.plan.mutable_spaces():
                if room.category.name is 'living':
                    father_nodes[room.floor.level] = room
                    break

        if not father_nodes:
            return True

        start_level = list(father_nodes.keys())[0]

        father_node = [room for room in self.plan.spaces if
                       room.floor.level is not start_level and "startingStep"
                       in room.components_category_associated()]

        for f in father_node:
            father_nodes[f.floor.level] = f

        for node in self.connectivity_graph.nodes():
            if not self.connectivity_graph.has_path(node, father_nodes[node.floor.level]):
                path, cost = self.draw_path(self.reachable_edges[father_nodes[node.floor.level]],
                                            self.reachable_edges[node], node.floor.level)
                self.circulation_cost += cost
                self.actualize_path(path, node.floor.level)
                self.connectivity_graph.add_edge(node, father_nodes[node.floor.level])

    @staticmethod
    def get_edge_path(path: List['Vertex']) -> List['Edge']:
        """
        from a list of vertices, gets the list of edges connecting those vertices
        :param path:
        :return:
        """
        edge_path = []
        for v, vert in enumerate(path[:-1]):
            for edge in vert.edges:
                if edge.end is path[v + 1]:
                    edge_path.append(edge)
                    break
        return edge_path

    def set_growing_directions(self, edge_path, level):
        """
        for each edge of a circulation path, gets the direction in which the corridor has to grow
        so as to bite preferentially on rooms which area is still higher than the minimum spec
        when amputated by the corridor.
        process:
        *decompose the path into its straight portions
        *for each edge of a considered straight portion, computes the most adapted growth direction
        *deduce the most adapted growth direction for the considered portion
        :param edge_path:
        :param level:
        :return:
        """

        def get_lines_of_path(edges: List['Edge']) -> List[List['Edge']]:
            """
            gets the straight portions from a circulation path
            :param edges:
            :return:
            """
            lines = []
            if not edges:
                return lines
            e = edges[0]
            count = 0
            while e:
                current_line = []
                # going forward
                current = e
                while current and current in edges:
                    current_line.append(current)
                    current = current.aligned_edge
                    count += 1
                lines.append(current_line)
                if count < len(edges):
                    e = edges[count]
                else:
                    return lines

        def get_score(space: 'Space', e: 'Edge') -> float:
            """
            returns min_required_space_area-edge.length*corridor_width if positive, else zero
            :param space:
            :param e:
            :return:
            """
            corridor_width = 90
            spec_items = self.spec.items[:]
            corresponding_items = list(filter(lambda i: i.category.name == space.category.name,
                                              spec_items))
            best_item = min(corresponding_items,
                            key=lambda i: math.fabs(i.required_area - space.cached_area()),
                            default=None)

            shift = best_item.min_size.area - (space.cached_area() - corridor_width * e.length)
            return shift * (shift > 0)

        def get_growing_direction(path_line: List['Edge']):
            """
            for a given line of edges, gets the direction the corridor has to grow so as to bite
            preferentially on rooms which area is still higher than the minimum spec
            when amputated by the corridor.
            :param path_line:
            :return:
            """
            dir_ccw = path_line[0].normal
            dir_cw = opposite_vector(dir_ccw)
            score_ccw = 0
            score_cw = 0
            for e in path_line:
                space_ccw = self.plan.get_space_of_edge(e)
                if not space_ccw or not space_ccw.mutable:
                    # corridor cannot grow outside of the plan or on a non mutable space
                    return dir_cw
                else:
                    score_ccw += get_score(space_ccw, e)
                space_cw = self.plan.get_space_of_edge(e.pair)
                if not space_cw or not space_cw.mutable:
                    # corridor cannot grow outside of the plan or on a non mutable space
                    return dir_ccw
                else:
                    score_cw += get_score(space_cw, e)
            return dir_ccw if score_ccw < score_cw else dir_cw

        path_lines = get_lines_of_path(edge_path)
        for line in path_lines:
            growing_direction = get_growing_direction(line)
            for edge in line:
                self.growing_directions[level][edge] = growing_direction

    def actualize_path(self, path: List['Vertex'], level: int) -> List[
        'Space']:
        """
        update based on computed circulation path
        :return: the list of spaces connected by path
        """
        self.connecting_paths[level].append(path)
        edge_path = self.get_edge_path(path)
        self.connecting_edges[level].append(edge_path)

        self.set_growing_directions(edge_path, level)

        # when a circulation has been set, it can be used to connect every other spaces
        # without cost increase
        # self.path_calculator.set_corridor_to_zero_cost(path, level)

        connected_rooms = []  # will contain the list of rooms connected by the path
        for e in edge_path:
            connected = self.plan.get_space_of_edge(e)
            if connected and connected not in connected_rooms and connected.mutable:
                connected_rooms.append(connected)
            connected = self.plan.get_space_of_edge(e.pair)
            if connected and connected not in connected_rooms and connected.mutable:
                connected_rooms.append(connected)

        return connected_rooms

    def connect_space_to_circulation_graph(self, space) -> List['Space']:
        """
        connects the given space with a circulation space of the plan
        :return: the list of spaces connected by the circulation drawn to connect space
        """
        path_min = None
        connected_room = None
        cost_min = None
        # compute path between space and every circulation spaces
        for other in self.plan.circulation_spaces():
            if other is not space and space.floor.level is other.floor.level:
                path, cost = self.draw_path(self.reachable_edges[space],
                                            self.reachable_edges[other], space.floor.level)
                if cost_min is None or cost < cost_min:
                    cost_min = cost
                    path_min = path
                    connected_room = other
        # compute path betwenn space and every existing circulation path
        for edge_path in self.connecting_edges[space.floor.level]:
            if edge_path:
                path, cost = self.draw_path(self.reachable_edges[space], edge_path,
                                            space.floor.level)
                if cost_min is None or cost < cost_min:
                    cost_min = cost
                    path_min = path

        connected_rooms = []
        if path_min is not None:
            connected_rooms = self.actualize_path(path_min, space.floor.level)
            self.circulation_cost += cost_min

        if connected_room not in connected_rooms:
            connected_rooms.append(connected_room)
        return connected_rooms

    def connect(self):
        """
        detects isolated rooms and generate a path to connect them
        :return:
        """
        self.clear()
        self.init_reachable_edges()
        self.init_connectivity_graph()
        self.expand_connectivity_graph()

    def plot(self, show: bool = False, save: bool = True, plot_edge=False):
        """
        plots plan with circulation paths
        :return:
        """

        ax = self.plan.plot(show=show, save=False)

        number_of_floors = self.plan.floor_count

        if plot_edge:
            for f in self.plan.list_level:
                _ax = ax[f] if number_of_floors > 1 else ax
                paths = self.connecting_edges[f]
                for path in paths:
                    for edge in path:
                        edge.plot(ax=_ax, color='blue')
                        # representing the growing direction
                        pt_tmp = move_point([edge.start.x, edge.start.y], edge.vector, 0.5)
                        pt = move_point(pt_tmp, self.growing_directions[f][edge], edge.length / 5)
                        _ax.scatter(pt[0], pt[1], marker='o', color='k')
        else:
            for f in self.plan.list_level:
                _ax = ax[f] if number_of_floors > 1 else ax
                paths = self.connecting_paths[f]
                for path in paths:
                    if len(path) == 1:
                        _ax.scatter(path[0].x, path[0].y, marker='o', s=15, facecolor='blue')
                    else:
                        for i in range(len(path) - 1):
                            v1 = path[i]
                            v2 = path[i + 1]
                            x_coords = [v1.x, v2.x]
                            y_coords = [v1.y, v2.y]
                            _ax.plot(x_coords, y_coords, 'k',
                                     linewidth=2,
                                     color="blue",
                                     solid_capstyle='butt')

        plot_save(save, show)


class PathCalculator:
    """
    PathCalculator class
    builds and manages a graph that can be used by a circulator so as to compute shortest path
    between two spaces independant from the library used to build the graph
    """

    def __init__(self, plan: Plan, cost_rules: Dict = None, graph_lib: str = 'networkx'):
        self.plan = plan
        self.graph_lib = graph_lib
        self.graph = None
        self.cost_rules = cost_rules

        window_cat = [cat for cat in LINEAR_CATEGORIES.keys() if
                      LINEAR_CATEGORIES[cat].window_type]
        self.component_edges = {'duct_edges': self.plan.category_edges('duct'),
                                'window_edges': self.plan.category_edges(*window_cat)}

    def __repr__(self):
        output = 'Grapher:\n'
        output += 'graph library :' + self.graph_lib + '\n'
        return output

    def build(self):
        """
        runs through space edges and adds branches to the graph, for each branch computes a weight
        :return:
        """

        self.graph = {level: EdgeGraph(self.graph_lib) for level in self.plan.list_level}
        # self.graph = EdgeGraph(self.graph_lib)
        for space in self.plan.spaces:
            if space.mutable:
                self._update(space)

    def _update(self, space: Space):
        """
        add edge to the graph and computes its cost
        return:
        """
        graph = self.graph[space.floor.level]

        def get_space_info():
            # info needed on the space to attribute a cost to each edge of this space
            num_ducts = space.count_ducts()
            num_windows = space.count_windows()
            needed_ducts = list(
                needed_space for needed_space in space.category.needed_spaces if
                needed_space.name is 'duct')
            needed_windows = list(
                needed_linear for needed_linear in space.category.needed_linears if
                needed_linear.window_type)
            info = {
                "num_ducts": num_ducts,
                "num_windows": num_windows,
                "needed_ducts": needed_ducts,
                "needed_windows": needed_windows,
            }
            return info

        space_info = get_space_info()
        for edge in space.edges:
            cost = self.cost(edge, space_info)
            graph.add_edge(edge, cost)

    def set_corridor_to_zero_cost(self, path: List, level: int):
        """
        sets the const of circulation edges to zero
        :return:
        """
        nb_vert = len(path)
        if nb_vert > 1:
            for v in range(nb_vert - 1):
                vert1 = path[v]
                vert2 = path[v + 1]
                self.graph[level].add_edge_by_vert(vert1, vert2, 0)

    def rule_type(self, edge: Edge, space_info: Dict) -> str:
        """
        gets the rule for edge cost computation
        :return: float
        """
        rule = 'default'

        if (edge.pair and edge.pair in self.component_edges['duct_edges']
                and space_info["needed_ducts"]):
            if space_info["num_ducts"] <= 2:
                rule = 'water_room_less_than_two_ducts'
            else:
                rule = 'water_room_default'

        elif edge in self.component_edges['window_edges'] and space_info["needed_windows"]:
            if space_info["num_windows"] <= 2:
                rule = 'window_room_less_than_two_windows'
            else:
                rule = 'window_room_default'

        elif edge in self.component_edges['window_edges']:
            rule = 'circulation_along_window'

        return rule

    def cost(self, edge: Edge, space_info: Dict) -> float:
        """
        computes the cost of an edge
        :return: float
        """
        cost = edge.length

        rule = self.rule_type(edge, space_info)
        if rule not in self.cost_rules.keys():
            raise ValueError('The rule dict does not contain this rule {0}'.format(rule))
        cost += self.cost_rules[rule]

        return cost


COST_RULES = {
    'water_room_less_than_two_ducts': 10e5,
    'water_room_default': 1000,
    'window_room_less_than_two_windows': 10e10,
    'window_room_default': 5000,
    'circulation_along_window': 5000,
    'default': 0
}

if __name__ == '__main__':
    import libs.io.reader as reader
    from libs.modelers.seed import SEEDERS
    from libs.modelers.grid import GRIDS
    from libs.space_planner.space_planner import SPACE_PLANNERS
    from libs.plan.category import SPACE_CATEGORIES
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--plan_index", help="choose plan index",
                        default=0)

    args = parser.parse_args()
    plan_index = int(args.plan_index)

    logging.getLogger().setLevel(logging.DEBUG)


    def test_duplex():
        """
        Test
        :return:
        """
        boundaries = [(0, 500), (400, 500), (400, 0), (1500, 0), (1500, 700), (1000, 700),
                      (1000, 800),
                      (0, 800)]
        boundaries_2 = [(0, 500), (400, 500), (400, 400), (1000, 400), (1000, 800), (0, 800)]

        plan = Plan("Solution_Tests_Multiple_floors")
        floor_1 = plan.add_floor_from_boundary(boundaries, floor_level=0)
        floor_2 = plan.add_floor_from_boundary(boundaries_2, floor_level=1)

        terrace_coords = [(400, 400), (400, 200), (1300, 200), (1300, 700), (1000, 700),
                          (1000, 400)]
        plan.insert_space_from_boundary(terrace_coords, SPACE_CATEGORIES["terrace"], floor_1)
        garden_coords = [(400, 200), (400, 0), (1500, 0), (1500, 700), (1300, 700), (1300, 200)]
        plan.insert_space_from_boundary(garden_coords, SPACE_CATEGORIES["garden"], floor_1)
        duct_coords = [(350, 500), (400, 500), (400, 520), (350, 520)]
        plan.insert_space_from_boundary(duct_coords, SPACE_CATEGORIES["duct"], floor_1)
        duct_coords = [(350, 780), (400, 780), (400, 800), (350, 800)]
        plan.insert_space_from_boundary(duct_coords, SPACE_CATEGORIES["duct"], floor_1)
        hole_coords = [(400, 700), (650, 700), (650, 800), (400, 800)]
        plan.insert_space_from_boundary(hole_coords, SPACE_CATEGORIES["hole"], floor_1)
        plan.insert_linear((650, 800), (650, 700), LINEAR_CATEGORIES["startingStep"], floor_1)
        plan.insert_linear((275, 500), (340, 500), LINEAR_CATEGORIES["frontDoor"], floor_1)
        plan.insert_linear((550, 400), (750, 400), LINEAR_CATEGORIES["doorWindow"], floor_1)
        plan.insert_linear((1000, 450), (1000, 650), LINEAR_CATEGORIES["doorWindow"], floor_1)
        plan.insert_linear((0, 700), (0, 600), LINEAR_CATEGORIES["window"], floor_1)

        duct_coords = [(350, 500), (400, 500), (400, 520), (350, 520)]
        plan.insert_space_from_boundary(duct_coords, SPACE_CATEGORIES["duct"], floor_2)
        duct_coords = [(350, 780), (400, 780), (400, 800), (350, 800)]
        plan.insert_space_from_boundary(duct_coords, SPACE_CATEGORIES["duct"], floor_2)
        hole_coords = [(400, 700), (650, 700), (650, 800), (400, 800)]
        plan.insert_space_from_boundary(hole_coords, SPACE_CATEGORIES["hole"], floor_2)
        plan.insert_linear((650, 800), (650, 700), LINEAR_CATEGORIES["startingStep"], floor_2)
        plan.insert_linear((500, 400), (600, 400), LINEAR_CATEGORIES["window"], floor_2)
        plan.insert_linear((1000, 550), (1000, 650), LINEAR_CATEGORIES["window"], floor_2)
        plan.insert_linear((0, 700), (0, 600), LINEAR_CATEGORIES["window"], floor_2)

        GRIDS["sequence_grid"].apply_to(plan)

        plan.plot()

        SEEDERS["simple_seeder"].apply_to(plan)

        plan.plot()

        spec = reader.create_specification_from_file("test_solution_duplex_setup.json")
        spec.plan = plan
        plan.plot()
        space_planner = SPACE_PLANNERS["standard_space_planner"]
        # best_solutions = space_planner.apply_to(spec)

        return space_planner


    def connect_plan():
        """
        Test
        :return:
        """

        space_planner = test_duplex()

        if space_planner.solutions_collector.solutions:
            for solution in space_planner.solutions_collector.best():
                circulator = Circulator(plan=solution.plan, spec=space_planner.spec,
                                        cost_rules=COST_RULES)
                circulator.connect()
                circulator.plot()
                logging.debug('connecting paths: {0}'.format(circulator.connecting_paths))


    connect_plan()
