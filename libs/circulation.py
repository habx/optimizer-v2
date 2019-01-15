# coding=utf-8
"""
Circulation module

used to detect isolated rooms and generate a path to connect them

"""

import logging
from typing import Dict, List, Tuple
from libs.plan import Space, Plan, Vertex
from libs.mesh import Edge
from libs.plot import plot_save
from libs.utils.graph import Graph_nx, EdgeGraph
from libs.category import LINEAR_CATEGORIES
from tools.build_plan import build_plan


# TODO : deal with load bearing walls by defining locations where they can be crossed

class Circulator:
    """
    Circulator Class
    contains utilities to detect isolated rooms connect them to circulation spaces
    """

    def __init__(self, plan: Plan, cost_rules: Dict = None):
        self.plan = plan
        self.path_calculator = PathCalculator(plan=self.plan, cost_rules=cost_rules)
        self.path_calculator.build()
        self.connectivity_graph = Graph_nx()
        self.connecting_paths = []

    def draw_path(self, space1: Space, space2: Space) -> Tuple['List[Vertex]', float]:
        """
        Finds the shortest path between two spaces in the plan
        :return list of vertices on the path and cost of the path
        """
        graph = self.path_calculator.graph
        path_min = None
        cost_min = None
        # tests all possible connections between both spaces
        # TODO : that's brutal, any more clever way to connect two sub graphs
        for edge1 in space1.edges:
            for edge2 in space2.edges:
                path, cost = graph.get_shortest_path(edge1, edge2)
                if cost_min is None or cost < cost_min:
                    cost_min = cost
                    path_min = path

        return path_min, cost_min

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
                connected_room = self.connect_space_to_circulation_graph(node)
                self.connectivity_graph.add_edge(connected_room, node)

    def set_circulation_path(self):
        """
        ensures circulation spaces are all connected
        :return:
        """
        father_node = None
        for room in self.plan.mutable_spaces():
            if room.category.name is 'entrance':
                father_node = room
                break
        else:
            for room in self.plan.mutable_spaces():
                if room.category.name is 'living':
                    father_node = room
                    break

        if not father_node:
            return True

        for node in self.connectivity_graph.nodes():
            if not self.connectivity_graph.has_path(node, father_node):
                path, cost = self.draw_path(father_node, node)
                self.actualize_path(path)
                self.connectivity_graph.add_edge(node, father_node)

    def actualize_path(self, path):
        """
        update based on computed corridor path
        :return:
        """
        self.connecting_paths.append(path)
        # when a circulation has been set, it can be used to connect every other spaces
        # without cost increase
        self.path_calculator.set_corridor_to_zero_cost(path)

    def connect_space_to_circulation_graph(self, space):
        """
        connects the given space with a circulation space of the plan
        :return:
        """
        path_min = None
        connected_room = None
        cost_min = None
        for other in self.plan.circulation_spaces():
            if other is not space:
                path, cost = self.draw_path(space, other)
                if cost_min is None or cost < cost_min:
                    cost_min = cost
                    path_min = path
                    connected_room = other
        if path_min is not None:
            self.actualize_path(path_min)

        return connected_room

    def connect(self):
        """
        detects isolated rooms and generate a path to connect them
        :return:
        """
        self.init_connectivity_graph()
        self.expand_connectivity_graph()

    def plot(self, show: bool = False, save: bool = True):
        """
        plots plan with circulation paths
        :return:
        """

        ax = self.plan.plot(show=show, save=False)

        paths = self.connecting_paths
        for path in paths:
            if len(path) == 1:
                ax.scatter(path[0].x, path[0].y, marker='o', s=15, facecolor='blue')
            else:
                for i in range(len(path) - 1):
                    v1 = path[i]
                    v2 = path[i + 1]
                    x_coords = [v1.x, v2.x]
                    y_coords = [v1.y, v2.y]
                    ax.plot(x_coords, y_coords, 'k',
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

    def __init__(self, plan: Plan, cost_rules: Dict = None, graph_lib: str = 'Dijkstar'):
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
        self.graph = EdgeGraph(self.graph_lib)
        graph = self.graph
        graph.init()

        for space in self.plan.spaces:
            if space.mutable:
                self._update(space)

        graph.set_cost_function()

    def _update(self, space: Space):
        """
        add edge to the graph and computes its cost
        return:
        """
        graph = self.graph
        for edge in space.edges:
            cost = self.cost(edge, space)
            graph.add_edge(edge, cost)

    def set_corridor_to_zero_cost(self, path):
        """
        sets the const of circulation edges to zero
        :return:
        """
        nb_vert = len(path)
        if nb_vert > 1:
            for v in range(nb_vert - 1):
                vert1 = path[v]
                vert2 = path[v + 1]
                self.graph.add_edge_by_vert(vert1, vert2, 0)

            self.graph.set_cost_function()

    def rule_type(self, edge: Edge, space: Space) -> str:
        """
        gets the rule for edge cost computation
        :return: float
        """
        rule = 'default'

        num_ducts = space.count_ducts()
        num_windows = space.count_windows()

        if (edge.pair and edge.pair in self.component_edges['duct_edges']
                and list(needed_space for needed_space in space.category.needed_spaces if
                         needed_space.name is 'duct')):
            if num_ducts <= 2:
                rule = 'water_room_less_than_two_ducts'
            else:
                rule = 'water_room_default'


        elif (edge in self.component_edges['window_edges'] and list(
                needed_linear for needed_linear in space.category.needed_linears if
                needed_linear.window_type)):
            if num_windows <= 2:
                rule = 'window_room_less_than_two_windows'
            else:
                rule = 'window_room_default'

        return rule

    def cost(self, edge: Edge, space: Space) -> float:
        """
        computes the cost of an edge
        :return: float
        """
        cost = edge.length / 100

        rule = self.rule_type(edge, space)
        # rule='default'
        if rule not in self.cost_rules.keys():
            raise ValueError('The rule dict does not contain this rule {0}'.format(rule))
        cost += self.cost_rules[rule]

        return cost

    def get_shortest_path(self, edge1: Edge, edge2: Edge) -> Tuple['List[Vertex]', float]:
        """
        get the shortest path between two edges
        :return list of vertices on the path and cost of the path
        """
        graph = self.graph
        return graph.get_shortest_path(self, edge1, edge2)


if __name__ == '__main__':
    import libs.reader as reader
    from libs.seed import Seeder, GROWTH_METHODS, FILL_METHODS
    from libs.selector import SELECTORS
    from libs.grid import GRIDS
    from libs.shuffle import SHUFFLES
    from libs.space_planner import SpacePlanner
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--plan_index", help="choose plan index",
                        default=0)

    args = parser.parse_args()
    plan_index = int(args.plan_index)

    logging.getLogger().setLevel(logging.DEBUG)


    def connect_plan():
        """
        Test
        :return:
        """
        input_file = reader.get_list_from_folder(reader.DEFAULT_BLUEPRINT_INPUT_FOLDER)[
            plan_index]  # 9 Antony B22, 13 Bussy 002

        plan = reader.create_plan_from_file(input_file)

        GRIDS["ortho_grid"].apply_to(plan)
        seeder = Seeder(plan, GROWTH_METHODS).add_condition(SELECTORS["seed_duct"], "duct")
        (seeder.plant()
         .grow()
         .shuffle(SHUFFLES["seed_square_shape"])
         .fill(FILL_METHODS, (SELECTORS["farthest_couple_middle_space_area_min_100000"], "empty"))
         .fill(FILL_METHODS, (SELECTORS["single_edge"], "empty"), recursive=True)
         .simplify(SELECTORS["fuse_small_cell"])
         .shuffle(SHUFFLES["seed_square_shape"]))

        input_setup = input_file[:-5] + "_setup.json"
        spec = reader.create_specification_from_file(input_setup)
        spec.plan = plan

        space_planner = SpacePlanner("test", spec)
        space_planner.solution_research()

        cost_rules = {
            'water_room_less_than_two_ducts': 10e5,
            'water_room_default': 1000,
            'window_room_less_than_two_windows': 10e10,
            'window_room_default': 5000,
            'default': 0
        }

        for solution in space_planner.solutions_collector.best():
            solution.plan.plot()
            circulator = Circulator(plan=solution.plan, cost_rules=cost_rules)
            circulator.connect()
            circulator.plot()
            logging.debug('connecting paths: {0}'.format(circulator.connecting_paths))


    connect_plan()
