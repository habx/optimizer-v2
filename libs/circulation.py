# coding=utf-8
"""
Circulation module

used to detect isolated rooms and generate a path to connect them

"""

import logging
from libs.plan import Space, Plan, Vertex
from libs.mesh import Edge
from libs.plot import plot_save
import dijkstar
import libs.utils.graph as gr
from libs.category import LINEAR_CATEGORIES
from typing import Dict, List, Tuple

from tools.build_plan import build_plan

COST_INITIALIZATION = 10e50


# TODO : deal with load bearing walls by defining locations where they can be crossed

class Circulator:
    """
    Circulator Class
    contains utilities to detect isolated rooms connect them to circulation spaces
    """

    def __init__(self, plan: Plan, graph_manager):
        self.plan = plan
        self.graph_manager = graph_manager
        self.connectivity_graph = gr.Graph_nx()

    def draw_path(self, space1: Space, space2: Space) -> Tuple['List[Vertex]', float]:
        """
        Finds the shortest path between two spaces in the plan
        :return list of vertices on the path and cost of the path
        """
        graph = self.graph_manager.graph
        cost_min = COST_INITIALIZATION
        path_min = None
        # tests all possible connections between both spaces
        for edge1 in space1.edges:
            for edge2 in space2.edges:
                path, cost = graph.get_shortest_path(edge1, edge2)
                if cost < cost_min:
                    cost_min = cost
                    path_min = path

        return path_min, cost_min

    def init_connectivity_graph(self):
        """
        builds a connectivity graph of the plan, each circulation space is a node
        :return:
        """
        circulation_spaces = []
        for space in self.plan.circulation_spaces():
            circulation_spaces.append(space)

        for space in circulation_spaces:
            self.connectivity_graph.add_node(space)

        # builds connectivity graph for circulation spaces
        for space in circulation_spaces:
            for other in circulation_spaces:
                if other is not space and other.adjacent_to(space):
                    # if spaces are adjacent, they are connected in the graph
                    self.connectivity_graph.add_edge(space, other)
            else:
                self.set_circulation_path()

    def expand_connectivity_graph(self):
        """
        connects each non circulation space of the plan to a circulation space
        :return:
        """
        circulation_spaces = []
        for space in self.plan.circulation_spaces():
            circulation_spaces.append(space)
        for space in self.plan.mutable_spaces():
            if space not in self.connectivity_graph.nodes():
                self.connectivity_graph.add_node(space)
                for other in circulation_spaces:
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
                self.actualize_graph_manager(path)
                self.connectivity_graph.add_edge(node, father_node)

    def actualize_graph_manager(self, path):
        """
        actualize the graph manager based on computed corridor path
        :return:
        """
        self.graph_manager.connecting_paths.append(path)
        # when a circulation has been set, it can be used to connect every other spaces
        # without cost increase
        self.graph_manager.set_corridor_to_zero_cost(path)

    def connect_space_to_circulation_graph(self, space):
        """
        connects the given space with a circulation space of the plan
        :return:
        """
        cost_min = COST_INITIALIZATION
        path_min = None
        connected_room = None
        for other in self.plan.circulation_spaces():
            if other is not space:
                path, cost = self.draw_path(space, other)
                if cost < cost_min:
                    cost_min = cost
                    path_min = path
                    connected_room = other
        if path_min is not None:
            self.actualize_graph_manager(path_min)

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
        paths = self.graph_manager.connecting_paths
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


class GraphManager:
    """
    Graph_manager class
    builds and manages a graph that can be used by a circulator so as to compute shortest path
    between two spaces independant from the library used to build the graph
    """

    def __init__(self, plan: Plan, cost_rules: Dict = None, graph_lib: str = 'Dijkstar'):
        self.plan = plan
        self.graph_lib = graph_lib
        self.graph = None
        self.cost_rules = cost_rules
        self.connecting_paths = []

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
        self.graph = Graph(self.graph_lib)
        graph = self.graph
        graph.init()

        for space in self.plan.spaces:
            if space.mutable:
                self.update(space)

        graph.set_cost_function()

    def update(self, space: Space):
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

        if edge.pair and edge.pair in self.component_edges['duct_edges'] \
                and space.category.needs_duct:
            if num_ducts <= 2:
                rule = 'water_room_less_than_two_ducts'
            else:
                rule = 'water_room_default'

        elif edge in self.component_edges['window_edges'] and space.category.needs_window:
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


class Graph:
    """
    Graph Graph:
    function to build and deal with graph, implementation for given graph libaries
    """

    def __init__(self, graph_lib: str = 'Dijkstar'):
        self.graph_lib = graph_lib
        self.cost_function = None
        self.graph_struct = None

    def __repr__(self):
        output = 'Graph:\n'
        output += 'graph library :' + self.graph_lib + '\n'
        return output

    def init(self):
        """
        graph initialization
        :return:
        """
        if self.graph_lib == 'Dijkstar':
            self.graph_struct = dijkstar.Graph()
        else:
            raise ValueError('graph library does not exit')

    def add_edge_by_vert(self, vert1: Vertex, vert2: Vertex, cost: float):
        """
        add edge to the graph
        :return:
        """
        if self.graph_lib == 'Dijkstar':
            self.graph_struct.add_edge(vert1, vert2, {'cost': cost})
        else:
            raise ValueError('graph library does not exit')

    def add_edge(self, edge: Edge, cost: float):
        """
        add edge to the graph
        :return:
        """
        if self.graph_lib == 'Dijkstar':
            self.graph_struct.add_edge(edge.start, edge.end, {'cost': cost})
            self.graph_struct.add_edge(edge.end, edge.start, {'cost': cost})
        else:
            raise ValueError('graph library does not exit')

    def set_cost_function(self):
        """
        sets the graph cost function
        :return:
        """
        if self.graph_lib == 'Dijkstar':
            self.cost_function = lambda u, v, e, prev_e: e['cost']
        else:
            raise ValueError('graph library does not exit')

    def get_shortest_path(self, edge1, edge2) -> Tuple['List[Vertex]', float]:
        """
        get the shortest path between two edges
        :return list of vertices on the path and cost of the path
        """
        if self.graph_lib == 'Dijkstar':
            search_tree_result = dijkstar.find_path(self.graph_struct, edge1.start, edge2.start,
                                                    cost_func=self.cost_function)
        else:
            raise ValueError('graph library does not exit')
        path = search_tree_result[0]
        cost = search_tree_result[3]
        return path, cost


if __name__ == '__main__':
    import libs.reader as reader
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

        plan = build_plan(input_file)

        cost_rules = {
            'water_room_less_than_two_ducts': 10e5,
            'water_room_default': 1000,
            'window_room_less_than_two_windows': 10e10,
            'window_room_default': 5000,
            'default': 0
        }

        graph_manager = GraphManager(plan=plan, cost_rules=cost_rules)
        graph_manager.build()

        circulator = Circulator(plan=plan, graph_manager=graph_manager)

        circulator.connect()
        logging.debug('connecting paths: {0}'.format(circulator.graph_manager.connecting_paths))

        circulator.plot()


    connect_plan()
