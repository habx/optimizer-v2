# coding=utf-8
"""
Circulation module

used to detect isolated rooms and generate a path to connect them

"""

import logging
from libs.plan import Space, Plan, Linear
from libs.mesh import Edge
import dijkstar
import libs.utils.graph as gr

from typing import Dict, List, Tuple
from tools.builds_plan import build_plan

COST_INITIALIZATION = 10e50


class Circulator:
    """
    Circulator Class
    """

    def __init__(self, plan: Plan, graph_manager):
        self.plan = plan
        self.graph_manager = graph_manager
        # self.isolated_rooms = self.get_isolated_rooms()
        self.connectivity_graph = gr.Graph_nx()

    def draw_path(self, space1: Space, space2: Space) -> Tuple['List[Vertex]', float]:
        graph = self.graph_manager.graph
        cost_min = COST_INITIALIZATION
        path_min = None
        for edge1 in space1.edges:
            for edge2 in space2.edges:
                path, cost = graph.get_shortest_path(edge1, edge2)
                if cost < cost_min:
                    cost_min = cost
                    path_min = path
        self.graph_manager.connecting_paths.append(path_min)
        self.graph_manager.set_corridor_to_zero_cost(path_min)
        return path_min, cost_min

    # todo : builds a graph of circulation rooms
    # if graph is not connected connect each component

    def init_connectivity_graph(self):

        circulation_spaces = []
        for space in self.plan.circulation_spaces():
            circulation_spaces.append(space)

        for room in self.plan.circulation_spaces():
            self.connectivity_graph.add_node(room)

        # building connectivity graph for circulation spaces
        for room in circulation_spaces:
            for other in circulation_spaces:
                if other is not room and other.adjacent_to(room):
                    self.connectivity_graph.add_edge(room, other)
            else:
                self.set_circulation_path()

    def expand_connectivity_graph(self):
        circulation_spaces = []
        for space in self.plan.circulation_spaces():
            circulation_spaces.append(space)
        for room in self.plan.mutable_spaces():
            if room not in list(self.connectivity_graph.nodes()):
                self.connectivity_graph.add_node(room)
                for other in circulation_spaces:
                    if other is not room and other.adjacent_to(room):
                        self.connectivity_graph.add_edge(room, other)

        for node in list(self.connectivity_graph.nodes()):
            if not self.connectivity_graph.node_connected(node):
                connected_room = self.connect_room_to_circulation_space(node)
                self.connectivity_graph.add_edge(connected_room, node)

    def set_circulation_path(self):
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

        for node in list(self.connectivity_graph.nodes()):
            if not self.connectivity_graph.has_path(node, father_node):
                self.draw_path(father_node, node)
                self.connectivity_graph.add_edge(node, father_node)

    def connect_room_to_circulation_graph(self, room):
        cost_min = COST_INITIALIZATION
        path_min = None
        connected_room = None
        for other in self.plan.circulation_spaces():
            if other is not room:
                path, cost = self.draw_path(room, other)
                if cost < cost_min:
                    cost = cost_min
                    path_min = path
                    connected_room = other
        # self.connecting_paths.append(path_min)
        # self.isolated_rooms.remove(room)
        # if connected_room in self.isolated_rooms():
        #    self.isolated_rooms.remove(connected_room)

        return connected_room

    def connect(self):
        self.init_connectivity_graph()
        self.expand_connectivity_graph()

    #
    # def get_isolated_rooms(self) -> List[Space]:
    #     list_isolated = []
    #     circulation_spaces = []
    #     for space in self.plan.circulation_spaces():
    #         circulation_spaces.append(space)
    #
    #     for room in self.plan.mutable_spaces():
    #         if room not in circulation_spaces:
    #             for circulation_space in circulation_spaces:
    #                 if room.adjacent_to(circulation_space):
    #                     break
    #             else:
    #                 list_isolated.append(room)
    #
    #     return list_isolated
    #
    # def connect(self):
    #     # first connect circulation rooms that are isolated
    #     while True:
    #         for room in self.isolated_rooms:
    #             if room in self.plan.circulation_spaces():
    #                 self.connect_room_to_circulation_space(room)
    #         else:
    #             break
    #     # connect other isolated rooms
    #     while True:
    #         for room in self.isolated_rooms:
    #             self.connect_room_to_circulation_space(room)
    #         else:
    #             break


"""
TODO Circulator:
-checks which rooms are isolated
-generates a path between two isolated rooms based on a graph
-non dependant of the used graph library
"""


class Graph_manager:
    """
    Graph_manager class
    builds and manages a graph that can be used by a circulator so as to compute shortest path between two spaces
    indendant from the library used to build the graph
    """

    """
    TODO Grapher:
    -separation of a class graph?
    -how to deal with typing as graph type is not known?
    """

    def __init__(self, plan: Plan, cost_rules: Dict = None, graph_lib: str = 'Dijkstar'):
        self.plan = plan
        self.graph_lib = graph_lib
        self.graph = None
        self.cost_rules = cost_rules
        self.connecting_paths = []

    def __repr__(self):
        output = 'Grapher:\n'
        output += 'graph library :' + self.graph_lib + '\n'
        return output

    def build(self):
        # runs through edges and adds branches to the graph, for each branch computes a weight
        plan = self.plan
        self.graph = Graph(self.graph_lib)
        graph = self.graph
        graph.init()
        seen = []

        for space in plan.spaces:
            if space.mutable:
                for edge in space.edges:
                    if edge not in seen:
                        added_pair = self.update(edge)
                        seen.append(edge)
                        if added_pair:
                            seen.append(edge.pair)

        graph.set_cost_function()

    def update(self, edge: Edge):
        added_pair = False
        graph = self.graph
        cost = self.cost(edge)
        graph.add_edge(edge, cost)
        if edge.pair and edge.pair.is_space_boundary:
            added_pair = True
            graph.add_edge(edge.pair, cost)

        return added_pair

    def set_corridor_to_zero_cost(self, path):
        nb_vert = len(path)
        if nb_vert > 1:
            for v in range(nb_vert - 1):
                vert1 = path[v]
                vert2 = path[v + 1]
                self.graph.add_edge_by_vert(vert1, vert2, 0)

            self.graph.set_cost_function()

    # def set_cost_fixed_items(self, cost_rules: Dict):
    #     for space in self.plan.mutable_spaces:
    #         for edge in space.edges:
    #             if edge.

    def cost(self, edge: Edge, cost_fixed_items: Dict = None):
        cost = edge.length
        # # TODO : add list of rules for cost
        # if not edge.is_mutable and edge in cost_fixed_items.keys():
        #     cost += cost_fixed_items[edge]
        # elif edge.pair:
        #     if not edge.pair.is_mutable and edge.pair in cost_fixed_items.keys():
        #         cost += cost_fixed_items[edge.pair.space]
        return cost

    def get_shortest_path(self, edge1: Edge, edge2: Edge):
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
        if self.graph_lib == 'Dijkstar':
            self.graph_struct = dijkstar.Graph()
        else:
            raise ValueError('graph library does not exit')

    def add_edge_by_vert(self, vert1, vert2, cost):
        if self.graph_lib == 'Dijkstar':
            self.graph_struct.add_edge(vert1, vert2, {'cost': cost})
        else:
            raise ValueError('graph library does not exit')

    def add_edge(self, edge, cost):
        if self.graph_lib == 'Dijkstar':
            self.graph_struct.add_edge(edge.start, edge.end, {'cost': cost})
        else:
            raise ValueError('graph library does not exit')

    def set_cost_function(self):
        if self.graph_lib == 'Dijkstar':
            self.cost_function = lambda u, v, e, prev_e: e['cost']
        else:
            raise ValueError('graph library does not exit')

    def get_shortest_path(self, edge1, edge2):
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


    def generate_path():
        """
        Test
        :return:
        """
        plan_index = 2
        input_file = reader.get_list_from_folder(reader.DEFAULT_BLUEPRINT_INPUT_FOLDER)[
            plan_index]  # 9 Antony B22, 13 Bussy 002

        input_file = "Antony_A22.json"
        plan = build_plan(input_file)

        graph_manager = Graph_manager(plan=plan)
        graph_manager.build()

        circulator = Circulator(plan=plan, graph_manager=graph_manager)

        link_space = []
        for space in plan.spaces:
            if space.mutable:
                link_space.append(space)
                # if (len(link_space) == 2):
                #    break
        ind_i = 5
        ind_f = 9

        path_min, cost_min = circulator.draw_path(link_space[ind_i], link_space[ind_f])

        logging.debug('path: {0}'.format(path_min))
        logging.debug('spaces: {0} - {1}'.format(link_space[ind_i], link_space[ind_f]))
        logging.debug('space center: {0} - {1}'.format(link_space[ind_i].as_sp.centroid.coords.xy,
                                                       link_space[ind_f].as_sp.centroid.coords.xy))
        plan.plot(show=True, path_min=path_min)


    def connect_complete():
        """
        Test
        :return:
        """
        plan_index = 2
        input_file = reader.get_list_from_folder(reader.DEFAULT_BLUEPRINT_INPUT_FOLDER)[
            plan_index]  # 9 Antony B22, 13 Bussy 002

        input_file = "Antony_A22.json"
        plan = build_plan(input_file)

        graph_manager = Graph_manager(plan=plan)
        graph_manager.build()

        circulator = Circulator(plan=plan, graph_manager=graph_manager)

        circulator.connect()
        print("CONNECTING PATH")
        print(circulator.graph_manager.connecting_paths)


    connect_complete()

# TODO :
# TODO : lier des espaces sachant la présence des couloirs existants
# TODO : gestion des murs porteurs : ajout d'edge travesant en projetant les points des espaces? => considération obligatoire des espaces non mutables dans lee graphe?
# TODO : ajout des règles par inamovible via liste
# TODO : on pourrait dans le graphe ne manipuler que des edge -> la circulation finale est l'ensemble des edges sauf le dernier, et juste le start de l'edge s'il n'y en a qu'un
