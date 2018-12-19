# coding=utf-8
"""
Circulation module

used to detect isolated rooms and generate a path to connect them

"""

import logging
from libs.plan import Space, PlanComponent, Plan, Linear
from libs.mesh import Edge
import dijkstar

from typing import Dict, List
from tools.builds_plan import build_plan


class Circulator:
    """
    Circulator Class
    """
    from libs.circulation import Graph_manager
    def __init__(self, plan: Plan, graph_manager: Graph_manager):
        self.plan = plan
        self.graph_manager = graph_manager

    def draw_path(self, space1: Space, space2: Space) -> List[Edge]:
        graph = self.graph_manager.graph
        cost_min = 10e50
        path_min = None
        for edge1 in space1.edges:
            for edge2 in space2.edges:
                path, cost = graph.get_shortest_path(edge1, edge2)
                if cost < cost_min:
                    cost_min = cost
                    path_min = path
        return path_min, cost_min


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

    def __repr__(self):
        output = 'Grapher:\n'
        output += 'graph library :' + self.graph_lib + '\n'
        return output

    def build(self):
        # runs through edges and adds branches to the graph, for each branch computes a weight
        plan = self.plan
        self.graph = Graph(self.graph_lib)
        graph = self.graph
        graph.init(self.graph_lib)
        seen = []

        for edge in plan.mutable_spaces.edges:
            if edge not in seen:
                added_pair = graph.update(graph, edge)
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

    # def set_cost_fixed_items(self, cost_rules: Dict):
    #     for space in self.plan.mutable_spaces:
    #         for edge in space.edges:
    #             if edge.

    def cost(self, edge: Edge, cost_fixed_items: Dict):
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
    Graph Class:
    builds a graph,
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
        cost = search_tree_result[1]
        return path, cost


if __name__ == '__main__':
    import libs.reader as reader
    from libs.grid import GRIDS
    from libs.selector import SELECTORS
    from libs.shuffle import SHUFFLES
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
        input_file = reader.get_list_from_folder(reader.DEFAULT_BLUEPRINT_INPUT_FOLDER)[
            plan_index]  # 9 Antony B22, 13 Bussy 002

        plan = build_plan(input_file)

        graph_manager = Graph_manager(plan=plan)
        graph_manager.build(plan)

        circulator = Circulator(plan=plan, graph_manager=graph_manager)

        link_space = []
        for space in plan.mutable_spaces:
            link_space.append(space)
            if (len(link_space) == 2):
                break

        path_min, cost_min = circulator.draw_path(link_space[0], link_space[1])


    generate_path()
