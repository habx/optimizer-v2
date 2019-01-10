# coding=utf-8
"""
Space Planner module

The space planner finds the best rooms layouts according to a plan with given seed spaces
and a customer input setup

"""
import logging
from libs.specification import Specification
from libs.solution import SolutionsCollector
from libs.plan import Plan
from libs.constraints_manager import ConstraintsManager
from libs.seed import Seeder, GROWTH_METHODS, FILL_METHODS
import networkx as nx


class SpacePlanner:
    """
    Space planner Class
    """

    def __init__(self, name: str, spec: 'Specification'):
        self.name = name
        self.spec = spec
        logging.debug(spec)

        self.manager = ConstraintsManager(self)

        self.spaces_adjacency_matrix = []
        self._init_spaces_adjacency()

        self.solutions_collector = SolutionsCollector(spec)

    def __repr__(self):
        output = 'SpacePlanner' + self.name
        return output

    def _init_spaces_adjacency(self) -> None:
        """
        spaces adjacency matrix init
        :return: None
        """
        for i, i_space in enumerate(self.spec.plan.mutable_spaces()):
            self.spaces_adjacency_matrix.append([])
            for j, j_space in enumerate(self.spec.plan.mutable_spaces()):
                if j != i:
                    self.spaces_adjacency_matrix[i].append(0)
                else:
                    self.spaces_adjacency_matrix[i].append(1)

        for i, i_space in enumerate(self.spec.plan.mutable_spaces()):
            for j, j_space in enumerate(self.spec.plan.mutable_spaces()):
                if j < i:
                    if i_space.adjacent_to(j_space):
                        self.spaces_adjacency_matrix[i][j] = 1
                        self.spaces_adjacency_matrix[j][i] = 1
                    else:
                        self.spaces_adjacency_matrix[i][j] = 0
                        self.spaces_adjacency_matrix[j][i] = 0

    def _check_adjacency(self, room_positions, connectivity_checker) -> bool:
        """
        Experimental function using BFS graph analysis in order to check wether each room is
        connected.
        A room is considered a subgraph of the voronoi graph.
        :param room_positions:
        :param connectivity_checker:
        :return: a boolean indicating wether each room is connected

        """
        # check for the connectivity of each room
        for i_item, item in enumerate(self.spec.items):
            # compute the number of fixed item in the room
            nbr_cells_in_room = sum(room_positions[i_item])
            # if a room has only one fixed item there is no need to check for adjacency
            if nbr_cells_in_room <= 1:
                continue
            # else check the connectivity of the subgraph composed of the fi inside the given room
            room_line = room_positions[i_item]
            fi_in_room = tuple([i for i, e in enumerate(room_line) if e])
            if not connectivity_checker(fi_in_room):
                return False

        return True

    def _check_validity(self) -> None:
        """
        check_connectivity of constraint programming solutions and remove wrong results of
        self.manager.solver.solutions
        :return: None
        """
        connectivity_checker = check_room_connectivity_factory(self.spaces_adjacency_matrix)

        sol_to_remove = []
        for sol in self.manager.solver.solutions:
            is_a_good_sol = self._check_adjacency(sol, connectivity_checker)
            if not is_a_good_sol:
                sol_to_remove.append(sol)

        if sol_to_remove:
            for sol in sol_to_remove:
                self.manager.solver.solutions.remove(sol)

    def _rooms_building(self, plan: 'Plan', matrix_solution) -> 'Plan':
        """
        Builds the rooms requested in the specification from the matrix and seed spaces.
        :param: plan
        :param: matrix_solution
        :return: built plan
        """
        dict_items_spaces = {}
        for i_item, item in enumerate(self.spec.items):
            item_space = []
            for j_space, space in enumerate(plan.mutable_spaces()):
                if matrix_solution[i_item][j_space] == 1:
                    space.category = item.category
                    item_space.append(space)
            dict_items_spaces[item] = item_space

        for item in self.spec.items:
            item_space = dict_items_spaces[item]
            if len(item_space) > 1:
                space_ini = item_space[0]
                i = 0
                while (len(item_space) > 1) and i < len(item_space) * len(item_space):
                    for space in item_space[1:]:
                        if space.adjacent_to(space_ini):
                            space_ini.merge(space)
                            plan.remove_null_spaces()
                            item_space.remove(space)
                            break
                    i += 1
        assert plan.check()
        return plan

    def solution_research(self) -> None:
        """
        Looks for all possible solutions then find the three best solutions
        :return: None
        """

        self.manager.solver.solve()

        if len(self.manager.solver.solutions) == 0:
            logging.warning('Plan without space planning solution')
        else:
            self._check_validity()
            logging.info('Plan with {0} solutions'.format(len(self.manager.solver.solutions)))
            logging.debug(self.spec.plan)
            for i, sol in enumerate(self.manager.solver.solutions):
                plan_solution = self.spec.plan.clone()
                plan_solution = self._rooms_building(plan_solution, sol)
                self.solutions_collector.add_plan(plan_solution)
                logging.debug(plan_solution)

            best_sol = self.solutions_collector.find_best_solutions()
            for sol in best_sol:
                logging.debug(sol)
                sol.plan.plot()

    def generate_best_solutions_files(self, best_sol: list('Solution')):
        """
        Generates the output files of the chosen solutions
        :return: None
        """


def adjacency_matrix_to_graph(matrix):
    """
    Converts adjacency matrix to a networkx graph structure,
    a value of 1 in the matrix correspond to an edge in the Graph
    :param matrix: an adjacency_matrix
    :return: a networkx graph structure
    """

    nb_cells = len(matrix)  # get the matrix dimensions
    graph = nx.Graph()
    edge_list = [(i, j) for i in range(nb_cells) for j in range(nb_cells) if
                 matrix[i][j] == 1]
    graph.add_edges_from(edge_list)

    return graph


def check_room_connectivity_factory(adjacency_matrix):
    """

    A factory to enable memoization on the check connectivity room

    :param adjacency_matrix: an adjacency_matrix
    :return: check_room_connectivity: a memoized function returning the connectivity of a room
    """

    connectivity_cache = {}
    # create graph from adjacency_matrix
    graph = adjacency_matrix_to_graph(adjacency_matrix)

    def check_room_connectivity(fi_in_room):
        """
        :param fi_in_room: a tuple indicating the fixed items present in the room
        :return: a Boolean indicating if the fixed items in the room are connected according to the
        graph
        """

        # check if the connectivity of these fixed items has already been checked
        # if it is the case fetch the result from the cache
        if fi_in_room in connectivity_cache:
            return connectivity_cache[fi_in_room]

        # else compute the connectivity and stores the result in the cache
        is_connected = nx.is_connected(graph.subgraph(fi_in_room))
        connectivity_cache[fi_in_room] = is_connected

        return is_connected

    # return the memorized function
    return check_room_connectivity


if __name__ == '__main__':

    import libs.reader as reader
    from libs.selector import SELECTORS
    from libs.grid import GRIDS
    from libs.shuffle import SHUFFLES
    import argparse

    logging.getLogger().setLevel(logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--plan_index", help="choose plan index",
                        default=0)

    args = parser.parse_args()
    plan_index = int(args.plan_index)


    def space_planning():
        """
        Test
        :return:
        """

        input_file = reader.get_list_from_folder(reader.DEFAULT_BLUEPRINT_INPUT_FOLDER)[
            plan_index]  # 9 Antony B22, 13 Bussy 002
        input_file = 'Antony_A22.json'  # 5 Levallois_Letourneur / Antony_A22
        plan = reader.create_plan_from_file(input_file)

        GRIDS['ortho_grid'].apply_to(plan)

        seeder = Seeder(plan, GROWTH_METHODS).add_condition(SELECTORS['seed_duct'], 'duct')
        (seeder.plant()
         .grow()
         .shuffle(SHUFFLES['seed_square_shape'])
         .fill(FILL_METHODS, (SELECTORS["farthest_couple_middle_space_area_min_100000"],
                              "empty"))
         .fill(FILL_METHODS, (SELECTORS["single_edge"], "empty"), recursive=True)
         .simplify(SELECTORS["fuse_small_cell"])
         .shuffle(SHUFFLES['seed_square_shape']))
        plan.plot()
        # input_file = 'Antony_A22_setup.json'
        input_file_setup = input_file[:-5]+"_setup.json"
        spec = reader.create_specification_from_file(input_file_setup)
        spec.plan = plan

        space_planner = SpacePlanner('test', spec)
        space_planner.solution_research()

    space_planning()
