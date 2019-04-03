# coding=utf-8
"""
Space Planner module

The space planner finds the best rooms layouts according to a plan with given seed spaces
and a customer input setup

"""
import logging
from typing import List, Optional, Dict
from libs.specification.specification import Specification, Item
from libs.specification.size import Size
from libs.space_planner.solution import SolutionsCollector, Solution
from libs.plan.plan import Plan, Space
from libs.space_planner.constraints_manager import ConstraintsManager
from libs.modelers.seed import Seeder, GROWTH_METHODS
from libs.plan.category import SPACE_CATEGORIES
from libs.modelers.shuffle import SHUFFLES
import libs.io.writer as writer
import networkx as nx

SQM = 10000


class SpacePlanner:
    """
    Space planner Class
    """

    def __init__(self, name: str, spec: 'Specification'):
        self.name = name
        self._init_spec(spec)
        self._plan_cleaner()
        logging.debug(self.spec)

        self.manager = ConstraintsManager(self)

        self.spaces_adjacency_matrix = []
        self._init_spaces_adjacency()

        self.solutions_collector = SolutionsCollector(self.spec)

    def __repr__(self):
        output = "SpacePlanner" + self.name
        return output

    def _init_spec(self, spec: 'Specification') -> None:
        """
        change reader specification :
        living + kitchen : opensOn --> livingKitchen
        area convergence
        :return: None
        """
        space_planner_spec = Specification('SpacePlannerSpecification', spec.plan)

        for item in spec.items:
            if ((item.category.name != "living" or "kitchen" not in item.opens_on) and
                    (item.category.name != "kitchen" or len(item.opens_on) == 0)):
                space_planner_spec.add_item(item)
            elif item.category.name == "living" and "kitchen" in item.opens_on:
                kitchens = spec.category_items("kitchen")
                for kitchen_item in kitchens:
                    if "living" in kitchen_item.opens_on:
                        size_min = Size(area=(kitchen_item.min_size.area + item.min_size.area))
                        size_max = Size(area=(kitchen_item.max_size.area + item.max_size.area))
                        opens_on = item.opens_on.remove("kitchen")
                        new_item = Item(SPACE_CATEGORIES["livingKitchen"], item.variant, size_min,
                                        size_max, opens_on, item.linked_to)
                        space_planner_spec.add_item(new_item)

        category_name_list = ["entrance", "toilet", "bathroom", "laundry", "dressing", "kitchen",
                              "living", "livingKitchen", "dining", "bedroom", "study", "misc",
                              "circulationSpace"]
        space_planner_spec.init_id(category_name_list)

        # area
        coeff = int(spec.plan.indoor_area) / int(sum(item.required_area for item in spec.items))
        for item in spec.items:
            item.min_size.area = item.min_size.area * coeff
            item.max_size.area = item.max_size.area * coeff
        logging.debug("SP - PLAN AREA : %i", int(spec.plan.indoor_area))
        logging.debug("SP - Setup AREA : %i", int(sum(item.required_area for item in spec.items)))
        self.spec = space_planner_spec

    def _plan_cleaner(self, min_area: float = 100) -> None:
        """
        Plan cleaner for little spaces
        :return: None
        """
        self.spec.plan.remove_null_spaces()
        for space in self.spec.plan.spaces:
            if space.area < min_area:
                self.spec.plan.remove(space)

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

    def _rooms_building(self, plan: 'Plan', matrix_solution) -> ('Plan', Dict['Item', 'Space']):
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

        # circulationSpace case :
        for j_space, space in enumerate(plan.mutable_spaces()):
            if space.category.name == "seed":
                space.category = SPACE_CATEGORIES["circulationSpace"]

        dict_items_space = {}
        for item in self.spec.items:
            item_space = dict_items_spaces[item]
            if len(item_space) > 1:
                space_ini = item_space[0]
                item_space.remove(item_space[0])
                i = 0
                iter_max = len(item_space) ** 2
                while (len(item_space) > 0) and i < iter_max:
                    i += 1
                    for space in item_space:
                        if space.adjacent_to(space_ini):
                            item_space.remove(space)
                            space_ini.merge(space)
                            plan.remove_null_spaces()
                            break
                dict_items_space[item] = space_ini
            else:
                dict_items_space[item] = item_space[0]
        # OPT-72: If we really want to enable it, it should be done through some execution context
        # parameters.
        # assert plan.check()

        return plan, dict_items_space

    def solution_research(self, show=False) -> Optional[List['Solution']]:
        """
        Looks for all possible solutions then find the three best solutions
        :return: None
        """

        self.manager.solver.solve()

        if len(self.manager.solver.solutions) == 0:
            logging.warning(
                "SpacePlanner : solution_research : Plan without space planning solution")
            return []
        else:
            self._check_validity()
            logging.info("SpacePlanner : solution_research : Plan with {0} solutions".format(
                len(self.manager.solver.solutions)))
            logging.debug(self.spec.plan)
            if len(self.manager.solver.solutions) > 0:
                for i, sol in enumerate(self.manager.solver.solutions):
                    plan_solution = self.spec.plan.clone()
                    plan_solution, dict_items_spaces = self._rooms_building(plan_solution, sol)
                    self.solutions_collector.add_solution(plan_solution, dict_items_spaces)
                    logging.debug(plan_solution)
                    if show:
                        plan_solution.plot()

                best_sol = self.solutions_collector.best()
                for sol in best_sol:
                    logging.debug(sol)
                    if show:
                        sol.plan.plot(save=True)
                return best_sol

    def generate_best_solutions_files(self, best_sol: ['Solution']):
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
    import libs.io.reader as reader
    from libs.operators.selector import SELECTORS
    from libs.modelers.grid import GRIDS
    import argparse
    import time

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
        # input_file = reader.get_list_from_folder(DEFAULT_BLUEPRINT_INPUT_FOLDER)[plan_index]
        input_file = "grenoble_101.json"
        t00 = time.clock()
        plan = reader.create_plan_from_file(input_file)
        logging.info("input_file %s", input_file)
        # print("input_file", input_file)
        logging.debug(("P2/S ratio : %i", round(plan.indoor_perimeter ** 2 / plan.indoor_area)))

        GRIDS['optimal_grid'].apply_to(plan)

        min_cell_area = 1 * SQM
        if plan.indoor_area > 105 * SQM and plan.floor_count < 2:
            min_cell_area = 2 * SQM
        elif plan.indoor_area > 130 * SQM and plan.floor_count < 2:
            min_cell_area = 3 * SQM

        seeder = Seeder(plan, GROWTH_METHODS).add_condition(SELECTORS['seed_duct'], 'duct')
        plan.plot()
        (seeder.plant()
         .grow(show=True)
         .divide_along_seed_borders(SELECTORS["not_aligned_edges"])
         .from_space_empty_to_seed()
         .merge_small_cells(min_cell_area=min_cell_area, excluded_components=["loadBearingWall"]))

        plan.plot()

        input_file_setup = input_file[:-5] + "_setup0.json"
        spec = reader.create_specification_from_file(input_file_setup)
        logging.debug(spec)
        spec.plan = plan
        spec.plan.remove_null_spaces()

        logging.debug("number of mutables spaces, %i",
                      len([space for space in spec.plan.spaces if space.mutable]))

        # surfaces control
        logging.debug("PLAN AREA : %i", int(spec.plan.indoor_area))
        logging.debug("Setup AREA : %i", int(sum(item.required_area for item in spec.items)))
        logging.debug("Setup max AREA : %i", int(sum(item.max_size.area for item in spec.items)))
        logging.debug("Setup min AREA : %i", int(sum(item.min_size.area for item in spec.items)))
        plan_ratio = round(spec.plan.indoor_perimeter ** 2 / spec.plan.indoor_area)
        logging.debug("PLAN Ratio : %i", plan_ratio)

        t0 = time.clock()
        space_planner = SpacePlanner("test", spec)
        logging.debug("space_planner time : %f", time.clock() - t0)
        t1 = time.clock()
        best_solutions = space_planner.solution_research()
        logging.debug("solution_research time: %f", time.clock() - t1)
        logging.debug(best_solutions)

        # Output
        for sol in best_solutions:
            solution_dict = writer.generate_output_dict_from_file(input_file, sol)
            writer.save_json_solution(solution_dict, sol.get_id)

        # shuffle
        if best_solutions:
            for sol in best_solutions:
                SHUFFLES['square_shape_shuffle_rooms'].run(sol.plan, show=True)
                sol.plan.plot()

        logging.info("total time : %f", time.clock() - t00)
        # print("total time :", time.clock() - t00)

        # Tests ordre des variables de prog par contraintes
        # category_name_list_test = ["entrance", "toilet", "bathroom", "laundry", "kitchen", "living",
        # "bedroom", "dressing"] #,
        # #"laundry", "dressing"]
        # #spaces_list = list(spec.plan.spaces)
        # best_name_list = None
        # best_failures = 10e15
        # best_branches = 10e15
        # worst_name_list = None
        # worst_failures = 0
        # worst_branches = 0
        # best_spaces_list = []
        # import itertools
        # perm_cat = list(itertools.permutations(category_name_list_test))
        # #perm_spaces = list(itertools.permutations(spaces_list))
        # print("nombre perm : ", len(perm_cat))
        # for cat_list in perm_cat:
        #     #spec.plan.spaces = spaces
        #     spec.init_id(cat_list)
        #     t0 = time.clock()
        #     space_planner = SpacePlanner("test", spec)
        #     print("space_planner time :", time.clock() - t0)
        #     t1 = time.clock()
        #     best_solutions = space_planner.solution_research()
        #     print("solution_research time:", time.clock() - t1)
        #     if space_planner.manager.solver.solver.Branches() < best_branches:
        #         best_branches = space_planner.manager.solver.solver.Branches()
        #         best_failures = space_planner.manager.solver.solver.Failures()
        #         best_name_list = cat_list
        #     if space_planner.manager.solver.solver.Branches() > worst_branches:
        #         worst_branches = space_planner.manager.solver.solver.Branches()
        #         worst_failures = space_planner.manager.solver.solver.Failures()
        #         worst_name_list = cat_list
        # print("BEST SOLUTION")
        # print("best_name_list", best_name_list)
        # print("best_failures", best_failures)
        # print("best_branches", best_branches)
        # print("worst_name_list", worst_name_list)
        # print("worst_failures", worst_failures)
        # print("worst_branches", worst_branches)


    space_planning()
