# coding=utf-8
"""
Space Planner

A space planner attributes the spaces of the plan created by the seeder to the items.
The spaces are allocated according to constraints using constraint programming

OR-Tools : google constraint programing solver
    https://developers.google.com/optimization/
    https://acrogenesis.com/or-tools/documentation/user_manual/index.html

"""
import logging
import matplotlib.pyplot as plt
import libs.utils.copy as copy

from libs.specification import Specification
from libs.solution import SolutionsCollector
from libs.plan import Plan
from libs.constraints_manager import ConstraintsManager
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
        self.init_spaces_adjacency()

        self.solutions_collector = SolutionsCollector(spec)

    def __repr__(self):
        # TODO
        output = 'SpacePlanner' + self.name
        return output

    def init_spaces_adjacency(self) -> None:
        """
        spaces adjacency matrix init
        :return: None
        """
        for i, i_space in enumerate(self.spec.plan.mutable_spaces):
            self.spaces_adjacency_matrix.append([])
            for j, j_space in enumerate(self.spec.plan.mutable_spaces):
                if j != i:
                    self.spaces_adjacency_matrix[i].append(0)
                else:
                    self.spaces_adjacency_matrix[i].append(1)

        for i, i_space in enumerate(self.spec.plan.mutable_spaces):
            for j, j_space in enumerate(self.spec.plan.mutable_spaces):
                if j < i:
                    if i_space.adjacent_to(j_space):
                        self.spaces_adjacency_matrix[i][j] = 1
                        self.spaces_adjacency_matrix[j][i] = 1
                    else:
                        self.spaces_adjacency_matrix[i][j] = 0
                        self.spaces_adjacency_matrix[j][i] = 0

    def check_adjacency(self, room_positions, connectivity_checker) -> bool:
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

    def check_validity(self) -> None:
        """
        check_connectivity of constraint programming solutions and remove wrong results of
        self.manager.solver.solutions
        :return: None
        """
        connectivity_checker = check_room_connectivity_factory(self.spaces_adjacency_matrix)

        sol_to_remove = []
        for sol in self.manager.solver.solutions:
            is_a_good_sol = self.check_adjacency(sol, connectivity_checker)
            if not is_a_good_sol:
                sol_to_remove.append(sol)

        if sol_to_remove:
            for sol in sol_to_remove:
                self.manager.solver.solutions.remove(sol)

    def rooms_building(self, plan: 'Plan', matrix_solution) -> 'Plan':
        """
        Rooms building
        :return: None
        """
        dict_items_spaces = {}
        for i_item, item in enumerate(self.spec.items):
            item_space = []
            for j_space, space in enumerate(plan.mutable_spaces):
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
                            space_ini._merge(space)
                            plan.remove_null_spaces()
                            item_space.remove(space)
                            break
                    i += 1
        return plan

    def solution_research(self) -> None:
        """
        Rooms building
        :return: None
        """

        self.manager.solver.solve()

        if len(self.manager.solver.solutions) == 0:
            logging.warning('Plan without space planning solution')
        else:
            self.check_validity()
            logging.info('Plan with {0} solutions'.format(len(self.manager.solver.solutions)))
            seed_plan = copy.plan_pickle(self.spec.plan, 'seed_plan')
            for sol in self.manager.solver.solutions:
                plan_solution = copy.load_pickle(seed_plan)
                plan_solution = self.rooms_building(plan_solution, sol)
                self.solutions_collector.add_plan(plan_solution)
                plan_solution.plot()


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
    import libs.seed
    from libs.selector import SELECTORS
    from libs.grid import GRIDS
    from libs.shuffle import SHUFFLES

    logging.getLogger().setLevel(logging.DEBUG)


    def space_planning():
        """
        Test
        :return:
        """

        input_file = 'Antony_A22.json'  # 5 Levallois_Letourneur / Antony_A22
        plan = reader.create_plan_from_file(input_file)

        seeder = libs.seed.Seeder(plan, libs.seed.GROWTH_METHODS)
        seeder.add_condition(SELECTORS['seed_duct'], 'duct')
        GRIDS['ortho_grid'].apply_to(plan)

        seeder.plant()
        seeder.grow(show=True)
        plan.plot(save=False)
        SHUFFLES['square_shape'].run(plan, show=True)

        ax = plan.plot(save=False)
        seeder.plot_seeds(ax)
        plt.title("seeding points")
        plt.show()

        plan.remove_null_spaces()
        plan.make_space_seedable("empty")

        seed_empty_furthest_couple_middle = SELECTORS[
            'seed_empty_furthest_couple_middle_space_area_min_100000']
        seed_empty_area_max_100000 = SELECTORS['area_max=100000']
        seed_methods = [
            (
                seed_empty_furthest_couple_middle,
                libs.seed.GROWTH_METHODS_FILL,
                "empty"
            ),
            (
                seed_empty_area_max_100000,
                libs.seed.GROWTH_METHODS_SMALL_SPACE_FILL,
                "empty"
            )
        ]

        filler = libs.seed.Filler(plan, seed_methods)
        filler.apply_to(plan)
        plan.remove_null_spaces()
        SHUFFLES['square_shape'].run(plan, show=True)

        input_file = 'Antony_A22_setup.json'
        spec = reader.create_specification_from_file(input_file)
        spec.plan = plan

        space_planner = SpacePlanner('test', spec)
        space_planner.solution_research()

        plan.plot(show=True)
        plt.show()
        assert spec.plan.check()


    space_planning()
