# coding=utf-8
"""
Space Planner

A space planner attributes the spaces of the plan created by the seeder to the items.
The spaces are allocated according to constraints using constraint programming

OR-Tools : google constraint programing solver
    https://developers.google.com/optimization/
    https://acrogenesis.com/or-tools/documentation/user_manual/index.html

"""
from typing import TYPE_CHECKING, List, Optional, Dict, Generator, Sequence
import logging
import copy

import matplotlib.pyplot as plt

from libs.plan import Space, PlanComponent, Plan, Linear, SeedSpace
from libs.specification import Specification, Item
from libs.utils.catalog import Catalog
from libs.plot import plot_point, Plot

from libs.category import SPACE_CATEGORIES

from ortools.constraint_solver import pywrapcp as ortools


class ConstraintSolver:
    """
    Constraint Solver
    """

    def __init__(self, items_nbr: int, spaces_nbr: int):
        self.items_nbr = items_nbr
        self.spaces_nbr = spaces_nbr
        # Create the solver
        self.solver = ortools.Solver('SpacePlanner')
        # Declare variables
        self.items_positions: List[List[ortools.IntVar]] = []
        for i_item in range(self.items_nbr):
            for j_space in range(self.spaces_nbr):
                self.items_positions[i_item][j_space] = self.solver.IntVar(0, 1,
                                                                           'items_positions[{0},{1}]'.format(i_item,
                                                                                                             j_space))
        # For the decision builder
        self.positions_flat: List[ortools.IntVar] = []
        self.positions_flat = [self.items_positions[i_item][j_space] for i_item in range(self.items_nbr) for
                               j_space in range(self.spaces_nbr)]

        self.solutions: Dict[int] = {}  # classe Solution : scoring

    def add_constraint(self, ct: ortools.Constraint) -> None:
        print(ct)
        self.solver.Add(ct)

    def solve(self):
        """
        search and solution
        :param j_space:
        :return: ct: ortools.Constraint
        """
        # Decision builder
        db = self.solver.Phase(self.positions_flat, self.solver.INT_VAR_DEFAULT, self.solver.ASSIGN_RANDOM_VALUE)

        self.solver.NewSearch(db)

        # Maximum number of solutions
        max_num_sol = 50000
        nbr_solutions = 0
        while self.solver.NextSolution():
            sol_positions = []
            for i_item in range(self.items_nbr):  # Rooms
                print(i_item, ":", [self.items_positions[i_item][j].Value() for j in range(self.spaces_nbr)])
                sol_positions.append([])
                for j_space in range(self.spaces_nbr):  # empty and seed spaces
                    sol_positions[i_item].append(self.items_positions[i_item][j_space].Value())
                    self.solutions[nbr_solutions] = sol_positions

            # Number of solutions
            nbr_solutions += 1
            if nbr_solutions >= max_num_sol:
                break

        self.solver.EndSearch()

        print('Statistics')
        print("num_solutions:", nbr_solutions)
        print("failures:", self.solver.Failures())
        print("branches:", self.solver.Branches())
        print("WallTime:", self.solver.WallTime())

    def space_attribution_constraint(self, j_space: int) -> ortools.Constraint:
        """
        Each space has to be associated with an item and one time only
        :param j_space:
        :return: ct: ortools.Constraint
        """
        ct = (self.solver.Sum(
            self.items_positions[i][j_space] for i in range(len(self.spec.items))) == 1)
        return ct

    def max_area_constraint(self, i_item: int, max_area: float) -> ortools.Constraint:
        """
        Maximum area constraint
        :param i_item:
        :param max_area: float
        :return: ct:  ortools.Constraint
        """
        ct = (self.solver.Sum(
            self.items_positions[i_item][j] * space.area for j, space in enumerate(self.spaces)) <= max_area)
        return ct

    def min_area_constraint(self, i_item: int, min_area: float) -> ortools.Constraint:
        """
        Minimum area constraint
        :param i_item:
        :param min_area: float
        :return: ct:  ortools.Constraint
        """
        ct = (self.solver.Sum(
            self.items_positions[i_item][j] * space.area for j, space in enumerate(self.spaces)) >= min_area)
        return ct

    def space_adjacency_constraint(self, i_item: int) -> ortools.Constraint:
        """
        Space adjacency constraint
        :param i_item:
        :return: ct:  ortools.Constraint
        """
        nbr_spaces_in_i_item = sum(self.items_positions[i_item][j] for j in range(len(self.spaces)))
        spaces_adjacency = self.solver.Sum(
            self.solver.Sum(int(self.plan.adjacent_spaces(j_space, k_space)) * self.items_positions[i_item][j] *
                            self.items_positions[i_item][k] for j, j_space in enumerate(self.spaces) if j > k)
            for k, k_space in enumerate(self.spaces))
        ct = (spaces_adjacency >= nbr_spaces_in_i_item - 1)
        return ct

    def item_adjacency_constraint(self, i_item: int, item_category: str) -> ortools.Constraint:
        """
        Item adjacency constraint
        :param i_item:
        :param category: str
        :return: ct:  ortools.Constraint
        """
        for j_item, item in enumerate(self.spec.items):
            if item.category.name == item_category:
                item_adjacency = self.solver.Sum(
                    self.solver.Sum(int(self.plan.adjacent_spaces(j_space, k_space)) * self.items_positions[i_item][j] *
                                    self.items_positions[j_item][k] for j, j_space in enumerate(self.spaces))
                    for k, k_space in enumerate(self.spaces))
        ct = (item_adjacency >= 1)
        return ct

    def category_adjacency_constraint(self, i_item: int, category: str) -> ortools.Constraint:
        """
        Category adjacency constraint
        :param i_item:
        :param category: str
        :return: ct:  ortools.Constraint
        """
        ct = (self.solver.Sum(
            self.items_positions[i_item][j] for j, space in enumerate(self.spaces)
            if category in space.immutable_categories_associated()) >= 1)
        return ct

    def cat1_or_cat2_adjacency_constraint(self, i_item: int, cat1: str, cat2: str) -> ortools.Constraint:
        """
        Category adjacency constraint
        :param i_item:
        :param cat1: category 1 str
        :param cat2: category 2 str
        :return: ct:  ortools.Constraint
        """
        ct1 = self.category_adjacency_constraint(i_item, cat1)
        ct2 = self.category_adjacency_constraint(i_item, cat2)
        ct = self.or_(ct1, ct2)
        return ct

    def category_no_adjacency_constraint(self, i_item: int, category: str) -> ortools.Constraint:
        """
        Category no adjacency constraint
        :param i_item:
        :param category: str
        :return: ct: ortools.Constraint
        """
        ct = (self.solver.Sum(
            self.items_positions[i_item][j] for j, space in enumerate(self.spaces)
            if category in space.immutable_categories_associated()) == 0)
        return ct

    def or_(self, ct1: ortools.Constraint, ct2: ortools.Constraint) -> ortools.Constraint:
        """
        Or between two constraints
        :param ct1
        :param ct2
        :return: ct: ortools.Constraint
        """
        ct = (self.solver.Max(ct1, ct2) == 1)
        return ct

    def and_(self, ct1: ortools.Constraint, ct2: ortools.Constraint) -> ortools.Constraint:
        """
        And between two constraints
        :param ct1
        :param ct2
        :return: ct: ortools.Constraint
        """
        ct = (self.solver.Min(ct1, ct2) == 1)
        return ct


class SpacePlanner:
    """
    Space planner Class
    """

    def __init__(self, name: str, spec: 'Specification'):
        self.name = name
        self.spec = spec
        logging.debug(spec)
        self.seed_spaces = []
        for space in self.spec.plan.get_spaces():  # empty and seed spaces
            if space.mutable and space.edge is not None:
                self.seed_spaces.append(space)
                logging.debug(self.seed_spaces)
                logging.debug(space.immutable_categories_associated())
        self.constraint_solver = ConstraintSolver(self.spec)

    def __repr__(self):
        # TODO
        output = 'SpacePlanner' + self.name
        return output

    def add_spaces_constraints(self) -> None:
        for j_space in range(len(self.seed_spaces)):
            self.constraint_solver.add_constraint(self.constraint_solver.space_attribution_constraint(j_space))

    def add_item_constraints(self) -> None:
        for i_item, item in enumerate(self.spec.items):
            self.constraint_solver.add_constraint(self.constraint_solver.space_adjacency_constraint(i_item))
            if item.category.name == 'living':
                self.constraint_solver.add_constraint(
                    self.constraint_solver.cat1_or_cat2_adjacency_constraint(i_item, 'doorWindow', 'window'))
                self.constraint_solver.add_constraint(
                    self.constraint_solver.item_adjacency_constraint(i_item, 'kitchen'))
            if item.category.name == 'kitchen':
                self.constraint_solver.add_constraint(
                    self.constraint_solver.cat1_or_cat2_adjacency_constraint(i_item, 'doorWindow', 'window'))
                self.constraint_solver.add_constraint(
                    self.constraint_solver.category_adjacency_constraint(i_item, 'duct'))
                self.constraint_solver.add_constraint(
                    self.constraint_solver.item_adjacency_constraint(i_item, 'living'))
            if item.category.name == 'entrance':
                self.constraint_solver.add_constraint(
                    self.constraint_solver.category_adjacency_constraint(i_item, 'frontDoor'))
            if item.category.name == 'bathroom':
                self.constraint_solver.add_constraint(
                    self.constraint_solver.category_adjacency_constraint(i_item, 'duct'))
            if item.category.name == 'wc':
                self.constraint_solver.add_constraint(
                    self.constraint_solver.category_adjacency_constraint(i_item, 'duct'))
            if item.category.name == 'bedroom':
                self.constraint_solver.add_constraint(
                    self.constraint_solver.cat1_or_cat2_adjacency_constraint(i_item, 'doorWindow', 'window'))

    def rooms_building(self):  # -> Plan:
        # plan_solution = copy.deepcopy(self.plan)
        self.constraint_solver.solve()

        for j_space, space in enumerate(self.spaces):  # empty and seed spaces
            for i_item, item in enumerate(self.spec.items):  # Rooms
                if self.constraint_solver.solutions[0][i_item][j_space] == 1:
                    space.category = item.category


if __name__ == '__main__':
    import libs.reader as reader
    import libs.seed as seed
    from libs.selector import SELECTORS
    from libs.grid import GRIDS
    from libs.shuffle import SHUFFLES

    logging.getLogger().setLevel(logging.DEBUG)


    def space_planning():
        """
        Test
        :return:
        """
        input_file = 'Antony_A22.json'  # 5 Levallois_Letourneur
        plan = reader.create_plan_from_file(input_file)

        seeder = seed.Seeder(plan, seed.GROWTH_METHODS)
        seeder.add_condition(SELECTORS['seed_duct'], 'duct')
        GRIDS['ortho_grid'].apply_to(plan)

        seeder.plant()
        seeder.grow(show=True)
        SHUFFLES['square_shape'].run(plan, show=True)

        print(plan)
        print(seeder)

        ax = plan.plot(save=True)
        # seeder.plot_seeds(ax)
        plt.show()
        assert plan.check()

        input_file = 'Antony_A22_setup.json'
        spec = reader.create_specification_from_file(input_file)

        space_planner = SpacePlanner('test', spec)
        space_planner.add_spaces_constraints()
        space_planner.add_item_constraints()
        space_planner.rooms_building()

        ax = plan.plot(save=True)
        # seeder.plot_seeds(ax)
        plt.show()


    space_planning()
