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


class SpacePlanner:
    """
    Space planner Class
    """
    def __init__(self, name: str, plan: 'Plan', spec: 'Specification'):
        self.name = name
        self.plan = plan
        print(plan)
        self.spaces: List['Space'] = []
        for space in self.plan.get_spaces():  # empty and seed spaces
            if space.mutable and space.edge is not None:
                self.spaces.append(space)
        print(self.spaces)
        for j, space in enumerate(self.spaces):
            print(space.immutable_categories_associated())
        self.spec = spec
        print(spec)
        self.solver: ortools.Solver
        self.planning_constraints: List['SpacePlannerConstraint'] = []
        self.items_positions: Dict[ortools.IntVar] = {}
        self.nbr_solutions = 0
        self.positions_flat: List[ortools.IntVar] = []

    def __repr__(self):
        # TODO
        output = 'SpacePlanner' + self.name
        return output

    def init_solver(self) -> None:

        # Create the solver
        self.solver = ortools.Solver("SpacePlanner" + self.name)

        # Declare variables
        for i_item in range(len(self.spec.items)):
            for j_space in range(len(self.spaces)):
                self.items_positions[i_item, j_space] = self.solver.IntVar(0, 1, 'items_positions[{0},{1}]'.format(i_item, j_space))

        # For the decision builder
        self.positions_flat = [self.items_positions[i_item, j_space] for i_item in range(len(self.spec.items)) for j_space in range(len(self.spaces))]

    def _add_constraint(self, ct: ortools.Constraint) -> None:
        print(ct)
        self.solver.Add(ct)

    def add_spaces_constraints(self) -> None:
        for j_space in range(len(self.spaces)):
            self._add_constraint(self._space_attribution_constraint(j_space))

    def add_item_constraints(self) -> None:
        for i_item, item in enumerate(self.spec.items):
            if item.category.name == 'living':
                self._add_constraint(self._cat1_or_cat2_adjacency_constraint(i_item, 'doorWindow', 'window'))
            if item.category.name == 'kitchen':
                self._add_constraint(self._cat1_or_cat2_adjacency_constraint(i_item, 'doorWindow', 'window'))
                self._add_constraint(self._category_adjacency_constraint(i_item, 'duct'))
            if item.category.name == 'entrance':
                self._add_constraint(self._category_adjacency_constraint(i_item, 'frontDoor'))
            if item.category.name == 'bathroom':
                self._add_constraint(self._category_adjacency_constraint(i_item, 'duct'))
            if item.category.name == 'wc':
                self._add_constraint(self._category_adjacency_constraint(i_item, 'duct'))
            if item.category.name == 'bedroom':
                self._add_constraint(self._cat1_or_cat2_adjacency_constraint(i_item, 'doorWindow', 'window'))

    def solve(self):
        #
        # search and solution
        #
        print('solve')
        # Decision builder
        db = self.solver.Phase(self.positions_flat, self.solver.INT_VAR_DEFAULT, self.solver.ASSIGN_RANDOM_VALUE)

        self.solver.NewSearch(db)

        # Maximum number of solutions
        num_sol = 50

        num_solutions = 0
        items_positions_sol = []
        while self.solver.NextSolution():
            sol_postions = []
            for i_item, item in enumerate(self.spec.items):  # Rooms
                print(item.category, ":", [self.items_positions[i_item, j].Value() for j in range(len(self.spaces))])
                sol_postions.append([])
                for j_space in range(len(self.spaces)):  # empty and seed spaces
                    #sol_postions[i_item].append(self.items_positions[i_item, j_space].Value())
                    items_positions_sol.append(sol_postions)

            # Number of solutions
            num_solutions += 1
            if num_solutions >= num_sol:
                break

        self.solver.EndSearch()

        print('Statistics')
        print("num_solutions:", num_solutions)
        print("failures:", self.solver.Failures())
        print("branches:", self.solver.Branches())
        print("WallTime:", self.solver.WallTime())

    def _space_attribution_constraint(self, j_space: int) -> ortools.Constraint:
        """
        Each space has to be associated with an item and one time only
        :param j_space:
        :return: ct: ortools.Constraint
        """
        ct = (self.solver.Sum(
                self.items_positions[i, j_space] for i in range(len(self.spec.items))) == 1)
        return ct

    def _max_area_constraint(self, i_item: int, max_area: float) -> ortools.Constraint:
        """
        Maximum area constraint
        :param i_item:
        :param max_area: float
        :return: ct:  ortools.Constraint
        """
        ct = (self.solver.Sum(
                self.items_positions[i_item, j]*space.area for j, space in enumerate(self.spaces)) <= max_area)
        return ct

    def _min_area_constraint(self, i_item: int, min_area: float) -> ortools.Constraint:
        """
        Minimum area constraint
        :param i_item:
        :param min_area: float
        :return: ct:  ortools.Constraint
        """
        ct = (self.solver.Sum(
                self.items_positions[i_item, j]*space.area for j, space in enumerate(self.spaces)) >= min_area)
        return ct

    def _space_adjacency_constraint(self, i_item: int, category: str) -> ortools.Constraint:
        """
        Space adjacency constraint
        :param i_item:
        :param category: str
        :return: ct:  ortools.Constraint
        """
        ct = (self.solver.Sum(
                self.items_positions[i_item, j] for j, space in enumerate(self.spaces)
                if category in space.immutable_categories_associated()) >= 1)
        return ct

    def _category_adjacency_constraint(self, i_item: int, category: str) -> ortools.Constraint:
        """
        Category adjacency constraint
        :param i_item:
        :param category: str
        :return: ct:  ortools.Constraint
        """
        ct = (self.solver.Sum(
                self.items_positions[i_item, j] for j, space in enumerate(self.spaces)
                if category in space.immutable_categories_associated()) >= 1)
        return ct

    def _cat1_or_cat2_adjacency_constraint(self, i_item: int, cat1: str, cat2: str) -> ortools.Constraint:
        """
        Category adjacency constraint
        :param i_item:
        :param cat1: category 1 str
        :param cat2: category 2 str
        :return: ct:  ortools.Constraint
        """
        ct1 = self._category_adjacency_constraint(i_item, cat1)
        ct2 = self._category_adjacency_constraint(i_item, cat2)
        ct = self._or(ct1, ct2)
        return ct

    def _category_no_adjacency_constraint(self, i_item: int, category: str) -> ortools.Constraint:
        """
        Category no adjacency constraint
        :param i_item:
        :param category: str
        :return: ct: ortools.Constraint
        """
        ct = (self.solver.Sum(
            self.items_positions[i_item, j] for j, space in enumerate(self.spaces)
            if category in space.immutable_categories_associated()) == 0)
        return ct

    def _or(self, ct1: ortools.Constraint, ct2: ortools.Constraint)-> ortools.Constraint:
        """
        Or between two constraints
        :param ct1
        :param ct2
        :return: ct: ortools.Constraint
        """
        ct = (self.solver.Max(ct1, ct2) == 1)
        return ct

    def _and(self, ct1: ortools.Constraint, ct2: ortools.Constraint)-> ortools.Constraint:
        """
        And between two constraints
        :param ct1
        :param ct2
        :return: ct: ortools.Constraint
        """
        ct = (self.solver.Min(ct1, ct2) == 1)
        return ct


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
        input_file = reader.BLUEPRINT_INPUT_FILES[6]  # 5 Levallois_Letourneur
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

        space_planner = SpacePlanner('test', plan, spec)
        space_planner.init_solver()
        space_planner.add_spaces_constraints()
        space_planner.add_item_constraints()
        space_planner.solve()

    space_planning()

