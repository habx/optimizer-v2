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
            if space.mutable:
                self.spaces.append(space)
        self.spec = spec
        print(spec)
        self.solver: ortools.Solver
        self.planning_constraints: List['SpacePlannerConstraint'] = []
        self.items_positions: List[ortools.IntVar] = []
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
        for i_item in range(len(self.spec.items)): # Rooms
            for j_space in range(len(self.spaces)): # empty and seed spaces
                self.items_positions[i_item, j_space] = ortools.IntVar(0, 1, "ItemPositions[%i_item,%j_space]" % (i_item, j_space))

        # For the decision builder
        self.positions_flat = [self.items_positions[i, j] for i in range(i_item) for j in range(j_space)]

    def _add_constraint(self, ct: ortools.Constraint) -> None:
        self.solver.Add(ct)

    def add_spaces_constraints(self) -> None:
        for j_space in range(len(self.spaces)):
            self._add_constraint(self, self._space_attribution_constraint(self,j_space))

    def add_item_constraints(self) -> None:
        for i_item, item in enumerate(self.spec.items):
            if item.category == 'living':
                self._add_constraint(self, self._category_adjacency_constraint(self, i_item, 'window'))
            if item.category == 'kitchen':
                self._add_constraint(self, self._category_adjacency_constraint(self, i_item, 'window'))
                self._add_constraint(self, self._category_adjacency_constraint(self, i_item, 'duct'))
            if item.category == 'entrance':
                self._add_constraint(self, self._category_adjacency_constraint(self, i_item, 'frontDoor'))
            if item.category == 'bathroom':
                self._add_constraint(self, self._category_adjacency_constraint(self, i_item, 'duct'))
            if item.category == 'wc':
                self._add_constraint(self, self._category_adjacency_constraint(self, i_item, 'duct'))
            if item.category == 'bedroom':
                window_ct = self._category_adjacency_constraint(self, i_item, 'window')
                doorWindow_ct = self._category_adjacency_constraint(self, i_item, 'doorWindow')
                ct = self._or(window_ct, doorWindow_ct)
                self._add_constraint(self, ct)

    def _solve(self):
        #
        # search and solution
        #

        # Decision builder
        db = self.solver.Phase(self.positions_flat, self.solver.INT_VAR_DEFAULT, self.solver.ASSIGN_RANDOM_VALUE)

        self.solver.NewSearch(db)

        # Maximum number of solutions
        num_sol = 50000

        num_solutions = 0
        RoomPositions = []
        while self.solver.NextSolution():
            # Save
            SolPostions = []
            for i_item, item in enumerate(self.spec.items):  # Rooms
                print(item.category, ":", [self.items_positions[i_item, j].Value() for j in range(len(self.spaces))])
                SolPostions.append([])
                for j_space in range(len(self.spaces)):  # empty and seed spaces
                    SolPostions[i_item].append(self.items_positions[i_item, j_space].Value())
            RoomPositions.append(SolPostions)

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
        ct = (self.space_planner.solver.Sum(
                self.space_planner.items_positions[i, j_space] for i in range(len(self.spec.items))) == 1)
        return ct

    def _category_adjacency_constraint(self, i_item: int, category: str) -> ortools.Constraint:
        """
        Category adjacency constraint
        :param i_item:
        :param category: str
        :return: ct:  ortools.Constraint
        """
        ct = (self.space_planner.solver.Sum(
                self.space_planner.items_positions[i_item, j] for j, space in enumerate(self.space_planner.spaces)
                if category in space.immutable_categories_associated) >= 1)
        return ct

    def _category_no_adjacency_constraint(self, i_item: int, category: str) -> ortools.Constraint:
        """
        Category no adjacency constraint
        :param i_item:
        :param category: str
        :return: ct: ortools.Constraint
        """
        ct = (self.space_planner.solver.Sum(
            self.space_planner.items_positions[i_item, j] for j, space in enumerate(self.space_planner.spaces)
            if category in space.immutable_categories_associated) == 0)
        return ct

    def _or(self, ct1: ortools.Constraint, ct2: ortools.Constraint)-> ortools.Constraint:
        """
        Or between two constraints
        :param ct1
        :param ct2
        :return: ct: ortools.Constraint
        """
        ct = self.space_planner.solver.Max(ct1, ct2)
        return ct

    def _and(self, ct1: ortools.Constraint, ct2: ortools.Constraint)-> ortools.Constraint:
        """
        And between two constraints
        :param ct1
        :param ct2
        :return: ct: ortools.Constraint
        """
        ct = self.space_planner.solver.Min(ct1, ct2)
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
        input_file = reader.BLUEPRINT_INPUT_FILES[5]  # 5 Levallois_Letourneur
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

        input_file = 'Levallois_Letourneur_setup.json'
        spec = reader.create_specification_from_file(input_file)

        space_planner = SpacePlanner('test', plan, spec)

        assert plan.check()

    space_planning()

