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
        self.positions: Dict[ortools.IntVar] = {}  # List[List[ortools.IntVar]] = [[]]
        for i_item in range(self.items_nbr):
            for j_space in range(self.spaces_nbr):
                self.positions[i_item, j_space] = self.solver.IntVar(0, 1,
                                                                     'positions[{0},{1}]'.format(
                                                                         i_item, j_space))
        # For the decision builder
        self.positions_flat: List[ortools.IntVar] = []
        self.positions_flat = [self.positions[i_item, j_space] for i_item in range(self.items_nbr)
                               for
                               j_space in range(self.spaces_nbr)]

        self.solutions: Dict[int] = {}  # classe Solution : scoring

    def add_constraint(self, ct: ortools.Constraint) -> None:
        print(ct)
        self.solver.Add(ct)

    def solve(self):
        """
        search and solution
        """
        # Decision builder
        db = self.solver.Phase(self.positions_flat, self.solver.INT_VAR_DEFAULT,
                               self.solver.ASSIGN_RANDOM_VALUE)

        self.solver.NewSearch(db)

        # Maximum number of solutions
        max_num_sol = 50
        nbr_solutions = 0
        while self.solver.NextSolution():
            sol_positions = []
            for i_item in range(self.items_nbr):  # Rooms
                print(i_item, ":",
                      [self.positions[i_item, j].Value() for j in range(self.spaces_nbr)])
                sol_positions.append([])
                for j_space in range(self.spaces_nbr):  # empty and seed spaces
                    sol_positions[i_item].append(self.positions[i_item, j_space].Value())
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


# SPACE_PLANNER_CONSTRAINTS = Catalog('space_planner_constraints')
# LIVING_SP_CONSTRAINTS = Catalog('living_sp_contraints')
# LIVING_SP_CONSTRAINTS.add(ItemSPConstraint())
#
# SP_CONSTRAINTS = {'living', []}


class SP_Constraint:
    """
    Space planner constraint Class
    """

    def __init__(self, sp: 'SpacePlanner', name: str = ''):
        self.name = name
        self.sp = sp

    def _add(self, ct: ortools.Constraint):
        self.sp.constraint_solver.add_constraint(ct)

    def or_(self, ct1: ortools.Constraint, ct2: ortools.Constraint) -> ortools.Constraint:
        """
        Or between two constraints
        :param ct1
        :param ct2
        :return: ct: ortools.Constraint
        """
        ct = (self.sp.constraint_solver.solver.Max(ct1, ct2) == 1)
        return ct

    def and_(self, ct1: ortools.Constraint, ct2: ortools.Constraint) -> ortools.Constraint:
        """
        And between two constraints
        :param ct1
        :param ct2
        :return: ct: ortools.Constraint
        """
        ct = (self.sp.constraint_solver.solver.Min(ct1, ct2) == 1)
        return ct


class SpaceAttributionConstraint(SP_Constraint):
    """
    Space planner constraint Class
    """

    def __init__(self, sp: 'SpacePlanner', j_space: int, name: str = ''):
        super().__init__(sp, name)
        self.j_space = j_space

    def __call__(self):
        self._add(self.attribution_constraint())

    def attribution_constraint(self) -> ortools.Constraint:
        """
        Each space has to be associated with an item and one time only
        :param j_space:
        :return: ct: ortools.Constraint
        """
        ct = (self.sp.constraint_solver.solver.Sum(
            self.sp.constraint_solver.positions[i, self.j_space] for i in
            range(len(self.sp.spec.items))) == 1)
        return ct


class AreaConstraint(SP_Constraint):
    """
    Space planner constraint Class
    """

    def __init__(self, sp: 'SpacePlanner', item: Item, min_max: str, name: str = ''):
        super().__init__(sp, name)
        self.item = item
        self.min_max = min_max

    def __call__(self):
        if self.min_max == 'max':
            ct = self.max_area_constraint()
        elif self.min_max == 'min':
            ct = self.min_area_constraint()
        else:
            ValueError('AreaConstraint')
        self._add(ct)

    def max_area_constraint(self) -> ortools.Constraint:
        """
        Maximum area constraint
        :param i_item:
        :param max_area: float
        :return: ct:  ortools.Constraint
        """
        ct = (self.sp.constraint_solver.solver.Sum(
            self.sp.constraint_solver.positions[self.item.number, j] * space.area for j, space in
            enumerate(self.sp.seed_spaces)) <= self.item.max_size)
        return ct

    def min_area_constraint(self) -> ortools.Constraint:
        """
        Minimum area constraint
        :param i_item:
        :param min_area: float
        :return: ct:  ortools.Constraint
        """
        ct = (self.sp.constraint_solver.solver.Sum(
            self.sp.constraint_solver.positions[self.item.number, j] * space.area for j, space in
            enumerate(self.sp.seed_spaces)) >= self.item.min_size)
        return ct


class InsideAdjacencyConstraint(SP_Constraint):
    """
    Space planner constraint Class
    TODO : not completed
    """

    def __init__(self, sp: 'SpacePlanner', item: Item, name: str = ''):
        super().__init__(sp, name)
        self.item = item

    def __call__(self):
        self._add(self.inside_adjacency_constraint())

    def inside_adjacency_constraint(self) -> ortools.Constraint:
        """
        Space adjacency constraint inside a given item
        :param i_item:
        :return: ct:  ortools.Constraint
        """
        nbr_spaces_in_i_item = sum(
            self.sp.constraint_solver.positions[self.item.number, j] for j in
            range(len(self.sp.seed_spaces)))
        spaces_adjacency = self.sp.constraint_solver.solver.Sum(
            self.sp.constraint_solver.solver.Sum(
                int(self.sp.spec.plan.adjacent_spaces(j_space, k_space)) *
                self.sp.constraint_solver.positions[self.item.number, j] *
                self.sp.constraint_solver.positions[self.item.number, k] for
                j, j_space in enumerate(self.sp.seed_spaces) if j > k)
            for k, k_space in enumerate(self.sp.seed_spaces))
        ct = (spaces_adjacency >= nbr_spaces_in_i_item - 1)
        return ct


class ItemAdjacencyConstraint(SP_Constraint):
    """
    Space planner constraint Class
    """

    def __init__(self, sp: 'SpacePlanner', item: Item, item_category: str, name: str = ''):
        super().__init__(sp, name)
        self.item = item
        self.item_category = item_category

    def __call__(self):
        self._add(self.item_adjacency_constraint(self.item_category))

    def item_adjacency_constraint(self, item_category: str) -> ortools.Constraint:
        """
        Item adjacency constraint :
        :param item_category: str
        :return: ct:  ortools.Constraint
        """
        for j_item, item in enumerate(self.sp.spec.items):
            if item.category.name == item_category:
                item_adjacency = self.sp.constraint_solver.solver.Sum(
                    self.sp.constraint_solver.solver.Sum(
                        int(self.sp.spec.plan.adjacent_spaces(j_space, k_space)) *
                        self.sp.constraint_solver.positions[self.item.number, j] *
                        self.sp.constraint_solver.positions[j_item, k] for
                        j, j_space in enumerate(self.sp.seed_spaces))
                    for k, k_space in enumerate(self.sp.seed_spaces))
        ct = (item_adjacency >= 1)
        return ct


class ComponentsAdjacencyConstraint(SP_Constraint):
    """
    Space planner constraint Class
    """

    def __init__(self, sp: 'SpacePlanner', item: Item, category: List[str], adj: bool, name: str = ''):
        super().__init__(sp, name)
        self.item = item
        self.category = category
        self.adj = adj

    def __call__(self):
        self._add(self.components_adjacency_constraint(self.category[0], self.adj))

    def components_adjacency_constraint(self, category: str, adj: bool) -> ortools.Constraint:
        """
        components adjacency constraint
        :param adj: bool
        :param category: str
        :return: ct:  ortools.Constraint
        """
        adjacency_sum = self.sp.constraint_solver.solver.Sum(
            self.sp.constraint_solver.positions[self.item.number, j] for j, space in
            enumerate(self.sp.seed_spaces)
            if category in space.components_associated())
        if adj == True:
            return adjacency_sum >= 1
        else:
            return adjacency_sum == 0


class SpacePlanner:
    """
    Space planner Class
    """

    def __init__(self, name: str, spec: 'Specification'):
        self.name = name
        self.spec = spec
        logging.debug(spec)
        self.seed_spaces = []
        print('self.spec.plan.get_spaces()', self.spec.plan)
        for space in self.spec.plan.get_spaces():  # empty and seed spaces
            if space.mutable and space.edge is not None:
                self.seed_spaces.append(space)
                logging.debug(self.seed_spaces)
                logging.debug(space.components_associated())
        self.constraint_solver = ConstraintSolver(len(self.spec.items), len(self.seed_spaces))

    def __repr__(self):
        # TODO
        output = 'SpacePlanner' + self.name
        return output

    def add_spaces_constraints(self) -> None:
        for j_space in range(len(self.seed_spaces)):
            SpaceAttributionConstraint(self, j_space)()

    def add_item_constraints(self) -> None:
        for item in self.spec.items:
            InsideAdjacencyConstraint(self, item)()
            if item.category.name == 'living':
                ComponentsAdjacencyConstraint(self, item, ['doorWindow'], True)()
                ItemAdjacencyConstraint(self, item, 'kitchen')()
            if item.category.name == 'kitchen':
                ComponentsAdjacencyConstraint(self, item, ['window'], True)()
                ComponentsAdjacencyConstraint(self, item, ['duct'], True)()
                ItemAdjacencyConstraint(self, item, 'living')()
            if item.category.name == 'entrance':
                ComponentsAdjacencyConstraint(self, item, ['frontDoor'], True)()
            if item.category.name == 'bathroom':
                ComponentsAdjacencyConstraint(self, item, ['duct'], True)()
            if item.category.name == 'wc':
                ComponentsAdjacencyConstraint(self, item, ['duct'], True)()
            if item.category.name == 'bedroom':
                ComponentsAdjacencyConstraint(self, item, ['doorWindow'], True)()

    def rooms_building(self):  # -> Plan:
        # plan_solution = copy.deepcopy(self.plan)
        self.constraint_solver.solve()

        if len(self.constraint_solver.solutions) >= 1:
            for j_space, space in enumerate(self.seed_spaces):  # empty and seed spaces
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
        spec.plan = plan

        space_planner = SpacePlanner('test', spec)
        space_planner.add_spaces_constraints()
        space_planner.add_item_constraints()
        space_planner.rooms_building()

        ax = plan.plot(save=True)
        # seeder.plot_seeds(ax)
        plt.show()


    space_planning()
