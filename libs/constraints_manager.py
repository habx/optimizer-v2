# coding=utf-8
"""
Constraints manager Module
Creates the following classes:
• ConstraintSolver: Encapsulation of the OR-Tools solver adapted to our problem
• ConstraintsManager : attributes the spaces of the plan created by the seeder to the items.
TODO : the adjacency constraint of spaces within the same room is not completed
TODO : fusion of the entrance for small apartment untreated

OR-Tools : google constraint programing solver
    https://developers.google.com/optimization/
    https://acrogenesis.com/or-tools/documentation/user_manual/index.html

"""
from typing import List, Callable, Optional, Sequence, TYPE_CHECKING
from ortools.constraint_solver import pywrapcp as ortools
from libs.specification import Item
from libs.utils.geometry import distance
import networkx as nx
import time
import logging

if TYPE_CHECKING:
    from libs.space_planner import SpacePlanner

WINDOW_ROOMS = ("living", "kitchen", "office", "dining", "bedroom")

DRESSING_NEIGHBOUR_ROOMS = ("entrance", "bedroom", "wc", "bathroom")

CIRCULATION_ROOMS = ("living", "dining", "entrance")

DAY_ROOMS = ("living", "dining", "kitchen", "cellar")

PRIVATE_ROOMS = ("bedroom", "bathroom", "laundry", "dressing", "entrance", "circulationSpace")

WINDOW_CATEGORY = ["window", "doorWindow"]

BIG_VARIANTS = ("m", "l", "xl")

SMALL_VARIANTS = ("xs", "s")

OPEN_ON_ADJACENCY_SIZE = 200
BIG_EXTERNAL_SPACE = 7000


class ConstraintSolver:
    """
    Constraint Solver
    Encapsulation of the OR-tools solver adapted to our problem
    """

    def __init__(self, items_nbr: int, spaces_nbr: int, multilevel: bool = False):
        self.items_nbr = items_nbr
        self.spaces_nbr = spaces_nbr
        self.multilevel = multilevel
        # Create the solver
        self.solver = ortools.Solver('SpacePlanner')
        # Declare variables
        self.cells_item: List[ortools.IntVar] = []
        self.positions = {}  # List[List[ortools.IntVar]] = [[]]
        self._init_positions()
        self.solutions = []

    def _init_positions(self) -> None:
        """
        variables initialization
        :return: None
        """
        # cells in [0, self.items_nbr-1], self.items_nbr for multilevel plans : circulationSpace
        if not self.multilevel:
            self.cells_item = [self.solver.IntVar(0, self.items_nbr-1,
                                                  "cells_item[{0}]".format(j_space))
                               for j_space in range(self.spaces_nbr)]
        else:
            self.cells_item = [self.solver.IntVar(0, self.items_nbr,
                                                  "cells_item[{0}]".format(j_space))
                               for j_space in range(self.spaces_nbr)]

        for i_item in range(self.items_nbr):
            for j_space in range(self.spaces_nbr):
                self.positions[i_item, j_space] = (self.cells_item[j_space] == i_item)

    def add_constraint(self, ct: ortools.Constraint) -> None:
        """
        add constraint
        :param ct: ortools.Constraint
        :return: None
        """
        if ct is not None:
            self.solver.Add(ct)

    def solve(self) -> None:
        """
        search and solution
        :return: None
        """
        t0 = time.clock()
        # Decision builder
        db = self.solver.Phase(self.cells_item, self.solver.CHOOSE_FIRST_UNBOUND,
                               self.solver.ASSIGN_MIN_VALUE)
        self.solver.NewSearch(db)

        # Maximum number of solutions
        max_num_sol = 500
        nbr_solutions = 0
        # noinspection PyArgumentList
        while self.solver.NextSolution():
            sol_positions = []
            for i_item in range(self.items_nbr):  # Rooms
                logging.debug("ConstraintSolver: Solution : {0}: {1}".format(i_item, [
                    self.cells_item[j].Value() == i_item for j in range(self.spaces_nbr)]))
                sol_positions.append([])
                for j_space in range(self.spaces_nbr):  # empty and seed spaces
                    sol_positions[i_item].append(self.cells_item[j_space].Value() == i_item)
            self.solutions.append(sol_positions)

            # Number of solutions
            nbr_solutions += 1
            if nbr_solutions >= max_num_sol:
                break

        # noinspection PyArgumentList
        self.solver.EndSearch()

        logging.debug("ConstraintSolver: Statistics")
        logging.debug("ConstraintSolver: num_solutions: %i", nbr_solutions)
        logging.debug("ConstraintSolver: failures: %i", self.solver.Failures())
        logging.debug("ConstraintSolver: branches:  %i", self.solver.Branches())
        # logging.debug("ConstraintSolver: WallTime:  %i", self.solver.WallTime())
        logging.debug("ConstraintSolver: Process time : %f", time.clock() - t0)

        print("ConstraintSolver: Statistics")
        print("ConstraintSolver: num_solutions:", nbr_solutions)
        print("ConstraintSolver: failures: ", self.solver.Failures())
        print("ConstraintSolver: branches: ", self.solver.Branches())
        # logging.debug("ConstraintSolver: WallTime:  %i", self.solver.WallTime())
        print("ConstraintSolver: Process time :", time.clock() - t0)


class ConstraintsManager:
    """
    Constraints manager Class
    A Constraints Manager attributes the spaces of the plan created by the seeder to the items.
    The spaces are allocated according to constraints using constraint programming
    All the possible solutions are given.
    """

    def __init__(self, sp: 'SpacePlanner', name: str = ''):
        self.name = name
        self.sp = sp
        if sp.spec.plan.floor_count < 2:
            self.solver = ConstraintSolver(len(self.sp.spec.items),
                                           self.sp.spec.plan.count_mutable_spaces())
        else:
            self.solver = ConstraintSolver(len(self.sp.spec.items),
                                           self.sp.spec.plan.count_mutable_spaces(), True)
        self.symmetry_breaker_memo = {}
        self.windows_length = {}
        self._init_windows_length()
        self.spaces_distance = []
        self._init_spaces_distance()
        self.space_graph = nx.Graph()
        self._init_spaces_graph()
        self.area_space_graph = nx.Graph()
        self._init_area_spaces_graph()
        self.item_constraints = {}
        self.add_spaces_constraints()
        self.add_item_constraints()
        self.sp.spec.plan.area

    def _init_windows_length(self) -> None:
        """
        Initialize the length of each window
        :return:
        """
        for item in self.sp.spec.items:
            length = 0
            for j, space in enumerate(self.sp.spec.plan.mutable_spaces()):
                for component in space.immutable_components():
                    if (component.category.name == "window"
                            or component.category.name == "doorWindow"):
                        length += (self.solver.positions[item.id, j]
                                   * int(component.length / 10))
            self.windows_length[str(item.id)] = length

    def _init_spaces_distance(self) -> None:
        """
        Initialize the spaces distance matrix
        :return:
        """
        for i, i_space in enumerate(self.sp.spec.plan.mutable_spaces()):
            self.spaces_distance.append([])
            for j, j_space in enumerate(self.sp.spec.plan.mutable_spaces()):
                self.spaces_distance[i].append(0)
                if i == j:
                    self.spaces_distance[i][i] = 0
                elif i_space.floor != j_space.floor:
                    self.spaces_distance[i][j] = 1e20
                else:
                    self.spaces_distance[i][j] = int(i_space.maximum_distance_to(j_space))

    def _init_spaces_graph(self) -> None:
        """
        Initialize the spaces graph
        :return:
        """

        for i, i_space in enumerate(self.sp.spec.plan.mutable_spaces()):
            for j, j_space in enumerate(self.sp.spec.plan.mutable_spaces()):
                if i != j:
                    if i_space.adjacent_to(j_space):
                        self.space_graph.add_edge(i, j, weight=1)

    def _init_area_spaces_graph(self) -> None:
        """
        Initialize the area spaces graph
        :return:
        """

        for i, i_space in enumerate(self.sp.spec.plan.mutable_spaces()):
            for j, j_space in enumerate(self.sp.spec.plan.mutable_spaces()):
                if i != j:
                    if i_space.adjacent_to(j_space):
                        self.area_space_graph.add_edge(i, j, weight=j_space.area)

    def add_spaces_constraints(self) -> None:
        """
        add spaces constraints
        - Each space has to be associated with an item and one time only :
        special case of stairs:
        they must be in a circulating room, otherwise: they are not allocated,
        they are created a circulationSpace
        :return: None
        """
        for j_space, space in enumerate(self.sp.spec.plan.mutable_spaces()):
            if ("startingStep" not in [component.category.name for component in
                                       space.immutable_components()]):
                self.solver.add_constraint(
                    space_attribution_constraint(self, j_space))

    def add_item_constraints(self) -> None:
        """
        add items constraints
        :return: None
        """
        for item in self.sp.spec.items:
            for constraint in GENERAL_ITEMS_CONSTRAINTS["all"]:
                self.add_item_constraint(item, constraint[0], **constraint[1])
            for constraint in GENERAL_ITEMS_CONSTRAINTS[item.category.name]:
                self.add_item_constraint(item, constraint[0], **constraint[1])
            if self.sp.spec.typology >= 3:
                for constraint in T3_MORE_ITEMS_CONSTRAINTS["all"]:
                    self.add_item_constraint(item, constraint[0], **constraint[1])
                for constraint in T3_MORE_ITEMS_CONSTRAINTS[item.category.name]:
                    self.add_item_constraint(item, constraint[0], **constraint[1])

    def add_item_constraint(self, item: Item, constraint_func: Callable, **kwargs) -> None:
        """
        add item constraint
        :param item: Item
        :param constraint_func: Callable
        :return: None
        """
        if kwargs is not {}:
            kwargs = {"item": item, **kwargs}
        else:
            kwargs = {"item": item}
        self.solver.add_constraint(constraint_func(self, **kwargs))

    def or_(self, ct1: ortools.Constraint, ct2: ortools.Constraint) -> ortools.Constraint:
        """
        Or between two constraints
        :param ct1: ortools.Constraint
        :param ct2: ortools.Constraint
        :return: ct: ortools.Constraint
        """
        ct = (self.solver.solver.Max(ct1, ct2) == 1)
        return ct

    def and_(self, ct1: ortools.Constraint, ct2: ortools.Constraint) -> ortools.Constraint:
        """
        And between two constraints
        :param ct1: ortools.Constraint
        :param ct2: ortools.Constraint
        :return: ct: ortools.Constraint
        """
        ct = (self.solver.solver.Min(ct1, ct2) == 1)
        return ct


def space_attribution_constraint(manager: 'ConstraintsManager',
                                 j_space: int) -> ortools.Constraint:
    """
    Each space has to be associated with an item and one time only
    :param manager: 'ConstraintsManager'
    :param j_space: int
    :return: ct: ortools.Constraint
    """
    ct = (manager.solver.solver.Sum(
        manager.solver.positions[i, j_space]
        for i in range(len(manager.sp.spec.items))) == 1)
    return ct


def item_attribution_constraint(manager: 'ConstraintsManager',
                                item: Item) -> ortools.Constraint:
    """
    Each item has to be associated with a space
    :param manager: 'ConstraintsManager'
    :param item: Item
    :return: ct: ortools.Constraint
    """
    ct = (manager.solver.solver.Sum(
        manager.solver.positions[item.id, j_space]
        for j_space in range(manager.sp.spec.plan.count_mutable_spaces())) >= 1)
    return ct


def area_constraint(manager: 'ConstraintsManager', item: Item,
                    min_max: str) -> ortools.Constraint:
    """
    Maximum area constraint
    :param manager: 'ConstraintsManager'
    :param item: Item
    :param min_max: str
    :return: ct: ortools.Constraint
    """
    ct = None
    max_area_coeff = 4 / 3
    min_area_coeff = 2 / 3

    if min_max == "max":
        ct = (manager.solver.solver
              .Sum(manager.solver.positions[item.id, j] * int(space.area)
                   for j, space in enumerate(manager.sp.spec.plan.mutable_spaces())) <=
              int(item.max_size.area * max_area_coeff))

    elif min_max == "min":
        ct = (manager.solver.solver
              .Sum(manager.solver.positions[item.id, j] * int(space.area)
                   for j, space in enumerate(manager.sp.spec.plan.mutable_spaces())) >=
              int(item.min_size.area * min_area_coeff))
    else:
        ValueError("AreaConstraint")

    return ct


def distance_constraint(manager: 'ConstraintsManager', item: Item) -> ortools.Constraint:
    """
    Maximum distance constraint between spaces (centroid) constraint
    :param manager: 'ConstraintsManager'
    :param item: Item
    :return: ct: ortools.Constraint
    # TODO : find best param
    # TODO : unit tests
    """
    if item.category.name in ["living", "dining"]:
        param = 2
    elif item.category.name in ["kitchen", "bedroom"]:
        param = 2
    else:
        param = 2

    max_distance = int(round(param * item.max_size.area**0.5))

    ct = None

    for j, j_space in enumerate(manager.sp.spec.plan.mutable_spaces()):
        for k, k_space in enumerate(manager.sp.spec.plan.mutable_spaces()):
            if j < k:
                if ct is None:
                    ct = ((manager.solver.positions[item.id, j] *
                           manager.solver.positions[item.id, k])
                          <= int(max_distance/manager.spaces_distance[j][k]))
                else:
                    new_ct = ((manager.solver.positions[item.id, j] *
                               manager.solver.positions[item.id, k])
                              <= int(max_distance/manager.spaces_distance[j][k]))
                    ct = manager.and_(ct, new_ct)

    return ct


def graph_constraint(manager: 'ConstraintsManager', item: Item) -> ortools.Constraint:
    """
    graph constraint
    :param manager: 'ConstraintsManager'
    :param item: Item
    :return: ct: ortools.Constraint
    # TODO : unit tests
    """

    ct = None
    for j, j_space in enumerate(manager.sp.spec.plan.mutable_spaces()):
        for k, k_space in enumerate(manager.sp.spec.plan.mutable_spaces()):
            if j < k:
                if ct is None:
                    if not nx.has_path(manager.space_graph, j, k):
                        ct = (manager.solver.positions[item.id, j] *
                              manager.solver.positions[item.id, k] == 0)
                    else:
                        path = nx.dijkstra_path(manager.area_space_graph, j, k)
                        area_path = int(sum(space.area
                                        for i, space in
                                        enumerate(manager.sp.spec.plan.mutable_spaces())
                                        if i in path))
                        ct1 = ((manager.solver.positions[item.id, j] *
                               manager.solver.positions[item.id, k] *
                               len(nx.dijkstra_path(manager.space_graph, j, k)))
                               <= manager.solver.solver.Sum(manager.solver.positions[item.id, l]
                                                            for l, l_space in enumerate(
                                                            manager.sp.spec.plan.mutable_spaces())))
                        ct2 = (manager.solver.positions[item.id, j] *
                               manager.solver.positions[item.id, k] * area_path <= int(item.max_size.area*4/3))
                        ct = manager.and_(ct1, ct2)
                        ct = ct

                else:
                    if not nx.has_path(manager.space_graph, j, k):
                        new_ct = (manager.solver.positions[item.id, j] *
                                  manager.solver.positions[item.id, k] == 0)
                    else:
                        path = nx.dijkstra_path(manager.area_space_graph, j, k)
                        area_path = int(sum(space.area
                                        for i, space in
                                        enumerate(manager.sp.spec.plan.mutable_spaces())
                                        if i in path))
                        ct1 = ((manager.solver.positions[item.id, j] *
                               manager.solver.positions[item.id, k] *
                               len(nx.dijkstra_path(manager.space_graph, j, k)))
                               <= manager.solver.solver.Sum(manager.solver.positions[item.id, l]
                                                            for l, l_space in enumerate(
                                                            manager.sp.spec.plan.mutable_spaces())))
                        ct2 = (manager.solver.positions[item.id, j] *
                               manager.solver.positions[item.id, k] * area_path <= int(item.max_size.area*4/3))
                        new_ct = manager.and_(ct1, ct2)
                    ct = manager.and_(ct, new_ct)

    return ct


def shape_constraint(manager: 'ConstraintsManager', item: Item) -> ortools.Constraint:
    """
    Shaape constraint
    :param manager: 'ConstraintsManager'
    :param item: Item
    :param min_max: str
    :return: ct: ortools.Constraint
    # TODO : find best param
    # TODO : unit tests
    """

    plan_ratio = round(manager.sp.spec.plan.indoor_perimeter ** 2 / manager.sp.spec.plan.indoor_area)

    if item.category.name in ["living", "dining"]:
        param = max(30, int(plan_ratio + 10))
    elif item.category.name in ["kitchen", "bedroom", "entrance"]:
        param = max(25, int(plan_ratio))
    else:
        param = 24

    item_area = manager.solver.solver.Sum(manager.solver.positions[item.id, j] * int(space.area)
                                          for j, space in
                                          enumerate(manager.sp.spec.plan.mutable_spaces()))

    cells_perimeter = manager.solver.solver.Sum(manager.solver.positions[item.id, j] *
                                                int(space.perimeter)
                                                for j, space in
                                                enumerate(manager.sp.spec.plan.mutable_spaces()))
    cells_adjacency = manager.solver.solver.Sum(manager.solver.positions[item.id, j] *
                                                manager.solver.positions[item.id, k] *
                                                int(j_space.adjacency_length(k_space))
                                                for j, j_space in
                                                enumerate(manager.sp.spec.plan.mutable_spaces())
                                                for k, k_space in
                                                enumerate(manager.sp.spec.plan.mutable_spaces())
                                                )
    item_perimeter = cells_perimeter - cells_adjacency
    ct = (item_perimeter*item_perimeter <= param*item_area)

    return ct


def windows_constraint(manager: 'ConstraintsManager', item: Item) -> Optional[bool]:
    """
    Windows length constraint
    :param manager: 'ConstraintsManager'
    :param item: Item
    :return: ct: ortools.Constraint
    """
    ct = None
    for j_item in manager.sp.spec.items:
        if item.required_area < j_item.required_area:
            if ct is None:
                ct = (manager.windows_length[str(item.id)] <=
                      manager.windows_length[str(j_item.id)])
            else:
                new_ct = (manager.windows_length[str(item.id)] <=
                          manager.windows_length[str(j_item.id)])
                ct = manager.solver.solver.Min(ct, new_ct)
    if ct is None:
        return ct
    else:
        return ct == 1


def opens_on_constraint(manager: 'ConstraintsManager', item: Item,
                        length: int) -> ortools.Constraint:
    """
    Opens on constraint : check the adjacency between two rooms if open on, otherwise impose the
    presence of a window in the room
    :param manager: 'ConstraintsManager'
    :param item: Item
    :param length: int
    :return: ct: ortools.Constraint
    """
    ct = None
    if item.opens_on:
        for category_name in item.opens_on:
            adjacency_sum = 0
            for other_item in manager.sp.spec.items:
                if other_item.category.name == category_name:
                    adjacency_sum += manager.solver.solver.Sum(
                        manager.solver.solver.Sum(
                            int(j_space.maximum_adjacency_length(k_space)) *
                            manager.solver.positions[item.id, j] *
                            manager.solver.positions[other_item.id, k] for
                            j, j_space in enumerate(manager.sp.spec.plan.mutable_spaces()))
                        for k, k_space in enumerate(manager.sp.spec.plan.mutable_spaces()))
            if ct is None:
                ct = (adjacency_sum >= length)
            else:
                ct = manager.and_(ct, (adjacency_sum >= length))
    else:
        ct = components_adjacency_constraint(manager, item, WINDOW_CATEGORY, addition_rule="Or")
    return ct


def symmetry_breaker_constraint(manager: 'ConstraintsManager',
                                item: Item) -> ortools.Constraint:
    """
    Symmetry Breaker constraint
    :param manager: 'ConstraintsManager'
    :param item: Item
    :return: ct: ortools.Constraint
    """
    ct = None
    item_sym_id = str(item.category.name + item.variant)
    if item_sym_id in manager.symmetry_breaker_memo.keys():
        for j, j_space in enumerate(manager.sp.spec.plan.mutable_spaces()):
            for k, k_space in enumerate(manager.sp.spec.plan.mutable_spaces()):
                if k < j:
                    if ct is None:
                        ct = (manager.solver.positions[
                                  manager.symmetry_breaker_memo[item_sym_id], j] *
                              manager.solver.positions[item.id, k] == 0)
                    else:
                        ct = manager.and_(ct, (manager.solver.positions[
                                                   manager.symmetry_breaker_memo[
                                                       item_sym_id], j] *
                                               manager.solver.positions[item.id, k] == 0))
    manager.symmetry_breaker_memo[item_sym_id] = item.id
    return ct


def inside_adjacency_constraint(manager: 'ConstraintsManager',
                                item: Item) -> ortools.Constraint:
    """
    Space adjacency constraint inside a given item
    :param manager: 'ConstraintsManager'
    :param item: Item
    :return: ct: ortools.Constraint
    """
    nbr_spaces_in_i_item = manager.solver.solver.Sum(
        manager.solver.positions[item.id, j] for j in
        range(manager.sp.spec.plan.count_mutable_spaces()))
    spaces_adjacency = manager.solver.solver.Sum(
        manager.solver.solver.Sum(
            int(j_space.adjacent_to(k_space)) *
            manager.solver.positions[item.id, j] *
            manager.solver.positions[item.id, k] for
            j, j_space in enumerate(manager.sp.spec.plan.mutable_spaces()) if j > k)
        for k, k_space in enumerate(manager.sp.spec.plan.mutable_spaces()))
    ct1 = (spaces_adjacency >= nbr_spaces_in_i_item - 1)

    ct2 = None
    for k, k_space in enumerate(manager.sp.spec.plan.mutable_spaces()):
        a = (manager.solver.positions[item.id, k] *
             manager.solver.solver
             .Sum(int(j_space.adjacent_to(k_space)) * manager.solver.positions[item.id, j]
                  for j, j_space in enumerate(manager.sp.spec.plan.mutable_spaces()) if k != j))

        if ct2 is None:
            ct2 = manager.solver.solver.Max(
                a >= manager.solver.positions[item.id, k],
                nbr_spaces_in_i_item == 1)
        else:
            ct2 = (manager.solver.solver
                   .Min(ct2, manager.solver.solver
                        .Max(a >= manager.solver.positions[item.id, k],
                             nbr_spaces_in_i_item == 1)))

    ct = (manager.solver.solver.Min(ct1, ct2) == 1)

    return ct


def item_adjacency_constraint(manager: 'ConstraintsManager', item: Item,
                              item_categories: List[str], adj: bool = True,
                              addition_rule: str = '') -> ortools.Constraint:
    """
    Item adjacency constraint :
    :param manager: 'ConstraintsManager'
    :param item: Item
    :param item_categories: List[str]
    :param adj: bool
    :param addition_rule: str
    :return: ct: ortools.Constraint
    """
    ct = None
    for cat in item_categories:
        adjacency_sum = 0
        if cat == 'circulationSpace':
            adjacency_sum += manager.solver.solver.Sum(
                manager.solver.solver.Sum(
                    int(j_space.adjacent_to(k_space)) *
                    manager.solver.positions[item.id, j] *
                    (manager.solver.solver.Sum(manager.solver.positions[x_item.id, k] for x_item in
                                               manager.sp.spec.items) == 0) for
                    j, j_space in enumerate(manager.sp.spec.plan.mutable_spaces()))
                for k, k_space in enumerate(manager.sp.spec.plan.mutable_spaces()))
        else:
            for num, num_item in enumerate(manager.sp.spec.items):
                if num_item.category.name == cat:
                    adjacency_sum += manager.solver.solver.Sum(
                        manager.solver.solver.Sum(
                            int(j_space.adjacent_to(k_space)) *
                            manager.solver.positions[item.id, j] *
                            manager.solver.positions[num, k] for
                            j, j_space in enumerate(manager.sp.spec.plan.mutable_spaces()))
                        for k, k_space in enumerate(manager.sp.spec.plan.mutable_spaces()))

        if adjacency_sum is not 0:
            if ct is None:
                if adj:
                    ct = (adjacency_sum >= 1)
                else:
                    ct = (adjacency_sum == 0)
            else:
                if adj:
                    if addition_rule == "Or":
                        ct = manager.or_(ct, (adjacency_sum >= 1))
                    elif addition_rule == "And":
                        ct = manager.and_(ct, (adjacency_sum >= 1))
                    else:
                        ValueError("ComponentsAdjacencyConstraint")
                else:
                    if addition_rule == "Or":
                        ct = manager.or_(ct, (adjacency_sum == 0))
                    elif addition_rule == "And":
                        ct = manager.and_(ct, (adjacency_sum == 0))
                    else:
                        ValueError("ComponentsAdjacencyConstraint")

    return ct


def components_adjacency_constraint(manager: 'ConstraintsManager', item: Item,
                                    category: Sequence[str], adj: bool = True,
                                    addition_rule: str = '') -> ortools.Constraint:
    """
    Components adjacency constraint
    :param manager: 'ConstraintsManager'
    :param item: Item
    :param category: List[str]
    :param adj: bool
    :param addition_rule: str
    :return: ct: ortools.Constraint
    """
    ct = None
    for c, cat in enumerate(category):
        adjacency_sum = manager.solver.solver.Sum(
            manager.solver.positions[item.id, j] for j, space in
            enumerate(manager.sp.spec.plan.mutable_spaces()) if
            cat in space.components_category_associated())
        if c == 0:
            if adj:
                ct = (adjacency_sum >= 1)
            else:
                ct = (adjacency_sum == 0)
        else:
            if adj:
                if addition_rule == "Or":
                    ct = manager.or_(ct, (adjacency_sum >= 1))
                elif addition_rule == "And":
                    ct = manager.and_(ct, (adjacency_sum >= 1))
                else:
                    ValueError("ComponentsAdjacencyConstraint")
            else:
                if addition_rule == "Or":
                    ct = manager.or_(ct, (adjacency_sum == 0))
                elif addition_rule == "And":
                    ct = manager.and_(ct, (adjacency_sum == 0))
                else:
                    ValueError("ComponentsAdjacencyConstraint")

    return ct


def externals_connection_constraint(manager: 'ConstraintsManager',
                                    item: Item) -> ortools.Constraint:
    """
    externals connection constraint
    :param manager: 'ConstraintsManager'
    :param item: Item
    :return: ct: ortools.Constraint
    """
    ct = None

    has_to_be_connected = False
    for space in manager.sp.spec.plan.spaces:
        if space.category.external and space.area > BIG_EXTERNAL_SPACE:
            has_to_be_connected = True
            break

    if has_to_be_connected:
        adjacency_sum = manager.solver.solver.Sum(
            manager.solver.positions[item.id, j] for j, space in
            enumerate(manager.sp.spec.plan.mutable_spaces())
            if max([ext_space.area for ext_space in space.connected_spaces()
                    if ext_space is not None and ext_space.category.external],
                   default=0) > BIG_EXTERNAL_SPACE)
        ct = (adjacency_sum >= 1)

    return ct


GENERAL_ITEMS_CONSTRAINTS = {
    "all": [
        [inside_adjacency_constraint, {}],
        [area_constraint, {"min_max": "min"}],
        [distance_constraint, {}],
        [graph_constraint, {}],
        [shape_constraint, {}],
        [windows_constraint, {}],
    ],
    "entrance": [
        [components_adjacency_constraint, {"category": ["frontDoor"], "adj": True}],  # ???
        [area_constraint, {"min_max": "max"}]
    ],
    "wc": [
        [item_attribution_constraint, {}],
        [components_adjacency_constraint, {"category": ["duct"], "adj": True}],
        [components_adjacency_constraint,
         {"category": WINDOW_CATEGORY, "adj": False, "addition_rule": "And"}],
        [components_adjacency_constraint, {"category": ["startingStep"], "adj": False}],
        [area_constraint, {"min_max": "max"}],
        [symmetry_breaker_constraint, {}]
    ],
    "bathroom": [
        [item_attribution_constraint, {}],
        [components_adjacency_constraint, {"category": ["duct"], "adj": True}],
        [components_adjacency_constraint, {"category": ["doorWindow"], "adj": False}],
        [components_adjacency_constraint, {"category": ["startingStep"], "adj": False}],
        [area_constraint, {"min_max": "max"}],
        [symmetry_breaker_constraint, {}]
    ],
    "living": [
        [item_attribution_constraint, {}],
        [components_adjacency_constraint,
         {"category": WINDOW_CATEGORY, "adj": True, "addition_rule": "Or"}],
        [item_adjacency_constraint,
         {"item_categories": ("kitchen", "dining"), "adj": True, "addition_rule": "Or"}]
    ],
    "dining": [
        [item_attribution_constraint, {}],
        [components_adjacency_constraint,
         {"category": WINDOW_CATEGORY, "adj": True, "addition_rule": "Or"}],
        [item_adjacency_constraint, {"item_categories": "kitchen"}]
    ],
    "kitchen": [
        [item_attribution_constraint, {}],
        [opens_on_constraint, {"length": 220}],
        [components_adjacency_constraint, {"category": ["duct"], "adj": True}],
        [area_constraint, {"min_max": "max"}],
        [item_adjacency_constraint,
         {"item_categories": ("living", "dining"), "adj": True, "addition_rule": "Or"}],
        [components_adjacency_constraint, {"category": ["startingStep"], "adj": False}]
    ],
    "bedroom": [
        [item_attribution_constraint, {}],
        [opens_on_constraint, {"length": 220}],
        [area_constraint, {"min_max": "max"}],
        [components_adjacency_constraint, {"category": ["startingStep"], "adj": False}],
        [symmetry_breaker_constraint, {}]
    ],
    "office": [
        [item_attribution_constraint, {}],
        [opens_on_constraint, {"length": 220}],
        [area_constraint, {"min_max": "max"}],
        [components_adjacency_constraint, {"category": ["startingStep"], "adj": False}],
        [symmetry_breaker_constraint, {}]
    ],
    "dressing": [
        [item_attribution_constraint, {}],
        [components_adjacency_constraint,
         {"category": WINDOW_CATEGORY, "adj": False, "addition_rule": "And"}],
        [components_adjacency_constraint, {"category": ["startingStep"], "adj": False}],
        [area_constraint, {"min_max": "max"}],
        [symmetry_breaker_constraint, {}]
    ],
    "laundry": [
        [item_attribution_constraint, {}],
        [components_adjacency_constraint, {"category": ["duct"], "adj": True}],
        [components_adjacency_constraint,
         {"category": WINDOW_CATEGORY, "adj": False, "addition_rule": "And"}],
        [components_adjacency_constraint, {"category": ["startingStep"], "adj": False}],
        [area_constraint, {"min_max": "max"}],
        [symmetry_breaker_constraint, {}]
    ]
}

T3_MORE_ITEMS_CONSTRAINTS = {
    "all": [

    ],
    "entrance": [

    ],
    "wc": [
        [item_adjacency_constraint,
         {"item_categories": PRIVATE_ROOMS, "adj": True, "addition_rule": "Or"}]
    ],
    "bathroom": [
        [item_adjacency_constraint,
         {"item_categories": PRIVATE_ROOMS, "adj": True, "addition_rule": "Or"}]
    ],
    "living": [
        [externals_connection_constraint, {}]
    ],
    "dining": [

    ],
    "kitchen": [

    ],
    "bedroom": [

    ],
    "office": [

    ],
    "dressing": [
        [item_adjacency_constraint,
         {"item_categories": PRIVATE_ROOMS, "adj": True, "addition_rule": "Or"}]
    ],
    "laundry": [
        [item_adjacency_constraint,
         {"item_categories": PRIVATE_ROOMS, "adj": True, "addition_rule": "Or"}]
    ]
}
