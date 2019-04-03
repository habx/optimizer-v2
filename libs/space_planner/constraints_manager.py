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
from libs.specification.specification import Item
import networkx as nx
import time
import logging

if TYPE_CHECKING:
    from libs.space_planner.space_planner import SpacePlanner

WINDOW_ROOMS = ["living", "kitchen", "livingKitchen", "study", "dining", "bedroom"]

DRESSING_NEIGHBOUR_ROOMS = ["entrance", "bedroom", "toilet", "bathroom"]

CIRCULATION_ROOMS = ["living", "livingKitchen", "dining", "entrance", "circulationSpace"]

DAY_ROOMS = ["living", "livingKitchen", "dining", "kitchen", "cellar"]

PRIVATE_ROOMS = ["bedroom", "bathroom", "laundry", "dressing", "entrance", "circulationSpace",
                 "toilet"]

WINDOW_CATEGORY = ["window", "doorWindow"]

BIG_VARIANTS = ["m", "l", "xl"]

SMALL_VARIANTS = ["xs", "s"]

OPEN_ON_ADJACENCY_SIZE = 200
BIG_EXTERNAL_SPACE = 7000

SQM = 10000
LBW_THICKNESS = 30
MAX_AREA_COEFF = 4 / 3
MIN_AREA_COEFF = 2 / 3
INSIDE_ADJACENCY_LENGTH = 40


class ConstraintSolver:
    """
    Constraint Solver
    Encapsulation of the OR-tools solver adapted to our problem
    """

    def __init__(self, items_nbr: int, spaces_nbr: int, spaces_adjacency_matrix: List[List[int]],
                 multilevel: bool = False):
        self.items_nbr = items_nbr
        self.spaces_nbr = spaces_nbr
        self.multilevel = multilevel
        self.spaces_adjacency_matrix = spaces_adjacency_matrix
        # Create the solver
        self.solver = ortools.Solver('SpacePlanner')
        # Declare variables
        self.cells_item: List[ortools.IntVar] = []
        self.positions = {}
        self._init_positions()
        self.solutions = []

    def _init_positions(self) -> None:
        """
        variables initialization
        :return: None
        """
        # cells in [0, self.items_nbr-1], self.items_nbr for multilevel plans : circulationSpace
        if not self.multilevel:
            self.cells_item = [self.solver.IntVar(0, self.items_nbr - 1,
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
        for i_item in range(self.items_nbr):
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

        connectivity_checker = check_room_connectivity_factory(self.spaces_adjacency_matrix)

        # Maximum number of solutions
        max_num_sol = 1000
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
            validity = self._check_adjacency(sol_positions, connectivity_checker)
            if validity:
                self.solutions.append(sol_positions)

                # Number of solutions
                nbr_solutions += 1
                if nbr_solutions >= max_num_sol or (time.clock() - t0 - 600) >= 0:
                    break

        # noinspection PyArgumentList
        self.solver.EndSearch()

        logging.debug("ConstraintSolver: Statistics")
        logging.debug("ConstraintSolver: num_solutions: %i", nbr_solutions)
        print("ConstraintSolver: num_solutions:", nbr_solutions)
        logging.debug("ConstraintSolver: failures: %i", self.solver.Failures())
        logging.debug("ConstraintSolver: branches:  %i", self.solver.Branches())
        # logging.debug("ConstraintSolver: WallTime:  %i", self.solver.WallTime())
        logging.debug("ConstraintSolver: Process time : %f", time.clock() - t0)


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

        self.spaces_adjacency_matrix = []
        self._init_spaces_adjacency()
        if sp.spec.plan.floor_count < 2:
            self.solver = ConstraintSolver(len(self.sp.spec.items),
                                           self.sp.spec.plan.count_mutable_spaces(),
                                           self.spaces_adjacency_matrix, False)
        else:
            self.solver = ConstraintSolver(len(self.sp.spec.items),
                                           self.sp.spec.plan.count_mutable_spaces(),
                                           self.spaces_adjacency_matrix, True)

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
                                   * int(round(component.length / 10)))
            self.windows_length[str(item.id)] = length

    def _init_spaces_distance(self) -> None:
        """
        Initialize the spaces distance matrix
        :return:
        """

        for i, i_space in enumerate(self.sp.spec.plan.mutable_spaces()):
            self.spaces_distance.append([])
            self.spaces_distance[i] = [0] * len(list(self.sp.spec.plan.mutable_spaces()))

        for i, i_space in enumerate(self.sp.spec.plan.mutable_spaces()):
            for j, j_space in enumerate(self.sp.spec.plan.mutable_spaces()):
                if i < j:
                    if i_space.floor != j_space.floor:
                        self.spaces_distance[i][j] = 1e20
                        self.spaces_distance[j][i] = 1e20
                    else:
                        self.spaces_distance[i][j] = int(i_space.maximum_distance_to(j_space))
                        self.spaces_distance[j][i] = int(i_space.maximum_distance_to(j_space))

    def _init_spaces_graph(self) -> None:
        """
        Initialize the graph of adjacent seed spaces
        :return:
        """

        for i, i_space in enumerate(self.sp.spec.plan.mutable_spaces()):
            for j, j_space in enumerate(self.sp.spec.plan.mutable_spaces()):
                if i < j:
                    if i_space.adjacent_to(j_space, INSIDE_ADJACENCY_LENGTH):
                        self.space_graph.add_edge(i, j, weight=1)
                        self.space_graph.add_edge(j, i, weight=1)

    def _init_area_spaces_graph(self) -> None:
        """
        Initialize the graph of adjacent seed spaces with weight = space area
        :return:
        """

        for i, i_space in enumerate(self.sp.spec.plan.mutable_spaces()):
            for j, j_space in enumerate(self.sp.spec.plan.mutable_spaces()):
                if i < j:
                    if i_space.adjacent_to(j_space, INSIDE_ADJACENCY_LENGTH):
                        self.area_space_graph.add_edge(i, j, weight=j_space.area + i_space.area)
                        self.area_space_graph.add_edge(j, i, weight=j_space.area + i_space.area)

    def _init_spaces_adjacency(self) -> None:
        """
        spaces adjacency matrix init
        :return: None
        """
        for i, i_space in enumerate(self.sp.spec.plan.mutable_spaces()):
            self.spaces_adjacency_matrix.append([])
            for j, j_space in enumerate(self.sp.spec.plan.mutable_spaces()):
                if j != i:
                    self.spaces_adjacency_matrix[i].append(0)
                else:
                    self.spaces_adjacency_matrix[i].append(1)

        for i, i_space in enumerate(self.sp.spec.plan.mutable_spaces()):
            for j, j_space in enumerate(self.sp.spec.plan.mutable_spaces()):
                if j < i:
                    if i_space.adjacent_to(j_space):
                        self.spaces_adjacency_matrix[i][j] = 1
                        self.spaces_adjacency_matrix[j][i] = 1
                    else:
                        self.spaces_adjacency_matrix[i][j] = 0
                        self.spaces_adjacency_matrix[j][i] = 0

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
            if self.sp.spec.typology >= 2 and self.sp.spec.number_of_items > 4:
                for constraint in T2_MORE_ITEMS_CONSTRAINTS["all"]:
                    self.add_item_constraint(item, constraint[0], **constraint[1])
                for constraint in T2_MORE_ITEMS_CONSTRAINTS.get(item.category.name, []):
                    self.add_item_constraint(item, constraint[0], **constraint[1])
            if self.sp.spec.typology >= 3:
                for constraint in T3_MORE_ITEMS_CONSTRAINTS["all"]:
                    self.add_item_constraint(item, constraint[0], **constraint[1])
                for constraint in T3_MORE_ITEMS_CONSTRAINTS.get(item.category.name, []):
                    self.add_item_constraint(item, constraint[0], **constraint[1])
            if self.sp.spec.typology >= 4:
                for constraint in T4_MORE_ITEMS_CONSTRAINTS["all"]:
                    self.add_item_constraint(item, constraint[0], **constraint[1])
                for constraint in T4_MORE_ITEMS_CONSTRAINTS.get(item.category.name, []):
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

    if min_max == "max":
        if ((item.variant in ["l", "xl"] or item.category.name in ["entrance"])
                and item.category.name not in ["living", "livingKitchen", "dining"]):
            max_area = round(item.max_size.area)
        else:
            max_area = round(max(item.max_size.area * MAX_AREA_COEFF, item.max_size.area + 1 * SQM))
        ct = (manager.solver.solver
              .Sum(manager.solver.positions[item.id, j] * round(space.area)
                   for j, space in enumerate(manager.sp.spec.plan.mutable_spaces())) <= max_area)

    elif min_max == "min":
        if item.variant in ["xs", "s"]:
            min_area = round(item.min_size.area)
        else:
            min_area = round(min(item.min_size.area * MIN_AREA_COEFF, item.min_size.area - 1 * SQM))
        ct = (manager.solver.solver
              .Sum(manager.solver.positions[item.id, j] * round(space.area)
                   for j, space in enumerate(manager.sp.spec.plan.mutable_spaces())) >= min_area)
    else:
        ValueError("AreaConstraint")

    return ct


def distance_constraint(manager: 'ConstraintsManager', item: Item) -> ortools.Constraint:
    """
    Maximum distance constraint between spaces constraint
    :param manager: 'ConstraintsManager'
    :param item: Item
    :return: ct: ortools.Constraint
    # TODO : find best param
    # TODO : unit tests
    """
    if item.category.name in ["living", "dining", "livingKitchen", "dressing", "laundry"]:
        param = 2.2
    elif item.category.name in ["bathroom", "study", "misc", "kitchen"]:
        param = 1.8
    else:
        param = 1.6

    max_distance = int(round(param * item.max_size.area ** 0.5))

    ct = None

    for j, j_space in enumerate(manager.sp.spec.plan.mutable_spaces()):
        for k, k_space in enumerate(manager.sp.spec.plan.mutable_spaces()):
            if j < k:
                if ct is None:
                    ct = ((manager.solver.positions[item.id, j] *
                           manager.solver.positions[item.id, k])
                          <= int(max_distance / manager.spaces_distance[j][k]))
                else:
                    new_ct = ((manager.solver.positions[item.id, j] *
                               manager.solver.positions[item.id, k])
                              <= int(max_distance / manager.spaces_distance[j][k]))
                    ct = manager.and_(ct, new_ct)

    return ct


def area_graph_constraint(manager: 'ConstraintsManager', item: Item) -> ortools.Constraint:
    """
    Graph constraint:
    - existing path between two seed space
    - shortest path area < max area (+ margin)
    :param manager: 'ConstraintsManager'
    :param item: Item
    :return: ct: ortools.Constraint
    # TODO : unit tests
    """

    ct = None
    if ((item.variant in ["l", "xl"] or item.category.name in ["entrance"])
            and item.category.name not in ["living", "livingKitchen", "dining"]):
        max_area = round(item.max_size.area)
    else:
        max_area = round(max(item.max_size.area * MAX_AREA_COEFF, item.max_size.area + 1 * SQM))

    for j, j_space in enumerate(manager.sp.spec.plan.mutable_spaces()):
        for k, k_space in enumerate(manager.sp.spec.plan.mutable_spaces()):
            if j < k:
                if ct is None:
                    if (j not in nx.nodes(manager.area_space_graph)
                            or k not in nx.nodes(manager.area_space_graph)
                            or not nx.has_path(manager.area_space_graph, j, k)):
                        ct = (manager.solver.positions[item.id, j] *
                              manager.solver.positions[item.id, k] == 0)
                    else:
                        path = nx.dijkstra_path(manager.area_space_graph, j, k)
                        area_path = sum(int(space.area)
                                        for i, space in
                                        enumerate(manager.sp.spec.plan.mutable_spaces())
                                        if i in path)
                        ct = (manager.solver.positions[item.id, j] *
                              manager.solver.positions[item.id, k] * area_path
                              <= max_area)

                else:
                    if (j not in nx.nodes(manager.area_space_graph)
                            or k not in nx.nodes(manager.area_space_graph)
                            or not nx.has_path(manager.area_space_graph, j, k)):
                        new_ct = (manager.solver.positions[item.id, j] *
                                  manager.solver.positions[item.id, k] == 0)
                    else:
                        path = nx.dijkstra_path(manager.area_space_graph, j, k)
                        area_path = sum(int(space.area)
                                        for i, space in
                                        enumerate(manager.sp.spec.plan.mutable_spaces())
                                        if i in path)
                        new_ct = (manager.solver.positions[item.id, j] *
                                  manager.solver.positions[item.id, k] * area_path
                                  <= max_area)
                    ct = manager.and_(ct, new_ct)

    return ct


def graph_constraint(manager: 'ConstraintsManager', item: Item) -> ortools.Constraint:
    """
    Graph constraint:
    - existing path between two seed space
    - shortest path number of seed spaces <= nbr of seed spaces
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
                    if j not in nx.nodes(manager.space_graph) or k not in nx.nodes(
                            manager.space_graph) or not nx.has_path(manager.space_graph, j, k):
                        ct = (manager.solver.positions[item.id, j] *
                              manager.solver.positions[item.id, k] == 0)
                    else:
                        ct = ((manager.solver.positions[item.id, j] *
                               manager.solver.positions[item.id, k] *
                               len(nx.dijkstra_path(manager.space_graph, j, k)))
                              <= manager.solver.solver.Sum(manager.solver.positions[item.id, l]
                                                           for l, l_space in enumerate(
                                    manager.sp.spec.plan.mutable_spaces())))

                else:
                    if j not in nx.nodes(manager.space_graph) or k not in nx.nodes(
                            manager.space_graph) or not nx.has_path(manager.space_graph, j, k):
                        new_ct = (manager.solver.positions[item.id, j] *
                                  manager.solver.positions[item.id, k] == 0)
                    else:
                        new_ct = ((manager.solver.positions[item.id, j] *
                                   manager.solver.positions[item.id, k] *
                                   len(nx.dijkstra_path(manager.space_graph, j, k)))
                                  <= manager.solver.solver.Sum(manager.solver.positions[item.id, l]
                                                               for l, l_space in enumerate(
                                    manager.sp.spec.plan.mutable_spaces())))
                    ct = manager.and_(ct, new_ct)

    return ct


def shape_constraint(manager: 'ConstraintsManager', item: Item) -> ortools.Constraint:
    """
    Shape constraint : perimeter**2/area
    :param manager: 'ConstraintsManager'
    :param item: Item
    :return: ct: ortools.Constraint
    # TODO : find best param
    # TODO : unit tests
    """

    plan_ratio = round(manager.sp.spec.plan.indoor_perimeter ** 2
                       / manager.sp.spec.plan.indoor_area)

    if item.category.name in ["living", "dining", "livingKitchen", "dressing", "laundry"]:
        param = min(max(28, int(plan_ratio + 10)), 40)
    elif item.category.name in ["bathroom", "study", "misc", "kitchen"]:
        param = min(max(22, int(plan_ratio)), 30)
    else:
        param = 20  # wc / bedroom / entrance / kitchen

    item_area = manager.solver.solver.Sum(manager.solver.positions[item.id, j] * int(space.area)
                                          for j, space in
                                          enumerate(manager.sp.spec.plan.mutable_spaces()))
    if item.category.name in ["toilet", "bathroom"]:
        cells_perimeter = manager.solver.solver.Sum(manager.solver.positions[item.id, j] *
                                                    int(space.perimeter_without_duct)
                                                    for j, space in
                                                    enumerate(
                                                        manager.sp.spec.plan.mutable_spaces()))
    else:
        cells_perimeter = manager.solver.solver.Sum(manager.solver.positions[item.id, j] *
                                                    int(space.perimeter)
                                                    for j, space in
                                                    enumerate(
                                                        manager.sp.spec.plan.mutable_spaces()))
    cells_adjacency = manager.solver.solver.Sum(manager.solver.positions[item.id, j] *
                                                manager.solver.positions[item.id, k] *
                                                int(j_space.contact_length(k_space))
                                                for j, j_space in
                                                enumerate(manager.sp.spec.plan.mutable_spaces())
                                                for k, k_space in
                                                enumerate(manager.sp.spec.plan.mutable_spaces())
                                                )
    item_perimeter = cells_perimeter - cells_adjacency
    ct = (item_perimeter * item_perimeter <= param * item_area)

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
        if (item.category.name in WINDOW_ROOMS and j_item.category.name in WINDOW_ROOMS
                and item.required_area < j_item.required_area):
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


def symmetry_breaker_constraint(manager: 'ConstraintsManager', item: Item) -> ortools.Constraint:
    """
    Symmetry Breaker constraint
    :param manager: 'ConstraintsManager'
    :param item: Item
    :return: ct: ortools.Constraint
    """
    ct = None
    item_sym_id = str(item.category.name + item.variant)
    if item_sym_id in manager.symmetry_breaker_memo.keys():
        memo = manager.solver.solver.Sum(
            2 ** j * manager.solver.positions[manager.symmetry_breaker_memo[item_sym_id], j]
            for j in range(manager.solver.spaces_nbr))
        current = manager.solver.solver.Sum(2 ** j * manager.solver.positions[item.id, j]
                                            for j in range(manager.solver.spaces_nbr))
        ct = manager.solver.solver.IsLessVar(memo, current) == 1

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
            int(j_space.adjacent_to(k_space, INSIDE_ADJACENCY_LENGTH)) *
            manager.solver.positions[item.id, j] *
            manager.solver.positions[item.id, k] for
            j, j_space in enumerate(manager.sp.spec.plan.mutable_spaces()) if j > k)
        for k, k_space in enumerate(manager.sp.spec.plan.mutable_spaces()))
    ct1 = (spaces_adjacency >= nbr_spaces_in_i_item - 1)

    ct2 = None
    for k, k_space in enumerate(manager.sp.spec.plan.mutable_spaces()):
        a = (manager.solver.positions[item.id, k] *
             manager.solver.solver
             .Sum(int(j_space.adjacent_to(k_space, INSIDE_ADJACENCY_LENGTH)) *
                  manager.solver.positions[item.id, j]
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

    ct = (manager.and_(ct1, ct2) == 1)

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
                    int(j_space.distance_to(k_space, "min") < LBW_THICKNESS) *
                    int(k_space.floor.level == j_space.floor.level) *
                    manager.solver.positions[item.id, j] *
                    (manager.solver.solver.Sum(manager.solver.positions[x_item.id, k] for x_item in
                                               manager.sp.spec.items) == 0) for
                    j, j_space in enumerate(manager.sp.spec.plan.mutable_spaces()))
                for k, k_space in enumerate(manager.sp.spec.plan.mutable_spaces()))
        else:
            for num, num_item in enumerate(manager.sp.spec.items):
                if num_item.category.name == cat and num_item != item:
                    adjacency_sum += manager.solver.solver.Sum(
                        manager.solver.solver.Sum(
                            int(j_space.distance_to(k_space, "min") < LBW_THICKNESS) *
                            int(k_space.floor.level == j_space.floor.level) *
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
        [graph_constraint, {}],
        [area_graph_constraint, {}],
        [distance_constraint, {}],
        [shape_constraint, {}],
        [windows_constraint, {}],
        [symmetry_breaker_constraint, {}]
    ],
    "entrance": [
        [components_adjacency_constraint, {"category": ["frontDoor"], "adj": True}],  # ???
        [area_constraint, {"min_max": "max"}]
    ],
    "toilet": [
        [area_constraint, {"min_max": "min"}],
        [item_attribution_constraint, {}],
        [components_adjacency_constraint, {"category": ["duct"], "adj": True}],
        [components_adjacency_constraint,
         {"category": WINDOW_CATEGORY, "adj": False, "addition_rule": "And"}],
        [components_adjacency_constraint, {"category": ["startingStep"], "adj": False}],
        [area_constraint, {"min_max": "max"}],
    ],
    "bathroom": [
        [area_constraint, {"min_max": "min"}],
        [item_attribution_constraint, {}],
        [components_adjacency_constraint, {"category": ["duct"], "adj": True}],
        [components_adjacency_constraint, {"category": ["doorWindow"], "adj": False}],
        [components_adjacency_constraint, {"category": ["startingStep"], "adj": False}],
        [area_constraint, {"min_max": "max"}],
    ],
    "living": [
        [area_constraint, {"min_max": "min"}],
        [item_attribution_constraint, {}],
        [components_adjacency_constraint,
         {"category": WINDOW_CATEGORY, "adj": True, "addition_rule": "Or"}],
        [item_adjacency_constraint,
         {"item_categories": ("kitchen", "dining"), "adj": True, "addition_rule": "Or"}]
    ],
    "livingKitchen": [
        [area_constraint, {"min_max": "min"}],
        [item_attribution_constraint, {}],
        [components_adjacency_constraint,
         {"category": WINDOW_CATEGORY, "adj": True, "addition_rule": "Or"}],
        [item_adjacency_constraint,
         {"item_categories": ("kitchen", "dining"), "adj": True, "addition_rule": "Or"}]
    ],
    "dining": [
        [area_constraint, {"min_max": "min"}],
        [item_attribution_constraint, {}],
        [opens_on_constraint, {"length": 220}],
        [components_adjacency_constraint,
         {"category": WINDOW_CATEGORY, "adj": True, "addition_rule": "Or"}],
        [item_adjacency_constraint, {"item_categories": ["kitchen"]}]
    ],
    "kitchen": [
        [area_constraint, {"min_max": "min"}],
        [item_attribution_constraint, {}],
        [opens_on_constraint, {"length": 220}],
        [components_adjacency_constraint, {"category": ["duct"], "adj": True}],
        [area_constraint, {"min_max": "max"}],
        [item_adjacency_constraint,
         {"item_categories": ("living", "dining"), "adj": True, "addition_rule": "Or"}],
        [components_adjacency_constraint, {"category": ["startingStep"], "adj": False}]
    ],
    "bedroom": [
        [area_constraint, {"min_max": "min"}],
        [item_attribution_constraint, {}],
        [opens_on_constraint, {"length": 220}],
        [area_constraint, {"min_max": "max"}],
        [components_adjacency_constraint, {"category": ["startingStep"], "adj": False}],
    ],
    "study": [
        [area_constraint, {"min_max": "min"}],
        [item_attribution_constraint, {}],
        [opens_on_constraint, {"length": 220}],
        [area_constraint, {"min_max": "max"}],
        [components_adjacency_constraint, {"category": ["startingStep"], "adj": False}],
    ],
    "dressing": [
        [area_constraint, {"min_max": "min"}],
        [item_attribution_constraint, {}],
        [components_adjacency_constraint,
         {"category": WINDOW_CATEGORY, "adj": False, "addition_rule": "And"}],
        [components_adjacency_constraint, {"category": ["startingStep"], "adj": False}],
        [area_constraint, {"min_max": "max"}],
    ],
    "laundry": [
        [area_constraint, {"min_max": "min"}],
        [item_attribution_constraint, {}],
        [components_adjacency_constraint, {"category": ["duct"], "adj": True}],
        [components_adjacency_constraint,
         {"category": WINDOW_CATEGORY, "adj": False, "addition_rule": "And"}],
        [components_adjacency_constraint, {"category": ["startingStep"], "adj": False}],
        [area_constraint, {"min_max": "max"}],
    ]
}

T2_MORE_ITEMS_CONSTRAINTS = {
    "all": [

    ],
    "livingKitchen": [
        [components_adjacency_constraint, {"category": ["duct"], "adj": True}],
    ]
}

T3_MORE_ITEMS_CONSTRAINTS = {
    "all": [

    ],
    "toilet": [
        [item_adjacency_constraint,
         {"item_categories": PRIVATE_ROOMS, "adj": True, "addition_rule": "Or"}],
        [item_adjacency_constraint, {"item_categories": ["toilet"], "adj": False}]
    ],
    "bathroom": [
        [item_adjacency_constraint, {"item_categories": ["bedroom"], "adj": True}],
    ],
    "living": [
        [externals_connection_constraint, {}]
    ],
    "livingKitchen": [
        [externals_connection_constraint, {}]
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
T4_MORE_ITEMS_CONSTRAINTS = {
    "all": [

    ],
    "bathroom": [
        [item_adjacency_constraint, {"item_categories": ["bathroom"], "adj": False}]
    ],
    "bedroom": [
        [item_adjacency_constraint,
         {"item_categories": PRIVATE_ROOMS, "adj": True, "addition_rule": "Or"}]
    ]
}
