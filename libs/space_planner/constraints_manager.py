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
from libs.plan.category import LinearCategory
import networkx as nx
import time
import logging

if TYPE_CHECKING:
    from libs.space_planner.space_planner import SpacePlanner

WINDOW_ROOMS = ["living", "kitchen", "livingKitchen", "study", "dining", "bedroom"]

CIRCULATION_ROOMS = ["living", "livingKitchen", "dining", "entrance", "circulation"]

DAY_ROOMS = ["living", "livingKitchen", "dining", "kitchen", "cellar", "study"]

PRIVATE_ROOMS = ["bedroom", "study", "bathroom", "laundry", "wardrobe", "entrance", "circulation",
                 "toilet"]

WINDOW_CATEGORY = ["window", "doorWindow"]

BIG_VARIANTS = ["m", "l", "xl"]

SMALL_VARIANTS = ["xs", "s"]

OPEN_ON_ADJACENCY_SIZE = 200


SQM = 10000
BIG_EXTERNAL_SPACE = 7*SQM
LBW_THICKNESS = 30
MAX_AREA_COEFF = 4 / 3
MIN_AREA_COEFF = 2 / 3
INSIDE_ADJACENCY_LENGTH = 20
ITEM_ADJACENCY_LENGTH = 100
SEARCH_TIME_LIMIT = 1800000  # millisecond
SEARCH_SOLUTIONS_LIMIT = 1000


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
        # cells in [0, self.items_nbr-1], self.items_nbr for multilevel plans : circulation
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
        t0 = time.process_time()
        decision_builder = self.solver.Phase(self.cells_item, self.solver.CHOOSE_FIRST_UNBOUND,
                               self.solver.ASSIGN_MIN_VALUE)
        time_limit = self.solver.TimeLimit(SEARCH_TIME_LIMIT)
        self.solver.NewSearch(decision_builder, time_limit)

        connectivity_checker = check_room_connectivity_factory(self.spaces_adjacency_matrix)

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
                if len(self.solutions) >= SEARCH_SOLUTIONS_LIMIT:
                    logging.warning("ConstraintSolver: SEARCH_SOLUTIONS_LIMIT: %d",
                                    len(self.solutions))
                    break
                if (time.process_time() - t0 - 600) >= 0:
                    logging.warning("ConstraintSolver: TIME_LIMIT - 10 min")
                    break

        # noinspection PyArgumentList
        self.solver.EndSearch()

        logging.debug("ConstraintSolver: Statistics")
        logging.debug("ConstraintSolver: num_solutions: %d", len(self.solutions))
        logging.debug("ConstraintSolver: failures: %d", self.solver.Failures())
        logging.debug("ConstraintSolver: branches:  %d", self.solver.Branches())
        logging.debug("ConstraintSolver: Process time : %f", time.process_time() - t0)
        if round(time.process_time() - t0) == round(SEARCH_TIME_LIMIT / 1000):
            logging.warning("ConstraintSolver: SEARCH_TIME_LIMIT - 30 min")


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
        self.spaces_item_adjacency_matrix = []
        self._init_spaces_item_adjacency()
        if sp.spec.plan.floor_count < 2:
            self.solver = ConstraintSolver(len(self.sp.spec.items),
                                           self.sp.spec.plan.count_mutable_spaces(),
                                           self.spaces_adjacency_matrix, False)
        else:
            self.solver = ConstraintSolver(len(self.sp.spec.items),
                                           self.sp.spec.plan.count_mutable_spaces(),
                                           self.spaces_adjacency_matrix, True)
        self.item_area = {}
        self._init_item_area()
        self.item_windows_area = {}
        self._init_item_windows_area()
        self.symmetry_breaker_memo = {}
        self.windows_length = {}
        self._init_windows_length()
        self.spaces_max_distance = []
        self.spaces_min_distance = []
        self._init_spaces_distance()
        self.space_graph = nx.Graph()
        self._init_spaces_graph()
        self.area_space_graph = nx.Graph()
        self._init_area_spaces_graph()
        self.centroid_space_graph = nx.Graph()
        self._init_centroid_spaces_graph()
        self.duct_next_to_entrance = []
        self._init_duct_next_to_entrance()
        self.toilet_entrance_proximity_constraint_first_pass = True


        self.item_constraints = {}
        self.add_spaces_constraints()
        self.add_item_constraints()

    def _init_item_area(self) -> None:
        """
        Initialize item area
        :return:
        """
        for item in self.sp.spec.items:
            self.item_area[item.id] = self.solver.solver.Sum(
                self.solver.positions[item.id, j] * round(space.cached_area())
                for j, space in
                enumerate(self.sp.spec.plan.mutable_spaces()))

    def _init_duct_next_to_entrance(self) -> None:
        """
        Initialize duct_next_to_entrance list
        :return:
        """
        min_distance_from_entrance = 400
        frontDoor = [lin for lin in self.sp.spec.plan.linears if lin.category.name == "frontDoor"]
        ducts = [space for space in self.sp.spec.plan.spaces if space.category.name == "duct"]
        for duct in ducts:
            if (frontDoor[0] and
                    duct.distance_to_linear(frontDoor[0], "min") < min_distance_from_entrance):
                self.duct_next_to_entrance.append(duct)

    def _init_item_windows_area(self) -> None:
        """
        Initialize item window area
        :return:
        """
        for item in self.sp.spec.items:
            area = 0
            for j, space in enumerate(self.sp.spec.plan.mutable_spaces()):
                for component in space.immutable_components():
                    if component.category.name == "window":
                        area += (self.solver.positions[item.id, j]
                                 * int(round(component.length * 100)))
                    elif component.category.name == "doorWindow":
                        area += (self.solver.positions[item.id, j]
                                   * int(round(component.length * 200)))
            self.item_windows_area[item.id] = area

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
            self.windows_length[item.id] = length

    def _init_spaces_distance(self) -> None:
        """
        Initialize the spaces distance matrix
        :return:
        """

        for i, i_space in enumerate(self.sp.spec.plan.mutable_spaces()):
            self.spaces_max_distance.append([])
            self.spaces_max_distance[i] = [0] * len(list(self.sp.spec.plan.mutable_spaces()))
            self.spaces_min_distance.append([])
            self.spaces_min_distance[i] = [0] * len(list(self.sp.spec.plan.mutable_spaces()))

        for i, i_space in enumerate(self.sp.spec.plan.mutable_spaces()):
            for j, j_space in enumerate(self.sp.spec.plan.mutable_spaces()):
                if i < j:
                    if i_space.floor != j_space.floor:
                        self.spaces_max_distance[i][j] = 1e20
                        self.spaces_max_distance[j][i] = 1e20
                        self.spaces_min_distance[i][j] = 1e20
                        self.spaces_min_distance[j][i] = 1e20
                    else:
                        self.spaces_max_distance[i][j] = int(i_space.maximum_distance_to(j_space))
                        self.spaces_max_distance[j][i] = int(i_space.maximum_distance_to(j_space))
                        self.spaces_min_distance[i][j] = int(i_space.distance_to(j_space, 'min'))
                        self.spaces_min_distance[j][i] = int(i_space.distance_to(j_space, 'min'))

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
                        self.area_space_graph.add_edge(i, j, weight=j_space.cached_area() +
                                                                    i_space.cached_area())
                        self.area_space_graph.add_edge(j, i, weight=j_space.cached_area() +
                                                                    i_space.cached_area())

    def _init_centroid_spaces_graph(self) -> None:
        """
        Initialize the graph of adjacent seed spaces with weight = centroid distance
        :return:
        """

        for i, i_space in enumerate(self.sp.spec.plan.mutable_spaces()):
            for j, j_space in enumerate(self.sp.spec.plan.mutable_spaces()):
                if i < j:
                    if i_space.adjacent_to(j_space, INSIDE_ADJACENCY_LENGTH):
                        centroid_distance = int(((j_space.centroid()[0] - i_space.centroid()[
                            0]) ** 2 + (j_space.centroid()[1] - i_space.centroid()[1]) ** 2) ** 0.5)
                        self.centroid_space_graph.add_edge(i, j, weight=centroid_distance)
                        self.centroid_space_graph.add_edge(j, i, weight=centroid_distance)

    def _init_spaces_adjacency(self) -> None:
        """
        spaces adjacency matrix init
        :return: None
        """
        self.spaces_adjacency_matrix = [
            [1 if i == j or i_space.adjacent_to(j_space) else 0 for i, i_space in
             enumerate(self.sp.spec.plan.mutable_spaces())] for j, j_space in
            enumerate(self.sp.spec.plan.mutable_spaces())]

    def _init_spaces_item_adjacency(self) -> None:
        """
        spaces adjacency matrix init
        :return: None
        """
        self.spaces_item_adjacency_matrix = [
            [1 if i == j or (i_space.as_sp.buffer(LBW_THICKNESS/2).intersection(
                j_space.as_sp.buffer(LBW_THICKNESS/2)).length/2> ITEM_ADJACENCY_LENGTH and
                             i_space.floor.level == j_space.floor.level) else 0 for i, i_space in
             enumerate(self.sp.spec.plan.mutable_spaces())] for j, j_space in
            enumerate(self.sp.spec.plan.mutable_spaces())]

    def add_spaces_constraints(self) -> None:
        """
        add spaces constraints
        - Each space has to be associated with an item and one time only :
        special case of stairs:
        they must be in a circulating room, otherwise: they are not allocated,
        they are created a circulation
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
            if self.sp.spec.typology <= 2:
                for constraint in T1_T2_ITEMS_CONSTRAINTS.get(item.category.name, []):
                    self.add_item_constraint(item, constraint[0], **constraint[1])
            if self.sp.spec.typology >= 2 and self.sp.spec.number_of_items > 4:
                for constraint in T2_MORE_ITEMS_CONSTRAINTS.get(item.category.name, []):
                    self.add_item_constraint(item, constraint[0], **constraint[1])
            if self.sp.spec.typology >= 3:
                for constraint in T3_MORE_ITEMS_CONSTRAINTS.get(item.category.name, []):
                    self.add_item_constraint(item, constraint[0], **constraint[1])
            if self.sp.spec.typology >= 4:
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
        if (item.variant in ["l", "xl"]
             and item.category.name not in ["living", "livingKitchen", "dining"]):
            max_area = round(item.max_size.area)
        else:
            max_area = round(max(item.max_size.area * MAX_AREA_COEFF, item.max_size.area + 1 * SQM))
        ct = manager.item_area[item.id] <= max_area

    elif min_max == "min":
        if item.variant in ["xs", "s"]:
            min_area = round(item.min_size.area)
        else:
            min_area = round(min(item.min_size.area * MIN_AREA_COEFF, item.min_size.area - 1 * SQM))
        ct = manager.item_area[item.id] >= min_area
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
    if item.category.name in ["living", "dining", "livingKitchen", "wardrobe", "laundry"]:
        param = 2
    elif item.category.name in ["bathroom"]:
        param = 1.9
    elif item.category.name in ["study", "misc", "kitchen"]:
        param = 1.8
    elif item.category.name in ["entrance"]:
        param = 2.5
    else:
        param = 1.8 # toilet, bedroom,

    max_distance = int(round(param * item.required_area ** 0.5))

    ct = None

    for j, j_space in enumerate(manager.sp.spec.plan.mutable_spaces()):
        for k, k_space in enumerate(manager.sp.spec.plan.mutable_spaces()):
            if j < k:
                if ct is None:
                    ct = ((manager.solver.positions[item.id, j] *
                           manager.solver.positions[item.id, k])
                          <= int(max_distance / manager.spaces_max_distance[j][k]))
                else:
                    new_ct = ((manager.solver.positions[item.id, j] *
                               manager.solver.positions[item.id, k])
                              <= int(max_distance / manager.spaces_max_distance[j][k]))
                    ct = manager.and_(ct, new_ct)
    ct = or_no_space_constraint(manager, item, ct)
    return ct

def item_max_distance_constraint(manager: 'ConstraintsManager', item: Item,
                                 item_categories: List[str], max_distance: int) -> ortools.Constraint:
    """
    Maximum distance constraint between item and an other type of item
    :param manager: 'ConstraintsManager'
    :param item: Item
    :param item_categories: List[str]
    :param max_distance: int
    :return: ct: ortools.Constraint
    # TODO : find best param
    # TODO : unit tests
    """
    current_ct = None
    for num, num_item in enumerate(manager.sp.spec.items):
        if num_item.category.name in item_categories and num_item != item:
            for j, j_space in enumerate(manager.sp.spec.plan.mutable_spaces()):
                for k, k_space in enumerate(manager.sp.spec.plan.mutable_spaces()):
                    if j!= k:
                        print(manager.spaces_min_distance[j][k])
                        if current_ct is None:
                            if manager.spaces_min_distance[j][k] == 0:
                                current_ct = (manager.solver.positions[item.id, j] *
                                              manager.solver.positions[num_item.id, k])
                            else:
                                current_ct = ((manager.solver.positions[item.id, j] *
                                              manager.solver.positions[num_item.id, k]) *
                                              ((manager.solver.positions[item.id, j] *
                                              manager.solver.positions[num_item.id, k])
                                             <= int(max_distance / manager.spaces_min_distance[j][k])))
                        else:
                            if manager.spaces_min_distance[j][k] == 0:
                                new_ct = (manager.solver.positions[item.id, j] *
                                               manager.solver.positions[num_item.id, k])
                                current_ct = manager.or_(current_ct, new_ct)
                            else:
                                new_ct = ((manager.solver.positions[item.id, j] *
                                              manager.solver.positions[num_item.id, k]) *
                                             ((manager.solver.positions[item.id, j] *
                                              manager.solver.positions[num_item.id, k])
                                             <= int(max_distance / manager.spaces_min_distance[j][k])))
                                current_ct = manager.or_(current_ct, new_ct)
    return current_ct

def max_distance_window_duct_constraint(manager: 'ConstraintsManager', item: Item,
                                        max_distance: int) -> ortools.Constraint:
    """
    Maximum distance constraint between window and duct constraint
    :param manager: 'ConstraintsManager'
    :param item: Item
    :param max_distance: int
    :return: ct: ortools.Constraint
    """
    additional_distance = 150 # 100 for window --> centroid and 50 for centroid --> duct
    ct = None
    for j, j_space in enumerate(manager.sp.spec.plan.mutable_spaces()):
        for j_space_component in j_space.immutable_components():
            if (type(j_space_component.category) == LinearCategory and
                    j_space_component.category.window_type):
                for k, k_space in enumerate(manager.sp.spec.plan.mutable_spaces()):
                    for k_space_component in k_space.immutable_components():
                        if k_space_component.category.name == "duct":
                            if ct is None:
                                if (j not in nx.nodes(manager.centroid_space_graph)
                                        or k not in nx.nodes(manager.centroid_space_graph)
                                        or not nx.has_path(manager.centroid_space_graph, j, k)):
                                    ct = (manager.solver.positions[item.id, j] *
                                          manager.solver.positions[item.id, k] == 0)
                                else:
                                    path_length, path = nx.single_source_dijkstra(
                                        manager.centroid_space_graph, j, k)
                                    path_length += additional_distance
                                    path_inside_room = 1
                                    for i_path in path:
                                        path_inside_room = (path_inside_room *
                                                        manager.solver.positions[item.id, i_path])
                                    ct = path_inside_room*(manager.solver.positions[item.id, j] *
                                          manager.solver.positions[item.id, k] * path_length
                                          <= max_distance)
                            else:
                                if (j not in nx.nodes(manager.centroid_space_graph)
                                        or k not in nx.nodes(manager.centroid_space_graph)
                                        or not nx.has_path(manager.centroid_space_graph, j, k)):
                                    new_ct = (manager.solver.positions[item.id, j] *
                                          manager.solver.positions[item.id, k] == 0)
                                else:
                                    path_length, path = nx.single_source_dijkstra(
                                        manager.centroid_space_graph, j, k)
                                    path_length += additional_distance
                                    path_inside_room = 1
                                    for i_path in path:
                                        path_inside_room = path_inside_room * manager.solver.positions[item.id, i_path]
                                    new_ct = path_inside_room*(manager.solver.positions[item.id, j] *
                                          manager.solver.positions[item.id, k] * path_length
                                          <= max_distance)
                                ct = manager.or_(ct, new_ct)
    return ct == 1

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
                        area_path = sum(int(space.cached_area())
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
                        area_path = sum(int(space.cached_area())
                                        for i, space in
                                        enumerate(manager.sp.spec.plan.mutable_spaces())
                                        if i in path)
                        new_ct = (manager.solver.positions[item.id, j] *
                                  manager.solver.positions[item.id, k] * area_path
                                  <= max_area)
                    ct = manager.and_(ct, new_ct)
    ct = or_no_space_constraint(manager, item, ct)
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
    ct = or_no_space_constraint(manager, item, ct)
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

    if item.category.name in ["living", "dining", "livingKitchen"]:
        param = min(max(25, plan_ratio + 10), 35)
    elif (item.category.name in ["study", "misc", "kitchen", "entrance", "wardrobe",
                                 "laundry"]
          or (item.category.name in ["bedroom", "bathroom"] and item.variant in ["l", "xl"])):
        param = min(max(25, plan_ratio), 32)
    elif item.category.name in ["bedroom", "bathroom"] and item.variant in ["s", "m"]:
        param = 26
    elif item.category.name in ["bedroom", "bathroom"] and item.variant in ["xs"]:
        param = 22
    else:
        param = 22 # toilet / entrance

    if item.category.name in ["toilet", "bathroom"]:
        cells_perimeter = manager.solver.solver.Sum(manager.solver.positions[item.id, j] *
                                                    int(round(space.perimeter_without_duct))
                                                    for j, space in
                                                    enumerate(
                                                        manager.sp.spec.plan.mutable_spaces()))
    else:
        cells_perimeter = manager.solver.solver.Sum(manager.solver.positions[item.id, j] *
                                                    int(round(space.perimeter))
                                                    for j, space in
                                                    enumerate(
                                                        manager.sp.spec.plan.mutable_spaces()))
    cells_adjacency = manager.solver.solver.Sum(manager.solver.positions[item.id, j] *
                                                manager.solver.positions[item.id, k] *
                                                int(round(j_space.contact_length(k_space)))
                                                for j, j_space in
                                                enumerate(manager.sp.spec.plan.mutable_spaces())
                                                for k, k_space in
                                                enumerate(manager.sp.spec.plan.mutable_spaces())
                                                )
    item_perimeter = cells_perimeter - cells_adjacency
    ct = (item_perimeter * item_perimeter <= int(param) * manager.item_area[item.id])
    ct = or_no_space_constraint(manager, item, ct)
    return ct


def windows_ordering_constraint(manager: 'ConstraintsManager',
                                item: Item) -> Optional[ ortools.Constraint]:
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
                ct = (manager.windows_length[item.id] <=
                      manager.windows_length[j_item.id])
            else:
                new_ct = (manager.windows_length[item.id] <=
                          manager.windows_length[j_item.id])
                ct = manager.solver.solver.Min(ct, new_ct)
        elif (item.category.name in WINDOW_ROOMS and j_item.category.name in WINDOW_ROOMS
              and item.required_area > j_item.required_area):
            if ct is None:
                ct = (manager.windows_length[item.id] >=
                      manager.windows_length[j_item.id])
            else:
                new_ct = (manager.windows_length[item.id] >=
                          manager.windows_length[j_item.id])
                ct = manager.solver.solver.Min(ct, new_ct)

    return ct

def windows_area_constraint(manager: 'ConstraintsManager', item: Item,
                            ratio: int) -> ortools.Constraint:
    """
    Windows area ratio constraint : NF HABITAT HQE
    :param manager: 'ConstraintsManager'
    :param item: Item
    :param ratio : minimum ratio between item area and windows area
    :return: ct: ortools.Constraint
    """
    ct = round(item.required_area * ratio) <= (
                manager.item_windows_area[item.id] * 100)
    return ct

def windows_constraint(manager: 'ConstraintsManager', item: Item) -> ortools.Constraint:
    """
    Windows constraint : windows_area_constraint or windows_ordering_constraint
    :param manager: 'ConstraintsManager'
    :param item: Item
    :return: ct: ortools.Constraint
    """
    ct = None

    if item.category.name in WINDOW_ROOMS:
        if item.category.name in ["living", "livingKitchen", "dining"]:
            ratio = 18
        elif item.category.name in ["bedroom"] and len(item.opens_on) == 0:
            ratio = 15
        elif item.category.name in ["kitchen", "study"] and len(item.opens_on) == 0:
            ratio = 10
        else:
            ratio = 0
            logging.warning("ConstraintsManager - windows_constraint : undefined ratio")

        ct1 = windows_ordering_constraint(manager, item)
        ct2 = windows_area_constraint(manager, item, ratio)
        if ct1 is None:
            ct = None
            logging.debug("ConstraintsManager - No window constraint")
        else:
            ct = manager.or_(ct1, ct2)

    return ct

def toilet_entrance_proximity_constraint(manager: 'ConstraintsManager', item: Item) -> ortools.Constraint:
    """
    warning : symmetry breaker
    :param manager: 'ConstraintsManager'
    :param item: Item
    :return: ct: ortools.Constraint
    """
    ct = None
    toilet_entrance_proximity = 0
    if manager.toilet_entrance_proximity_constraint_first_pass:
        for j, space in enumerate(manager.sp.spec.plan.mutable_spaces()):
            for component in space.immutable_components():
                if component in manager.duct_next_to_entrance:
                    toilet_entrance_proximity += manager.solver.positions[item.id, j]
        if toilet_entrance_proximity:
            ct = toilet_entrance_proximity >= 1

    manager.toilet_entrance_proximity_constraint_first_pass = False
    return ct

def large_windows_constraint(manager: 'ConstraintsManager',
                             item: Item) -> Optional[ortools.Constraint]:
    """
    Large Windows constraint
    :param manager: 'ConstraintsManager'
    :param item: Item
    :return: ct: ortools.Constraint
    """
    ct = None

    large_windows_sum = 0
    for j, space in enumerate(manager.sp.spec.plan.mutable_spaces()):
        for component in space.immutable_components():
            if component.category.name is "doorWindow" and component.length > 180:
                large_windows_sum += manager.solver.positions[item.id, j]
    if large_windows_sum:
        ct = large_windows_sum >= 1

    return ct

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
            if adjacency_sum != 0:
                if ct is None:
                    ct = (adjacency_sum >= length)
                else:
                    ct = manager.and_(ct, (adjacency_sum >= length))
            else:
                logging.warning("ConstraintSolver: opens_on inconsistency")
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
    if item_sym_id in manager.symmetry_breaker_memo:
        memo = 0
        current = 0
        for j in range(manager.solver.spaces_nbr):
            memo = manager.solver.solver.Max(j *
                    manager.solver.positions[manager.symmetry_breaker_memo[item_sym_id], j], memo)
            current = manager.solver.solver.Max(j * manager.solver.positions[item.id, j], current)
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

    ct = or_no_space_constraint(manager, item, ct)

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
        for num, num_item in enumerate(manager.sp.spec.items):
            if num_item.category.name == cat and num_item != item:
                adjacency_sum += manager.solver.solver.Sum(
                    manager.solver.solver.Sum(
                        int(manager.spaces_item_adjacency_matrix[j][k]) *
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
        if space.category.external and space.cached_area() > BIG_EXTERNAL_SPACE:
            has_to_be_connected = True
            break

    if has_to_be_connected:
        adjacency_sum = manager.solver.solver.Sum(
            manager.solver.positions[item.id, j] for j, space in
            enumerate(manager.sp.spec.plan.mutable_spaces())
            if (max([ext_space.cached_area() for ext_space in space.connected_spaces()
                    if ext_space is not None and ext_space.category.external],
                   default=0) > BIG_EXTERNAL_SPACE))
        ct = (adjacency_sum >= 1)
    return ct

def or_no_space_constraint(manager: 'ConstraintsManager', item: Item,
                           ct: Optional[ortools.Constraint]) -> Optional[ortools.Constraint]:
    """
    to apply the given constraint only if the item exists in the solution
    :param manager: 'ConstraintsManager'
    :param item: Item
    :param ct : ortools.Constraint
    :return: ct: ortools.Constraint
    """
    ct0 = (manager.solver.solver.Sum(manager.solver.positions[item.id, j]
                                     for j, space in enumerate(
                                        manager.sp.spec.plan.mutable_spaces())) == 0)
    if ct:
        return manager.or_(ct, ct0)
    else:
        return None

def optional_entrance_constraint(manager: 'ConstraintsManager',
                                    item: Item) -> ortools.Constraint:
    """
    optional entrance constraint
    :param manager: 'ConstraintsManager'
    :param item: Item
    :return: ct: ortools.Constraint
    """
    ct1 = components_adjacency_constraint(manager, item,["frontDoor"], True)
    ct = or_no_space_constraint(manager, item, ct1)

    return ct

def conditional_entrance_constraint(manager: 'ConstraintsManager',
                                    item: Item) -> ortools.Constraint:
    """
    conditional entrance constraint
    :param manager: 'ConstraintsManager'
    :param item: Item
    :return: ct: ortools.Constraint
    """
    ct1 = components_adjacency_constraint(manager, item, ["frontDoor"], True)

    front_door_space = None
    for space in manager.sp.spec.plan.mutable_spaces():
        if "frontDoor" in space.components_category_associated():
            front_door_space = space
            break

    if front_door_space and front_door_space.cached_area() > 5*SQM:
        ct = or_no_space_constraint(manager, item, ct1)
    else:
        ct = ct1

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
        [area_constraint, {"min_max": "max"}]
    ],
    "toilet": [
        [item_attribution_constraint, {}],
        [area_constraint, {"min_max": "min"}],
        [area_constraint, {"min_max": "max"}],
        [components_adjacency_constraint, {"category": ["duct"], "adj": True}],
        [components_adjacency_constraint,
         {"category": WINDOW_CATEGORY, "adj": False, "addition_rule": "And"}],
        [components_adjacency_constraint, {"category": ["startingStep", "frontDoor"], "adj": False,
                                           "addition_rule": "And"}],
        [toilet_entrance_proximity_constraint, {}],
        [item_adjacency_constraint,
         {"item_categories": PRIVATE_ROOMS, "adj": True, "addition_rule": "Or"}],
    ],
    "bathroom": [
        [item_attribution_constraint, {}],
        [area_constraint, {"min_max": "min"}],
        [area_constraint, {"min_max": "max"}],
        [components_adjacency_constraint, {"category": ["duct"], "adj": True}],
        [components_adjacency_constraint, {"category": ["startingStep", "frontDoor"], "adj": False,
                                           "addition_rule": "And"}],
    ],
    "living": [
        [item_attribution_constraint, {}],
        [area_constraint, {"min_max": "min"}],
        [components_adjacency_constraint,
         {"category": WINDOW_CATEGORY, "adj": True, "addition_rule": "Or"}],
        [item_adjacency_constraint,
         {"item_categories": ("kitchen", "dining"), "adj": True, "addition_rule": "Or"}]
    ],
    "livingKitchen": [
        [item_attribution_constraint, {}],
        [area_constraint, {"min_max": "min"}],
        [components_adjacency_constraint,
         {"category": WINDOW_CATEGORY, "adj": True, "addition_rule": "Or"}],
        [item_adjacency_constraint,
         {"item_categories": ("kitchen", "dining"), "adj": True, "addition_rule": "Or"}],
        [max_distance_window_duct_constraint, {"max_distance": 700}]
    ],
    "dining": [
        [item_attribution_constraint, {}],
        [area_constraint, {"min_max": "min"}],
        [opens_on_constraint, {"length": 220}],
        [components_adjacency_constraint,
         {"category": WINDOW_CATEGORY, "adj": True, "addition_rule": "Or"}],
        [components_adjacency_constraint, {"category": ["startingStep", "frontDoor"], "adj": False,
                                           "addition_rule": "And"}],
        [item_adjacency_constraint,
         {"item_categories": ["kitchen", "livingKitchen"], "adj": True, "addition_rule": "Or"}]
    ],
    "kitchen": [
        [item_attribution_constraint, {}],
        [area_constraint, {"min_max": "min"}],
        [area_constraint, {"min_max": "max"}],
        [opens_on_constraint, {"length": 220}],
        [components_adjacency_constraint, {"category": ["duct"], "adj": True}],
        [components_adjacency_constraint, {"category": ["startingStep", "frontDoor"], "adj": False,
                                           "addition_rule": "And"}],
        [item_adjacency_constraint,
         {"item_categories": ("living", "dining"), "adj": True, "addition_rule": "Or"}],
    ],
    "bedroom": [
        [item_attribution_constraint, {}],
        [area_constraint, {"min_max": "min"}],
        [area_constraint, {"min_max": "max"}],
        [opens_on_constraint, {"length": 220}],
        [components_adjacency_constraint, {"category": ["startingStep", "frontDoor"], "adj": False,
                                           "addition_rule": "And"}]
    ],
    "study": [
        [item_attribution_constraint, {}],
        [area_constraint, {"min_max": "min"}],
        [area_constraint, {"min_max": "max"}],
        [opens_on_constraint, {"length": 220}],
        [components_adjacency_constraint, {"category": ["startingStep", "frontDoor"], "adj": False,
                                           "addition_rule": "And"}]
    ],
    "wardrobe": [
        [item_attribution_constraint, {}],
        [area_constraint, {"min_max": "min"}],
        [area_constraint, {"min_max": "max"}],
        [components_adjacency_constraint,
         {"category": WINDOW_CATEGORY, "adj": False, "addition_rule": "And"}],
        [components_adjacency_constraint, {"category": ["startingStep", "frontDoor"], "adj": False,
                                           "addition_rule": "And"}],
        [item_adjacency_constraint,
         {"item_categories": PRIVATE_ROOMS, "adj": True, "addition_rule": "Or"}]
    ],
    "misc": [
        [item_attribution_constraint, {}],
        [area_constraint, {"min_max": "min"}],
        [area_constraint, {"min_max": "max"}],
        [components_adjacency_constraint, {"category": ["startingStep", "frontDoor"], "adj": False,
                                           "addition_rule": "And"}]
    ],
    "laundry": [
        [item_attribution_constraint, {}],
        [area_constraint, {"min_max": "min"}],
        [area_constraint, {"min_max": "max"}],
        [components_adjacency_constraint, {"category": ["duct"], "adj": True}],
        [components_adjacency_constraint,
         {"category": WINDOW_CATEGORY, "adj": False, "addition_rule": "And"}],
        [components_adjacency_constraint, {"category": ["startingStep", "frontDoor"], "adj": False,
                                           "addition_rule": "And"}]

    ]
}

T1_T2_ITEMS_CONSTRAINTS = {
    "entrance": [
        [optional_entrance_constraint,{}],
    ]
}

T2_MORE_ITEMS_CONSTRAINTS = {
    "livingKitchen": [
        [components_adjacency_constraint, {"category": ["duct"], "adj": True}],
    ]
}

T3_MORE_ITEMS_CONSTRAINTS = {
    "entrance": [
        [conditional_entrance_constraint, {}],
    ],
    "toilet": [
        [item_adjacency_constraint, {"item_categories": ["toilet"], "adj": False}]
    ],
    "bathroom": [
        [item_adjacency_constraint,
         {"item_categories": PRIVATE_ROOMS, "adj": True, "addition_rule": "Or"}],
        [item_adjacency_constraint, {"item_categories": ["bathroom"], "adj": False}],
        [item_max_distance_constraint, {"item_categories": ["bedroom"], "max_distance": 200}]
    ],
    "living": [
        [externals_connection_constraint, {}],
        [large_windows_constraint, {}]
    ],
    "livingKitchen": [
        [externals_connection_constraint, {}],
        [large_windows_constraint, {}]
    ],
    "bedroom": [
        [item_max_distance_constraint, {"item_categories": ["bathroom"], "max_distance": 500}]
    ]
}

T4_MORE_ITEMS_CONSTRAINTS = {
    "bathroom": [

    ],
    "bedroom": [

    ]
}
