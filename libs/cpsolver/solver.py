# coding=utf-8
"""
Custom Sat module
to attribute cells to spaces
based on DPLL

The solver attributes a space reference (a value) to each cells.


"""
import copy
import math
import logging

from typing import Optional, Dict, Set

from cpsolver.exception import Success, Conflict, Finished, Restart
import cpsolver.pick as pick
from cpsolver.constraint import (
    AreaConstraint,
    ComponentConstraint,
    ConnectivityConstraint,
    SymmetryBreakerConstraint
)
from cpsolver.variables import Cell, Value
from cpsolver.node import DecisionNode


class Solver:
    """
    The solver
    """
    def __init__(self, adjacency_matrix=None, params=None):
        self.cells: Dict[int, Cell] = {}
        self.values: Dict[int, Value] = {}
        self.adjacency = adjacency_matrix
        self.node: Optional[DecisionNode] = None  # root decision node
        # to store results
        self.params = params or {}

        self.cell_props = {}
        self._solutions: [[Cell]] = []
        self._fails = 0
        self._restart_num = 0
        self._tried_cells: [int] = []

    def print_solutions(self):
        """
        prints the solution in the shape of a square
        :return:
        """
        num_col = int(math.sqrt(len(self.cells)))
        i = 0
        output = "Found solutions:\n"
        for solution in self._solutions:
            output += "Solution n°{}".format(i) + "\n"
            for cell in solution:
                output += "C[{:02d}]:".format(cell.ix) + str(cell.value_ix()) + ("\n" if (cell.ix + 1) % num_col == 0 else " ")
            i += 1

        logging.info(output)
        logging.info("Found %i solutions", i)

    def with_restart(self):
        """
        Returns True if a restart strategy will be applied by the solver
        according to its parameters
        :return:
        """
        return self.params and self.params["num_restarts"] > 0

    def add_value(self, ix: int):
        """
        Adds a value to the solver
        :param ix:
        :return:
        """
        value = Value(ix)
        self.values[ix] = value

    def add_values(self, indexes: Set[int]):
        """
        Adds all value
        :param indexes:
        :return:
        """
        for ix in indexes:
            value = Value(ix)
            self.values[ix] = value

    def get_value(self, ix: int) -> Value:
        """
        Retrieve a value from its index
        :param ix:
        :return:
        """
        return self.values[ix]

    def add_cell(self, domain: [int], props: Dict, ix: int):
        """
        Creates a new cell and adds it to the solver
        """
        cell = Cell(domain, ix)
        self.cells[ix] = cell
        self.cell_props[ix] = props

    def get_props(self, ix: int, key: Optional[str] = None):
        """
        Returns the properties of the cell according to index
        :param ix:
        :param key
        :return:
        """
        if key is None:
            return self.cell_props[ix]
        else:
            return self.cell_props[ix][key]

    def get_cell(self, ix: int) -> 'Cell':
        """
        Retrieves the cell corresponding to the index
        :param ix:
        :return:
        """
        return self.cells[ix]

    def add_area_constraint(self, value_ix: int, min_area: float, max_area: float):
        """
        Adds an area constraint to the solver
        :return: nothing
        """
        value = self.get_value(value_ix)
        constraint = AreaConstraint(min_area, max_area)
        value.bind(constraint)

    def add_component_constraint(self, value_ix: int, components: Dict):
        """
        Adds a component constraint to the solver
        :return: nothing
        """
        value = self.get_value(value_ix)
        constraint = ComponentConstraint(components)
        value.bind(constraint)

    def add_symmetry_breaker_constraint(self, value_ix: int, other_ix: int):
        """
        Adds a symmetry breaker constraint to the solver
        :return: nothing
        """
        value = self.get_value(value_ix)
        other = self.get_value(other_ix)
        constraint = SymmetryBreakerConstraint(other)
        value.bind(constraint)

    def add_connectivity_constraint(self, value_ix: int, matrix: [[bool]]):
        """
        Adds a connectivity constraint to the solver
        :return: nothing
        """
        value = self.get_value(value_ix)
        constraint = ConnectivityConstraint(matrix)
        value.bind(constraint)

    def has_been_tried(self, cell_ix: int) -> bool:
        """
        Returns true if the cell has been tried as a starting point
        :param cell_ix:
        :return:
        """
        return cell_ix in self._tried_cells

    def add_as_tried(self, cell_ix: int):
        """
        Adds the cell index to the tried list
        :param cell_ix:
        :return:
        """
        self._tried_cells.append(cell_ix)

    def order_tried(self):
        """
        returns a ordered list of the tried indexes
        :return:
        """
        output = self._tried_cells[:]
        output.sort()
        return output

    def solve(self) -> [[[int]]]:
        """
        Runs the solver
        :return:
        """
        self.node = DecisionNode(self.cells)  # root node

        max_solutions = self.params.get("num_solutions", 0)
        max_fails = self.params.get("num_fails", 0)
        max_restarts = self.params.get("num_restarts", 0)
        # the number of restart cannot be more than the number of cells minus one
        max_restarts = min(len(self.cells) - 1, max_restarts)

        while True:
            try:
                self.execute_node()
                if len(self._solutions) >= max_solutions > 0:
                    raise Finished
                if self._fails >= max_fails > 0:
                    raise Restart
                if self._restart_num >= max_restarts > 0:
                    raise Finished

            except Finished:
                break

            except Restart:
                self.restart()

        logging.info("Finished !")
        logging.info("# of fails: {}".format(self._fails))
        logging.info("# of restarts: {}".format(self._restart_num))
        logging.info("Tried cells: {}".format(self._tried_cells))
        self.print_solutions()

        return self._solutions

    def restart(self):
        """
        Restart the solver
        :return:
        """
        logging.info("RESTARTING : {} - Fails {} - # Solutions {}"
                     .format(self._tried_cells, self._fails, len(self._solutions)))

        self._restart_num += 1
        self._fails = 0
        self.node = DecisionNode(self.cells)

    def execute_node(self):
        """
        Runs the solver
        :return:
        """
        try:
            if self.node.is_completely_bound():
                logging.debug("Node completely bound %s", self.node)
                raise Success

            self.branch()

        except Conflict:
            self._fails += 1
            self.node = self.previous(self.node)

        except Success:
            self._fails += 1
            self._solutions.append(copy.deepcopy(list(self.node.cells)))
            self.node = self.previous(self.node)

    def branch(self):
        """
        Creates a new decision node
        :return:
        """
        node = self.node

        # pick a cell and a value
        cell_ix = self.pick_cell(node)
        value_ix = self.pick_value(node, cell_ix)

        logging.debug("BRANCHING: Picking value %i for cell %i", value_ix, cell_ix)
        self.node = node.child(cell_ix, value_ix)

        # set value to pick cell
        self.node.cell.set_value(value_ix, self.node, self)

    def pick_cell(self, node: 'DecisionNode') -> int:
        """
        chose the cell to bound
        :param node
        :return:
        """
        # Initial strategy
        if node.is_root():
            initial_cell_ix = pick.random_untried_cell(node, self)
            self.add_as_tried(initial_cell_ix)
            return initial_cell_ix

        ix = pick.first_unbound_adjacent_cell(node, self)
        return ix

    @staticmethod
    def pick_value(node: 'DecisionNode', cell_ix: int) -> int:
        """
        Chose the value to assign
        :param node
        :param cell_ix
        :return:
        """
        # simple strategy : choose most bound value
        ix = pick.most_bound_value(node, cell_ix)
        return ix

    def previous(self, node: 'DecisionNode') -> 'DecisionNode':
        """
        Backtracks the decision tree
        :param node
        :return:
        """
        logging.debug(" ^ PREVIOUS ^ refuting value %i for cell %i",
                      node.value_ix if node.value_ix else -1,
                      node.cell.ix if node.cell else -1)

        if node.is_root():
            raise Finished

        if node.parent.is_root() and self.with_restart():
            raise Restart

        try:
            # prune the value that has been refuted
            value = self.get_value(node.value_ix)
            node.parent.get_cell(node.cell.ix).prune(value, node.parent, self)
        except Success:
            return self.previous(node.parent)
        except Conflict:
            return self.previous(node.parent)

        return node.parent
