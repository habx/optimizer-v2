# coding=utf-8
"""
Variables module
Contains Cell and Value classes
"""
import logging
from typing import TYPE_CHECKING, Set

from libs.cpsolver.exception import Conflict

if TYPE_CHECKING:
    from libs.cpsolver.constraint import Constraint
    from libs.cpsolver.solver import DecisionNode
    from libs.cpsolver.solver import Solver


class Value:
    """
    A value that can be taken by a cell
    In our example a value will be a Space
    """
    def __init__(self, ix: int):
        self.ix = ix
        self.constraints: ['Constraint'] = []

    def __repr__(self):
        return "Value n° {}".format(self.ix)

    def bind(self, constraint):
        """
        Binds a constraint to the value
        :param constraint:
        :return:
        """
        self.constraints.append(constraint)

    def propagate(self, node: 'DecisionNode', solver: 'Solver'):
        """
        Propagates the constraints bounded with the provided value
        :param node:
        :param solver:
        :return:
        """

        while True:

            value_changed = False

            for constraint in self.constraints:
                change = constraint.propagate(self, node, solver)
                value_changed = value_changed or change

            if not value_changed:
                break


class Cell:
    """
    A variable
    """
    def __init__(self, domain: Set[int], ix: int):
        self.domain = domain.copy()  # possible value of the cell
        self.ix = ix  # cell index

    def __repr__(self):
        return "Cell {0}: {1}".format(self.ix, self.domain)

    def clone(self) -> 'Cell':
        """
        Returns a cloned Cell
        :return:
        """
        return Cell(self.domain, self.ix)

    def min(self) -> int:
        """
        Returns the min value of the domain
        :return:
        """
        return min(self.domain)

    def max(self) -> int:
        """
        Returns the max value of the domain
        :return:
        """
        return max(self.domain)

    def is_bound(self) -> bool:
        """
        Returns true if domain has only one value left
        :return:
        """
        return len(self.domain) == 1

    def is_empty(self) -> bool:
        """
        Returns True if there are no values possible for the variable
        :return:
        """
        return len(self.domain) == 0

    def has_value_ix(self, value_ix: int) -> bool:
        """
        Returns True if a value is in the domain
        :param value_ix:
        :return:
        """
        return value_ix in self.domain

    def value_ix(self) -> int:
        """
        Returns the unique value of the cell
        :return:
        """
        return next(iter(self.domain))

    def is_adjacent(self, other: 'Cell', matrix) -> bool:
        """
        Returns True if the two cells are adjacent according to the specified adjacency matrix
        :param other:
        :param matrix
        :return:
        """
        return matrix[self.ix][other.ix]

    def prune(self, value: Value, node: 'DecisionNode', solver: 'Solver'):
        """
        Removes the value from the domain
        :param value
        :param node
        :param solver
        :return:
        """
        value_ix = value.ix

        if self.has_value_ix(value_ix):
            logging.debug("Pruning value %i from cell %i", value_ix, self.ix)
            self.domain.remove(value_ix)

            if self.is_empty():
                logging.debug("Conflict! No more value for cell %i!", self.ix)
                raise Conflict("No more value for cell %i", self.ix)

            # we need to propagate the removed value constraints
            value.propagate(node, solver)

            # we also need to propagate the constraint of a newly bound value
            if self.is_bound():
                bound_value = solver.get_value(self.value_ix())
                bound_value.propagate(node, solver)

    def set_value(self, value_ix: int, node: 'DecisionNode', solver: 'Solver'):
        """
        Bounds the cell to the provided value. We do this by pruning all other values.
        :param value_ix:
        :param node:
        :param solver:
        :return:
        """
        logging.debug("Setting value %i for cell %i", value_ix, self.ix)
        found_value = False
        for ix in list(self.domain):
            if ix == value_ix:
                found_value = True
                continue
            value = solver.get_value(ix)
            self.prune(value, node, solver)

        if not found_value:
            raise ValueError("Cannot set a cell to a value not in its domain")
