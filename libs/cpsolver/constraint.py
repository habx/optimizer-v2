# coding=utf-8
"""
Constraints module

Contains the constraint available for the solver :
• AreaConstraint : a space must have an area between min and max
• Component constraint : a space must have certain component
• Connectivity constraint : a space must be connected
• SymmetryBreaker constraint : two spaces that are the same

"""

from typing import TYPE_CHECKING, Dict
import logging

import networkx as nx

from libs.cpsolver.exception import Conflict

if TYPE_CHECKING:
    from libs.cpsolver.solver import DecisionNode, Solver
    from libs.cpsolver.variables import Cell, Value


class Constraint:
    """
    A constraint
    """
    def propagate(self, value: 'Value', node: 'DecisionNode', solver: 'Solver'):
        """
        Propagates the constraint
        :return:
        """
        raise NotImplementedError("A constraint should implement the propagate method")

    def __repr__(self):
        return self.__class__.__name__


class AreaConstraint(Constraint):
    """
    A sum constraint
    """
    def __init__(self, min_sum: float, max_sum):
        self.min = min_sum
        self.max = max_sum
        self.conflict = False

    def propagate(self, value: 'Value', node: 'DecisionNode', solver: 'Solver') -> bool:
        """
        propagate the constraint after bounding a cell
        :param value
        :param node
        :param solver
        :return:
        """
        logging.debug("propagating area constraint for value %i", value.ix)
        domain_changed = False
        value_ix = value.ix

        # remove value_ix from cell that are too big
        bounded_sum = sum(solver.get_props(cell.ix, "area") for cell in node.has_value_ix(value_ix))

        if bounded_sum > self.max:
            logging.debug("Conflict! Max area unsat !")
            raise Conflict("Max area unsat")

        for cell in node.has_in_domain(value_ix):
            if not cell.is_bound() and bounded_sum + solver.get_props(cell.ix, "area") > self.max:
                cell.prune(value, node, solver)
                domain_changed = True

        # check if they are still cells that will able to augment the min size
        total_sum = sum(solver.get_props(cell.ix, "area")
                        for cell in node.has_in_domain(value_ix))

        if total_sum < self.min:
            logging.debug("Conflict! Min area unsat !")
            raise Conflict("Min area unsat", value_ix)

        return domain_changed


class ComponentConstraint(Constraint):
    """
    A component constraint
    Enforces the fact that a space must include a cell with a specific component
    Expect a components dict of the form :
        {
            "window" : 2,
            "duct": 1
            ...
        }
    """
    def __init__(self, components: Dict):
        if components == {}:
            raise ValueError("You must provide a non empty components "
                             "dictionary to create a proper component constraint")
        self.components = components

    def create_components_counter(self) -> Dict:
        """
        Returns a component counter dict
        :return:
        """
        return {key: 0 for key in self.components}

    @staticmethod
    def cell_has_component(ix: int, component: str, solver: 'Solver') -> bool:
        """
        Returns True if the cell has a given component
        :param ix:
        :param component:
        :param solver
        :return:
        """
        components = solver.get_props(ix, "components")
        return component in components

    def propagate(self, value: 'Value', node: 'DecisionNode', solver: 'Solver') -> bool:
        """
        Propagate the constraints along the domain
        We cannot prune anything really as the constraint in the minimum not a maximum.
        We can only raise a conflict when no more component are available.
        :param value
        :param node:
        :param solver:
        :return:
        """
        logging.debug("propagating component constraint for value %i", value.ix)

        value_ix = value.ix
        has_changed = False
        # count possible components
        possible_components_counter = self.create_components_counter()
        for cell in node.has_in_domain(value_ix):
            for component_key in possible_components_counter:
                if component_key in solver.get_props(cell.ix, "components"):
                    possible_components_counter[component_key] += 1

        # compare with the constraint
        for key, min_number in self.components.items():
            # remove from the domain every cells with an undesired component
            if min_number == -1 and possible_components_counter[key] > 0:
                for cell in node.has_in_domain(value_ix):
                    if self.cell_has_component(cell.ix, key, solver):
                        cell.prune(value, node, solver)
                        has_changed = True
            # check if enough components remain in domain
            elif possible_components_counter[key] < min_number:
                logging.debug("CONFLICT : Component constraint unsat for value %i", value_ix)
                raise Conflict("Component constraint unsat for value %i", value_ix)
            # assign the value to every cell if the domain has exactly the number
            # of needed components
            elif possible_components_counter[key] == min_number:
                for cell in node.has_in_domain(value_ix):
                    if not cell.is_bound() and self.cell_has_component(cell.ix, key, solver):
                        cell.set_value(value_ix, node, solver)
                        has_changed = True

        return has_changed


class ConnectivityConstraint(Constraint):
    """
    An adjacency constraint
    """
    def __init__(self, adjacency_matrix):
        self.matrix = adjacency_matrix
        self.graph = self.create_graph(adjacency_matrix)

    @staticmethod
    def create_graph(matrix) -> nx.Graph:
        """
        Creates a networkx graph from an adjacency matrix
        We expect a symetric matrix to create an undirected graph
        :param matrix:
        :return:
        """
        graph = nx.Graph()
        for i in range(len(matrix)):
            for j in range(i+1, len(matrix[0])):
                if matrix[i][j]:
                    graph.add_edge(i, j)

        return graph

    def create_sub_graph(self, value_ix: int, cells: ['Cell']) -> nx.Graph:
        """
        Creates a subgraph of the graph with the given cells index
        :param value_ix:
        :param cells:
        :return:
        """
        indexes = set()
        for cell in cells:
            if cell.has_value_ix(value_ix):
                indexes.add(cell.ix)

        return self.graph.subgraph(indexes)

    def propagate(self, value: 'Value', node: 'DecisionNode', solver: 'Solver') -> bool:
        """
        Propagate the constraint after bounding a cell
        :param value:
        :param node:
        :param solver:
        :return: the if a domain was changed
        """
        logging.debug("propagating adjacency constraint for value %i", value.ix)
        subgraph = self.create_sub_graph(value.ix, node.cells)
        has_changed = False

        # if the graph is connected do nothing
        if nx.is_connected(subgraph):
            logging.debug("Subgraph connected for value %i", value.ix)
            return has_changed

        # find first bound variables
        for cell in node.cells:
            if cell.is_bound() and cell.has_value_ix(value.ix):
                bound_cell_ix = cell.ix
                break
        else:
            logging.debug("Subgraph with no bound cell for value %i", value.ix)
            return has_changed

        connected_nodes = nx.node_connected_component(subgraph, bound_cell_ix)

        for cell in node.cells:
            if cell.ix not in connected_nodes:
                cell.prune(value, node, solver)
                has_changed = True

        return has_changed


class SymmetryBreakerConstraint(Constraint):
    """
    A Symmetry Breaker constraint
    Enforces the fact that two spaces are similar and should not be swapped.
    An artificial order is created between the two spaces according to the min. cell index.
    Per convention the first value should have a lower cell index than the other value.
    """
    def __init__(self, other: 'Value'):
        self.other = other

    def propagate(self, value: 'Value', node: 'DecisionNode', solver: 'Solver') -> bool:
        """
        Propagate the constraints along the domain
        :param value:
        :param node:
        :param solver:
        :return:
        """
        has_changed = False

        # Find the min cell index of the cells that have the value in their domain
        min_ix = None
        for cell in node.has_in_domain(value.ix):
            if min_ix is None or cell.ix < min_ix:
                min_ix = cell.ix

        if min_ix is None:
            raise Exception("This should not happen a value has been completely pruned")

        # remove all the other value from cell
        # with an index inferior or equal to the min bound index
        for cell in node.has_in_domain(self.other.ix):
            if cell.ix <= min_ix:
                cell.prune(self.other, node, solver)
                has_changed = True

        return has_changed
