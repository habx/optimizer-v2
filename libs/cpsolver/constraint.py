# coding=utf-8
"""
Constraints module

Contains the constraint available for the solver :
• AreaConstraint : a space must have an area between min and max
• Component constraint : a space must have certain component
• Connectivity constraint : a space must be connected
• SymmetryBreaker constraint : two spaces that are the same

"""

from typing import TYPE_CHECKING, Dict, Union, Tuple
from itertools import chain

import logging

import networkx as nx

from libs.cpsolver.exception import Conflict

if TYPE_CHECKING:
    from libs.cpsolver.solver import DecisionNode, Solver
    from libs.cpsolver.variables import Value


class Constraint:
    """
    A constraint
    Should implement a propagate method that will prune value of the node cells
    according to the constraint.
    If all possible values have been pruned for a cell, the
    propagate method should raise a conflict Exception.
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
        bounded_sum = sum(solver.get_props(cell.ix, "area")
                          for cell in node.cells_with_value_ix(value_ix, bound=True))

        if bounded_sum > self.max:
            logging.debug("Conflict! Max area unsat !")
            raise Conflict("Max area unsat")

        total_sum = bounded_sum

        for cell in node.cells_with_value_ix(value_ix, bound=False):
            cell_area = solver.get_props(cell.ix, "area")
            total_sum += cell_area
            if bounded_sum + cell_area > self.max:
                cell.prune(value, node, solver)
                domain_changed = True

        # check if they are still cells that will able to augment the min size
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
            ("window", "doorWindow") : 2,
            "duct": 1,
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
    def cell_has_component(ix: int, component: Union[str, Tuple[str]], solver: 'Solver') -> bool:
        """
        Returns True if the cell has a given component
        :param ix:
        :param component:
        :param solver
        :return:
        """
        components = solver.get_props(ix, "components")
        # a tuple as a component means that we can provide either of the specified components
        # ex. : ("doorWindow", "window")
        if isinstance(component, tuple):
            for alternative_component in component:
                if alternative_component in components:
                    return True
            return False
        else:
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
        for cell in node.cells_with_value_ix(value_ix):
            for component_key in possible_components_counter:
                if self.cell_has_component(cell.ix, component_key, solver):
                    possible_components_counter[component_key] += 1

        # compare with the constraint
        for key, min_number in self.components.items():
            # remove from the domain every cells with an undesired component
            if min_number == -1 and possible_components_counter[key] > 0:
                for cell in node.cells_with_value_ix(value_ix):
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
                for cell in node.cells_with_value_ix(value_ix):
                    if not cell.is_bound() and self.cell_has_component(cell.ix, key, solver):
                        cell.set_value(value_ix, node, solver)
                        has_changed = True

        return has_changed


class GraphConstraint(Constraint):
    """
    A specific class of constraint that need the adjacency graph of the cells
    """
    def __init__(self, adjacency_matrix: [[int]]):
        self.matrix = adjacency_matrix
        self.graph = self.create_graph(adjacency_matrix)

    @staticmethod
    def create_graph(matrix) -> nx.Graph:
        """
        Creates a networkx graph from an adjacency matrix
        We expect a symmetric matrix to create an undirected graph
        :param matrix:
        :return:
        """
        graph = nx.Graph()
        for i in range(len(matrix)):
            for j in range(i + 1, len(matrix[0])):
                if matrix[i][j]:
                    graph.add_edge(i, j)

        return graph

    def create_sub_graph(self, value_ix: int, node: 'DecisionNode') -> nx.Graph:
        """
        Creates a subgraph of the graph with the given cells index
        :param value_ix:
        :param node:
        :return:
        """
        indexes = set()
        for cell in node.cells_with_value_ix(value_ix):
                indexes.add(cell.ix)

        return self.graph.subgraph(indexes)

    def propagate(self, value: 'Value', node: 'DecisionNode', solver: 'Solver'):
        """
        Propagates the constraint
        :return:
        """
        raise NotImplementedError("A constraint should implement the propagate method")


class ConnectivityConstraint(GraphConstraint):
    """
    An adjacency constraint
    """
    def propagate(self, value: 'Value', node: 'DecisionNode', solver: 'Solver') -> bool:
        """
        Propagate the constraint after bounding a cell
        :param value:
        :param node:
        :param solver:
        :return: the if a domain was changed
        """
        logging.debug("propagating adjacency constraint for value %i", value.ix)
        subgraph = self.create_sub_graph(value.ix, node)
        number_of_nodes = nx.number_of_nodes(subgraph)
        if not number_of_nodes:
            raise Conflict("Solver: Connectivity constraint unsat")
        has_changed = False

        # if the graph is connected do nothing
        if nx.is_connected(subgraph):
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
        logging.debug("propagating symmetry breaker constraint for value %i", value.ix)

        has_changed = False

        # Find the min cell index of the cells that have the value in their domain
        min_ix = min(map(lambda c: c.ix, node.cells_with_value_ix(value.ix)))

        if min_ix is None:
            raise Exception("This should not happen a value has been completely pruned")

        # remove all the other value from cell
        # with an index inferior or equal to the min bound index
        for cell in node.cells_with_value_ix(self.other.ix):
            if cell.ix <= min_ix:
                cell.prune(self.other, node, solver)
                has_changed = True

        return has_changed


class AdjacencyConstraint(GraphConstraint):
    """
    An adjacency constraint
    Enforces the fact that a space must be adjacent to one or several other spaces
    """
    def __init__(self, adjacency_matrix: [[int]], others: ['Value']):
        super().__init__(adjacency_matrix)
        self.others = others

    def propagate(self, value: 'Value', node: 'DecisionNode', solver: 'Solver') -> bool:
        """
        Propagate the constraint
        :param value:
        :param node:
        :param solver:
        :return:
        """
        has_changed = False
        # create set of cell indices who have the value in their domain
        value_cells = {cell.ix for cell in node.cells if cell.has_value_ix(value.ix)}
        # check each adjacency constraint
        for other in self.others:
            other_cells = {cell.ix for cell in node.cells if cell.has_value_ix(other.ix)}
            # if the two value sets overlaps the constraint can be satisfied
            if value_cells & other_cells:
                continue
            # create the boundary set
            boundary = set(chain.from_iterable(self.graph[v] for v in other_cells)) - other_cells
            # look for an adjacent cell (intersection of value_cells and the boundary)
            adjacent_cells = boundary & value_cells
            if not adjacent_cells:
                logging.debug("Adjacency Constraint unsat, %i, %i", value.ix, other.ix)
                raise Conflict("Adjacency Constraint unsat !")
            # if we only find one boundary cell : we have to set it
            if len(adjacent_cells) == 1:
                (adjacent_cell_ix,) = adjacent_cells
                node.get_cell(adjacent_cell_ix).set_value(value.ix, node, solver)
                has_changed = True

        return has_changed


class MaxSizeConstraint(Constraint):
    """
    A Constraint on the max length between two cells of the same space.
    The goal is to prevent spaces with too thin shapes and to enable the solver
    to prune distant cells more efficiently
    """

    def __init__(self, max_value: float):
        self.max_value = max_value

    def propagate(self, value: 'Value', node: 'DecisionNode', solver: 'Solver'):
        """
        Propagate the constraint
        :param value:
        :param node:
        :param solver:
        :return:
        """
        logging.debug("propagating max size constraint for value %i", value.ix)

        has_changed = False

        bound_cells = node.cells_with_value_ix(value.ix, bound=True)
        domain_cells = node.cells_with_value_ix(value.ix, bound=False)

        for bound_cell in bound_cells:
            for domain_cell in domain_cells:
                if solver.distances[bound_cell.ix][domain_cell.ix] > self.max_value:
                    domain_cell.prune(value, node, solver)
                    has_changed = True

        return has_changed


class MaxPerimeterConstraint(Constraint):
    """
    A constraint on the perimeter of the space
    """
    def __init__(self, max_perimeter: float):
        self.max = max_perimeter

    @staticmethod
    def perimeter(cells, solver: 'Solver') -> float:
        """
        Returns the perimeter of the space
        :param cells
        :param solver:
        :return:
        """
        total_perimeter = sum(map(lambda c: solver.get_props(c.ix, "perimeter"), cells))
        shared_perimeter = sum(solver.adjacency[i.ix][j.ix] for i in cells for j in cells)
        return total_perimeter - shared_perimeter

    def propagate(self, value: 'Value', node: 'DecisionNode', solver: 'Solver'):
        """
        propagate the constraint
        :param value:
        :param node:
        :param solver:
        :return:
        """
        has_changed = False
        # we only check this constraint if only one cell is unbound in the domain
        unbound_cells = list(node.cells_with_value_ix(value.ix, bound=False))
        if len(unbound_cells) > 1:
            return has_changed

        logging.debug("propagating max perimeter constraint for value %i", value.ix)

        cells = list(node.cells_with_value_ix(value.ix, bound=True)) + unbound_cells
        perimeter = self.perimeter(cells, solver)

        if perimeter > self.max:
            if not unbound_cells:
                logging.debug("Perimeter Constraint unsat, %i, %f", value.ix, self.max)
                raise Conflict
            unbound_cells[0].prune(value, node, solver)
            has_changed = True

        return has_changed
