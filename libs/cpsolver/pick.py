# coding=utf-8
"""
Branching strategies
"""

from typing import TYPE_CHECKING
from random import randrange


if TYPE_CHECKING:
    from libs.cpsolver.node import DecisionNode
    from libs.cpsolver.solver import Solver

# Cell picking


def random_untried_cell(node: 'DecisionNode', solver: 'Solver') -> int:
    """
    Returns a random cell that has not yet been tried
    :param node:
    :param solver:
    :return:
    """
    max_try = 100
    while True:
        max_try -= 1
        random_ix = randrange(len(list(node.cells)))
        if not solver.has_been_tried(random_ix):
            break
    return random_ix


def farthest_untried_cell(_: 'DecisionNode', solver: 'Solver') -> int:
    """
    Returns the farthest cell from previously tried ones.
    Note: expects the cell indexes to be a continuous increasing list of integers
    ex: [1, 2, 3, 4, 5, 6, 7]
    :param _:
    :param solver:
    :return:
    """
    tried_cells_ix = solver.order_tried()
    max_dist = 0
    previous_ix = tried_cells_ix[0]
    head_ix = None
    tail_ix = None
    for ix in tried_cells_ix[1:]:
        if ix - previous_ix > max_dist:
            head_ix = ix
            tail_ix = previous_ix

    return int((head_ix - tail_ix) / 2) + tail_ix


def first_unbound_cell(node: 'DecisionNode', _: 'Solver') -> int:
    """
    Picks the first adjacent cell
    :param node:
    :param _:
    :return:
    """
    # simple first unbound strategy
    for cell in node.unbounded():
        return cell.ix


def first_unbound_adjacent_cell(node: 'DecisionNode', solver: 'Solver') -> int:
    """
    Picks the first adjacent cell
    :param node:
    :param solver:
    :return:
    """
    # first unbound adjacent to current cell
    if node.cell and solver.adjacency:
        for cell in node.unbounded():
            if cell.is_adjacent(node.cell, solver.adjacency):
                return cell.ix

    # simple first unbound strategy
    return first_unbound_cell(node, solver)


# value picking

def min_value(node: 'DecisionNode', cell_ix: int) -> int:
    """
    Returns the value with the smallest index
    :param node:
    :param cell_ix
    :return:
    """
    # simple strategy choose min value
    cell = node.get_cell(cell_ix)
    value_ix = min(cell.domain)
    return value_ix


def most_bound_value(node: 'DecisionNode', cell_ix: int) -> int:
    """
    Returns the value with the smallest index
    :param node:
    :param cell_ix
    :return:
    """
    # simple strategy choose min value
    cell = node.get_cell(cell_ix)
    bound_max = 0
    value_ix = -1
    for ix in cell.domain:
        bound_count = sum(cell.value_ix() == ix for cell in node.bounded())
        if bound_count > bound_max or value_ix == -1:
            bound_max = bound_count
            value_ix = ix

    return value_ix


def least_bound_value(node: 'DecisionNode', cell_ix: int) -> int:
    """
    Returns the value with the smallest index
    :param node:
    :param cell_ix
    :return:
    """
    # simple strategy choose min value
    cell = node.get_cell(cell_ix)
    bound_min = len(list(node.cells))
    value_ix = -1
    for ix in cell.domain:
        bound_count = sum(cell.value_ix() == ix for cell in node.bounded())
        if bound_count < bound_min or value_ix == -1:
            bound_min = bound_count
            value_ix = ix
            # if we find a totally unbound value not need to search further
            if bound_count == 0:
                break

    return value_ix
