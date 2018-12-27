# coding=utf-8
"""
Node module
"""
import math
from typing import TYPE_CHECKING, Optional, Generator, Dict

if TYPE_CHECKING:
    from libs.cpsolver.variables import Cell


class DecisionNode:
    """
    A node of the decision tree
    """

    def __init__(self,
                 cells: Dict[int, 'Cell'],
                 cell_ix: Optional[int] = None,
                 value_ix: Optional[int] = None,
                 parent: Optional['DecisionNode'] = None):

        self._cells = {ix: cell.clone() for ix, cell in cells.items()}
        self.cell: Optional['Cell'] = self._cells.get(cell_ix, None)
        self.value_ix = value_ix
        self.parent = parent

    def __repr__(self):
        """
        Tries to print the node domain has a square
        """
        node = self
        chain = ""
        while node.cell and node.parent:
            chain += "C{}:{} ".format(node.cell.ix, node.value_ix)
            node = node.parent
        output = "Node:" + chain + "\n"
        num_col = int(math.sqrt(len(list(self.cells))))
        for cell in self.cells:
            output += (str(cell.domain) +
                       ("\n" if (cell.ix + 1) % num_col == 0
                        else " " + (" "*(9 - len(str(cell.domain))))))
        return output

    @property
    def cells(self)-> Generator['Cell', None, None]:
        """
        Property
        :return:
        """
        for _, cell in self._cells.items():
            yield cell

    def child(self, cell_ix: int, value_ix: int) -> 'DecisionNode':
        """
        Returns a child node
        :param cell_ix:
        :param value_ix:
        :return:
        """
        return DecisionNode(self._cells, cell_ix, value_ix, self)

    def is_root(self) -> bool:
        """
        Returns true if we found the root node
        :return:
        """
        return self.parent is None

    def bounded(self) -> Generator['Cell', None, None]:
        """
        yield the bounded cells
        :return:
        """
        for cell in self.cells:
            if cell.is_bound():
                yield cell

    def is_completely_bound(self):
        """
        Returns True if the node has all its cells bound
        :return:
        """
        return not len(list(self.unbounded()))

    def unbounded(self) -> Generator['Cell', None, None]:
        """
        yield the unbounded cells
        :return:
        """
        for cell in self.cells:
            if not cell.is_bound():
                yield cell

    def has_in_domain(self, value_ix: int) -> Generator['Cell', None, None]:
        """
        Yields cells that can take the value
        :param value_ix:
        :return:
        """
        for cell in self.cells:
            if cell.has_value_ix(value_ix):
                yield cell

    def has_value_ix(self, value: int) -> Generator['Cell', None, None]:
        """
        Yield the cells bounded to the value
        :param value:
        :return:
        """
        for cell in self.cells:
            if cell.is_bound() and cell.value_ix() == value:
                yield cell

    def get_cell(self, ix: int) -> 'Cell':
        """
        gets a cell from its index
        :param ix:
        :return:
        """
        return self._cells[ix]
