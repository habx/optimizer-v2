# coding=utf-8
"""
Size class
"""
from typing import Optional

from libs.utils.geometry import pseudo_equal


EPSILON_SIZE = 1.0


class Size:
    """
    A class defining the size of a space
    """
    def __init__(self,
                 area: Optional[float] = None,
                 width: Optional[float] = None,
                 depth: Optional[float] = None):
        self.area = float(area) if area else None
        self.width = float(width) if width else None
        self.depth = float(depth) if depth else None

    def __repr__(self):
        return 'Size: area {0}, width {1}, depth {2}'.format(self.area, self.width, self.depth)

    def is_equal(self, other: 'Size', epsilon: float = EPSILON_SIZE):
        """
        Returns True if the size are equal
        :param other:
        :param epsilon:
        :return:
        """
        _is_equal = True
        if self.area is not None:
            if other.area is None:
                return False
            _is_equal = _is_equal and pseudo_equal(self.area, other.area, epsilon**2)

        if self.width is not None:
            if self.width is None:
                return False
            _is_equal = _is_equal and pseudo_equal(self.width, other.width, epsilon)

        if self.depth is not None:
            if other.depth is None:
                return False
            _is_equal = _is_equal and pseudo_equal(self.depth, other.depth, epsilon)

        return _is_equal

    def distance(self, other: 'Size') -> float:
        """
        Computes the distance between the two sizes
        :param other:
        :return:
        """
        output = 0
        if self.area is not None:
            if other.area is not None:
                output += (self.area - other.area)**2

        if self.width is not None:
            if other.width is not None:
                output += (self.width - other.width)**2

        if self.depth is not None:
            if other.depth is not None:
                output += (self.depth - other.depth)**2

        return output

    def __eq__(self, other):
        return self.is_equal(other)

    def __le__(self, other):
        is_less = True
        if self.area is not None:
            if other.area is not None:
                is_less = is_less and self.area <= other.area

        if self.width is not None:
            if other.width is not None:
                is_less = is_less and self.width <= other.width

        if self.depth is not None:
            if other.depth is not None:
                is_less = is_less and self.depth <= other.depth

        return is_less

    def __ge__(self, other):
        is_greater = True
        if self.area is not None:
            if other.area is not None:
                is_greater = is_greater and self.area >= other.area

        if self.width is not None:
            if other.width is not None:
                is_greater = is_greater and self.width >= other.width

        if self.depth is not None:
            if other.depth is not None:
                is_greater = is_greater and self.depth >= other.depth

        return is_greater

