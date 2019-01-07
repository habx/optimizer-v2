# coding=utf-8
"""
Size class
"""
from typing import Optional
import logging

from libs.utils.geometry import pseudo_equal


COORD_EPSILON = 1.0
FLOAT_DECIMAL = 2


class Size:
    """
    A class defining the size of a space
    """
    def __init__(self,
                 area: Optional[float] = None,
                 width: Optional[float] = None,
                 depth: Optional[float] = None,
                 epsilon: float = COORD_EPSILON):
        self.area = round(float(area), FLOAT_DECIMAL) if area else None
        self.width = round(float(width), FLOAT_DECIMAL) if width else None
        self.depth = round(float(depth), FLOAT_DECIMAL) if depth else None
        self.epsilon = epsilon

    def __repr__(self):
        return 'Size: area {0}, width {1}, depth {2}'.format(self.area, self.width, self.depth)

    def is_equal(self, other: 'Size', epsilon: float = COORD_EPSILON):
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

        self_area = self.area or 0
        other_area = other.area or 0
        output += (self_area - other_area)**2

        self_width = self.width or 0
        other_width = other.width or 0
        output += (self_width - other_width)**2

        self_depth = self.depth or 0
        other_depth = other.depth or 0
        output += (self_depth - other_depth)**2

        return output

    def __eq__(self, other):
        return self.is_equal(other)

    def __le__(self, other):
        is_less = True
        if self.area is not None:
            if other.area is not None:
                le_area = self.area <= other.area + other.epsilon**2
                if not le_area:
                    logging.debug('Max area reached : {0} > {1}'.format(self.area, other.area))
                is_less = is_less and le_area
            else:
                return False

        if self.width is not None:
            if other.width is not None:
                le_width = self.width <= other.width + other.epsilon
                if not le_width:
                    logging.debug('Max width reached : {0} > {1}'.format(self.width, other.width))
                is_less = is_less and le_width
            else:
                return False

        if self.depth is not None:
            if other.depth is not None:
                le_depth = self.depth <= other.depth + other.epsilon
                if not le_depth:
                    logging.debug('Max depth reached : {0} > {1}'.format(self.depth, other.depth))
                is_less = is_less and le_depth
            else:
                return False

        return is_less

    def __ge__(self, other):
        is_greater = True
        if self.area is not None:
            if other.area is not None:
                is_greater = is_greater and self.area >= other.area - other.epsilon**2
        else:
            return False

        if self.width is not None:
            if other.width is not None:
                is_greater = is_greater and self.width >= other.width - other.epsilon
        else:
            return False

        if self.depth is not None:
            if other.depth is not None:
                is_greater = is_greater and self.depth >= other.depth - other.epsilon
        else:
            return False

        return is_greater
