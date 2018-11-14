# coding=utf-8
"""
Size class
"""
from typing import Optional


class Size:
    """
    The desired size of a space
    """
    def __init__(self,
                 min_area: float,
                 max_area: float,
                 min_width: Optional[float] = None,
                 max_width: Optional[float] = None,
                 min_depth: Optional[float] = None,
                 max_depth: Optional[float] = None):
        self.min_area = float(min_area)
        self.max_area = float(max_area)
        self.min_width = min_width
        self.max_width = max_width
        self.min_depth = min_depth
        self.max_depth = max_depth

    def __repr__(self):
        output = 'Size: area min {0}, max {1}'.format(self.min_area, self.max_area)
        output += 'min_width {0}'.format(self.min_width) if self.min_width is not None else ''
        output += 'max_width {0}'.format(self.max_width) if self.max_width is not None else ''
        output += 'min_depth {0}'.format(self.min_depth) if self.min_depth is not None else ''
        output += 'max_depth {0}'.format(self.max_depth) if self.max_depth is not None else ''
        return output
