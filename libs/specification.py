# coding=utf-8
"""
Specification module
Describes the specifications of the different spaces
"""

from libs.category import SpaceCategory
from libs.plan import Plan

from typing import List, Optional


class Specification:
    """
    The wishes of the user describing its flan
    """
    def __init__(self, name: str = '', plan: Plan = None, items: Optional[List['Item']] = None):
        self.name = name
        self.plan = plan
        self.items = items or []

    def __repr__(self):
        output = 'Specification: ' + self.name + '\n'
        i = 1
        for item in self.items:
            output += str(i) + ' • ' + item.__repr__() + '\n'
            i += 1
        return output

    def add_item(self, value):
        """
        Adds a specification item to the specification
        :param value:
        :return:
        """
        self.items.append(value)


class Item:
    """
    The items of the specification
    """
    def __init__(self, category: SpaceCategory,
                 variant: str, size: 'Size',
                 adjacencies: Optional[List['Item']] = None):
        self.category = category
        self.variant = variant
        self.size = size
        self.adjacencies = adjacencies

    def __repr__(self):
        return 'Item: ' + self.category.name + ' ' + self.variant + ', ' + self.size.__repr__()


class Size:
    """
    The desired size of the specification item
    """
    def __init__(self, min_area: float, max_area: float):
        self.min_area = float(min_area)
        self.max_area = float(max_area)

    def __repr__(self):
        return 'Size: min {0} - max {1}'.format(self.min_area, self.max_area)
