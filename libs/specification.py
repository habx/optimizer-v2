# coding=utf-8
"""
Specification module
Describes the specifications of the different spaces
"""

from libs.category import SpaceCategory
from libs.plan import Plan
from libs.size import Size

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
                 variant: str, min_size: 'Size', max_size: 'Size',
                 adjacencies: Optional[List['Item']] = None):
        self.category = category
        self.variant = variant
        self.min_size = min_size
        self.max_size = max_size
        self.adjacencies = adjacencies

    def __repr__(self):
        return 'Item: ' + self.category.name + ' ' + self.variant + ', ' + str(self.min_size) + ', ' + str(
            self.max_size).__repr__()
