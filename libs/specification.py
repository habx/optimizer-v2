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

    def __init__(self, name: str = '', plan: Optional[Plan] = None,
                 items: Optional[List['Item']] = None):
        self.name = name
        self.plan = plan
        self.items = items or []

    def __repr__(self):
        output = 'Specification: ' + self.name + '\n'
        for item in self.items:
            output += str(item.id) + ' • ' + item.__repr__() + '\n'

        return output

    def init_id(self) -> None:
        """
        Returns the number of rooms from the specification
        :return:
        """
        i = 0
        for item in self.items:
            item.id = i
            i += 1

    @property
    def number_of_items(self):
        """
        Returns the number of rooms from the specification
        :return:
        """
        return len(self.items)

    @property
    def typology(self):
        """
        Returns the typology of the specification
        :return:
        """
        apartment_type = 1
        for item in self.items:

            if item.category.name in ['bedroom', 'office']:

                apartment_type += 1
        return apartment_type

    def add_item(self, value: 'Item'):
        """
        Adds a specification item to the specification
        :param value:
        :return:
        """
        value.id = len(self.items)
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
        self.id = 0

    def __repr__(self):
        return 'Item: ' + self.category.name + ' ' + self.variant + ', Area : ' + \
               str(self.required_area)

    @property
    def required_area(self) -> float:
        """
        Returns the required size of the item
        :return:
        """
        return (self.min_size.area + self.max_size.area)/2
