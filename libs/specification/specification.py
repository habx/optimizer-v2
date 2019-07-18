# coding=utf-8
"""
Specification module
Describes the specifications of the different spaces
"""

from libs.plan.category import SpaceCategory
from libs.plan.plan import Plan
from libs.specification.size import Size
from libs.plan.category import SPACE_CATEGORIES

from typing import List, Optional, Dict
SQM = 10000


class Specification:
    """
    The wishes of the user describing its flan
    """

    def __init__(self, name: str = '', plan: Optional[Plan] = None,
                 items: Optional[List['Item']] = None):
        self.name = name
        self.plan = plan
        self.items = items or []
        self.init_id()

    def __repr__(self):
        output = 'Specification: ' + self.name + '\n'
        for item in self.items:
            output += str(item.id) + ' • ' + item.__repr__() + '\n'

        return output

    def get_item_from_id(self, _id: int):
        """
        Returns
        :param _id:
        :return:
        """
        for item in self.items:
            if item.id == _id:
                return item
        raise ValueError("Specification: Item not found for id %i", _id)

    def init_id(self, category_name_list: Optional[List[str]] = None) -> None:
        """
        Returns the number of rooms from the specification
        :return:
        """
        if category_name_list:
            new_items_list = []
            i = 0
            for name in category_name_list:
                for item in self.items:
                    if item.category.name == name:
                        item.id = i
                        i += 1
                        new_items_list.append(item)
            self.items = new_items_list
        else:
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
            if item.category.name in ['bedroom', 'study']:
                apartment_type += 1
        return apartment_type

    def category_items(self, category_name: str) -> ['Item']:
        """
        Returns the items of the category given
        :return:
        """
        items_list = []
        for item in self.items:
            if item.category.name == category_name:
                items_list.append(item)
        return items_list

    def add_item(self, value: 'Item'):
        """
        Adds a specification item to the specification
        :param value:
        :return:
        """
        value.id = len(self.items)
        self.items.append(value)

    def serialize(self) -> Dict:
        """
        Serialize the specification
        :return:
        """
        output = {"rooms": [i.serialize() for i in self.items],
                  "name": self.name}
        return output

    @classmethod
    def deserialize(cls, data: Dict) -> 'Specification':
        """
        Returns a specification object
        :param data:
        :return:
        """
        specification = cls()
        specification.name = data["name"]
        for item in data["rooms"]:
            new_item = Item.deserialize(item)
            specification.add_item(new_item)
        return specification

    def area_checker(self):
        """
        Returns if plan area and specification area are matching
        :return:
        """
        limit_area_dict = {
            1 : 22*SQM,
            2 : 37*SQM,
            3 : 57*SQM,
            4 : 73*SQM,
            5 : 92*SQM
        }
        if limit_area_dict[self.typology] > sum(space.area for space in self.plan.mutable_spaces()):
            return False
        else:
            return True

class Item:
    """
    The items of the specification
    """

    def __init__(self, category: SpaceCategory,
                 variant: str, min_size: 'Size', max_size: 'Size',
                 opens_on: Optional[List['str']] = None, linked_to: Optional[List['str']] = None,
                 tags: Optional[List['str']] = None):
        self.category = category
        self.variant = variant
        self.min_size = min_size
        self.max_size = max_size
        self.opens_on = opens_on or []
        self.linked_to = linked_to or []
        self.tags = tags or []
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

    def serialize(self) -> Dict:
        """
        Returns the dictionary format to save as json
        format :    {
                      "linkedTo": [],
                      "opensOn": [],
                      "requiredArea": {
                        "max": 100000,
                        "min": 80000
                      },
                      "tags": [],
                      "type": "bedroom",
                      "variant": "xs"
                    },
        :return:
        """
        output = {
            "linkedTo": self.linked_to,
            "opensOn": self.opens_on,
            "requiredArea": {
                "max": self.max_size.area,
                "min": self.min_size.area
            },
            "tags": self.tags,
            "type": self.category.name,
            "variant": self.variant
        }

        return output

    @classmethod
    def deserialize(cls, data: Dict) -> 'Item':
        """
        Creates an Item instance from a dictionary
        :param data:
        :return:
        """
        _category = data["type"]
        if _category not in SPACE_CATEGORIES:
            raise ValueError(
                "Space type not present in space categories: {0}".format(_category))
        required_area = data["requiredArea"]
        size_min = Size(area=required_area["min"])
        size_max = Size(area=required_area["max"])
        variant = data.get("variant", "m")
        opens_on = data.get("opensOn", [])
        linked_to = data.get("linkedTo", [])
        tags = data.get("tags", [])
        item = cls(SPACE_CATEGORIES[_category], variant, size_min, size_max, opens_on,
                   linked_to, tags)

        return item
