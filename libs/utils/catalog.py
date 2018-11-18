# coding=utf-8
"""
Catalog class
A catalog is a container for instances of different classes.
A class instance can be stored in a catalog if the class has a name property
A catalog can be used to store other catalogs
"""
from typing import Any, Optional
import logging


class Catalog:
    """
    Catalog class
    """
    def __init__(self, name: str):
        self.name = name
        self._items = {}
        self._factories = {}

    def __repr__(self):
        output = 'Catalog: {0}'.format(self.name)
        for key in self._items:
            output += '\n • {0}'.format(key)
        return output

    def add(self, *items: Any) -> 'Catalog':
        """
        Adds one or multiple items to the catalog
        :param items:
        :return:
        """
        for item in items:
            if not hasattr(item, 'name'):
                raise ValueError('An item must have a name attribute in order to be stored' +
                                 ' in a catalog: {0}'.format(item))

            if item.name in self._items:
                logging.info('Item already in catalog')
                return self

            self._items[item.name] = item
        return self

    @property
    def factory(self):
        """
        Returns the factories of the catalog
        :return:
        """
        return self._factories

    def add_factory(self, *factories: Any) -> 'Catalog':
        """
        Adds a factory item
        :param value:
        :return:
        """
        for item in factories:
            if not hasattr(item, 'name'):
                raise ValueError('A factory must have a name attribute in order to be stored' +
                                 ' in a catalog: {0}'.format(item))

            if item.name in self._factories:
                logging.info('Factory already in catalog')
                return self

            self._factories[item.name] = item
        return self

    def __call__(self, item_name: str, default: Optional[Any] = None) -> Any:
        """
        Retrieves an item from the catalog
        :param item_name:
        :param default: a default item to return if an item is not in the catalog
        :return:
        """
        if item_name not in self._items:
            return default
        return self._items[item_name]

    def __getitem__(self, item_name):
        """
        Returns an item of the catalog as a subscriptable
        :param item_name:
        :return:
        """
        if item_name not in self._items:
            return None
        return self._items[item_name]

    def __contains__(self, key):
        return key in self._items
