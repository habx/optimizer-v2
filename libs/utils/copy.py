# coding=utf-8
"""
copy module
Contains utility for copying, pickling
"""

import sys
import dill as pickle
import copy

from typing import Any


def get_deep_copy(element: Any) -> Any:
    """
    Returns deepcopy of current element
    :return:
    """
    sys.setrecursionlimit(10000)  # increase default recursion limit
    element_copy = copy.deepcopy(element)
    return element_copy


def load_pickle(path: str) -> Any:
    """
    loads a pickled element
    :return:
    """
    with open(path, 'rb') as pickle_file:
        pickled = pickle.load(pickle_file)

    return pickled
