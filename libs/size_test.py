# coding=utf-8
"""
Test Size Module
"""

from libs.size import Size


def test_greater_than():
    """
    Test
    :return:
    """
    size_a = Size(100, 10, 10)
    size_b = Size(100, 10, 10)

    assert(size_a <= size_b)


def test_distance():
    """
    Test
    :return:
    """
    size_a = Size(100, 10, 10)
    size_b = Size(100, 12, 12)

    assert(size_a.distance(size_b) == 8.0)
