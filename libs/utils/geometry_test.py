# coding=utf-8
"""
Test module for geometry
"""

import libs.utils.geometry as geometry


def test_rectangle():
    """
    Test the creation of a simple rectangle
    :return:
    """
    point = (0, 0)
    vector = (1, 0)
    width = 100
    height = 50

    rectangle = geometry.rectangle(point, vector, width, height)
    assert rectangle == [(0, 0), (100, 0), (100, 50), (0, 50)]


def test_other_rectangle():
    """
    Test the creation of a simple rectangle
    :return:
    """
    point = (0, 0)
    vector = (1, 1)
    width = 100
    height = 50

    rectangle = geometry.rectangle(point, vector, width, height)
    assert rectangle == [(0, 0),
                         (70.71067811865474, 70.71067811865474),
                         (35.35533905932736, 106.06601717798212),
                         (-35.35533905932738, 35.35533905932738)]


def test_line_intersection():
    """
    Test
    :return:
    """
    assert geometry.lines_intersection(((0, 0), (1, 1)), ((0, 10), (1, 0))) == (10, 10)
    assert geometry.lines_intersection(((10, 5), (0, 2)), ((0, 0), (1, 1))) == (10, 10)
    assert geometry.lines_intersection(((0, 0), (1, 1)), ((10, 0), (-1, 1))) == (5, 5)
    assert geometry.lines_intersection(((0, 0), (-1, -1)), ((10, 0), (-1, 1))) == (5, 5)
    assert geometry.lines_intersection(((0, 0), (1, 1)), ((10, 0), (1, 1))) is None


def test_segment_projection():
    """
    Test
    :return:
    """
    assert geometry.project_point_on_segment((0, 0), (1, 1), ((-10, 10), (10, 10))) == (10, 10)
    assert geometry.project_point_on_segment((10, 5), (0, 2), ((0, 0), (12, 12))) == (10, 10)
    assert geometry.project_point_on_segment((0, 0), (1, 1), ((10, 0), (0, 10))) == (5, 5)
    assert geometry.project_point_on_segment((0, 0), (-1, -1), ((10, 0), (0, 10))) is None
    assert geometry.project_point_on_segment((0, 0), (1, 1), ((10, 0), (10, 9.9))) is None
