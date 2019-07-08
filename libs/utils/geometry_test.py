# coding=utf-8
"""
Test module for geometry
"""

import libs.utils.geometry as geometry
import math


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
    assert geometry.project_point_on_segment((0, 0), (1, 1), ((10, 0), (10, 8.9))) is None
    assert geometry.project_point_on_segment((0, 0), (1, 1),
                                             ((10, 0), (10, 9.5)), epsilon=0.5) == (10, 10)
    assert geometry.project_point_on_segment((0, 0), (1, 1), ((10, 0), (10, 9.5)),
                                             epsilon=0.4) is None


def test_min_depth():
    """
    Test
    :return:
    """
    assert geometry.min_section([(0, 0), (10, 0), (5, 5)]) == 5.0
    assert geometry.min_section([(0, 0), (90, 0), (100, -10), (200, -10), (200, 0),
                                 (400, 0), (400, 80), (100, 80), (70, 500), (0, 500)]) == 70.0
    assert geometry.min_section([(0, 0), (90, 0), (100, -10), (220, -10), (200, 0),
                                 (400, 0), (400, 80), (100, 80), (70, 500), (0, 500)]) == 10.0

    assert geometry.min_section([(0, 0), (100, 0), (100, 100), (60, 100), (50, 20), (40, 100),
                                 (0, 100)]) == 20.0


def test_rotate():
    rect = ((1, 1), (5, 1), (5, 2), (1, 2))
    rotated = ((1, 1), (1, 5), (0, 5), (0, 1))
    computed = geometry.rotate(rect, rect[0], 90)
    for i, point in enumerate(rotated):
        assert math.isclose(point[0], computed[i][0], abs_tol=0.000001)
        assert math.isclose(point[1], computed[i][1], abs_tol=0.000001)
