# coding=utf-8
"""
Plot module : used for plotting function for meshes and plans
"""
from random import randint

from shapely.geometry import LineString
from typing import Sequence
from libs.utils.custom_types import Vector2d, Coords2d

from libs.utils.geometry import normalized_vector, move_point, magnitude, ccw_angle, unit_vector


def plot_polygon(_ax, data1: Sequence[float], data2: Sequence[float],
                 options: Sequence[str] = ('vertices', 'border', 'fill'),
                 color: str = 'b'):
    """
    Simple convenience function to plot a mesh face with matplotlib
    :param _ax:
    :param data1:
    :param data2:
    :param options: tuple of strings indicating what to show on plot
    :param color: string, matplotlib color
    :return:
    """
    if 'fill' in options:
        _ax.fill(data1, data2, alpha=0.3, color=color)
    if 'border' in options:
        _ax.plot(data1, data2, 'k', color=color)
    if 'vertices' in options:
        _ax.plot(data1, data2, 'ro', color=color)

    return _ax


def make_arrow(point: Coords2d, vector: Vector2d, normal: Vector2d) -> LineString:
    """
    Returns a polyline of the form o a semi-arrow translated from original edge
    Used to plot half-edge for debugging purposes
    :param point: starting point of the edge
    :param vector:
    :param normal:
    :return: LineString
    """
    distance_to_point = 10
    arrow_angle = 20
    arrow_head_length = 10
    vector_unit = normalized_vector(vector)

    start_point = move_point(point, normal, distance_to_point)
    end_point = move_point(start_point, vector)

    # reduce size of the arrow for better clarity
    if magnitude(vector) > 2 * distance_to_point:
        end_point = move_point(end_point, vector_unit, -1 * distance_to_point)
        start_point = move_point(start_point, vector_unit, distance_to_point)

    relative_angle = ccw_angle(vector) + 180 - arrow_angle
    arrow_head_vector = unit_vector(relative_angle)
    arrow_head_point = move_point(end_point, arrow_head_vector, arrow_head_length)
    arrow = LineString([start_point, end_point, arrow_head_point])

    return arrow


def random_color() -> str:
    """
    Convenient function for matplotlib
    :return: string
    """
    matplotlib_colors = ('b', 'g', 'r', 'c', 'm', 'y', 'k')
    return matplotlib_colors[randint(0, len(matplotlib_colors) - 1)]
