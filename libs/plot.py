# coding=utf-8
"""
Plot module : used for plotting function for meshes and plans
"""
from random import randint
import os
import datetime
import logging
from typing import Optional

from shapely.geometry import LineString
from typing import Sequence
import matplotlib.pyplot as plt

from libs.utils.custom_types import Vector2d, Coords2d
from libs.utils.geometry import (
    normalized_vector,
    move_point,
    magnitude,
    ccw_angle,
    unit_vector
)

SAVE_PATH = "../output/plots"
module_path = os.path.dirname(__file__)
output_path = os.path.join(module_path, SAVE_PATH)

if not os.path.exists(output_path):
    os.makedirs(output_path)


def plot_save(save: bool = True, show: bool = False):
    """
    Saves or displays the plot
    :param save:
    :param show:
    :return:
    """

    if not save and not show:
        return

    if save:
        logging.info('Saving plot')
        file_name = str(datetime.datetime.utcnow()).replace('/', '-') + '.svg'
        plt.savefig(os.path.join(output_path, file_name), format='svg')
        plt.close()  # need to close the plot (otherwise matplotlib keeps it in memory)

    if show:
        plt.show()


def plot_edge(x_coords: Sequence[float],
              y_coords: Sequence[float],
              _ax=None,
              color: str = 'b',
              width: float = 1.0,
              alpha=1,
              save: Optional[bool] = None):

    """
    Plots an edge
    :param _ax:
    :param x_coords:
    :param y_coords:
    :param color:
    :param width:
    :param alpha
    :param save: whether to save the plot
    :return:
    """
    if _ax is None:
        fig, _ax = plt.subplots()
        _ax.set_aspect('equal')
        save = True if save is None else save

    _ax.plot(x_coords, y_coords, 'k',
             linewidth=width,
             color=color,
             alpha=alpha,
             solid_capstyle='butt')

    plot_save(save)

    return _ax


def plot_polygon(_ax,
                 data1: Sequence[float],
                 data2: Sequence[float],
                 options: Sequence[str] = ('vertices', 'border', 'fill'),
                 color: str = 'b',
                 should_save: Optional[bool] = None):
    """
    Simple convenience function to plot a mesh face with matplotlib
    :param _ax:
    :param data1:
    :param data2:
    :param options: tuple of strings indicating what to show on plot
    :param color: string, matplotlib color
    :param should_save: whether to save the plot
    :return:
    """

    if _ax is None:
        fig, _ax = plt.subplots()
        _ax.set_aspect('equal')
        should_save = True if should_save is None else should_save

    if 'fill' in options:
        _ax.fill(data1, data2, alpha=0.3, color=color)
    if 'border' in options:
        _ax.plot(data1, data2, 'k', color=color)
    if 'vertices' in options:
        _ax.plot(data1, data2, 'ro', color=color)

    plot_save(should_save)

    return _ax


def make_arrow(point: Coords2d, vector: Vector2d, normal: Vector2d) -> LineString:
    """
    Returns a polyline of the form o a semi-arrow translated from original edge
    Used to plot half-edge for debugging purposes
    TODO : make a nicer arrow
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
    return '#%02X%02X%02X' % (randint(0, 255), randint(0, 255), randint(0, 255))
