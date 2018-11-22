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
from matplotlib.patches import Polygon
import numpy as np

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
        ax = plt.gca()
        date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_name = ax.get_title() + '_' + date_time + '.svg'
        plt.savefig(os.path.join(output_path, file_name), format='svg')
        plt.close()  # need to close the plot (otherwise matplotlib keeps it in memory)

    if show:
        plt.show()


def plot_point(x_coords: Sequence[float],
               y_coords: Sequence[float],
               _ax=None,
               color: str = 'r',
               save: Optional[bool] = None):
    """
    Plots a point
    :param x_coords:
    :param y_coords:
    :param _ax:
    :param color:
    :param save
    :return:
    """

    if _ax is None:
        fig, _ax = plt.subplots()
        _ax.set_aspect('equal')
        save = True if save is None else save

    _ax.plot(x_coords, y_coords, 'ro', color=color)

    plot_save(save)

    return _ax


def plot_edge(x_coords: Sequence[float],
              y_coords: Sequence[float],
              _ax=None,
              color: str = 'b',
              width: float = 1.0,
              alpha: float = 1,
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
        ls = ':' if 'dash' in options else 'solid'
        lw = 0.5 if 'dash' in options else 1.0
        _ax.plot(data1, data2, 'k', color=color, ls=ls, lw=lw)
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


class Plot:
    """
    Plot class
    """
    def __init__(self):
        _fig = plt.figure()
        _ax = _fig.add_subplot(111)
        _ax.set_aspect('equal')
        self.ax = _ax
        self.fig = _fig
        self.space_figs = {}
        self.face_figs = {}

    def draw(self, plan):
        """
        Draw a plan
        :param plan:
        :return:
        """
        self._draw_boundary(plan)

        for space in plan.spaces:
            self._draw_space(space)
            for face in space.faces:
                self._draw_face(face)

        for linear in plan.linears:
            self._draw_linear(linear)

    def _draw_boundary(self, plan):
        xy = zip(*plan.boundary_as_sp.coords)
        self.ax.plot(*xy, 'k', color='k',
                     linewidth=2.0, alpha=1.0,
                     solid_capstyle='butt')

    def _draw_linear(self, linear):
        xy = zip(*linear.as_sp.coords)
        self.ax.plot(*xy, 'k', color=linear.category.color,
                     linewidth=linear.category.width, alpha=0.6,
                     solid_capstyle='butt')

    def _draw_space(self, space):
        color = space.category.color
        xy = space.as_sp.exterior.coords
        new_patch = Polygon(np.array(xy), color=color, alpha=0.3, ls='-', lw=2.0)
        self.ax.add_patch(new_patch)
        self.space_figs[id(space)] = new_patch

    def _draw_face(self, face):
        color = face.space.category.color if face.space else 'r'
        xy = face.as_sp.exterior.xy
        new_line, = self.ax.plot(*xy, color=color, ls=':', lw=0.5)
        self.face_figs[id(face)] = new_line

    def update(self, spaces):
        """
        Updates the plan with the space data
        :param spaces:
        :return:
        """
        for space in spaces:
            _id = id(space)
            xy = space.as_sp.exterior.coords if space.edge is not None else None
            if _id not in self.space_figs:
                if xy is None:
                    continue
                self._draw_space(space)
            else:
                if xy is None:
                    self.space_figs[_id].set_visible(False)
                else:
                    self.space_figs[_id].set_xy(np.array(xy))
                    self.space_figs[_id].set_visible(True)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def show(self):
        """
        Show the plot
        :return:
        """
        self.fig.show()
