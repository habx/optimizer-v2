# coding=utf-8
"""
Plot module : used for plotting function for meshes and plans
"""
from random import randint
import os
import datetime
import logging
from typing import TYPE_CHECKING, Optional, List

from shapely.geometry import LineString
from typing import Sequence
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np

from libs.utils.custom_types import Vector2d, Coords2d
from libs.utils.geometry import (
    unit,
    move_point,
    magnitude,
    ccw_angle,
    unit_vector,
    barycenter
)
from output import DEFAULT_PLOTS_OUTPUT_FOLDER

if TYPE_CHECKING:
    from libs.plan.plan import Plan, Floor, Space, Linear

output_path = DEFAULT_PLOTS_OUTPUT_FOLDER

if not os.path.exists(output_path):
    os.makedirs(output_path)


def plot_save(save: bool = True, show: bool = False, name: Optional[str] = None):
    """
    Saves or displays the plot
    :param save:
    :param show:
    :param name:
    :return:
    """

    if not save and not show:
        return

    if save:
        logging.info('Plot: Saving plot')
        ax = plt.gca()
        date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if name is None:
            file_name = ax.get_title().replace(':', '') + '_' + date_time + '.svg'
        else:
            file_name = name + '.svg'

        plt.gcf().tight_layout()
        plt.savefig(os.path.join(output_path, file_name), format='svg')
        plt.close()  # need to close the plot (otherwise matplotlib keeps it in memory)

    if show:
        plt.show()


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
                 should_save: Optional[bool] = None,
                 alpha: Optional[float] = 0.3):
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
        # if alpha != 0.3:
        #     if alpha == 0:
        #         color = 'white'
        #     elif 0 < alpha < 0.2:
        #         color = 'bisque'
        #     elif 0.2 < alpha <= 0.4:
        #         color = 'lightsalmon'
        #     elif 0.4 <= alpha < 0.6:
        #         color = 'orange'
        #     elif 0.6 <= alpha < 0.8:
        #         color = 'darkorange'
        #     elif 0.8 <= alpha < 0.99:
        #         color = 'chocolate'
        #     elif alpha > 0.99:
        #         color = 'sienna'
        #_ax.fill(data1, data2, alpha=max(min(1-alpha, 1),0), color=color)
        # if alpha < 0.8:
        #     color = 'white'
        _ax.fill(data1, data2, alpha=max(min(alpha, 1),0), color=color)
    if 'border' in options:
        ls = ':' if 'dash' in options else 'solid'
        lw = 0.5 if 'dash' in options else 1.5
        if 'dash' in options:
            _ax.plot(data1, data2, 'k', color=color, ls=ls, lw=lw)
        else:
            _ax.plot(data1, data2, 'k', color='k', ls=ls, lw=lw)
    if 'vertices' in options:
        _ax.plot(data1, data2, 'ro', color='white')

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
    vector_unit = unit(vector)

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
    def __init__(self, plan: 'Plan'):
        _fig, _ax = plt.subplots(1, plan.floor_count)
        if plan.floor_count > 1:
            for _sub in _ax:
                _sub.set_aspect('equal')
        else:
            _ax.set_aspect('equal')
        _fig.subplots_adjust(hspace=0.4)  # needed to prevent overlapping of subplots title
        # we must create a lookup dict to assign to each floor a plotting ax
        self._floor_to_ax = {floor.id: i for i, floor in enumerate(plan.floors.values())}
        self.ax = _ax
        self.fig = _fig
        self.space_figs = {}
        self.face_figs = {}

    def get_ax(self, floor: 'Floor'):
        """
        Returns the ax corresponding to the specified floor
        :param floor:
        :return:
        """
        if len(self._floor_to_ax) > 1:
            return self.ax[self._floor_to_ax[floor.id]]
        return self.ax

    def draw(self, plan: 'Plan'):
        """
        Draw a plan
        :param plan:
        :return:
        """
        self._draw_boundary(plan)

        for space in plan.spaces:
            self._draw_space(space)
            for face in space.faces:
                self._draw_face(face, space)

        for linear in plan.linears:
            self._draw_linear(linear)

    def _draw_boundary(self, plan: 'Plan'):
        for floor in plan.floors.values():
            xy = zip(*floor.boundary_as_sp.coords)
            self.get_ax(floor).plot(*xy, 'k', color='k', linewidth=2.0, alpha=1.0,
                                    solid_capstyle='butt')

    def _draw_linear(self, linear: 'Linear'):
        ax = self.get_ax(linear.floor)
        xy = zip(*linear.as_sp.coords)
        ax.plot(*xy, 'k', color=linear.category.color,
                     linewidth=linear.category.width, alpha=0.6,
                     solid_capstyle='butt')

    def _draw_space(self, space: 'Space'):
        """
        NOTE : this does not plot correctly holes in spaces.
        :param space:
        :return:
        """
        if space.edge is None:
            return
        ax = self.get_ax(space.floor)
        color = space.category.color
        xy = space.as_sp.exterior.coords
        new_patch = Polygon(np.array(xy), color=color, alpha=0.3, ls='-', lw=2.0)
        ax.add_patch(new_patch)
        self.space_figs[(space.id, space.floor.id)] = new_patch

    def _draw_face(self, face, space: 'Space'):
        if face.edge is None:
            return
        color = 'r'
        ax = self.get_ax(space.floor)
        xy = face.as_sp.exterior.xy
        new_line, = ax.plot(*xy, color=color, ls=':', lw=0.5)
        self.face_figs[(face.id, space.floor.id)] = new_line

    def update_faces(self, spaces: List['Space']):
        """
        Updates the faces of the spaces
        :param spaces:
        :return:
        """
        if len(spaces) == 0:
            return

        plan: 'Plan' = spaces[0].plan

        for space in spaces:
            for face in space.faces:
                _id = (face.id, space.floor.id)
                xy = face.as_sp.exterior.xy if face.edge is not None else None
                if _id not in self.face_figs:
                    if xy is None:
                        continue
                    self._draw_face(face, space)
                else:
                    if xy is None:
                        self.face_figs[_id].set_visible(False)
                    else:
                        self.face_figs[_id].set_data(np.array(xy))
                        self.face_figs[_id].set_visible(True)

        for face_id, floor_id in self.face_figs:
            floor = plan.get_floor_from_id(floor_id)
            if not floor.mesh.has_face(face_id):
                self.face_figs[(face_id, floor_id)].set_visible(False)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def update(self, spaces):
        """
        Updates the plan with the space data
        :param spaces:
        :return:
        """
        for space in spaces:
            if space is None:
                continue
            _id = space.id, space.floor.id
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

    def draw_seeds_points(self, seeder):
        """
        Draw the seed point on fixed items
        :param seeder:
        :return:
        """
        seed_distance_to_edge = 15  # per convention

        for seed in seeder.seeds:
            if seed.space is None:
                continue
            _ax = self.get_ax(seed.space.floor)
            point = barycenter(seed.edge.start.coords, seed.edge.end.coords, 0.5)
            point = move_point(point, seed.edge.normal, seed_distance_to_edge)
            _ax.plot([point[0]], [point[1]], 'ro', color='r')

    def show(self):
        """
        Show the plot
        :return:
        """
        self.fig.show()
