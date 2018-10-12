# coding=utf-8
"""
Grid Module
We define a grid as a mesh with specific functionalities

Ideas :
- a face should have a type (space, object, reserved_space) and a nature from type:
"""
import math
from libs.mesh import Mesh
import matplotlib.pyplot as plt


class Plan:
    """
    Main class containing the floor plan of the appartement
    • mesh
    • spaces
    • walls
    • objects
    • input
    """


class Transformation:
    """
    Transformation class
    Describes a transformation of a mesh
    A transformation queries the edges of the mesh and modify them as prescribed
    • a query
    • a transformation
    """

    pass


class Factory:
    """
    Grid Factory Class
    Creates a mesh according to transformations, boundaries and fixed-items input
    and specific rules
    """
    pass


class Query:
    """
    Queries a mesh and returns a generator with the required edges
    """


def get_perimeter(infos):
    """
    Returns a vertices list of the perimeter points of an apartment
    :param infos:
    :return:
    """
    apartment = infos.input_floor_plan_dict['apartment']
    perimeter_walls = apartment['externalWalls']
    vertices = apartment['vertices']
    return [(vertices[i]['x'], vertices[i]['y']) for i in perimeter_walls]


def get_fixed_items_perimeters(infos):
    """
    Returns a list with the perimeter of each fixed items.
    NOTE: we are using the dataframe because we do not want to recalculate the absolute geometry
    of each fixed items. As a general rule, it would be much better to read and store the geometry
    of each fixed items as list of vertices instead of the way it's done by using barycentric and
    width data. It would be faster and enable us any fixed item shape.
    :param infos:
    :return: list
    """
    fixed_items_polygons = infos.floor_plan.fixed_items['Polygon']
    output = [polygon.exterior.coords[1:] for polygon in fixed_items_polygons]
    return output

