# coding=utf-8

"""
Test module for plan module
"""
import json
import logging
import os

from libs.plan import Plan
from libs.category import space_categories
import libs.logsetup as ls
from libs.utils.geometry import (
    barycenter,
    direction_vector,
    normal,
    move_point,
    point_dict_to_tuple
)

ls.init()


def get_perimeter(input_floor_plan_dict):
    """
    Returns a vertices list of the perimeter points of an apartment
    :param input_floor_plan_dict:
    :return:
    """
    apartment = input_floor_plan_dict['apartment']
    perimeter_walls = apartment['externalWalls']
    vertices = apartment['vertices']
    return [(vertices[i]['x'], vertices[i]['y']) for i in perimeter_walls]


def get_fixed_item_perimeter(fixed_item, vertices):
    """calculates the polygon perimeter of the fixed item
    Input dict expected with following attributes
    {
        "type": "doorWindow",
        "vertex2": 1,
        "width": 80,
        "coef1": 286,
        "vertex1": 0,
        "coef2": 540
      },
    """
    width = fixed_item['width']
    vertex_1 = point_dict_to_tuple(vertices[fixed_item['vertex1']])
    vertex_2 = point_dict_to_tuple(vertices[fixed_item['vertex2']])
    coef_1 = fixed_item['coef1']
    coef_2 = fixed_item['coef2']
    point_1 = barycenter(vertex_1, vertex_2, coef_1 / 1000)
    point_2 = barycenter(vertex_1, vertex_2, coef_2 / 1000)

    vector = direction_vector(point_1, point_2)
    normal_vector = normal(vector)

    point_3 = move_point(point_2, normal_vector, width)
    point_4 = move_point(point_1, normal_vector, width)

    return point_1, point_2, point_3, point_4


def get_fixed_items_perimeters(input_floor_plan_dict):
    """
    Returns a list with the perimeter of each fixed items.
    NOTE: we are using the pandas dataframe because we do not want to recalculate
    the absolute geometry of each fixed items. As a general rule,
    it would be much better to read and store the geometry
    of each fixed items as list of vertices instead of the way it's done by using barycentric and
    width data. It would be faster and enable us any fixed item shape.
    :param input_floor_plan_dict:
    :return: list
    """
    apartment = input_floor_plan_dict['apartment']
    vertices = apartment['vertices']
    fixed_items = apartment['fixedItems']
    output = []
    for fixed_item in fixed_items:
        coords = get_fixed_item_perimeter(fixed_item, vertices)
        output.append((coords, fixed_item['type']))

    return output


def get_floor_plans_dicts():
    """
    Test
    :return:
    """
    input_files = [
        "Antony_A22.json",
        "Bussy_A001.json",
        "Bussy_B104.json",
        "Levallois_Parisot.json"
    ]
    input_folder = "../resources/blueprints"
    module_path = os.path.dirname(__file__)
    input_path = os.path.join(module_path, input_folder)

    input_files_path = [os.path.join(input_path, file_path) for file_path in input_files]

    logging.debug("input_files_path = %s", input_files_path)

    # retrieve data from json files
    infos_array = []
    for input_file in input_files_path:
        with open(os.path.abspath(input_file)) as floor_plan_file:
            input_floor_plan_dict = json.load(floor_plan_file)
            infos_array.append(input_floor_plan_dict)

    return infos_array


def test_floor_plan():
    """
    Test
    :return:
    """
    floor_plans_dicts = get_floor_plans_dicts()[3]  # first floor plan
    perimeter = get_perimeter(floor_plans_dicts)

    plan = Plan().from_boundary(perimeter)
    empty_space = plan.empty_space

    fixed_items = get_fixed_items_perimeters(floor_plans_dicts)

    for fixed_item in fixed_items:
        empty_space.insert_space(fixed_item[0], category=space_categories[fixed_item[1]])

    for edge in list(empty_space.edges):
        empty_space.cut_at_barycenter(edge)

    plan.plot()

    assert plan.check()


def test_add_duct_to_space():
    """
    Test
    :return:
    """

    perimeter = [(0, 0), (1000, 0), (1000, 1000), (0, 1000)]
    duct = [(200, 0), (400, 0), (400, 400), (200, 400)]

    duct_category = space_categories['duct']

    # add border duct
    plan = Plan().from_boundary(perimeter)
    plan.empty_space.insert_space(duct, duct_category)

    # add inside duct
    inside_duct = [(600, 200), (800, 200), (800, 400), (600, 400)]
    plan.empty_space.insert_space(inside_duct, duct_category)

    # add touching duct
    touching_duct = [(0, 800), (200, 800), (200, 1000), (0, 1000)]
    plan.empty_space.insert_space(touching_duct, duct_category)

    # add separating duct
    separating_duct = [(700, 800), (1000, 700), (1000, 800), (800, 1000), (700, 1000)]
    plan.empty_space.insert_space(separating_duct, duct_category)

    # add single touching point
    point_duct = [(0, 600), (200, 500), (200, 700)]
    plan.empty_space.insert_space(point_duct, duct_category)

    # add complex duct
    complex_duct = [(300, 1000), (300, 600), (600, 600), (600, 800), (500, 1000),
                    (450, 800), (400, 1000), (350, 1000)]
    plan.empty_space.insert_space(complex_duct, duct_category)

    plan.plot()

    assert plan.check()


def test_add_face():
    """
    Test
    :return:
    """
    perimeter = [(0, 0), (1000, 0), (1000, 1000), (0, 1000)]

    plan = Plan().from_boundary(perimeter)

    # add complex face
    complex_face = [(700, 800), (1000, 700), (1000, 800), (800, 1000), (700, 1000)]
    plan.empty_space.face.insert_face_from_boundary(complex_face)

    face_to_remove = list(plan.empty_space.faces)[1]
    plan.empty_space.remove_face(face_to_remove)

    plan.empty_space.add_face(face_to_remove)

    assert plan.check()
