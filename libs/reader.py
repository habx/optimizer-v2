# coding=utf-8
"""
Reader module : Used to read file from json input and create a plan.
"""
from typing import Dict, Sequence, Tuple, List
import os
import json
import logging
from libs import plan
from libs.category import SPACE_CATEGORIES, LINEAR_CATEGORIES
from libs.specification import Specification, Item, Size
from libs.plan import Plan

from libs.utils.geometry import (
    point_dict_to_tuple,
    barycenter,
    direction_vector,
    normal_vector,
    move_point
)
from libs.utils.custom_types import Coords2d, FourCoords2d, ListCoords2d
from libs.writer import DEFAULT_PLANS_OUTPUT_FOLDER, DEFAULT_MESHES_OUTPUT_FOLDER

LOAD_BEARING_WALL_WIDTH = 15.0
DEFAULT_BLUEPRINT_INPUT_FOLDER = "../resources/blueprints"
DEFAULT_SPECIFICATION_INPUT_FOLDER = "../resources/specifications"


def get_list_from_folder(path: str = DEFAULT_BLUEPRINT_INPUT_FOLDER):
    """
    Returns a list containing names of all files contained in specified folder - for tests purposes
    :param path
    :return:
    """
    list_files = []
    for filename in os.listdir(path):
        list_files.append(filename)

    return list_files


def _get_perimeter(input_blueprint_dict: Dict) -> Sequence[Coords2d]:
    """
    Returns a vertices list of the perimeter points of an blueprint
    :param input_floor_plan_dict:
    :return:
    """
    perimeter_walls = input_blueprint_dict["externalWalls"]
    floor_vertices = input_blueprint_dict["vertices"]
    return [(floor_vertices[i]['x'], floor_vertices[i]['y']) for i in perimeter_walls]


def _get_not_floor_space(input_blueprint_dict: Dict, my_plan: 'Plan'):
    """
    Returns a vertices list of the perimeter points of an blueprint
    :param input_floor_plan_dict:
    :return:
    """
    floor_vertices = input_blueprint_dict["vertices"]
    if "stairsObstacles" in input_blueprint_dict.keys():
        stairs_obstacles = input_blueprint_dict["stairsObstacles"]
        if stairs_obstacles:
            for stairs_obstacle in stairs_obstacles:
                stairs_obstacles_poly = [(floor_vertices[i]['x'], floor_vertices[i]['y']) for i in
                                         stairs_obstacle]
                if stairs_obstacles_poly[0] == stairs_obstacles_poly[len(stairs_obstacles_poly)-1]:
                    stairs_obstacles_poly.remove(stairs_obstacles_poly[len(stairs_obstacles_poly)-1])
                my_plan.insert_space_from_boundary(stairs_obstacles_poly,
                                                   category=SPACE_CATEGORIES["stairsObstacle"],
                                                   floor=my_plan.floor_of_given_level(
                                                       input_blueprint_dict["level"]))
    if "holes" in input_blueprint_dict.keys():
        holes = input_blueprint_dict["holes"]
        if holes:
            for hole in holes:
                hole_poly = [(floor_vertices[i]['x'], floor_vertices[i]['y']) for i in
                             hole]
                if hole_poly[0] == hole_poly[len(hole_poly)-1]:
                    hole_poly.remove(hole_poly[len(hole_poly)-1])
                my_plan.insert_space_from_boundary(hole_poly,
                                                   category=SPACE_CATEGORIES["hole"],
                                                   floor=my_plan.floor_of_given_level(
                                                       input_blueprint_dict["level"]))


def _get_fixed_item_perimeter(fixed_item: Dict,
                              vertices: Sequence[Coords2d]) -> FourCoords2d:
    """calculates the polygon perimeter of the fixed item
    Input dict expected with following attributes
    Should be useless if the dict gives us the fixed item geometry directly
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
    return _rectangle_from_segment((point_1, point_2), width)


def _rectangle_from_segment(segment: Tuple[Coords2d, Coords2d], width: float) -> FourCoords2d:
    """
    Creates a rectangle from a segment and a width
    :param segment:
    :param width:
    :return:
    """
    point_1, point_2 = segment
    vector = direction_vector(point_1, point_2)
    vector = normal_vector(vector)

    point_3 = move_point(point_2, vector, width)
    point_4 = move_point(point_1, vector, width)

    return point_1, point_2, point_3, point_4


def _get_external_spaces(input_blueprint_dict: Dict, my_plan: 'Plan'):
    """
    Returns a list with the perimeter of external space.
    :param input_blueprint_dict:
    :return:
    """
    external_spaces = input_blueprint_dict['externalSpaces']
    floor_vertices = input_blueprint_dict["vertices"]

    output = []
    for external_space in external_spaces:
        if 'polygon' in external_space.keys():
            external_space_points = external_space["polygon"]
            if external_space_points[0] == external_space_points[len(external_space_points) - 1]:
                external_space_points.remove(external_space_points[len(external_space_points) - 1])
            my_plan.insert_space_from_boundary(external_space_points,
                                               category=SPACE_CATEGORIES[external_space["type"]],
                                               floor=my_plan.floor_of_given_level(
                                                   input_blueprint_dict["level"]))


def _get_fixed_items_perimeters(input_blueprint_dict: Dict) -> Sequence[Tuple[Coords2d, Dict]]:
    """
    Returns a list with the perimeter of each fixed items.
    NOTE: we are using the pandas dataframe because we do not want to recalculate
    the absolute geometry of each fixed items. As a general rule,
    it would be much better to read and store the geometry
    of each fixed items as list of vertices instead of the way it's done by using barycentric and
    width data. It would be faster and enable us any fixed item shape.
    :param input_blueprint_dict:
    :return: list
    """
    vertices = input_blueprint_dict['vertices']
    fixed_items = input_blueprint_dict['fixedItems']
    output = []
    for fixed_item in fixed_items:
        coords = _get_fixed_item_perimeter(fixed_item, vertices)
        output.append((coords, fixed_item['type']))

    return output


def _get_load_bearing_wall_perimeter(load_bearing_wall: List[int],
                                     vertices: Sequence[Dict[str, str]]) -> Sequence[Coords2d]:
    """
    Returns a rectangular perimeter around the wall
    Expect the following data model :
    "loadBearingWalls": [
      [
        9,
        8
      ]
    ],
    :param load_bearing_wall:
    :param vertices:
    :return:
    """

    wall_points = [point_dict_to_tuple(vertices[ix]) for ix in load_bearing_wall]
    normal = normal_vector(direction_vector(*wall_points))
    point_1 = move_point(wall_points[0], normal, 1)
    point_2 = move_point(wall_points[1], normal, 1)

    return _rectangle_from_segment((point_1, point_2), LOAD_BEARING_WALL_WIDTH)


def _get_load_bearings_walls(input_blueprint_dict: Dict) -> Sequence[Tuple[Coords2d, str]]:
    """
    Returns a proper format with type and coordinates for each load bearing wall
    :param input_blueprint_dict:
    :return:
    """
    vertices = input_blueprint_dict['vertices']
    load_bearing_walls = input_blueprint_dict['loadBearingWalls']
    output = []
    for load_bearing_wall in load_bearing_walls:
        coords = _get_load_bearing_wall_perimeter(load_bearing_wall, vertices)
        output.append((coords, 'loadBearingWall'))

    return output


def get_json_from_file(file_path: str = 'Antony_A22.json',
                       input_folder: str = DEFAULT_BLUEPRINT_INPUT_FOLDER) -> Dict:
    """
    Retrieves the data dictionary from an optimizer json input
    :return:
    """

    module_path = os.path.dirname(__file__)
    input_file_path = os.path.join(module_path, input_folder, file_path)

    # retrieve data from json file
    with open(os.path.abspath(input_file_path)) as floor_plan_file:
        input_floor_plan_dict = json.load(floor_plan_file)

    return input_floor_plan_dict


def get_plan_from_json(file_name: str = 'Antony_A22',
                       input_folder: str = DEFAULT_PLANS_OUTPUT_FOLDER) -> Dict:
    """
    Retrieves the data dictionary from an optimizer json input
    :return:
    """
    file_path = file_name + ".json"
    return get_json_from_file(file_path, input_folder)


def get_mesh_from_json(file_name: str,
                       input_folder: str = DEFAULT_MESHES_OUTPUT_FOLDER) -> Dict:
    """
    Retrieves the data dictionary from an optimizer json input
    :return:
    """
    file_path = file_name + ".json"
    return get_json_from_file(file_path, input_folder)


def create_plan_from_file(input_file: str) -> plan.Plan:
    """
    Creates a plan object from the data retrieved from the given file
    :param input_file: the path to a json file
    :return: a plan object
    """
    floor_plan_dict = get_json_from_file(input_file)
    file_name = os.path.splitext(os.path.basename(input_file))[0]
    my_plan = plan.Plan(file_name)
    apartment = floor_plan_dict['apartment']

    for blueprint_dict in apartment["blueprints"]:
        perimeter = _get_perimeter(blueprint_dict)
        my_plan.add_floor_from_boundary(perimeter, floor_level=blueprint_dict["level"])
        _get_external_spaces(blueprint_dict, my_plan)
        _get_not_floor_space(blueprint_dict, my_plan)
        fixed_items = _get_load_bearings_walls(blueprint_dict)
        fixed_items += _get_fixed_items_perimeters(blueprint_dict)
        linears = (fixed_item for fixed_item in fixed_items if fixed_item[1] in LINEAR_CATEGORIES)
        spaces = (fixed_item for fixed_item in fixed_items if fixed_item[1] in SPACE_CATEGORIES)

        for linear in linears:
            my_plan.insert_linear(linear[0][0], linear[0][1], category=LINEAR_CATEGORIES[linear[1]],
                                  floor=my_plan.floor_of_given_level(blueprint_dict["level"]))

        for space in spaces:
            my_plan.insert_space_from_boundary(space[0], category=SPACE_CATEGORIES[space[1]],
                                               floor=my_plan.floor_of_given_level(
                                                   blueprint_dict["level"]))

    return my_plan


def create_specification_from_file(input_file: str):
    """
    Creates a specification object from a json data
    The model is in the form:
    {
      "setup": [
        {
          "type": "entrance",
          "variant": "s",
          "requiredArea": {
            "min": 25000,
            "max": 50000
          }
        },
        {
          "type": "livingKitchen",
          "variant": "s",
          "requiredArea": {
            "min": 240000,
            "max": 360000
          }
        }]
    }
    """
    spec_dict = get_json_from_file(input_file, DEFAULT_SPECIFICATION_INPUT_FOLDER)
    specification = Specification(input_file)
    for item in spec_dict["setup"]:
        _category = item["type"]
        if _category not in SPACE_CATEGORIES:
            raise ValueError("Space type not present in space categories: {0}".format(_category))
        required_area = item["requiredArea"]
        size_min = Size(area=required_area["min"])
        size_max = Size(area=required_area["max"])
        variant = item["variant"]
        opens_on = []
        linked_to = []
        if "opensOn" in list(item.keys()):
            opens_on = item["opensOn"]
        if "linkedTo" in list(item.keys()):
            linked_to = item["linkedTo"]
        new_item = Item(SPACE_CATEGORIES[_category], variant, size_min, size_max, opens_on,
                        linked_to)
        specification.add_item(new_item)

    return specification


if __name__ == '__main__':
    def specification_read():
        """
        Test
        :return:
        """
        input_file = "begles-carrelets_C304_setup.json"
        spec = create_specification_from_file(input_file)
        print(spec)


    def plan_read():
        input_file = "begles-carrelets_C304.json"
        my_plan = create_plan_from_file(input_file)
        my_plan.plot()
        print(my_plan)


    specification_read()
    plan_read()
