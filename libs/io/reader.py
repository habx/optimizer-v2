# coding=utf-8
"""
Reader module : Used to read file from json input and create a plan.
"""
from typing import Dict, Sequence, Tuple, List, Optional
import os
import json
from libs.plan.category import SPACE_CATEGORIES, LINEAR_CATEGORIES
from libs.specification.specification import Specification, Item, Size
import libs.plan.plan as plan

from libs.utils.geometry import (
    point_dict_to_tuple,
    barycenter,
    direction_vector,
    normal_vector,
    move_point
)
from libs.utils.custom_types import Coords2d, FourCoords2d
from libs.mesh.mesh import COORD_EPSILON
from resources import DEFAULT_BLUEPRINT_INPUT_FOLDER, DEFAULT_SPECIFICATION_INPUT_FOLDER
from output import DEFAULT_PLANS_OUTPUT_FOLDER, DEFAULT_MESHES_OUTPUT_FOLDER

LOAD_BEARING_WALL_WIDTH = 15.0


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
    :param input_blueprint_dict: Dict
    :return:
    """
    perimeter_walls = input_blueprint_dict["externalWalls"]
    floor_vertices = input_blueprint_dict["vertices"]
    return [(floor_vertices[i]['x'], floor_vertices[i]['y']) for i in perimeter_walls]


def _add_not_floor_space(input_blueprint_dict: Dict, my_plan: 'plan.Plan'):
    """
    Insert stairsObstacles and holes on a plan
    :param input_blueprint_dict: Dict
    :param my_plan: 'Plan'
    :return:
    """
    floor_level = input_blueprint_dict["level"]
    floor_vertices = input_blueprint_dict["vertices"]

    if "stairsObstacles" in input_blueprint_dict.keys():
        stairs_obstacles = input_blueprint_dict["stairsObstacles"] or []
        for stairs_obstacle in stairs_obstacles:
            stairs_obstacle_poly = [(floor_vertices[i]['x'], floor_vertices[i]['y'])
                                    for i in stairs_obstacle]
            if stairs_obstacle_poly[0] == stairs_obstacle_poly[len(stairs_obstacle_poly) - 1]:
                stairs_obstacle_poly.remove(stairs_obstacle_poly[len(stairs_obstacle_poly) - 1])
            my_plan.insert_space_from_boundary(stairs_obstacle_poly,
                                               category=SPACE_CATEGORIES["stairsObstacle"],
                                               floor=my_plan.floor_of_given_level(floor_level))
    if "holes" in input_blueprint_dict.keys():
        holes = input_blueprint_dict["holes"] or []
        for hole in holes:
            hole_poly = [(floor_vertices[i]['x'], floor_vertices[i]['y']) for i in hole]
            if hole_poly[0] == hole_poly[len(hole_poly) - 1]:
                hole_poly.remove(hole_poly[len(hole_poly) - 1])
            my_plan.insert_space_from_boundary(hole_poly,
                                               category=SPACE_CATEGORIES["hole"],
                                               floor=my_plan.floor_of_given_level(floor_level))


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


def _add_external_spaces(input_blueprint_dict: Dict, my_plan: 'plan.Plan'):
    """
    Returns a list with the perimeter of external space.
    :param input_blueprint_dict:
    :return:
    """
    external_spaces = input_blueprint_dict['externalSpaces']

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


def _clean_perimeter(perimeter: Sequence[Coords2d]) -> List[Coords2d]:
    """
    Remove points that are too close for Optimizer Epsilon
    """
    new_perimeter = [perimeter[0]]
    for coord in perimeter:
        if (((coord[0] - new_perimeter[-1][0]) ** 2) + (
                (coord[1] - new_perimeter[-1][1]) ** 2)) ** 0.5 > COORD_EPSILON:
            new_perimeter.append(coord)
    if (((new_perimeter[0][0] - new_perimeter[-1][0]) ** 2) + (
            (new_perimeter[0][1] - new_perimeter[-1][1]) ** 2)) ** 0.5 < COORD_EPSILON:
        del new_perimeter[-1]
    return new_perimeter


def get_json_from_file(file_name: str = '011.json',
                       input_folder: str = DEFAULT_BLUEPRINT_INPUT_FOLDER) -> Dict:
    """
    Retrieves the data dictionary from an optimizer json input
    :return:
    """
    assert file_name[len(file_name) - 5:] == ".json", "The filename must have a .json extension"
    module_path = os.path.dirname(__file__)
    input_file_path = os.path.join(module_path, input_folder, file_name)

    # retrieve data from json file
    with open(os.path.abspath(input_file_path)) as floor_plan_file:
        input_floor_plan_dict = json.load(floor_plan_file)

    return input_floor_plan_dict


def get_plan_from_json(file_name: str = '011.json',
                       input_folder: str = DEFAULT_PLANS_OUTPUT_FOLDER) -> Dict:
    """
    Retrieves the data dictionary from an optimizer json input
    :return:
    """
    file_path = file_name
    return get_json_from_file(file_path, input_folder)


def get_mesh_from_json(file_name: str,
                       input_folder: str = DEFAULT_MESHES_OUTPUT_FOLDER) -> Dict:
    """
    Retrieves the data dictionary from an optimizer json input
    :return:
    """
    file_path = file_name
    return get_json_from_file(file_path, input_folder)


def create_plan_from_file(input_file_name: str) -> plan.Plan:
    """
    Creates a plan object from the data retrieved from the given file
    :param input_file_name: the path to a json file
    :return: a plan object
    """
    floor_plan_dict = get_json_from_file(input_file_name)
    return create_plan_from_data(floor_plan_dict)


def create_plan_from_data(floor_plan_dict: Dict, name: Optional[str] = None) -> plan.Plan:
    """
    Create a plan object from data as found in blueprint files
    :return: a plan object
    """
    if "v2" in floor_plan_dict.keys():
        lot_slug = floor_plan_dict["meta"]["slug"]
        project_slug = floor_plan_dict["meta"]["projectSlug"]
        plan_name = "_".join((project_slug, lot_slug)) if name is None else name
        return create_plan_from_v2_data(floor_plan_dict["v2"], plan_name)
    elif "v1" in floor_plan_dict.keys():
        plan_name = floor_plan_dict["v1"]["apartment"]["id"] if name is None else name
        return create_plan_from_v1_data(floor_plan_dict["v1"], plan_name)
    else:
        plan_name = floor_plan_dict["apartment"]["id"] if name is None else name
        return create_plan_from_v1_data(floor_plan_dict, plan_name)


def create_plan_from_v2_data(v2_data: Dict, name: str) -> plan.Plan:
    """
    Creates a plan object from the data retrieved from the specified dictionary
    The function uses the version 2 of the blueprint data format.
    """

    my_plan = plan.Plan(name)

    # get vertices data
    vertices_by_id: Dict[int, Coords2d] = {}
    for vertex_data in v2_data["vertices"]:
        vertices_by_id[vertex_data["id"]] = (vertex_data["x"], vertex_data["y"])

    # get data for each floor of the apartment
    for floor_data in v2_data["floors"]:
        # empty perimeter (per convention we create the floor with the first empty space)
        empty_space_data = next(space_data for space_data in v2_data["spaces"]
                                if space_data["id"] in floor_data["elements"]
                                and space_data["category"] == "empty")
        perimeter = [vertices_by_id[vertex_id] for vertex_id in empty_space_data["geometry"]]
        my_plan.add_floor_from_boundary(perimeter, floor_level=floor_data["level"])
        floor = my_plan.floor_of_given_level(floor_data["level"])

        # add linears except steps
        # note : we need to add linears before spaces because the insertion of spaces
        # can sometimes split the edges on the perimeter that should have received a linear
        # (for example when we add an internal duct or load bearing wall)
        for linear_data in v2_data["linears"]:
            if (linear_data["id"] in floor_data["elements"]
                    and linear_data["category"] != "startingStep"):
                p1 = vertices_by_id[linear_data["geometry"][0]]
                p2 = vertices_by_id[linear_data["geometry"][1]]
                category = LINEAR_CATEGORIES[linear_data["category"]]
                my_plan.insert_linear(p1, p2, category=category, floor=floor)

        # add other spaces
        for space_data in v2_data["spaces"]:
            if (space_data["id"] in floor_data["elements"]
                    and space_data["id"] != empty_space_data["id"]):
                space_points = [vertices_by_id[vertex_id] for vertex_id in space_data["geometry"]]
                space_points = _clean_perimeter(space_points)
                category = SPACE_CATEGORIES[space_data["category"]]
                my_plan.insert_space_from_boundary(space_points, category=category, floor=floor)

        # add steps linear "StartingStep". This linear must be inserted after the steps space.
        for linear_data in v2_data["linears"]:
            if (linear_data["category"] == "startingStep"
                    and linear_data["id"] in floor_data["elements"]):
                p1 = vertices_by_id[linear_data["geometry"][0]]
                p2 = vertices_by_id[linear_data["geometry"][1]]
                category = LINEAR_CATEGORIES[linear_data["category"]]
                my_plan.insert_linear(p1, p2, category=category, floor=floor)

    return my_plan


def create_plan_from_v1_data(v1_data: Dict, name: str) -> plan.Plan:
    """
    Creates a plan object from the data retrieved from the specified dictionary
    The function uses the version 1 of the blueprint data format.
    Note: only 1 floor per apartment
    :param name: name for the plan
    :param v1_data:
    :return:
    """
    my_plan = plan.Plan(name)

    apartment = v1_data["apartment"]

    for blueprint_dict in apartment["blueprints"]:

        # create floor
        perimeter = _get_perimeter(blueprint_dict)
        my_plan.add_floor_from_boundary(perimeter, floor_level=blueprint_dict["level"])
        _add_external_spaces(blueprint_dict, my_plan)
        _add_not_floor_space(blueprint_dict, my_plan)

        # add fixed items
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


def create_specification_from_file(input_file: str) -> Specification:
    """
    Creates a specification object from a json file name
    """
    spec_dict = get_json_from_file(input_file, DEFAULT_SPECIFICATION_INPUT_FOLDER)
    return create_specification_from_data(spec_dict, input_file)


def create_specification_from_data(input_data: dict,
                                   spec_name: str = "unnamed spec") -> Specification:
    """
    Creates a specification object from a dict
    The model is in the form:
    {
      "rooms": [
        {
          "type": "circulation",
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
    specification = Specification(spec_name)
    for item in input_data["rooms"]:
        _category = item["type"]
        if _category not in SPACE_CATEGORIES:
            raise ValueError("Space type not present in space categories: {0}".format(_category))
        required_area = item["requiredArea"]
        size_min = Size(area=required_area["min"])
        size_max = Size(area=required_area["max"])
        variant = item.get("variant", "m")
        opens_on = item.get("opensOn", [])
        linked_to = item.get("linkedTo", [])
        tags = item.get("tags", [])
        new_item = Item(SPACE_CATEGORIES[_category], variant, size_min, size_max, opens_on,
                        linked_to, tags)
        specification.add_item(new_item)

    return specification


if __name__ == '__main__':
    import matplotlib
    matplotlib.use("TkAgg")

    def specification_read():
        """
        Test
        :return:
        """
        input_file = "015_setup0.json"
        spec = create_specification_from_file(input_file)
        print(spec)

        spec_dict = spec.serialize()
        print(spec_dict)


    def plan_read():
        """
        Test
        :return:
        """
        input_file = "A2E2H05.json" # (2875.051, 556.2034), (2759.2167, 498.7173)
        my_plan = create_plan_from_file(input_file)
        my_plan.plot()
        print(my_plan)
        assert my_plan.check()


    # specification_read()
    plan_read()
