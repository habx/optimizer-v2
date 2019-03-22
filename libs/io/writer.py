# coding=utf-8
"""
Writer module : Used to read file from json input and create a plan.
"""
from typing import Dict, Optional
import os
import json
import logging
import libs.io.reader as reader
from libs.space_planner.solution import Solution


def save_as_json(data: Dict, output_folder: str, name: Optional[str] = None):
    """
    Saves the data to a json file
    :param data:
    :param output_folder:
    :param name:
    :return:
    """
    name = name or data.get("name", "unnamed")
    file_path = "{}.json".format(name)
    module_path = os.path.dirname(__file__)
    output_folder_path = os.path.join(module_path, output_folder)
    output_path = os.path.join(module_path, output_folder, file_path)

    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    with open(os.path.abspath(output_path), 'w') as fp:
        json.dump(data, fp, sort_keys=True, indent=2)


def save_plan_as_json(data: Dict, name: Optional[str] = None):
    """
    Saves the serialized plan as a json
    :param data: the serialized data
    :param name: the name of the file (without the extension)
    """
    save_as_json(data, reader.DEFAULT_PLANS_OUTPUT_FOLDER, name)


def save_mesh_as_json(data: Dict, name: Optional[str] = None):
    """
    Saves the serialized mesh as a json
    :param data: the serialized data
    :param name: the name of the file (without the extension)
    """
    save_as_json(data, reader.DEFAULT_MESHES_OUTPUT_FOLDER, name)


def generate_output_dict(input_data: dict, solution: Solution) -> dict:
    # deep copy is not thread safe, dict comprehension is not deep, so we use this small hack
    output_dict = json.loads(json.dumps(input_data))

    points = output_dict["vertices"]
    spaces = output_dict["spaces"]
    floors = output_dict["floors"]
    vertices_max_id = 0
    room_max_id = 0

    for space in spaces:
        if space["category"] == "empty":
            space["category"] = "border"

    for i, room in enumerate(solution.plan.mutable_spaces()):
        room_max_id += 1
        room_dict = {
            "area": room.area,
            "category": room.category.name,
            "geometry": [],
            "id": int("70" + str(room_max_id))}
        for edge in list(room.edges):
            vertices_max_id += 1
            point_dict = {
                "id": int("50" + str(vertices_max_id)),
                "x": edge.start.x,
                "y": edge.start.y
            }
            room_dict["geometry"].append(int("50" + str(vertices_max_id)))
            points.append(point_dict)

        spaces.append(room_dict)

        for floor in floors:
            if floor["level"] == room.floor.level:
                floor["elements"].append(int("70" + str(room_max_id)))

    output_dict = {"v2": output_dict}

    return output_dict


def generate_output_dict_from_file(input_file_name: str, solution: Solution) -> dict:
    floor_plan_dict = reader.get_json_from_file(input_file_name)

    if "v2" in floor_plan_dict.keys():
        input_dict = floor_plan_dict["v2"]
    else:
        logging.warning("Writer : v1 input plan")
        return {}

    return generate_output_dict(input_dict, solution)


def save_json_solution(data, num_sol):
    with open(os.path.join(reader.DEFAULT_PLANS_OUTPUT_FOLDER,
                           "output" + str(num_sol) + ".json"), "w") as f:
        json.dump(data, f, sort_keys=True, indent=4)
