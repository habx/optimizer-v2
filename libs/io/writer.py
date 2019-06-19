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


def save_as_json(data: Dict, output_folder: str, file_name: Optional[str] = None):
    """
    Saves the data to a json file
    :param data:
    :param output_folder:
    :param file_name:
    :return:
    """
    if file_name:
        assert file_name[len(file_name) - 5:] == ".json", "The filename must have a .json extension"

    file_name = file_name or data.get("name", "unnamed") + ".json"
    module_path = os.path.dirname(__file__)
    output_folder_path = os.path.join(module_path, output_folder)
    output_path = os.path.join(module_path, output_folder, file_name)

    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    with open(os.path.abspath(output_path), 'w') as fp:
        json.dump(data, fp, sort_keys=True, indent=2)


def save_plan_as_json(data: Dict,
                      file_name: Optional[str] = None,
                      output_folder: Optional[str] = None):
    """
    Saves the serialized plan as a json
    :param data: the serialized data
    :param file_name: the name of the file (without the extension)
    """
    output_folder = output_folder or reader.DEFAULT_PLANS_OUTPUT_FOLDER
    if file_name is not None and file_name[-5:] != ".json":
        file_name += ".json"
    save_as_json(data, output_folder, file_name)


def save_mesh_as_json(data: Dict, name: Optional[str] = None):
    """
    Saves the serialized mesh as a json
    :param data: the serialized data
    :param name: the name of the file (without the extension)
    """
    save_as_json(data, reader.DEFAULT_MESHES_OUTPUT_FOLDER, name)


def generate_output_dict(input_data: dict, solution: Solution) -> dict:
    """
    Creates the data dict from the input_data
    :param input_data:
    :param solution:
    :return:
    """
    # deep copy is not thread safe, dict comprehension is not deep
    # so we use this hack (fast enough)
    assert "v2" in input_data.keys()

    output_data_v2 = json.loads(json.dumps(input_data["v2"]))

    points = output_data_v2["vertices"]
    spaces = output_data_v2["spaces"]
    floors = output_data_v2["floors"]
    apt_infos = output_data_v2["aptInfos"]
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

        for i_ref, ref_edge in enumerate(room.reference_edges):
            room_dict["geometry"].append([])
            for edge in room.siblings(ref_edge):
                vertices_max_id += 1
                point_dict = {
                    "id": int("50" + str(vertices_max_id)),
                    "x": edge.start.x,
                    "y": edge.start.y
                }
                room_dict["geometry"][i_ref].append(int("50" + str(vertices_max_id)))
                points.append(point_dict)

        spaces.append(room_dict)

        for floor in floors:
            if floor["level"] == room.floor.level:
                floor["elements"].append(int("70" + str(room_max_id)))

    apt_infos["score"] = solution.score

    return {"v2": output_data_v2}


def generate_output_dict_from_file(input_file_name: str, solution: Solution) -> dict:
    """
    Compute the data dict from a specified file
    :param input_file_name:
    :param solution:
    :return:
    """
    floor_plan_dict = reader.get_json_from_file(input_file_name)

    if "v2" not in floor_plan_dict.keys():
        logging.warning("Writer : v1 input plan")
        return {}

    return generate_output_dict(floor_plan_dict, solution)


def save_json_solution(data, num_sol):
    """
    Save a json to file
    :param data:
    :param num_sol:
    :return:
    """
    with open(os.path.join(reader.DEFAULT_PLANS_OUTPUT_FOLDER,
                           "output" + str(num_sol) + ".json"), "w") as f:
        json.dump(data, f, sort_keys=True, indent=4)
