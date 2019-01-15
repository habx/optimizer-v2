# coding=utf-8
"""
Writer module : Used to read file from json input and create a plan.
"""
from typing import Dict, Optional

import os
import json

DEFAULT_PLANS_OUTPUT_FOLDER = "../output/plans"
DEFAULT_MESHES_OUTPUT_FOLDER = "../output/meshes"


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
    save_as_json(data, DEFAULT_PLANS_OUTPUT_FOLDER, name)


def save_mesh_as_json(data: Dict, name: Optional[str] = None):
    """
    Saves the serialized mesh as a json
    :param data: the serialized data
    :param name: the name of the file (without the extension)
    """
    save_as_json(data, DEFAULT_MESHES_OUTPUT_FOLDER, name)
