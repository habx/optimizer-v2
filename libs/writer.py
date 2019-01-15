# coding=utf-8
"""
Writer module : Used to read file from json input and create a plan.
"""
from typing import Dict, Optional

import os
import json

DEFAULT_PLANS_OUTPUT_FOLDER = "../output/plans"


def save_plan_as_json(data: Dict, name: Optional[str] = None,
                      output_folder: str = DEFAULT_PLANS_OUTPUT_FOLDER):
    """
    Saves the serialized plan as a json
    :param data: the serialized data
    :param name: the name of the file (without the extension)
    :param output_folder: where to store the file
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
