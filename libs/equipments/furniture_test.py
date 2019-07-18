import json
from libs.io import reader
import libs.equipments.furniture as furniture
from libs.equipments.doors import place_doors
from libs.space_planner.solution import reference_plan_solution


def test_ref_plan():
    plan_data = reader.get_json_from_file("ARCH001_plan.json")
    setup_data = reader.get_json_from_file("ARCH001_setup.json",
                                           reader.DEFAULT_SPECIFICATION_INPUT_FOLDER)

    ref_plan = reader.create_plan_from_data(plan_data)
    setup_spec = reader.create_specification_from_data(setup_data)
    ref_solution = reference_plan_solution(ref_plan, setup_spec)
    place_doors(ref_solution.spec.plan)
    furniture.GARNISHERS['default'].apply_to(ref_solution)
    ref_solution.spec.plan.check()