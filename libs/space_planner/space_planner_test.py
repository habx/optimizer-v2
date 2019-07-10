# coding=utf-8
"""
Space planner Module Tests
"""

import pytest

from libs.inout import reader
from libs.modelers.seed import SEEDERS
from libs.plan.plan import Plan
from libs.modelers.grid import GRIDS
from libs.plan.category import SPACE_CATEGORIES, LINEAR_CATEGORIES
from libs.space_planner.space_planner import SPACE_PLANNERS

test_files = [("009.json", "009_setup0.json"),
              ("012.json", "012_setup0.json")]


@pytest.mark.parametrize("input_file, input_setup", test_files)
def test_space_planner(input_file, input_setup):
    """
    Test
    :return:
    """

    plan = reader.create_plan_from_file(input_file)

    GRIDS['002'].apply_to(plan)
    SEEDERS["directional_seeder"].apply_to(plan)

    spec = reader.create_specification_from_file(input_setup)
    spec.plan = plan

    space_planner = SPACE_PLANNERS["standard_space_planner"]
    best_solutions = space_planner.apply_to(spec, 3)

    if input_file == "009.json":
        assert len(space_planner.solutions_collector.solutions) == 55
        assert len(best_solutions) == 3
    elif input_file == "012.json":
        assert len(space_planner.solutions_collector.solutions) == 597
        assert len(best_solutions) == 3


def test_duplex():
    """
    Test
    :return:
    """
    boundaries = [(0, 0), (1000, 0), (1000, 500), (200, 500)]
    boundaries_2 = [(0, 0), (800, 0), (800, 500), (200, 500)]

    plan = Plan("SpacePlanner_Tests_Multiple_floors")
    floor_1 = plan.add_floor_from_boundary(boundaries, floor_level=0)
    floor_2 = plan.add_floor_from_boundary(boundaries_2, floor_level=1)

    balcony_coords = [(800, 0), (1000, 0), (1000, 500), (800, 500)]
    plan.insert_space_from_boundary(balcony_coords, SPACE_CATEGORIES["balcony"], floor_1)
    duct_coords = [(600, 475), (650, 475), (650, 500), (600, 500)]
    plan.insert_space_from_boundary(duct_coords, SPACE_CATEGORIES["duct"], floor_1)
    hole_coords = [(450, 0), (600, 0), (600, 150), (450, 150)]
    plan.insert_space_from_boundary(hole_coords, SPACE_CATEGORIES["hole"], floor_1)
    plan.insert_linear((800, 50), (800, 250), LINEAR_CATEGORIES["doorWindow"], floor_1)
    plan.insert_linear((800, 350), (800, 450), LINEAR_CATEGORIES["doorWindow"], floor_1)
    plan.insert_linear((250, 0), (350, 0), LINEAR_CATEGORIES["window"], floor_1)
    plan.insert_linear((350, 500), (250, 500), LINEAR_CATEGORIES["window"], floor_1)
    plan.insert_linear((550, 500), (475, 500), LINEAR_CATEGORIES["frontDoor"], floor_1)
    plan.insert_linear((450, 150), (525, 150), LINEAR_CATEGORIES["startingStep"], floor_1)

    plan.insert_space_from_boundary(duct_coords, SPACE_CATEGORIES["duct"], floor_2)
    plan.insert_space_from_boundary(hole_coords, SPACE_CATEGORIES["hole"], floor_2)
    hole_coords = [(600, 0), (800, 0), (800, 300), (600, 300)]
    plan.insert_space_from_boundary(hole_coords, SPACE_CATEGORIES["hole"], floor_2)
    plan.insert_linear((100, 0), (200, 0), LINEAR_CATEGORIES["window"], floor_2)
    plan.insert_linear((300, 0), (400, 0), LINEAR_CATEGORIES["window"], floor_2)
    plan.insert_linear((525, 150), (600, 150), LINEAR_CATEGORIES["startingStep"], floor_2)

    GRIDS["002"].apply_to(plan)

    SEEDERS["directional_seeder"].apply_to(plan)

    spec = reader.create_specification_from_file("test_space_planner_duplex_setup.json")
    spec.plan = plan

    space_planner = SPACE_PLANNERS["standard_space_planner"]
    space_planner.apply_to(spec, 3)
