# coding=utf-8
"""
Space planner Module Tests
"""

import pytest

from libs import reader
from libs.seed import Seeder, GROWTH_METHODS, FILL_METHODS

from libs.grid import GRIDS
from libs.selector import SELECTORS
from libs.shuffle import SHUFFLES
from libs.plan import Plan
from libs.space_planner import SpacePlanner
from libs.solution import SolutionsCollector
from libs.category import SPACE_CATEGORIES, LINEAR_CATEGORIES

test_files = [("Antony_A22.json", "Antony_A22_setup.json"),
              ("Levallois_Parisot.json", "Levallois_Parisot_setup.json")]


@pytest.mark.parametrize("input_file, input_setup", test_files)
def test_solution(input_file, input_setup):
    """
    Test
    :return:
    """

    plan = reader.create_plan_from_file(input_file)

    GRIDS["ortho_grid"].apply_to(plan)
    seeder = Seeder(plan, GROWTH_METHODS).add_condition(SELECTORS["seed_duct"], "duct")
    (seeder.plant()
     .grow()
     .shuffle(SHUFFLES["seed_square_shape"])
     .fill(FILL_METHODS, (SELECTORS["farthest_couple_middle_space_area_min_100000"], "empty"))
     .fill(FILL_METHODS, (SELECTORS["single_edge"], "empty"), recursive=True)
     .simplify(SELECTORS["fuse_small_cell"])
     .shuffle(SHUFFLES["seed_square_shape"]))

    spec = reader.create_specification_from_file(input_setup)
    spec.plan = plan

    space_planner = SpacePlanner("test", spec)
    space_planner.solution_research()


def test_solution():
    """
    Test
    :return:
    """
    boundaries = [(0, 0), (1000, 0), (1000, 500), (200, 500)]
    boundaries_2 = [(0, 0), (800, 0), (800, 500), (200, 500)]

    plan = Plan("multiple_floors")
    floor_1 = plan.add_floor_from_boundary(boundaries, floor_level=0)
    floor_2 = plan.add_floor_from_boundary(boundaries_2, floor_level=1)

    balcony_coords = [(800, 0), (1000, 0), (1000, 500), (800, 500)]
    plan.insert_space_from_boundary(balcony_coords, SPACE_CATEGORIES["balcony"], floor_1)
    duct_coords = [(600, 475), (650, 475), (650, 500), (600, 500)]
    plan.insert_space_from_boundary(duct_coords, SPACE_CATEGORIES["duct"], floor_1)
    hole_coords = [(500, 0), (600, 0), (600, 100), (500, 100)]
    plan.insert_space_from_boundary(hole_coords, SPACE_CATEGORIES["hole"], floor_1)
    plan.insert_linear((800, 100), (800, 400), LINEAR_CATEGORIES["doorWindow"], floor_1)
    plan.insert_linear((350, 0), (450, 0), LINEAR_CATEGORIES["window"], floor_1)
    plan.insert_linear((200, 500), (300, 500), LINEAR_CATEGORIES["window"], floor_1)
    plan.insert_linear((600, 500), (650, 500), LINEAR_CATEGORIES["frontDoor"], floor_1)
    plan.insert_linear((500, 100), (550, 100), LINEAR_CATEGORIES["startingStep"], floor_1)

    plan.insert_space_from_boundary(duct_coords, SPACE_CATEGORIES["duct"], floor_2)
    plan.insert_space_from_boundary(hole_coords, SPACE_CATEGORIES["hole"], floor_2)
    plan.insert_linear((150, 0), (250, 0), LINEAR_CATEGORIES["window"], floor_2)
    plan.insert_linear((350, 0), (450, 0), LINEAR_CATEGORIES["window"], floor_2)
    plan.insert_linear((550, 100), (600, 100), LINEAR_CATEGORIES["startingStep"], floor_2)

    GRIDS["ortho_grid"].apply_to(plan)

    plan.plot()

    seeder = Seeder(plan, GROWTH_METHODS).add_condition(SELECTORS['seed_duct'], 'duct')
    (seeder.plant()
     .grow()
     .shuffle(SHUFFLES['seed_square_shape'])
     .fill(FILL_METHODS, (SELECTORS["farthest_couple_middle_space_area_min_100000"], "empty"))
     .fill(FILL_METHODS, (SELECTORS["single_edge"], "empty"), recursive=True)
     .simplify(SELECTORS["fuse_small_cell"])
     .shuffle(SHUFFLES['seed_square_shape']))

    spec = reader.create_specification_from_file("Antony_A22_setup.json")
    spec.plan = plan

    space_planner = SpacePlanner("test", spec)
    space_planner.solution_research()


if __name__ == '__main__':
    test_solution()