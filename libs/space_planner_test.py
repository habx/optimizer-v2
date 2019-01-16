# coding=utf-8
"""
Space planner Module Tests
"""

import pytest

from libs import reader
from libs.seed import Seeder, GROWTH_METHODS, FILL_METHODS, FILL_METHODS_HOMOGENEOUS
from libs.plan import Plan
from libs.grid import GRIDS
from libs.selector import SELECTORS
from libs.shuffle import SHUFFLES
from libs.category import SPACE_CATEGORIES, LINEAR_CATEGORIES
from libs.space_planner import SpacePlanner

test_files = [("Antony_A22.json", "Antony_A22_setup.json"),
              ("Levallois_Letourneur.json", "Levallois_Letourneur_setup.json")]


@pytest.mark.parametrize("input_file, input_setup", test_files)
def test_space_planner(input_file, input_setup):
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


def test_duplex():
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

    GRIDS["simple_grid"].apply_to(plan)

    plan.plot()

    seeder = Seeder(plan, GROWTH_METHODS).add_condition(SELECTORS['seed_duct'], 'duct')
    (seeder.plant()
     .grow(show=True)
     .shuffle(SHUFFLES['seed_square_shape'], show=True)
     .fill(FILL_METHODS_HOMOGENEOUS, (SELECTORS["farthest_couple_middle_space_area_min_100000"],
                                      "empty"), show=True)
     .fill(FILL_METHODS_HOMOGENEOUS, (SELECTORS["single_edge"], "empty"), recursive=True,
           show=True)
     .simplify(SELECTORS["fuse_small_cell_without_components"], show=True)
     .shuffle(SHUFFLES['seed_square_shape'], show=True))

    plan.plot()

    spec = reader.create_specification_from_file("test_space_planner_duplex_setup.json")
    spec.plan = plan

    space_planner = SpacePlanner("test", spec)
    space_planner.solution_research()


if __name__ == '__main__':

    import logging
    logging.getLogger().setLevel(logging.DEBUG)

    test_duplex()