# coding=utf-8
"""
Circulation Module Tests
"""

import pytest

from libs import reader
from libs.seed import Seeder, GROWTH_METHODS, FILL_METHODS_HOMOGENEOUS

from libs.grid import GRIDS
from libs.selector import SELECTORS
from libs.shuffle import SHUFFLES
from libs.circulation import Circulator, COST_RULES

from libs.space_planner import SpacePlanner

test_files = [("Antony_A22.json", "Antony_A22_setup.json")]


@pytest.mark.parametrize("input_file, input_setup", test_files)
def test_circulation(input_file, input_setup):
    """
    Test
    :return:
    """

    plan = reader.create_plan_from_file(input_file)

    GRIDS["finer_ortho_grid"].apply_to(plan)
    seeder = Seeder(plan, GROWTH_METHODS).add_condition(SELECTORS["seed_duct"], "duct")
    (seeder.plant()
     .grow(show=True)
     .fill(FILL_METHODS_HOMOGENEOUS, (SELECTORS["farthest_couple_middle_space_area_min_50000"],
                                      "empty"), show=True)
     .fill(FILL_METHODS_HOMOGENEOUS, (SELECTORS["single_edge"], "empty"), recursive=True,
           show=True)
     .simplify(SELECTORS["fuse_small_cell_without_components"], show=True)
     .shuffle(SHUFFLES['seed_square_shape_component_aligned'], show=True)
     .empty(SELECTORS["corner_big_cell_area_70000"])
     .fill(FILL_METHODS_HOMOGENEOUS, (SELECTORS["farthest_couple_middle_space_area_min_50000"],
                                      "empty"), show=True)
     .simplify(SELECTORS["fuse_small_cell_without_components"], show=True)
     .shuffle(SHUFFLES['seed_square_shape_component_aligned'], show=True))

    spec = reader.create_specification_from_file(input_setup)
    spec.plan = plan

    space_planner = SpacePlanner("test", spec)
    space_planner.solution_research()

    if space_planner.solutions_collector.solutions:
        for solution in space_planner.solutions_collector.best():
            circulator = Circulator(plan=solution.plan, cost_rules=COST_RULES)
            circulator.connect()
