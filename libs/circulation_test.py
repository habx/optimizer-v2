# coding=utf-8
"""
Circulation Module Tests
"""

import pytest

from libs import reader
from libs.seed import Seeder, GROWTH_METHODS, FILL_METHODS

from libs.grid import GRIDS
from libs.selector import SELECTORS
from libs.shuffle import SHUFFLES
from libs.circulation import Circulator

from libs.space_planner import SpacePlanner

test_files = [("Antony_A22.json", "Antony_A22_setup.json"),
              ("Vernouillet_A105.json", "Vernouillet_A105_setup.json")]


@pytest.mark.parametrize("input_file, input_setup", test_files)
def test_circulation(input_file, input_setup):
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

    cost_rules = {
        'water_room_less_than_two_ducts': 10e5,
        'water_room_default': 1000,
        'window_room_less_than_two_windows': 10e10,
        'window_room_default': 5000,
        'default': 0
    }

    for solution in space_planner.solutions_collector.best():
        circulator = Circulator(plan=solution.plan, cost_rules=cost_rules)
        circulator.connect()
