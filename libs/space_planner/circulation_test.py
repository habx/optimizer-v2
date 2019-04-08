# coding=utf-8
"""
Circulation Module Tests
"""

import pytest

from libs.io import reader
from libs.modelers.seed import SEEDERS
from libs.modelers.grid import GRIDS
from libs.space_planner.circulation import Circulator, COST_RULES

from libs.space_planner.space_planner import SpacePlanner

test_files = [("grenoble_101.json", "grenoble_101_setup0.json")]


@pytest.mark.parametrize("input_file, input_setup", test_files)
def test_circulation(input_file, input_setup):
    """
    Test
    :return:
    """

    plan = reader.create_plan_from_file(input_file)

    GRIDS["optimal_grid"].apply_to(plan)
    SEEDERS["simple_seeder"].apply_to(plan)

    spec = reader.create_specification_from_file(input_setup)
    spec.plan = plan

    space_planner = SpacePlanner("test", spec)
    best_solutions = space_planner.solution_research()

    if space_planner.solutions_collector.solutions:
        for solution in best_solutions:
            circulator = Circulator(plan=solution.plan, cost_rules=COST_RULES)
            circulator.connect()
