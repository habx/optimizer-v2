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

    GRIDS["optimal_grid"].apply_to(plan)
    seeder = Seeder(plan, GROWTH_METHODS).add_condition(SELECTORS['seed_duct'], 'duct')
    (seeder.plant()
     .grow(show=True)
     .divide_from_seeds(SELECTORS["not_aligned_edges"])
     .from_space_empty_to_seed())

    spec = reader.create_specification_from_file(input_setup)
    spec.plan = plan

    space_planner = SpacePlanner("test", spec)
    best_solutions = space_planner.solution_research()

    if space_planner.solutions_collector.solutions:
        for solution in best_solutions:
            circulator = Circulator(plan=solution.plan, cost_rules=COST_RULES)
            circulator.connect()
