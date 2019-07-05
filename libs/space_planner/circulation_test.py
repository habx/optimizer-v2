# coding=utf-8
"""
Circulation Module Tests
"""

import pytest

from libs.io import reader
from libs.modelers.seed import SEEDERS
from libs.modelers.grid import GRIDS
from libs.space_planner.circulation import Circulator

from libs.space_planner.space_planner import SPACE_PLANNERS

test_files = [("062.json", "062_setup0.json")]


@pytest.mark.parametrize("input_file, input_setup", test_files)
def test_circulation(input_file, input_setup):
    """
    Test
    :return:
    """

    plan = reader.create_plan_from_file(input_file)

    GRIDS["001"].apply_to(plan)
    SEEDERS["directional_seeder"].apply_to(plan)

    spec = reader.create_specification_from_file(input_setup)
    spec.plan = plan

    space_planner = SPACE_PLANNERS["standard_space_planner"]
    best_solutions = space_planner.apply_to(spec, 3,1)

    for solution in best_solutions:
        circulator = Circulator(plan=solution.spec.plan, spec=spec)
        circulator.connect()
        circulator.plot()
