# coding=utf-8
"""
Test module for grid module
"""

import pytest

from libs import reader, reader_test
from libs.grid import GRIDS, Grid
from libs.selector import SELECTORS
from libs.mutation import MUTATIONS

INPUT_FILES = reader_test.BLUEPRINT_INPUT_FILES


@pytest.mark.parametrize("input_file", INPUT_FILES)
def test_floor_plan_grid(input_file):
    """
    Test. We create a simple grid on several real blue prints.
    :return:
    """
    plan = reader.create_plan_from_file(input_file)
    new_plan = GRIDS['ortho_grid'].apply_to(plan)

    new_plan.plot()
    assert new_plan.check()


def test_simple_grid():
    """
    Test
    :return:
    """
    simple_grid = Grid('simple_grid', [(SELECTORS["previous_angle_salient_ortho"],
                                        MUTATIONS['ortho_projection_cut'])])

    plan = reader.create_plan_from_file("Antony_A22.json")
    new_plan = simple_grid.apply_to(plan)
    assert new_plan.check()
