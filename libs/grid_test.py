# coding=utf-8
"""
Test module for grid module
"""

import pytest

import libs.reader as reader
from libs.grid import sequence_grid

INPUT_FILES = reader.BLUEPRINT_INPUT_FILES


@pytest.mark.parametrize("input_file", INPUT_FILES)
def test_floor_plan_grid(input_file):
    """
    Test. We create a simple grid on several real blue prints.
    :return:
    """
    plan = reader.create_plan_from_file(input_file)
    new_plan = sequence_grid.apply_to(plan)

    new_plan.plot()
    assert new_plan.check()
