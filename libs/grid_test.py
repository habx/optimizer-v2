# coding=utf-8
"""
Test module for grid module
"""

import pytest

import libs.reader as reader
from libs.grid import rectilinear_grid

INPUT_FILES = reader.INPUT_FILES


def test_create_a_grid():
    """
    Test
    :return:
    """
    plan = reader.create_plan_from_file("Edison_10.json")

    new_plan = rectilinear_grid.apply_to(plan)
    assert new_plan.check()


@pytest.mark.parametrize("input_file", INPUT_FILES)
def test_floor_plan_grid(input_file):
    """
    Test. We create a simple grid on several real blue prints.
    :return:
    """
    plan = reader.create_plan_from_file(input_file)
    new_plan = rectilinear_grid.apply_to(plan)

    new_plan.plot()
    assert new_plan.check()
