# coding=utf-8
"""
Test module for grid module
"""
import libs.reader as reader
from libs.grid import rectilinear_grid


def test_create_a_grid():
    """
    Test
    :return:
    """
    plan = reader.create_plan_from_file("Edison_10.json")

    new_plan = rectilinear_grid.apply_to(plan)
    assert new_plan.check()
