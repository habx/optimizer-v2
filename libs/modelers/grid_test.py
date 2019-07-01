# coding=utf-8
"""
Test module for grid module
"""

import pytest

from libs.inout import reader, reader_test
from libs.modelers.grid import GRIDS, Grid
from libs.operators.selector import SELECTORS
from libs.operators.mutation import MUTATIONS
from libs.plan.plan import Plan

INPUT_FILES = reader_test.BLUEPRINT_INPUT_FILES


def rectangular_plan(width: float, depth: float) -> Plan:
    """
    a simple rectangular plan

   0, depth   width, depth
     +------------+
     |            |
     |            |
     |            |
     +------------+
    0, 0     width, 0

    :return:
    """
    boundaries = [(0, 0), (width, 0), (width, depth), (0, depth)]
    plan = Plan("square")
    plan.add_floor_from_boundary(boundaries)
    return plan


@pytest.mark.parametrize("input_file", INPUT_FILES)
def test_floor_plan_grid(input_file):
    """
    Test. We create a simple grid on several real blue prints.
    :return:
    """
    plan = reader.create_plan_from_file(input_file)
    new_plan = GRIDS["001"].apply_to(plan)

    assert new_plan.check()


def test_simple_ortho_grid():
    """
    Test
    :return:
    """
    simple_ortho_grid = Grid('simple_ortho_grid', [(SELECTORS["previous_angle_salient_ortho"],
                                                    MUTATIONS['ortho_projection_cut'], False)])

    plan = reader.create_plan_from_file("011.json")
    new_plan = simple_ortho_grid.apply_to(plan)

    assert new_plan.check()


def test_simple_grid():
    """
    Test
    :return:
    """
    simple_grid = GRIDS["simple_grid"]
    plan = rectangular_plan(500, 500)
    plan = simple_grid.apply_to(plan)
    assert len(plan.empty_space._faces_id) == 64


def test_multiple_floors_grid():
    """
    Test a plan with multiple floors
    :return:
    """
    boundaries = [(0, 0), (1000, 0), (1000, 700), (0, 700)]
    boundaries_2 = [(0, 0), (800, 0), (900, 500), (0, 250)]
    plan = Plan()
    plan.add_floor_from_boundary(boundaries)
    plan.add_floor_from_boundary(boundaries_2, 1)

    GRIDS["finer_ortho_grid"].apply_to(plan)
    assert plan.check()
