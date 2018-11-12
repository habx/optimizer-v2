# coding=utf-8
"""
Seeder Module Tests
"""
import pytest

import libs.reader as reader
from libs.seeder import Seeder

from libs.grid import edge_length, sequence_grid
from libs.reader import BLUEPRINT_INPUT_FILES


@pytest.mark.parametrize("input_file", BLUEPRINT_INPUT_FILES)
def test_grow_a_plan(input_file):
    """
    Test
    :return:
    """
    plan = reader.create_plan_from_file(input_file)
    new_plan = sequence_grid.apply_to(plan)
    seeder = Seeder(new_plan)
    seeder.add_condition(edge_length(50.0), 'duct')
    seeder.grow()

    new_plan.plot(save=True, options=('fill', 'border'))

    assert(new_plan.check())
