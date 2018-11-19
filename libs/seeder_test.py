# coding=utf-8
"""
Seeder Module Tests
"""
import pytest

import libs.reader as reader
from libs.seeder import Seeder

from libs.grid import GRIDS
from libs.reader import BLUEPRINT_INPUT_FILES
from libs.selector import edge_length

from libs.plot import plot_save


@pytest.mark.parametrize("input_file", BLUEPRINT_INPUT_FILES)
def test_grow_a_plan(input_file):
    """
    Test
    :return:
    """
    plan = reader.create_plan_from_file(input_file)
    new_plan = GRIDS['sequence_grid'].apply_to(plan)
    seeder = Seeder(new_plan)
    seeder.add_condition(edge_length(50.0), 'duct')
    seeder.grow()

    ax = new_plan.plot(save=False, options=('fill', 'border', 'face'))
    seeder.plot(ax)
    plot_save()

    assert new_plan.check()
