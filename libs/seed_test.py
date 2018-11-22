# coding=utf-8
"""
Seeder Module Tests
"""
import pytest

import libs.reader as reader
from libs.seed import Seeder, GROWTH_METHODS

from libs.grid import GRIDS
from libs.reader import BLUEPRINT_INPUT_FILES
from libs.selector import SELECTORS

from libs.plot import plot_save


@pytest.mark.parametrize("input_file", BLUEPRINT_INPUT_FILES)
def test_grow_a_plan(input_file):
    """
    Test
    :return:
    """
    plan = reader.create_plan_from_file(input_file)

    GRIDS['sequence_grid'].apply_to(plan)

    seeder = Seeder(plan, GROWTH_METHODS)
    seeder.add_condition(SELECTORS['seed_duct'], 'duct')
    seeder.plant()
    seeder.grow()

    ax = plan.plot(save=False, options=('fill', 'border', 'face'))
    seeder.plot_seeds(ax)
    plot_save()

    assert plan.check()
