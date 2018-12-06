# coding=utf-8
"""
Seeder Module Tests
"""
import pytest

import libs.reader as reader
from libs.seed import Seeder, Filler, GROWTH_METHODS
from libs.shuffle import few_corner_shuffle

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

    seeder = Seeder(plan, GROWTH_METHODS)
    seeder.add_condition(SELECTORS['seed_duct'], 'duct')
    GRIDS['ortho_grid'].apply_to(plan)

    seeder.plant()
    seeder.grow()
    few_corner_shuffle.run(plan, show=True)

    plan.remove_null_spaces()

    seed_empty_furthest_couple = SELECTORS['seed_empty_furthest_couple']
    seed_empty_area_max_100000 = SELECTORS['area_max=100000']
    seed_methods = [
        (
            seed_empty_furthest_couple,
            GROWTH_METHODS,
            "empty"
        ),
        (
            seed_empty_area_max_100000,
            GROWTH_METHODS,
            "empty"
        )
    ]

    filler = Filler(plan, seed_methods)
    filler.apply_to(plan)

    ax = plan.plot(save=False, options=('fill', 'border', 'face'))
    seeder.plot_seeds(ax)
    plot_save()

    plan.remove_null_spaces()

    assert plan.check()
