# coding=utf-8
"""
Seeder Module Tests
"""

import pytest

from libs import reader
from libs.seed import Seeder, Filler, GROWTH_METHODS, GROWTH_METHODS_FILL, \
    GROWTH_METHODS_SMALL_SPACE_FILL

from libs.grid import GRIDS
from libs.selector import SELECTORS
from libs.shuffle import SHUFFLES

from libs.space_planner import SpacePlanner

test_files = [('Levallois_Letourneur.json','Levallois_Letourneur_setup.json')]
@pytest.mark.parametrize("input_file, input_setup", test_files)
def test_space_planner(input_file, input_setup):
    """
    Test
    :return:
    """

    plan = reader.create_plan_from_file(input_file)

    seeder = Seeder(plan, GROWTH_METHODS)
    seeder.add_condition(SELECTORS['seed_duct'], 'duct')
    GRIDS['ortho_grid'].apply_to(plan)

    seeder.plant()
    seeder.grow(show=True)
    SHUFFLES['square_shape'].run(plan, show=False)

    seed_empty_furthest_couple_middle = SELECTORS[
        'seed_empty_furthest_couple_middle_space_area_min_100000']
    seed_empty_area_max_100000 = SELECTORS['area_max=100000']
    seed_methods = [
        (
            seed_empty_furthest_couple_middle,
            GROWTH_METHODS_FILL,
            "empty"
        ),
        (
            seed_empty_area_max_100000,
            GROWTH_METHODS_SMALL_SPACE_FILL,
            "empty"
        )
    ]

    filler = Filler(plan, seed_methods)
    filler.apply_to(plan)
    plan.remove_null_spaces()
    fuse_selector = SELECTORS['fuse_small_cell']
    filler.fusion(fuse_selector)

    SHUFFLES['square_shape'].run(plan, show=False)

    spec = reader.create_specification_from_file(input_setup)
    spec.plan = plan

    space_planner = SpacePlanner('test', spec)
    space_planner.add_spaces_constraints()
    space_planner.add_item_constraints()
    space_planner.rooms_building()

    assert plan.check()


