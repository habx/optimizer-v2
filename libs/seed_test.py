# coding=utf-8
"""
Seeder Module Tests
"""
import pytest

from libs import reader
from libs.seed import Seeder,Filler, GROWTH_METHODS
from libs.shuffle import few_corner_shuffle

from libs.grid import GRIDS
from libs.reader_test import BLUEPRINT_INPUT_FILES
from libs.selector import SELECTORS

from libs.plot import plot_save
from libs.plan import Plan
from libs.category import SPACE_CATEGORIES


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
    plan.make_space_seedable("empty")

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
    return Plan("square").from_boundary(boundaries)


def plan_with_duct(width: float, depth: float) -> Plan:
    """
    Returns a plan with a duct
    0, depth   width, depth
     +-----------+
     |           |
     |  +------+ | <- depth * 3/4
     |  |      | |
     +-----------+
    0, 0 ^     ^ width, 0
         |     |
  width * 1/4  width * 3/4
    """
    my_plan = rectangular_plan(width, depth)
    duct = [(width/4, 0), (width*3/4, 0), (width*3/4, depth*3/4), (width/4, depth*3/4)]
    my_plan.insert_space_from_boundary(duct, SPACE_CATEGORIES["duct"])
    return my_plan


def test_simple_seed_test():
    """
    Test
    :return:
    """
    my_plan = plan_with_duct(300, 300)

    seeder = Seeder(my_plan, GROWTH_METHODS)
    seeder.add_condition(SELECTORS['seed_duct'], 'duct')

    GRIDS['ortho_grid'].apply_to(my_plan)

    seeder.plant()
    seeder.grow()
    ax = my_plan.plot()
    seeder.plot_seeds(ax)
