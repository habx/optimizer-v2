# coding=utf-8
"""
Seeder Module Tests
"""
import pytest

from libs.inout import reader
from libs.modelers.seed import SEEDERS

from libs.modelers.grid import GRIDS
from libs.inout.reader_test import BLUEPRINT_INPUT_FILES

from libs.plan.plan import Plan
from libs.plan.category import SPACE_CATEGORIES, LINEAR_CATEGORIES


def test_seed_multiple_floors():
    """
    Test
    :return:
    """
    boundaries = [(0, 0), (1000, 0), (1000, 700), (0, 700)]
    boundaries_2 = [(0, 0), (800, 0), (900, 500), (0, 250)]

    plan = Plan("multiple_floors")
    floor_1 = plan.add_floor_from_boundary(boundaries)
    floor_2 = plan.add_floor_from_boundary(boundaries_2, floor_level=1)

    plan.insert_linear((50, 0), (100, 0), LINEAR_CATEGORIES["window"], floor_1)
    plan.insert_linear((200, 0), (300, 0), LINEAR_CATEGORIES["window"], floor_1)
    plan.insert_linear((50, 0), (100, 0), LINEAR_CATEGORIES["window"], floor_2)
    plan.insert_linear((400, 0), (500, 0), LINEAR_CATEGORIES["window"], floor_2)

    GRIDS['optimal_grid'].apply_to(plan)

    SEEDERS["simple_seeder"].apply_to(plan)

    assert plan.check()


@pytest.mark.parametrize("input_file", BLUEPRINT_INPUT_FILES)
def test_grow_a_plan(input_file):
    """
    Test
    :return:
    """
    plan = reader.create_plan_from_file(input_file)
    GRIDS['001'].apply_to(plan)
    SEEDERS["directional_seeder"].apply_to(plan)
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
    plan = Plan("square")
    plan.add_floor_from_boundary(boundaries)
    return plan


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
    duct = [(width / 4, 0), (width * 3 / 4, 0), (width * 3 / 4, depth * 3 / 4),
            (width / 4, depth * 3 / 4)]
    my_plan.insert_space_from_boundary(duct, SPACE_CATEGORIES["duct"])
    return my_plan


def test_simple_seed_test():
    """
    Test
    :return:
    """
    my_plan = plan_with_duct(300, 300)
    GRIDS['optimal_grid'].apply_to(my_plan)
    SEEDERS["simple_seeder"].apply_to(my_plan)
    assert my_plan.check()


def test_directionnal_seed_test():
    """
    Test
    :return:
    """
    my_plan = plan_with_duct(300, 300)
    GRIDS['optimal_finer_grid'].apply_to(my_plan)
    SEEDERS["directional_seeder"].apply_to(my_plan)
    assert my_plan.check()
