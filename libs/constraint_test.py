# coding=utf-8
"""
Constraint Test Module
"""
import pytest
from libs.constraint import CONSTRAINTS
from libs.plan import Plan
from libs.category import SPACE_CATEGORIES


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


def square_plan(size: float) -> Plan:
    """
    a simple square plan

   0, size   size, size
     +-------+
     |       |
     |       |
     |       |
     +-------+
    0, 0     size, 0

    :return:
    """
    return rectangular_plan(size, size)


def plan_with_hole(width: float, depth: float) -> Plan:
    """
    Returns a plan with a hole
    0, depth   width, depth
     +--+--------+
     |  |        |
     |  +------+ | <- depth * 3/4
     |  |      | |
     |  +------+ | <- depth * 1/4
     +-----------+
    0, 0 ^     ^ width, 0
         |     |
  width * 1/4  width * 3/4
    """
    my_plan = rectangular_plan(width, depth)
    duct = [(width/4, depth/4), (width*3/4, depth/4), (width*3/4, depth*3/4), (width/4, depth*3/4)]
    my_plan.insert_space_from_boundary(duct, SPACE_CATEGORIES["duct"])
    return my_plan


@pytest.fixture
def weird_plan() -> Plan:
    """
    Creates a weirdly shaped plan

                 500, 1200       1200, 1200
                     +---------------+
                     |               |
                   |                 |
        0, 500   |                   |
           +--+ 200, 500   1000, 400 |
           |                   +-----+ 1200, 400
           |                   |
           |                   |
           +-------------------+
         0, 0              1000, 0

    :return:
    """

    boundaries = [(0, 0), (1000, 0), (1000, 400), (1200, 400), (1200, 1200),
                  (500, 1200), (200, 500), (0, 500)]
    plan = Plan("weird_shaped")
    plan.add_floor_from_boundary(boundaries)
    return plan


def test_min_size_fail():
    """
    Test max_size
    :return:
    """
    my_plan = square_plan(50.0)
    constraint = CONSTRAINTS["min_size"]
    assert not constraint.check(my_plan.empty_space)


def test_min_size_check():
    """
    Test max_size
    :return:
    """
    my_plan = square_plan(65.0)
    constraint = CONSTRAINTS["min_size"]
    assert constraint.check(my_plan.empty_space)


def test_max_size_seed_check():
    """
    Test max_size
    :return:
    """
    my_plan = square_plan(316.0)
    constraint = CONSTRAINTS["max_size_seed"]
    my_plan.empty_space.category = SPACE_CATEGORIES["seed"]
    assert constraint.check(my_plan.spaces[0])


def test_max_size_seed_fail():
    """
    Test max_size
    :return:
    """
    my_plan = square_plan(317.0)
    constraint = CONSTRAINTS["max_size_seed"]
    my_plan.empty_space.category = SPACE_CATEGORIES["seed"]
    assert not constraint.check(my_plan.spaces[0])


def test_max_depth_seed_check():
    """
    Test max_size
    :return:
    """
    my_plan = rectangular_plan(10, 900)
    constraint = CONSTRAINTS["max_size_seed"]
    my_plan.empty_space.category = SPACE_CATEGORIES["seed"]
    assert constraint.check(my_plan.spaces[0])


def test_max_depth_seed_fail():
    """
    Test max_size
    :return:
    """
    my_plan = rectangular_plan(7, 1050)
    my_plan.empty_space.category = SPACE_CATEGORIES["seed"]
    constraint = CONSTRAINTS["max_size_seed"]
    assert not constraint.check(my_plan.spaces[0])


def test_max_width_seed_check():
    """
    Test max_size
    :return:
    """
    my_plan = rectangular_plan(990, 10)
    my_plan.empty_space.category = SPACE_CATEGORIES["seed"]
    constraint = CONSTRAINTS["max_size_seed"]
    assert constraint.check(my_plan.spaces[0])


def test_max_width_seed_fail():
    """
    Test max_size
    :return:
    """
    my_plan = rectangular_plan(1005, 8)
    my_plan.empty_space.category = SPACE_CATEGORIES["seed"]
    constraint = CONSTRAINTS["max_size_seed"]
    assert not constraint.check(my_plan.spaces[0])


def test_weird_shape(weird_plan):
    """
    Test
    :param weird_plan:
    :return:
    """
    constraint = CONSTRAINTS["max_size"]
    assert not constraint.check(weird_plan.empty_space)
    assert weird_plan.empty_space.size.width == 1200
    assert weird_plan.empty_space.size.depth == 1200
    assert weird_plan.empty_space.size.area == (1200*1200 - 200*400 - 700*500 + 300*700/2)


def test_max_size_check_with_hole():
    """
    Test max_size
    :return:
    """
    my_plan = plan_with_hole(340, 340)  # area
    constraint = CONSTRAINTS["max_size"]
    assert my_plan.empty_space.area == 340*340*3/4
    assert my_plan.empty_space.size.width == 340
    assert my_plan.empty_space.size.depth == 340
    assert constraint.check(my_plan.empty_space)


def test_few_corners_check():
    """
    Test few corners constraint
    :return:
    """
    my_plan = square_plan(100)
    constraint = CONSTRAINTS["few_corners"]
    assert constraint.score(my_plan.empty_space) == 0
    assert constraint.check(my_plan.empty_space)


def test_few_corners_fail(weird_plan):
    """
    Test few corners constraint
    :return:
    """
    constraint = CONSTRAINTS["few_corners"]
    assert constraint.score(weird_plan.empty_space) == 4.0/8.0
    assert not constraint.check(weird_plan.empty_space)


def test_few_corners_check_with_hole():
    """
    Test few corners constraint
    :return:
    """
    my_plan = plan_with_hole(400, 400)
    constraint = CONSTRAINTS["few_corners"]
    assert constraint.check(my_plan.empty_space)
