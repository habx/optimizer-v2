# coding=utf-8
"""
Shuffle Module Test
"""
from libs.plan import Plan
from libs.shuffle import SHUFFLES
from libs.grid import GRIDS
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
    return Plan("square").from_boundary(boundaries)


def test_seed_square_shape():
    """
    Test
    :return:
    """
    simple_grid = GRIDS["simple_grid"]
    plan = rectangular_plan(500, 500)
    plan = simple_grid.apply_to(plan)

    new_space_boundary = [(0, 0), (62.5, 0), (62.5, 62.5), (0, 62.5)]
    plan.insert_space_from_boundary(new_space_boundary, SPACE_CATEGORIES["seed"])
    # plan.empty_space.category = SPACE_CATEGORIES["seed"]
    plan.plot()
    plan.check()
