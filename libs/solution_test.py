# coding=utf-8
"""
Solution Module Tests
"""

from libs.solution import SolutionsCollector
from libs.category import SPACE_CATEGORIES
from libs.specification import Specification
from libs.plan import Plan


def test_distance_plans():
    """
    Test
    :return:
    """
    boundaries = [(0, 0), (1000, 0), (1000, 1000), (0, 1000)]

    spec = Specification()
    collector = SolutionsCollector(spec)

    plan1 = Plan("1")
    floor_1 = plan1.add_floor_from_boundary(boundaries, floor_level=0)
    living_coords = [(0, 0), (500, 0), (500, 1000), (0, 1000)]
    bedroom_coords = [(500, 0), (1000, 0), (1000, 500), (500, 500)]
    bathroom_coords = [(500, 500), (1000, 500), (1000, 1000), (500, 1000)]
    plan1.insert_space_from_boundary(living_coords, SPACE_CATEGORIES["living"], floor_1)
    plan1.insert_space_from_boundary(bedroom_coords, SPACE_CATEGORIES["bedroom"], floor_1)
    plan1.insert_space_from_boundary(bathroom_coords, SPACE_CATEGORIES["bathroom"], floor_1)
    plan1.remove_null_spaces()

    plan2 = plan1.clone("2")
    plan2.spaces[1].category = SPACE_CATEGORIES["kitchen"]
    plan2.spaces[2].category = SPACE_CATEGORIES["bedroom"]

    collector.add_solution(plan1)
    collector.add_solution(plan2)

    assert collector.solutions[0].distance(collector.solutions[1]) == 25, "Wrong distance"

    plan3 = plan1.clone("3")
    plan3.spaces[1].category = SPACE_CATEGORIES["kitchen"]
    plan3.spaces[2].category = SPACE_CATEGORIES["entrance"]

    collector.add_solution(plan3)

    assert collector.solutions[0].distance(collector.solutions[2]) == 50, "Wrong distance"

    plan4 = plan1.clone("4")
    collector.add_solution(plan4)

    assert collector.solutions[0].distance(collector.solutions[3]) == 0, "Wrong distance"

    plan5 = plan1.clone("5")
    plan5.spaces[0].category = SPACE_CATEGORIES["bedroom"]
    plan5.spaces[1].category = SPACE_CATEGORIES["living"]
    plan5.spaces[2].category = SPACE_CATEGORIES["kitchen"]
    collector.add_solution(plan5)

    assert collector.solutions[0].distance(collector.solutions[4]) == 100, "Wrong distance"
