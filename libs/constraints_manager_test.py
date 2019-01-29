# coding=utf-8
"""
Constraint manager Module Tests
"""

import pytest

from libs import reader
from libs.seed import Seeder, GROWTH_METHODS, FILL_METHODS, FILL_METHODS_HOMOGENEOUS
from libs.plan import Plan
from libs.grid import GRIDS
from libs.selector import SELECTORS
from libs.shuffle import SHUFFLES
from libs.category import SPACE_CATEGORIES, LINEAR_CATEGORIES
from libs.space_planner import SpacePlanner
from libs.specification import Specification, Item, Size


def test_basic_case():
    """
    Test
    :return:
    """
    boundaries = [(0, 0), (1000, 0), (1000, 1000), (0, 1000)]

    plan = Plan("Constraint_manager_basic_test")
    floor_1 = plan.add_floor_from_boundary(boundaries, floor_level=0)

    for i in range(10):
        plan.insert_linear((25 + i * 100, 0), (75 + i * 100, 0), LINEAR_CATEGORIES["window"],
                           floor_1)
        plan.insert_linear((1000, 25 + i * 100), (1000, 75 + i * 100), LINEAR_CATEGORIES["window"],
                           floor_1)
        plan.insert_linear((0, 75 + i * 100), (0, 25 + i * 100), LINEAR_CATEGORIES["window"],
                           floor_1)
        plan.insert_linear((75 + i * 100, 1000), (25 + i * 100, 1000), LINEAR_CATEGORIES["window"],
                           floor_1)

    GRIDS["ortho_grid"].apply_to(plan)

    plan.plot()

    seeder = Seeder(plan, GROWTH_METHODS).add_condition(SELECTORS["seed_duct"], "duct")
    (seeder.plant()
     .grow(show=True)
     .fill(FILL_METHODS_HOMOGENEOUS, (SELECTORS["farthest_couple_middle_space_area_min_50000"],
                                      "empty"), show=True)
     .fill(FILL_METHODS_HOMOGENEOUS, (SELECTORS["single_edge"], "empty"), recursive=True,
           show=True)
     .empty(SELECTORS["farthest_couple_middle_space_area_min_50000"])
     .fill(FILL_METHODS_HOMOGENEOUS, (SELECTORS["farthest_couple_middle_space_area_min_50000"],
                                      "empty"), show=True)
     )

    plan.plot()


def t1_test():
    """
    Test
    Tested : inside_adjacency_constraint / area_constraint / components_adjacency_constraint /
    item_attribution_constraint
    :return:
    """
    boundaries = [(0, 0), (400, 0), (400, 600), (0, 600)]

    plan_t1 = Plan("SpacePlanner_simple_test_t1")
    floor_1 = plan_t1.add_floor_from_boundary(boundaries, floor_level=0)

    duct_coords = [(0, 150), (40, 150), (40, 200), (0, 200)]
    plan_t1.insert_space_from_boundary(duct_coords, SPACE_CATEGORIES["duct"], floor_1)
    plan_t1.insert_linear((210, 0), (290, 0), LINEAR_CATEGORIES["frontDoor"], floor_1)
    plan_t1.insert_linear((250, 600), (50, 600), LINEAR_CATEGORIES["doorWindow"], floor_1)
    plan_t1.plot()

    bathroom_coords = [(0, 0), (200, 0), (200, 150), (0, 150)]
    plan_t1.insert_space_from_boundary(bathroom_coords, SPACE_CATEGORIES["seed"], floor_1)
    entrance_coords = [(200, 0), (400, 0), (400, 150), (200, 150)]
    plan_t1.insert_space_from_boundary(entrance_coords, SPACE_CATEGORIES["seed"], floor_1)
    living_coord = [(40, 150), (400, 150), (400, 600), (0, 600), (0, 200), (40, 200)]
    plan_t1.insert_space_from_boundary(living_coord, SPACE_CATEGORIES["seed"], floor_1)
    plan_t1.remove_null_spaces()
    plan_t1.plot()

    bathroom = Item(SPACE_CATEGORIES["bathroom"], "xs", Size(area=25000), Size(area=35000))
    entrance = Item(SPACE_CATEGORIES["entrance"], "xs", Size(area=25000), Size(area=35000))
    living = Item(SPACE_CATEGORIES["living"], "s", Size(area=140000), Size(area=200000))
    spec = Specification("simple_test", plan_t1, [bathroom, entrance, living])

    space_planner_t1 = SpacePlanner("t1", spec)
    best_t1 = space_planner_t1.solution_research()
    assert len(space_planner_t1.solutions_collector.solutions) == 1


def t1_bis_test():
    """
    Test
    tested : opens_on_constraint
    :return:
    """
    boundaries = [(0, 0), (400, 0), (400, 600), (0, 600)]
    plan_t1_bis = Plan("SpacePlanner_simple_test_t1_bis")
    floor_1 = plan_t1_bis.add_floor_from_boundary(boundaries, floor_level=0)

    duct_coords = [(0, 150), (40, 150), (40, 200), (0, 200)]
    plan_t1_bis.insert_space_from_boundary(duct_coords, SPACE_CATEGORIES["duct"], floor_1)
    plan_t1_bis.insert_linear((210, 0), (290, 0), LINEAR_CATEGORIES["frontDoor"], floor_1)
    plan_t1_bis.insert_linear((150, 600), (50, 600), LINEAR_CATEGORIES["doorWindow"], floor_1)
    plan_t1_bis.plot()

    bathroom_coords = [(0, 0), (200, 0), (200, 150), (0, 150)]
    plan_t1_bis.insert_space_from_boundary(bathroom_coords, SPACE_CATEGORIES["seed"], floor_1)
    entrance_coords = [(200, 0), (400, 0), (400, 150), (200, 150)]
    plan_t1_bis.insert_space_from_boundary(entrance_coords, SPACE_CATEGORIES["seed"], floor_1)
    living_coord = [(40, 150), (200, 150), (200, 600), (0, 600), (0, 200), (40, 200)]
    plan_t1_bis.insert_space_from_boundary(living_coord, SPACE_CATEGORIES["seed"], floor_1)
    bedroom_coord = [(200, 150), (400, 150), (400, 600), (200, 600)]
    plan_t1_bis.insert_space_from_boundary(bedroom_coord, SPACE_CATEGORIES["seed"], floor_1)
    plan_t1_bis.remove_null_spaces()
    plan_t1_bis.plot()

    bathroom = Item(SPACE_CATEGORIES["bathroom"], "xs", Size(area=25000), Size(area=35000))
    entrance = Item(SPACE_CATEGORIES["entrance"], "xs", Size(area=25000), Size(area=35000))
    living = Item(SPACE_CATEGORIES["living"], "s", Size(area=80000), Size(area=100000), ["bedroom"])
    bedroom = Item(SPACE_CATEGORIES["bedroom"], "s", Size(area=80000), Size(area=100000), ["living"])
    spec = Specification("simple_test", plan_t1_bis, [bathroom, entrance, living, bedroom])

    space_planner_t1_bis = SpacePlanner("t1_bis", spec)
    best_t1_bis = space_planner_t1_bis.solution_research()
    assert len(space_planner_t1_bis.solutions_collector.solutions) == 1


def t3_test():
    """
    Test
    :return:
    """
    #  Tested :
    boundaries = [(0, 0), (400, 0), (400, 150), (1000, 150), (1000, 600), (0, 600)]
    plan_t3 = Plan("SpacePlanner_simple_test_t3")
    floor_1 = plan_t3.add_floor_from_boundary(boundaries, floor_level=0)

    duct_coords = [(0, 150), (40, 150), (40, 200), (0, 200)]
    plan_t3.insert_space_from_boundary(duct_coords, SPACE_CATEGORIES["duct"], floor_1)
    plan_t3.insert_linear((210, 0), (290, 0), LINEAR_CATEGORIES["frontDoor"], floor_1)
    plan_t3.insert_linear((250, 600), (50, 600), LINEAR_CATEGORIES["doorWindow"], floor_1)
    plan_t3.insert_linear((650, 600), (450, 600), LINEAR_CATEGORIES["doorWindow"], floor_1)
    plan_t3.insert_linear((950, 600), (750, 600), LINEAR_CATEGORIES["doorWindow"], floor_1)
    plan_t3.plot()

    bathroom_coords = [(0, 0), (200, 0), (200, 150), (0, 150)]
    plan_t3.insert_space_from_boundary(bathroom_coords, SPACE_CATEGORIES["seed"], floor_1)
    entrance_coords = [(200, 0), (400, 0), (400, 150), (200, 150)]
    plan_t3.insert_space_from_boundary(entrance_coords, SPACE_CATEGORIES["seed"], floor_1)
    living_coord = [(40, 150), (400, 150), (400, 600), (0, 600), (0, 200), (40, 200)]
    plan_t3.insert_space_from_boundary(living_coord, SPACE_CATEGORIES["seed"], floor_1)
    bedroom_coord = [(400, 150), (700, 150), (700, 600), (400, 600)]
    plan_t3.insert_space_from_boundary(bedroom_coord, SPACE_CATEGORIES["seed"], floor_1)
    plan_t3.remove_null_spaces()
    plan_t3.plot()

    bathroom = Item(SPACE_CATEGORIES["bathroom"], "xs", Size(area=25000), Size(area=35000))
    entrance = Item(SPACE_CATEGORIES["entrance"], "xs", Size(area=25000), Size(area=35000))
    living = Item(SPACE_CATEGORIES["living"], "s", Size(area=140000), Size(area=200000))
    bedroom1 = Item(SPACE_CATEGORIES["bedroom"], "s", Size(area=100000), Size(area=140000))
    bedroom2 = Item(SPACE_CATEGORIES["bedroom"], "s", Size(area=100000), Size(area=140000))
    spec = Specification("simple_test", plan_t3, [bathroom, entrance, living, bedroom1, bedroom2])

    space_planner_t3 = SpacePlanner("t3", spec)
    best_t3 = space_planner_t3.solution_research()
    assert len(space_planner_t3.solutions_collector.solutions) == 3


def t3_balcony_test():
    """
    Test
    Tested : externals spaces
    :return:
    """
    boundaries = [(0, 0), (400, 0), (400, 150), (1000, 150), (1000, 600), (0, 600)]
    plan_t3_balcony = Plan("SpacePlanner_simple_test_t3_balcony")
    floor_1 = plan_t3_balcony.add_floor_from_boundary(boundaries, floor_level=0)

    duct_coords = [(0, 150), (40, 150), (40, 200), (0, 200)]
    plan_t3_balcony.insert_space_from_boundary(duct_coords, SPACE_CATEGORIES["duct"], floor_1)
    plan_t3_balcony.insert_linear((210, 0), (290, 0), LINEAR_CATEGORIES["frontDoor"], floor_1)
    plan_t3_balcony.insert_linear((250, 600), (50, 600), LINEAR_CATEGORIES["doorWindow"], floor_1)
    plan_t3_balcony.insert_linear((650, 600), (450, 600), LINEAR_CATEGORIES["doorWindow"], floor_1)
    plan_t3_balcony.insert_linear((950, 600), (750, 600), LINEAR_CATEGORIES["doorWindow"], floor_1)
    balcony_coords = [(0, 600), (400, 600), (400, 800), (0, 800)]
    plan_t3_balcony.insert_space_from_boundary(balcony_coords, SPACE_CATEGORIES["balcony"], floor_1)
    plan_t3_balcony.plot()

    bathroom_coords = [(0, 0), (200, 0), (200, 150), (0, 150)]
    plan_t3_balcony.insert_space_from_boundary(bathroom_coords, SPACE_CATEGORIES["seed"], floor_1)
    entrance_coords = [(200, 0), (400, 0), (400, 150), (200, 150)]
    plan_t3_balcony.insert_space_from_boundary(entrance_coords, SPACE_CATEGORIES["seed"], floor_1)
    living_coord = [(40, 150), (400, 150), (400, 600), (0, 600), (0, 200), (40, 200)]
    plan_t3_balcony.insert_space_from_boundary(living_coord, SPACE_CATEGORIES["seed"], floor_1)
    bedroom_coord = [(400, 150), (700, 150), (700, 600), (400, 600)]
    plan_t3_balcony.insert_space_from_boundary(bedroom_coord, SPACE_CATEGORIES["seed"], floor_1)
    bedroom_coord = [(700, 150), (1000, 150), (1000, 600), (700, 600)]
    plan_t3_balcony.insert_space_from_boundary(bedroom_coord, SPACE_CATEGORIES["seed"], floor_1)
    plan_t3_balcony.remove_null_spaces()
    plan_t3_balcony.plot()

    bathroom = Item(SPACE_CATEGORIES["bathroom"], "xs", Size(area=25000), Size(area=35000))
    entrance = Item(SPACE_CATEGORIES["entrance"], "xs", Size(area=25000), Size(area=35000))
    living = Item(SPACE_CATEGORIES["living"], "s", Size(area=140000), Size(area=200000))
    bedroom1 = Item(SPACE_CATEGORIES["bedroom"], "s", Size(area=100000), Size(area=140000))
    bedroom2 = Item(SPACE_CATEGORIES["bedroom"], "s", Size(area=100000), Size(area=140000))
    spec = Specification("simple_test", plan_t3_balcony, [bathroom, entrance, living, bedroom1,
                                                          bedroom2])

    space_planner_t3_balcony = SpacePlanner("t3_balcony", spec)
    best_t3 = space_planner_t3_balcony.solution_research()
    assert len(space_planner_t3_balcony.solutions_collector.solutions) == 1


def t3_bis_test():
    """
    Test
    Tested : symmetry_breaker / windows_constraint
    :return:
    """
    boundaries = [(0, 0), (400, 0), (400, 150), (1000, 150), (1000, 600), (0, 600)]
    plan_t3_bis = Plan("SpacePlanner_simple_test_t3_bis")
    floor_1 = plan_t3_bis.add_floor_from_boundary(boundaries, floor_level=0)

    duct_coords = [(0, 150), (40, 150), (40, 200), (0, 200)]
    plan_t3_bis.insert_space_from_boundary(duct_coords, SPACE_CATEGORIES["duct"], floor_1)
    plan_t3_bis.insert_linear((210, 0), (290, 0), LINEAR_CATEGORIES["frontDoor"], floor_1)
    plan_t3_bis.insert_linear((350, 600), (50, 600), LINEAR_CATEGORIES["doorWindow"], floor_1)
    plan_t3_bis.insert_linear((600, 600), (500, 600), LINEAR_CATEGORIES["doorWindow"], floor_1)
    plan_t3_bis.insert_linear((900, 600), (800, 600), LINEAR_CATEGORIES["doorWindow"], floor_1)
    plan_t3_bis.plot()

    bathroom_coords = [(0, 0), (200, 0), (200, 150), (0, 150)]
    plan_t3_bis.insert_space_from_boundary(bathroom_coords, SPACE_CATEGORIES["seed"], floor_1)
    entrance_coords = [(200, 0), (400, 0), (400, 150), (200, 150)]
    plan_t3_bis.insert_space_from_boundary(entrance_coords, SPACE_CATEGORIES["seed"], floor_1)
    living_coord = [(40, 150), (400, 150), (400, 600), (0, 600), (0, 200), (40, 200)]
    plan_t3_bis.insert_space_from_boundary(living_coord, SPACE_CATEGORIES["seed"], floor_1)
    bedroom_coord = [(400, 150), (700, 150), (700, 600), (400, 600)]
    plan_t3_bis.insert_space_from_boundary(bedroom_coord, SPACE_CATEGORIES["seed"], floor_1)
    plan_t3_bis.remove_null_spaces()
    plan_t3_bis.plot()

    bathroom = Item(SPACE_CATEGORIES["bathroom"], "xs", Size(area=25000), Size(area=35000))
    entrance = Item(SPACE_CATEGORIES["entrance"], "xs", Size(area=25000), Size(area=35000))
    living = Item(SPACE_CATEGORIES["living"], "s", Size(area=140000), Size(area=200000))
    bedroom1 = Item(SPACE_CATEGORIES["bedroom"], "s", Size(area=100000), Size(area=140000))
    bedroom2 = Item(SPACE_CATEGORIES["bedroom"], "s", Size(area=100000), Size(area=140000))
    spec = Specification("simple_test", plan_t3_bis, [bathroom, entrance, living, bedroom1,
                                                      bedroom2])

    space_planner_t3_bis = SpacePlanner("t3_bis", spec)
    best_t3_bis = space_planner_t3_bis.solution_research()
    assert len(space_planner_t3_bis.solutions_collector.solutions) == 1




