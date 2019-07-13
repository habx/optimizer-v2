# coding=utf-8
"""
Constraint manager Module Tests
"""

from libs.plan.plan import Plan
from libs.plan.category import SPACE_CATEGORIES, LINEAR_CATEGORIES
from libs.space_planner.space_planner import SPACE_PLANNERS
from libs.specification.specification import Specification, Item, Size


def test_t1():
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

    bathroom_coords = [(0, 0), (200, 0), (200, 150), (0, 150)]
    plan_t1.insert_space_from_boundary(bathroom_coords, SPACE_CATEGORIES["seed"], floor_1)
    circulation_coords = [(200, 0), (400, 0), (400, 150), (200, 150)]
    plan_t1.insert_space_from_boundary(circulation_coords, SPACE_CATEGORIES["seed"], floor_1)
    living_coord = [(40, 150), (400, 150), (400, 600), (0, 600), (0, 200), (40, 200)]
    plan_t1.insert_space_from_boundary(living_coord, SPACE_CATEGORIES["seed"], floor_1)
    plan_t1.remove_null_spaces()

    bathroom = Item(SPACE_CATEGORIES["bathroom"], "xs", Size(area=25000), Size(area=35000))
    circulation = Item(SPACE_CATEGORIES["circulation"], "xs", Size(area=25000), Size(area=35000))
    living = Item(SPACE_CATEGORIES["living"], "s", Size(area=140000), Size(area=200000))
    spec = Specification("simple_test", plan_t1, [bathroom, circulation, living])

    space_planner_t1 = SPACE_PLANNERS["standard_space_planner"]
    best_solutions = space_planner_t1.apply_to(spec, 3,processes=1)
    assert len(space_planner_t1.solutions_collector.solutions) == 2


def test_t1_bis():
    """
    Test
    tested : opens_on_constraint
    :return:
    """
    boundaries = [(0, 0), (400, 0), (400, 400), (0, 400)]
    plan_t1_bis = Plan("SpacePlanner_simple_test_t1_bis")
    floor_1 = plan_t1_bis.add_floor_from_boundary(boundaries, floor_level=0)

    duct_coords = [(0, 150), (40, 150), (40, 200), (0, 200)]
    plan_t1_bis.insert_space_from_boundary(duct_coords, SPACE_CATEGORIES["duct"], floor_1)
    plan_t1_bis.insert_linear((210, 0), (290, 0), LINEAR_CATEGORIES["frontDoor"], floor_1)
    plan_t1_bis.insert_linear((175, 400), (25, 400), LINEAR_CATEGORIES["doorWindow"], floor_1)

    bathroom_coords = [(0, 0), (200, 0), (200, 150), (0, 150)]
    plan_t1_bis.insert_space_from_boundary(bathroom_coords, SPACE_CATEGORIES["seed"], floor_1)
    circulation_coords = [(200, 0), (400, 0), (400, 150), (200, 150)]
    plan_t1_bis.insert_space_from_boundary(circulation_coords, SPACE_CATEGORIES["seed"], floor_1)
    living_coord = [(40, 150), (200, 150), (200, 400), (0, 400), (0, 200), (40, 200)]
    plan_t1_bis.insert_space_from_boundary(living_coord, SPACE_CATEGORIES["seed"], floor_1)
    bedroom_coord = [(200, 150), (400, 150), (400, 400), (200, 400)]
    plan_t1_bis.insert_space_from_boundary(bedroom_coord, SPACE_CATEGORIES["seed"], floor_1)
    plan_t1_bis.remove_null_spaces()

    bathroom = Item(SPACE_CATEGORIES["bathroom"], "xs", Size(area=25000), Size(area=35000))
    circulation = Item(SPACE_CATEGORIES["circulation"], "xs", Size(area=25000), Size(area=35000))
    living = Item(SPACE_CATEGORIES["living"], "s", Size(area=40000), Size(area=60000), ["bedroom"])
    bedroom = Item(SPACE_CATEGORIES["bedroom"], "s", Size(area=40000), Size(area=60000),
                   ["living"])
    spec = Specification("simple_test", plan_t1_bis, [bathroom, circulation, living, bedroom])

    space_planner_t1_bis = SPACE_PLANNERS["standard_space_planner"]
    best_solutions = space_planner_t1_bis.apply_to(spec, 3,processes=1)
    assert len(space_planner_t1_bis.solutions_collector.solutions) == 1


def test_t3_balcony():
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
    plan_t3_balcony.insert_linear((275, 600), (25, 600), LINEAR_CATEGORIES["doorWindow"], floor_1)
    plan_t3_balcony.insert_linear((675, 600), (425, 600), LINEAR_CATEGORIES["doorWindow"], floor_1)
    plan_t3_balcony.insert_linear((975, 600), (725, 600), LINEAR_CATEGORIES["doorWindow"], floor_1)
    balcony_coords = [(400, 600), (700, 600), (700, 900), (400, 900)]
    plan_t3_balcony.insert_space_from_boundary(balcony_coords, SPACE_CATEGORIES["balcony"], floor_1)

    bathroom_coords = [(0, 0), (200, 0), (200, 150), (0, 150)]
    plan_t3_balcony.insert_space_from_boundary(bathroom_coords, SPACE_CATEGORIES["seed"], floor_1)
    circulation_coords = [(200, 0), (400, 0), (400, 150), (200, 150)]
    plan_t3_balcony.insert_space_from_boundary(circulation_coords, SPACE_CATEGORIES["seed"], floor_1)
    living_coord = [(40, 150), (400, 150), (400, 600), (0, 600), (0, 200), (40, 200)]
    plan_t3_balcony.insert_space_from_boundary(living_coord, SPACE_CATEGORIES["seed"], floor_1)
    bedroom_coord = [(400, 150), (700, 150), (700, 600), (400, 600)]
    plan_t3_balcony.insert_space_from_boundary(bedroom_coord, SPACE_CATEGORIES["seed"], floor_1)
    bedroom_coord = [(700, 150), (1000, 150), (1000, 600), (700, 600)]
    plan_t3_balcony.insert_space_from_boundary(bedroom_coord, SPACE_CATEGORIES["seed"], floor_1)
    plan_t3_balcony.remove_null_spaces()

    bathroom = Item(SPACE_CATEGORIES["bathroom"], "xs", Size(area=25000), Size(area=35000))
    circulation = Item(SPACE_CATEGORIES["circulation"], "xs", Size(area=25000), Size(area=35000))
    living = Item(SPACE_CATEGORIES["living"], "s", Size(area=100000), Size(area=150000))
    bedroom1 = Item(SPACE_CATEGORIES["bedroom"], "m", Size(area=100000), Size(area=150000))
    bedroom2 = Item(SPACE_CATEGORIES["bedroom"], "s", Size(area=150000), Size(area=200000))
    spec = Specification("simple_test", plan_t3_balcony, [bathroom, circulation, living, bedroom1,
                                                          bedroom2])

    space_planner_t3_balcony = SPACE_PLANNERS["standard_space_planner"]
    plan_t3_balcony.plot()
    best_solutions = space_planner_t3_balcony.apply_to(spec, 3,processes=1)

    assert len(space_planner_t3_balcony.solutions_collector.solutions) == 1


def test_t3_bis():
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
    plan_t3_bis.insert_linear((650, 600), (500, 600), LINEAR_CATEGORIES["doorWindow"], floor_1)
    plan_t3_bis.insert_linear((950, 600), (800, 600), LINEAR_CATEGORIES["doorWindow"], floor_1)

    bathroom_coords = [(0, 0), (200, 0), (200, 150), (0, 150)]
    plan_t3_bis.insert_space_from_boundary(bathroom_coords, SPACE_CATEGORIES["seed"], floor_1)
    circulation_coords = [(200, 0), (400, 0), (400, 150), (200, 150)]
    plan_t3_bis.insert_space_from_boundary(circulation_coords, SPACE_CATEGORIES["seed"], floor_1)
    living_coord = [(40, 150), (400, 150), (400, 600), (0, 600), (0, 200), (40, 200)]
    plan_t3_bis.insert_space_from_boundary(living_coord, SPACE_CATEGORIES["seed"], floor_1)
    bedroom_coord = [(400, 150), (700, 150), (700, 600), (400, 600)]
    plan_t3_bis.insert_space_from_boundary(bedroom_coord, SPACE_CATEGORIES["seed"], floor_1)
    bedroom1_coord = [(700, 150), (1000, 150), (1000, 600), (700, 600)]
    plan_t3_bis.insert_space_from_boundary(bedroom1_coord, SPACE_CATEGORIES["seed"], floor_1)
    plan_t3_bis.remove_null_spaces()

    bathroom = Item(SPACE_CATEGORIES["bathroom"], "xs", Size(area=25000), Size(area=35000))
    circulation = Item(SPACE_CATEGORIES["circulation"], "xs", Size(area=25000), Size(area=35000))
    living = Item(SPACE_CATEGORIES["living"], "m", Size(area=140000), Size(area=200000))
    bedroom1 = Item(SPACE_CATEGORIES["bedroom"], "s", Size(area=90000), Size(area=110000))
    bedroom2 = Item(SPACE_CATEGORIES["bedroom"], "s", Size(area=90000), Size(area=110000))
    spec = Specification("simple_test", plan_t3_bis, [bathroom, circulation, living, bedroom1,
                                                      bedroom2])

    space_planner_t3_bis = SPACE_PLANNERS["standard_space_planner"]
    best_solutions = space_planner_t3_bis.apply_to(spec, 3,processes=1)

    assert len(space_planner_t3_bis.solutions_collector.solutions) == 1




