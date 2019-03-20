# coding=utf-8
"""
Solution Module Tests
"""

from libs.space_planner.solution import SolutionsCollector
from libs.plan.category import SPACE_CATEGORIES, LINEAR_CATEGORIES
from libs.specification import Specification
from libs.plan.plan import Plan
from libs.modelers.grid import GRIDS
from libs.modelers.seed import Seeder, GROWTH_METHODS, FILL_METHODS_HOMOGENEOUS
from libs.operators.selector import SELECTORS
from libs.io import reader
from libs.space_planner import SpacePlanner
from libs.modelers.shuffle import SHUFFLES


def test_solution_distance():
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

    collector.add_solution(plan1, {})
    collector.add_solution(plan2, {})

    assert collector.solutions[0].distance(collector.solutions[1]) == 25, "Wrong distance"

    plan3 = plan1.clone("3")
    plan3.spaces[1].category = SPACE_CATEGORIES["kitchen"]
    plan3.spaces[2].category = SPACE_CATEGORIES["entrance"]

    collector.add_solution(plan3, {})

    assert collector.solutions[0].distance(collector.solutions[2]) == 50, "Wrong distance"

    plan4 = plan1.clone("4")
    collector.add_solution(plan4, {})

    assert collector.solutions[0].distance(collector.solutions[3]) == 0, "Wrong distance"

    plan5 = plan1.clone("5")
    plan5.spaces[0].category = SPACE_CATEGORIES["bedroom"]
    plan5.spaces[1].category = SPACE_CATEGORIES["living"]
    plan5.spaces[2].category = SPACE_CATEGORIES["kitchen"]
    collector.add_solution(plan5, {})

    assert collector.solutions[0].distance(collector.solutions[4]) == 100, "Wrong distance"


def test_duplex():
    """
    Test
    :return:
    """
    boundaries = [(0, 500), (400, 500), (400, 0), (1500, 0), (1500, 700), (1000, 700), (1000, 800),
                  (0, 800)]
    boundaries_2 = [(0, 500), (400, 500), (400, 400), (1000, 400), (1000, 800), (0, 800)]

    plan = Plan("Solution_Tests_Multiple_floors")
    floor_1 = plan.add_floor_from_boundary(boundaries, floor_level=0)
    floor_2 = plan.add_floor_from_boundary(boundaries_2, floor_level=1)

    terrace_coords = [(400, 400), (400, 200), (1300, 200), (1300, 700), (1000, 700), (1000, 400)]
    plan.insert_space_from_boundary(terrace_coords, SPACE_CATEGORIES["terrace"], floor_1)
    garden_coords = [(400, 200), (400, 0), (1500, 0), (1500, 700), (1300, 700), (1300, 200)]
    plan.insert_space_from_boundary(garden_coords, SPACE_CATEGORIES["garden"], floor_1)
    duct_coords = [(350, 500), (400, 500), (400, 520), (350, 520)]
    plan.insert_space_from_boundary(duct_coords, SPACE_CATEGORIES["duct"], floor_1)
    duct_coords = [(350, 780), (400, 780), (400, 800), (350, 800)]
    plan.insert_space_from_boundary(duct_coords, SPACE_CATEGORIES["duct"], floor_1)
    hole_coords = [(400, 700), (650, 700), (650, 800), (400, 800)]
    plan.insert_space_from_boundary(hole_coords, SPACE_CATEGORIES["hole"], floor_1)
    plan.insert_linear((650, 800), (650, 700), LINEAR_CATEGORIES["startingStep"], floor_1)
    plan.insert_linear((275, 500), (340, 500), LINEAR_CATEGORIES["frontDoor"], floor_1)
    plan.insert_linear((550, 400), (750, 400), LINEAR_CATEGORIES["doorWindow"], floor_1)
    plan.insert_linear((1000, 450), (1000, 650), LINEAR_CATEGORIES["doorWindow"], floor_1)
    plan.insert_linear((0, 700), (0, 600), LINEAR_CATEGORIES["window"], floor_1)

    duct_coords = [(350, 500), (400, 500), (400, 520), (350, 520)]
    plan.insert_space_from_boundary(duct_coords, SPACE_CATEGORIES["duct"], floor_2)
    duct_coords = [(350, 780), (400, 780), (400, 800), (350, 800)]
    plan.insert_space_from_boundary(duct_coords, SPACE_CATEGORIES["duct"], floor_2)
    hole_coords = [(400, 700), (650, 700), (650, 800), (400, 800)]
    plan.insert_space_from_boundary(hole_coords, SPACE_CATEGORIES["hole"], floor_2)
    plan.insert_linear((650, 800), (650, 700), LINEAR_CATEGORIES["startingStep"], floor_2)
    plan.insert_linear((500, 400), (600, 400), LINEAR_CATEGORIES["window"], floor_2)
    plan.insert_linear((1000, 550), (1000, 650), LINEAR_CATEGORIES["window"], floor_2)
    plan.insert_linear((0, 700), (0, 600), LINEAR_CATEGORIES["window"], floor_2)

    GRIDS["sequence_grid"].apply_to(plan)

    plan.plot()

    seeder = Seeder(plan, GROWTH_METHODS).add_condition(SELECTORS['seed_duct'], 'duct')
    (seeder.plant()
     .grow(show=True)
     .shuffle(SHUFFLES['seed_square_shape_component_aligned'], show=True)
     .fill(FILL_METHODS_HOMOGENEOUS, (SELECTORS["farthest_couple_middle_space_area_min_100000"],
                                      "empty"), show=True)
     .fill(FILL_METHODS_HOMOGENEOUS, (SELECTORS["single_edge"], "empty"), recursive=True,
           show=True)
     .simplify(SELECTORS["fuse_small_cell_without_components"], show=True)
     .shuffle(SHUFFLES['seed_square_shape_component_aligned'], show=True)
     .simplify(SELECTORS["fuse_small_cell_without_components"], show=True)
     .shuffle(SHUFFLES['seed_square_shape_component_aligned'], show=True))

    plan.plot()

    spec = reader.create_specification_from_file("test_solution_duplex_setup.json")
    spec.plan = plan

    space_planner = SpacePlanner("test", spec)
    best_solutions = space_planner.solution_research()
