# coding=utf-8
"""
Space planner Module Tests
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

test_files = [("Antony_A22.json", "Antony_A22_setup.json"),
              ("begles-carrelets_C304.json", "begles-carrelets_C304_setup.json")]


@pytest.mark.parametrize("input_file, input_setup", test_files)
def test_space_planner(input_file, input_setup):
    """
    Test
    :return:
    """

    plan = reader.create_plan_from_file(input_file)

    GRIDS["ortho_grid"].apply_to(plan)
    seeder = Seeder(plan, GROWTH_METHODS).add_condition(SELECTORS["seed_duct"], "duct")
    (seeder.plant()
     .grow()
     .shuffle(SHUFFLES["seed_square_shape"])
     .fill(FILL_METHODS, (SELECTORS["farthest_couple_middle_space_area_min_100000"], "empty"))
     .fill(FILL_METHODS, (SELECTORS["single_edge"], "empty"), recursive=True)
     .simplify(SELECTORS["fuse_small_cell"])
     .shuffle(SHUFFLES["seed_square_shape"]))

    spec = reader.create_specification_from_file(input_setup)
    spec.plan = plan

    space_planner = SpacePlanner("test", spec)
    best_solutions = space_planner.solution_research()


def test_duplex():
    """
    Test
    :return:
    """
    boundaries = [(0, 0), (1000, 0), (1000, 500), (200, 500)]
    boundaries_2 = [(0, 0), (800, 0), (800, 500), (200, 500)]

    plan = Plan("SpacePlanner_Tests_Multiple_floors")
    floor_1 = plan.add_floor_from_boundary(boundaries, floor_level=0)
    floor_2 = plan.add_floor_from_boundary(boundaries_2, floor_level=1)

    balcony_coords = [(800, 0), (1000, 0), (1000, 500), (800, 500)]
    plan.insert_space_from_boundary(balcony_coords, SPACE_CATEGORIES["balcony"], floor_1)
    duct_coords = [(600, 475), (650, 475), (650, 500), (600, 500)]
    plan.insert_space_from_boundary(duct_coords, SPACE_CATEGORIES["duct"], floor_1)
    hole_coords = [(450, 0), (600, 0), (600, 150), (450, 150)]
    plan.insert_space_from_boundary(hole_coords, SPACE_CATEGORIES["hole"], floor_1)
    plan.insert_linear((800, 50), (800, 250), LINEAR_CATEGORIES["doorWindow"], floor_1)
    plan.insert_linear((800, 350), (800, 450), LINEAR_CATEGORIES["doorWindow"], floor_1)
    plan.insert_linear((250, 0), (350, 0), LINEAR_CATEGORIES["window"], floor_1)
    plan.insert_linear((350, 500), (250, 500), LINEAR_CATEGORIES["window"], floor_1)
    plan.insert_linear((550, 500), (475, 500), LINEAR_CATEGORIES["frontDoor"], floor_1)
    plan.insert_linear((450, 150), (525, 150), LINEAR_CATEGORIES["startingStep"], floor_1)

    plan.insert_space_from_boundary(duct_coords, SPACE_CATEGORIES["duct"], floor_2)
    plan.insert_space_from_boundary(hole_coords, SPACE_CATEGORIES["hole"], floor_2)
    hole_coords = [(600, 0), (800, 0), (800, 300), (600, 300)]
    plan.insert_space_from_boundary(hole_coords, SPACE_CATEGORIES["hole"], floor_2)
    plan.insert_linear((100, 0), (200, 0), LINEAR_CATEGORIES["window"], floor_2)
    plan.insert_linear((300, 0), (400, 0), LINEAR_CATEGORIES["window"], floor_2)
    plan.insert_linear((525, 150), (600, 150), LINEAR_CATEGORIES["startingStep"], floor_2)

    GRIDS["simple_grid"].apply_to(plan)

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
     .empty(SELECTORS["corner_big_cell_area_70000"])
     .fill(FILL_METHODS_HOMOGENEOUS, (SELECTORS["farthest_couple_middle_space_area_min_50000"],
                                      "empty"), show=True)
     .simplify(SELECTORS["fuse_small_cell_without_components"], show=True)
     .shuffle(SHUFFLES['seed_square_shape_component_aligned'], show=True))

    plan.plot()
    spec = reader.create_specification_from_file("test_space_planner_duplex_setup.json")
    spec.plan = plan

    space_planner = SpacePlanner("test", spec)
    best_solutions = space_planner.solution_research()


def simple_test():
    """
    Test
    :return:
    """
    boundaries = [(0, 0), (400, 0), (400, 600), (0, 600)]

    #  Tested : inside_adjacency_constraint / area_constraint / components_adjacency_constraint /
    # item_attribution_constraint
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
    print(spec)

    space_planner_t1 = SpacePlanner("t1", spec)
    best_t1 = space_planner_t1.solution_research()
    assert len(best_t1) == 1

    #  tested : opens_on_constraint
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
    print(spec)

    space_planner_t1_bis = SpacePlanner("t1_bis", spec)
    best_t1_bis = space_planner_t1_bis.solution_research()
    assert len(best_t1_bis) == 1

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
    print(spec)
    print("plan_t3", plan_t3)
    space_planner_t3 = SpacePlanner("t3", spec)
    best_t3 = space_planner_t3.solution_research()
    assert len(space_planner_t3.solutions_collector.solutions) == 3

    #  Tested : externals spaces
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
    plan_t3_balcony.remove_null_spaces()
    plan_t3_balcony.plot()
    print("plan_t3_balcony", plan_t3_balcony)

    bathroom = Item(SPACE_CATEGORIES["bathroom"], "xs", Size(area=25000), Size(area=35000))
    entrance = Item(SPACE_CATEGORIES["entrance"], "xs", Size(area=25000), Size(area=35000))
    living = Item(SPACE_CATEGORIES["living"], "s", Size(area=140000), Size(area=200000))
    bedroom1 = Item(SPACE_CATEGORIES["bedroom"], "s", Size(area=100000), Size(area=140000))
    bedroom2 = Item(SPACE_CATEGORIES["bedroom"], "s", Size(area=100000), Size(area=140000))
    spec = Specification("simple_test", plan_t3_balcony, [bathroom, entrance, living, bedroom1, bedroom2])
    print(spec)

    space_planner_t3_balcony = SpacePlanner("t3_balcony", spec)
    best_t3 = space_planner_t3_balcony.solution_research()
    assert len(space_planner_t3_balcony.solutions_collector.solutions) == 1

    #  Tested : symmetry_breaker / windows_constraint
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
    spec = Specification("simple_test", plan_t3_bis, [bathroom, entrance, living, bedroom1, bedroom2])
    print(spec)

    space_planner_t3_bis = SpacePlanner("t3_bis", spec)
    best_t3_bis = space_planner_t3_bis.solution_research()
