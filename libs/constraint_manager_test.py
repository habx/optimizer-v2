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
        plan.insert_linear((0, 75 + i * 100), (0, 25 + i * 100),LINEAR_CATEGORIES["window"],
                           floor_1)
        plan.insert_linear((75 + i * 100, 1000), (25 + i * 100, 1000), LINEAR_CATEGORIES["window"],
                           floor_1)
    plan.plot()
    GRIDS["ortho_grid"].apply_to(plan)

    plan.plot()

    seeder = Seeder(plan, GROWTH_METHODS).add_condition(SELECTORS["seed_duct"], "duct")
    (seeder.plant()
     .grow()
     .shuffle(SHUFFLES["seed_square_shape"])
     .fill(FILL_METHODS, (SELECTORS["farthest_couple_middle_space_area_min_100000"],
                          "empty"))
     .fill(FILL_METHODS, (SELECTORS["single_edge"], "empty"), recursive=True)
     .simplify(SELECTORS["fuse_small_cell"])
     .shuffle(SHUFFLES["seed_square_shape"]))



