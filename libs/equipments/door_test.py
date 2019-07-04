# coding=utf-8
"""
Test module for door module
"""

from libs.modelers.grid import GRIDS
from libs.modelers.seed import SEEDERS
from libs.equipments.doors import place_door_between_two_spaces


def test_simple_plan():
    ################ GRID ################
    from libs.modelers.grid_test import rectangular_plan
    simple_grid = GRIDS["simple_grid"]
    plan = rectangular_plan(500, 500)
    from libs.plan.category import LINEAR_CATEGORIES
    plan.insert_linear((400, 500), (300, 500), LINEAR_CATEGORIES["window"], plan.floor)
    plan = simple_grid.apply_to(plan)
    SEEDERS["directional_seeder"].apply_to(plan)

    sp_0 = list(plan.spaces)[0]
    for adj in sp_0.adjacent_spaces():
        place_door_between_two_spaces(sp_0, adj)

    plan.check()


test_simple_plan()
