# coding=utf-8
"""
Test module for corridor module
"""

from libs.modelers.grid import GRIDS
from libs.modelers.seed import SEEDERS
from libs.modelers.corridor import Corridor
from libs.space_planner.circulation import Circulator, CostRules
from libs.specification.specification import Specification

CORRIDOR_RULES = {
    "layer_width": 25,
    "nb_layer": 5,
    "recursive_cut_length": 400,
    "width": 100,
    "penetration_length": 90,
    "layer_cut": True
}


def test_simple_grid():
    def get_following_edge(edge):
        return edge.next.pair.next

    def get_internal_edge(plan):
        internal_face = None
        for face in plan.mesh.faces:
            for edge in face.edges:
                if not edge.pair.face:
                    break
            else:
                internal_face = face
                break
        e = internal_face.edge.pair
        return e

    def build_a_path(plan):
        edge1 = get_internal_edge(plan)
        path_vert = [edge1.start]
        path_edge = [edge1]

        edge_list = []
        e = edge1
        for i in range(3):
            edge_list.append(e)
            e = get_following_edge(e)
            path_vert.append(e.start)
        path_vert.append(e.end)
        path_edge.append(e)
        edge_corner = e.pair.previous.pair
        edge_list.append(edge_corner)
        e = edge_corner
        for i in range(3):
            edge_list.append(e)
            e = get_following_edge(e)
            path_vert.append(e.start)
            path_edge.append(e)
        path_vert.append(e.end)
        return [path_vert, path_edge]

    ################ GRID ################
    from libs.modelers.grid_test import rectangular_plan
    simple_grid = GRIDS["simple_grid"]
    plan = rectangular_plan(500, 500)
    from libs.plan.category import LINEAR_CATEGORIES
    plan.insert_linear((400, 500), (300, 500), LINEAR_CATEGORIES["window"], plan.floor)
    plan = simple_grid.apply_to(plan)
    SEEDERS["simple_seeder"].apply_to(plan)

    ################ circulation path ################
    circulation_paths = build_a_path(plan)
    circulator = Circulator(plan, spec=Specification(), cost_rules=CostRules)
    for edge in circulation_paths[1]:
        circulator.directions[0][edge] = 1

    ################ corridor build ################
    corridor = Corridor(corridor_rules=CORRIDOR_RULES)

    corridor._clear()
    corridor.plan = plan
    corridor.circulator = circulator
    corridor.cut(circulation_paths[0])
    plan.check()
    corridor.grow(circulation_paths[0])
    plan.remove_null_spaces()
    plan.plot()
    plan.check()


test_simple_grid()
