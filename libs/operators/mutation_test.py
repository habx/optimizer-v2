# coding=utf-8
"""
Test Module of Mutation Module
"""
import pytest

from libs.plan.plan import Plan
from libs.operators.mutation import MUTATIONS, MUTATION_FACTORIES


@pytest.fixture
def l_plan():
    """
    Creates a weirdly shaped plan

                 500, 1000       1200, 1200
                     +---------------+
                     |               |
                   |                 |
        0, 500   |                   |
           +--+ 200, 500   1000, 400 |
           |                   +-----+ 1200, 400
           |      500, 200     |
           |      ---*--       |
           |   ---      ---    |
           +---            ----+
         0, 0              1000, 0

    :return:
    """
    boundaries = [(0, 0), (500, 200), (1000, 0), (1000, 400), (1200, 400), (1200, 1200),
                  (500, 1000), (200, 500), (0, 500)]
    plan = Plan("L_shaped")
    plan.add_floor_from_boundary(boundaries)
    return plan


def test_ortho_cut(l_plan):
    """
    Test
    :return:
    """
    mutation = MUTATIONS["ortho_projection_cut"]
    mutation.apply_to(l_plan.empty_space.edge.next, l_plan.empty_space)
    assert l_plan.check()


def test_barycenter_cut(l_plan):
    """
    Test
    :param l_plan:
    :return:
    """
    mutation = MUTATION_FACTORIES["barycenter_cut"](0.5)
    mutation.apply_to(l_plan.empty_space.edge.next, l_plan.empty_space)
    assert l_plan.check()


def test_remove_edge(l_plan):
    """
    Test
    :param l_plan:
    :return:
    """
    cut_mutation = MUTATION_FACTORIES["barycenter_cut"](0.5)
    cut_mutation.apply_to(l_plan.empty_space.edge, l_plan.empty_space)
    edge = list(l_plan.empty_space.faces)[0].edge.next
    remove_mutation = MUTATIONS["remove_edge"]
    remove_mutation.apply_to(edge, l_plan.empty_space)
    assert l_plan.check()


def test_swap_faces(l_plan):
    """
    Test
    :param l_plan:
    :return:
    """
    from libs.plan.category import SPACE_CATEGORIES

    duct = [(1200, 500), (1200, 800), (1000, 800), (1000, 500)]
    duct_space = l_plan.insert_space_from_boundary(duct, SPACE_CATEGORIES["duct"])
    cut_mutation = MUTATION_FACTORIES["barycenter_cut"](0.5)
    cut_mutation.apply_to(duct_space.edge, duct_space)

    swap_mutation = MUTATIONS["swap_face"]
    edge = duct_space.edge.previous
    modified_spaces = swap_mutation.apply_to(edge, duct_space)
    swap_mutation.reverse(modified_spaces)

    assert l_plan.check()


def test_rectangle_cut(l_plan):
    """
    Test the mutation
    :param l_plan:
    :return:
    """
    rectangle_mutation = MUTATION_FACTORIES["rectangle_cut"](100, 100)
    edge = l_plan.mesh.boundary_edge.pair
    rectangle_mutation.apply_to(edge, l_plan.empty_space)

    assert l_plan.check()


def test_rectangle_cut_split(l_plan):
    """
    Test the mutation
    :param l_plan:
    :return:
    """
    rectangle_mutation = MUTATION_FACTORIES["rectangle_cut"](100, 100, 20)
    edge = l_plan.mesh.boundary_edge.pair
    rectangle_mutation.apply_to(edge, l_plan.empty_space)

    assert l_plan.check()


def test_slice_cut(l_plan):
    rectangle_mutation = MUTATION_FACTORIES["slice_cut"](100)
    edge = l_plan.mesh.boundary_edge.pair
    rectangle_mutation.apply_to(edge, l_plan.empty_space)

    assert l_plan.check()


def test_slice_cut_too_close():
    boundary = [(0, 0), (100, 0), (100, 100), (0, 100)]
    plan = Plan("Slice too close")
    plan.add_floor_from_boundary(boundary)
    rectangle_mutation = MUTATION_FACTORIES["slice_cut"](90)
    edge = plan.mesh.boundary_edge.pair
    rectangle_mutation.apply_to(edge, plan.empty_space)

    assert plan.check()


def test_remove_line(l_plan):
    """
    Remove a whole line
    :return:
    """
    from libs.modelers.grid import GRIDS

    plan = GRIDS["finer_ortho_grid"].apply_to(l_plan)
    edge = plan.mesh.boundary_edge.pair.next.next
    MUTATIONS["remove_line"].apply_to(edge, plan.empty_space)

    assert plan.check()


def test_add_aligned_edges(l_plan):
    """
    Add all aligned faces to the space
    :return:
    """
    from libs.modelers.grid import GRIDS
    from libs.plan.plan import Space

    plan = GRIDS["optimal_grid"].apply_to(l_plan)
    plan.empty_space.remove_face(plan.mesh.faces[0])
    plan.empty_space.remove_face(plan.mesh.faces[1])
    plan.empty_space.remove_face(plan.mesh.faces[6])
    other = Space(plan, plan.floor, plan.mesh.faces[0].edge)
    other.add_face(plan.mesh.faces[1])
    other.add_face(plan.mesh.faces[6])
    plan = GRIDS["finer_ortho_grid"].apply_to(l_plan)

    edge = list(other.exterior_edges)[15]
    MUTATIONS["add_aligned_face"].apply_to(edge, other)

    assert plan.check()
