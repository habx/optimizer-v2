# coding=utf-8
"""
Test Module of Mutation Module
"""
import pytest

from libs.plan import Plan
from libs.mutation import MUTATIONS, MUTATION_FACTORIES


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
    l_plan.plot()
    l_plan.mesh.plot()
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
    l_plan.plot()
    assert l_plan.check()


def test_swap_faces(l_plan):
    """
    Test
    :param l_plan:
    :return:
    """
    from libs.category import SPACE_CATEGORIES

    duct = [(1200, 500), (1200, 800), (1000, 800), (1000, 500)]
    duct_space = l_plan.insert_space_from_boundary(duct, SPACE_CATEGORIES["duct"])
    cut_mutation = MUTATION_FACTORIES["barycenter_cut"](0.5)
    cut_mutation.apply_to(duct_space.edge, duct_space)

    swap_mutation = MUTATIONS["swap_face"]
    edge = duct_space.edge.previous
    modified_spaces = swap_mutation.apply_to(edge, duct_space)
    swap_mutation.reverse(modified_spaces)

    l_plan.plot()

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

    l_plan.plot()

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

    l_plan.plot()

    assert l_plan.check()
