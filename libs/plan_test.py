# coding=utf-8

"""
Test module for plan module
"""

import pytest

from libs.plan import Plan
from libs.category import space_categories
import libs.logsetup as ls
import libs.reader as reader


ls.init()

INPUT_FILES = reader.INPUT_FILES


@pytest.mark.parametrize("input_file", INPUT_FILES)
def test_floor_plan(input_file):
    """
    Test. We create a simple grid on several real blue prints.
    :return:
    """
    plan = reader.create_plan_from_file(input_file)
    """
    empty_space = plan.empty_space
    boundary_edges = list(empty_space.edges)

    for edge in boundary_edges:
        if edge.length > 30:
            empty_space.cut_at_barycenter(edge, 0)
            empty_space.cut_at_barycenter(edge, 1)
    """
    plan.plot()

    assert plan.check()


def test_add_duct_to_space():
    """
    Test. Add various space inside an emptySpace.
    We test different cases such as an internal duct, a touching duct etc.
    TODO : split this in several tests.
    :return:
    """

    perimeter = [(0, 0), (1000, 0), (1000, 1000), (0, 1000)]
    duct = [(200, 0), (400, 0), (400, 400), (200, 400)]

    duct_category = space_categories['duct']

    # add border duct
    plan = Plan().from_boundary(perimeter)
    plan.empty_space.insert_space(duct, duct_category)

    # add inside duct
    inside_duct = [(600, 200), (800, 200), (800, 400), (600, 400)]
    plan.empty_space.insert_space(inside_duct, duct_category)

    # add touching duct
    touching_duct = [(0, 800), (200, 800), (200, 1000), (0, 1000)]
    plan.empty_space.insert_space(touching_duct, duct_category)

    # add separating duct
    separating_duct = [(700, 800), (1000, 700), (1000, 800), (800, 1000), (700, 1000)]
    plan.empty_space.insert_space(separating_duct, duct_category)

    # add single touching point
    point_duct = [(0, 600), (200, 500), (200, 700)]
    plan.empty_space.insert_space(point_duct, duct_category)

    # add complex duct
    complex_duct = [(300, 1000), (300, 600), (600, 600), (600, 800), (500, 1000),
                    (450, 800), (400, 1000), (350, 1000)]
    plan.empty_space.insert_space(complex_duct, duct_category)

    assert plan.check()


def test_add_face():
    """
    Test. Create a new face, remove it, then add it again.
    :return:
    """
    perimeter = [(0, 0), (1000, 0), (1000, 1000), (0, 1000)]

    plan = Plan().from_boundary(perimeter)

    # add complex face
    complex_face = [(700, 800), (1000, 700), (1000, 800), (800, 1000), (700, 1000)]
    plan.empty_space.face.insert_face_from_boundary(complex_face)

    face_to_remove = list(plan.empty_space.faces)[1]
    plan.empty_space.remove_face(face_to_remove)

    plan.empty_space.add_face(face_to_remove)

    assert plan.check()


def test_cut_to_inside_space():
    """
    Test a cut to a space inside another space.
    The cut should stop and not cut the internal space
    :return:
    """
    perimeter = [(0, 0), (1000, 0), (1000, 1000), (0, 1000)]
    plan = Plan().from_boundary(perimeter)
    duct = [(200, 200), (800, 200), (800, 800), (200, 800)]
    plan.empty_space.insert_space(duct, space_categories['duct'])
    plan.empty_space.cut_at_barycenter(list(plan.empty_space.edges)[7])

    assert plan.check()
