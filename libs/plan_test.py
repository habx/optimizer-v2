# coding=utf-8

"""
Test module for plan module
"""

import pytest

from libs.plan import Plan, Space
from libs.category import SPACE_CATEGORIES
import libs.logsetup as ls
import libs.reader as reader


ls.init()

INPUT_FILES = reader.BLUEPRINT_INPUT_FILES


@pytest.mark.parametrize("input_file", INPUT_FILES)
def test_floor_plan(input_file):
    """
    Test. We create a simple grid on several real blue prints.
    :return:
    """
    plan = reader.create_plan_from_file(input_file)

    for empty_space in plan.empty_spaces:
        boundary_edges = list(empty_space.edges)
        
        for edge in boundary_edges:
            if edge.length > 30:
                empty_space.barycenter_cut(edge, 0)
                empty_space.barycenter_cut(edge, 1)

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

    duct_category = SPACE_CATEGORIES('duct')

    # add border duct
    plan = Plan().from_boundary(perimeter)
    plan.insert_space_from_boundary(duct, duct_category)

    # add inside duct
    inside_duct = [(600, 200), (800, 200), (800, 400), (600, 400)]
    plan.insert_space_from_boundary(inside_duct, duct_category)

    # add touching duct
    touching_duct = [(0, 800), (200, 800), (200, 1000), (0, 1000)]
    plan.insert_space_from_boundary(touching_duct, duct_category)

    # add separating duct
    separating_duct = [(700, 800), (1000, 700), (1000, 800), (800, 1000), (700, 1000)]
    plan.insert_space_from_boundary(separating_duct, duct_category)

    # add single touching point
    point_duct = [(0, 600), (200, 500), (200, 700)]
    plan.insert_space_from_boundary(point_duct, duct_category)

    # add complex duct
    complex_duct = [(300, 1000), (300, 600), (600, 600), (600, 800), (500, 1000),
                    (450, 800), (400, 1000), (350, 1000)]
    plan.insert_space_from_boundary(complex_duct, duct_category)

    assert plan.check()


def test_add_face():
    """
    Test. Create a new face, remove it, then add it again.
    :return:
    """
    perimeter = [(0, 0), (1000, 0), (1000, 1000), (0, 1000)]

    plan = Plan().from_boundary(perimeter)

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
    plan.insert_space_from_boundary(duct, SPACE_CATEGORIES('duct'))
    plan.empty_space.barycenter_cut(list(plan.empty_space.edges)[7])

    assert plan.check()


def test_add_overlapping_face():
    """
    Test. Create a new face, remove it, then add it again.
    :return:
    """
    perimeter = [(0, 0), (500, 0), (500, 500), (0, 500)]
    hole = [(200, 200), (300, 200), (300, 300), (200, 300)]
    hole_2 = [(50, 150), (150, 150), (150, 300), (50, 300)]

    plan = Plan().from_boundary(perimeter)

    plan.empty_space.face.insert_face_from_boundary(hole)
    face_to_remove = list(plan.empty_space.faces)[1]
    plan.empty_space.remove_face(face_to_remove)

    plan.empty_space.face.insert_face_from_boundary(hole_2)
    face_to_remove = list(plan.empty_space.faces)[1]
    plan.empty_space.remove_face(face_to_remove)

    assert plan.check()


def test_add_border_overlapping_face():
    """
    Test. Create a new face, remove it, then add it again.
    :return:
    """
    perimeter = [(0, 0), (500, 0), (500, 500), (0, 500)]
    hole = [(200, 200), (300, 200), (300, 300), (200, 300)]
    hole_2 = [(0, 150), (150, 150), (150, 300), (0, 300)]

    plan = Plan().from_boundary(perimeter)

    plan.empty_space.face.insert_face_from_boundary(hole)
    face_to_remove = list(plan.empty_space.faces)[1]
    plan.empty_space.remove_face(face_to_remove)

    plan.empty_space.face.insert_face_from_boundary(hole_2)
    face_to_remove = list(plan.empty_space.faces)[1]
    plan.empty_space.remove_face(face_to_remove)

    assert plan.check()


def test_add_face_touching_internal_edge():
    """
    Test. Create a new face, remove it, then add it again.
    :return:
    """
    perimeter = [(0, 0), (500, 0), (500, 500), (0, 500)]
    hole = [(200, 200), (300, 200), (300, 300), (200, 300)]
    hole_2 = [(50, 150), (150, 150), (150, 200), (50, 200)]
    hole_3 = [(50, 200), (150, 200), (150, 300), (50, 300)]

    plan = Plan().from_boundary(perimeter)

    plan.empty_space.face.insert_face_from_boundary(hole)
    face_to_remove = list(plan.empty_space.faces)[1]
    plan.empty_space.remove_face(face_to_remove)

    plan.empty_space.face.insert_face_from_boundary(hole_2)
    face_to_remove = list(plan.empty_space.faces)[1]
    plan.empty_space.remove_face(face_to_remove)

    plan.empty_space.face.insert_face_from_boundary(hole_3)
    face_to_remove = list(plan.empty_space.faces)[1]
    plan.empty_space.remove_face(face_to_remove)

    assert plan.check()


def test_add_two_face_touching_internal_edge_and_border():
    """
    Test. Create a new face, remove it, then add it again.
    :return:
    """
    perimeter = [(0, 0), (500, 0), (500, 500), (0, 500)]
    hole = [(200, 200), (300, 200), (300, 300), (200, 300)]
    hole_2 = [(0, 150), (150, 150), (150, 200), (0, 200)]
    hole_3 = [(0, 200), (150, 200), (150, 300), (0, 300)]

    plan = Plan().from_boundary(perimeter)

    plan.empty_space.face.insert_face_from_boundary(hole)
    face_to_remove = list(plan.empty_space.faces)[1]
    plan.empty_space.remove_face(face_to_remove)

    plan.empty_space.face.insert_face_from_boundary(hole_2)
    face_to_remove = list(plan.empty_space.faces)[1]
    plan.empty_space.remove_face(face_to_remove)

    plan.empty_space.face.insert_face_from_boundary(hole_3)
    face_to_remove = list(plan.empty_space.faces)[1]
    plan.empty_space.remove_face(face_to_remove)

    assert plan.check()


def test_insert_separating_wall():
    """
    Test
    :return:
    """
    perimeter = [(0, 0), (500, 0), (500, 500), (0, 500)]
    wall = [(250, 0), (300, 0), (300, 500), (250, 500)]
    plan = Plan('Plan_test_wall').from_boundary(perimeter)

    plan.insert_space_from_boundary(wall, category=SPACE_CATEGORIES('loadBearingWall'))

    assert plan.check()


def test_remove_middle_space():
    """
    Test
    :return:
    """
    perimeter = [(0, 0), (500, 0), (500, 500), (0, 500)]
    plan = Plan('my plan').from_boundary(perimeter)

    plan.empty_space.barycenter_cut(coeff=0.3)
    plan.empty_space.barycenter_cut()
    plan.empty_space.barycenter_cut()

    middle_face = list(plan.empty_space.faces)[1]

    plan.empty_space.remove_face(middle_face)
    plan.empty_space.add_face(middle_face)

    assert plan.check()


def test_remove_enclosing_space():
    """
    Test. Add various space inside an emptySpace.
    We test different cases such as an internal duct, a touching duct etc.
    TODO : split this in several tests.
    :return:
    """

    perimeter = [(0, 0), (1000, 0), (1000, 1000), (0, 1000)]

    # add border duct
    plan = Plan().from_boundary(perimeter)

    # add single touching point
    point_duct = [(0, 600), (200, 500), (200, 700)]
    plan.mesh.faces[0].insert_face_from_boundary(point_duct)
    hole_face = plan.mesh.faces[0]

    plan.empty_space.remove_face(hole_face)
    plan.empty_space.add_face(hole_face)

    assert plan.check()


def test_remove_u_space():
    """
    Test.
    :return:
    """

    perimeter = [(0, 0), (1000, 0), (1000, 1000), (0, 1000)]

    # add border duct
    plan = Plan().from_boundary(perimeter)

    # add single touching point
    point_duct = [(0, 600), (200, 500), (200, 700), (0, 700)]
    plan.mesh.faces[0].insert_face_from_boundary(point_duct)
    hole_face = plan.mesh.faces[1]

    plan.empty_space.remove_face(hole_face)
    plan.empty_space.add_face(hole_face)

    assert plan.check()


def test_remove_middle_b_space():
    """
    Test
    :return:
    """
    perimeter = [(0, 0), (500, 0), (500, 500), (200, 500), (200, 200), (0, 200)]
    plan = Plan('my plan').from_boundary(perimeter)
    list(plan.mesh.faces[0].edges)[4].barycenter_cut(0)

    duct = [(200, 200), (300, 200), (300, 300)]
    list(plan.spaces[0].faces)[1].insert_face_from_boundary(duct)

    plan.empty_space.remove_face(plan.mesh.faces[0])
    plan.empty_space.add_face(plan.mesh.faces[0])

    assert plan.check()


def test_remove_middle_u_space():
    """
    Test
    :return:
    """
    perimeter = [(0, 0), (500, 0), (500, 500), (200, 500), (200, 200), (0, 200)]
    plan = Plan('my plan').from_boundary(perimeter)
    list(plan.mesh.faces[0].edges)[4].barycenter_cut(0)

    duct = [(200, 300), (200, 150), (300, 150), (300, 300)]
    list(plan.spaces[0].faces)[1].insert_face_from_boundary(duct)

    plan.empty_space.remove_face(plan.mesh.faces[0])
    plan.empty_space.add_face(plan.mesh.faces[0])

    assert plan.check()


def test_remove_middle_c_space():
    """
    Test
    :return:
    """
    perimeter = [(0, 0), (500, 0), (500, 500), (200, 500), (200, 200), (0, 200)]
    plan = Plan('my plan').from_boundary(perimeter)
    list(plan.mesh.faces[0].edges)[4].barycenter_cut(0)

    duct = [(200, 200), (400, 200), (400, 400), (200, 400)]
    list(plan.spaces[0].faces)[1].insert_face_from_boundary(duct)

    plan.empty_space.remove_face(plan.mesh.faces[2])
    plan.empty_space.add_face(plan.mesh.faces[2])

    assert plan.check()


def test_remove_d_space():
    """
    Test
    :return:
    """
    perimeter = [(0, 0), (500, 0), (500, 500), (0, 500)]
    duct = [(0, 400), (100, 400), (100, 500), (50, 500)]

    plan = Plan('my plan').from_boundary(perimeter)
    plan.empty_space.face.insert_face_from_boundary(duct)

    plan.empty_space.remove_face(plan.mesh.faces[1])
    plan.empty_space.add_face(plan.mesh.faces[1])

    assert plan.check()


def test_remove_middle_e_space():
    """
    Test
    :return:
    """
    perimeter = [(0, 0), (800, 0), (800, 800), (0, 800)]
    duct = [(0, 0), (500, 0), (500, 300), (300, 300), (300, 500), (0, 500)]
    plan = Plan('my plan').from_boundary(perimeter)

    duct_2 = [(200, 100), (300, 100), (300, 300), (200, 300)]
    plan.empty_space.face.insert_face_from_boundary(duct)
    plan.mesh.faces[0].insert_face_from_boundary(duct_2)

    plan.empty_space.remove_face(plan.mesh.faces[0])
    plan.empty_space.add_face(plan.mesh.faces[0])
    plan.empty_space.remove_face(plan.mesh.faces[0])
    plan.empty_space.add_face(plan.mesh.faces[0])

    assert plan.check()


def test_bounding_box():
    """
    Test
    :return:
    """
    perimeter = [(100, 0), (150, 50), (400, 0), (600, 0), (500, 400), (400, 400), (400, 500),
                 (0, 500), (0, 400), (200, 400), (200, 200), (0, 200)]
    plan = Plan().from_boundary(perimeter)
    box = plan.empty_space.bounding_box((1, 0))
    assert box == (600.0, 500.0)


def test_remove_face_along_internal_edge():
    """
    Test
    :return:
    """
    perimeter = [(0, 0), (500, 0), (500, 500), (0, 500)]
    plan = Plan('my plan').from_boundary(perimeter)

    duct = [(150, 150), (300, 150), (300, 300), (150, 300)]
    plan.insert_space_from_boundary(duct)
    list(plan.mesh.faces[1].edges)[-1].pair.barycenter_cut(1)
    my_space = plan.empty_space
    my_space.remove_face(plan.mesh.faces[0])
    my_space.add_face(plan.mesh.faces[0])

    assert plan.check()


def test_remove_encapsulating_face():
    """
    Test
    :return:
    """
    perimeter = [(0, 0), (500, 0), (500, 500), (0, 500)]
    plan = Plan('my plan').from_boundary(perimeter)

    duct = [(150, 150), (300, 150), (300, 300), (150, 300)]
    plan.empty_space.face.insert_face_from_boundary(duct)
    list(plan.mesh.faces[1].edges)[0].pair.barycenter_cut(1)

    my_space = plan.empty_space
    my_space.remove_face(plan.mesh.faces[0])
    new_space = Space(plan, plan.mesh.faces[0].edge)
    plan.add_space(new_space)

    plan.mesh.faces[1].space.remove_face(plan.mesh.faces[1])
    new_space.add_face(plan.mesh.faces[1])

    assert plan.check()
