# coding=utf-8

"""
Test module for plan module
"""

import pytest

from libs.plan.plan import Plan, Space
from libs.plan.category import SPACE_CATEGORIES, LINEAR_CATEGORIES
from libs.io import reader, reader_test

INPUT_FILES = reader_test.BLUEPRINT_INPUT_FILES


@pytest.mark.parametrize("input_file", INPUT_FILES)
def test_read_floor_plan(input_file):
    """
    Test. We create a simple grid on several real blue prints.
    :return:
    """
    plan = reader.create_plan_from_file(input_file)
    plan.plot()

    assert plan.check()


def rectangular_plan(width: float, depth: float) -> Plan:
    """
    a simple rectangular plan

   0, depth   width, depth
     +------------+
     |            |
     |            |
     |            |
     +------------+
    0, 0     width, 0

    :return:
    """
    boundaries = [(0, 0), (width, 0), (width, depth), (0, depth)]
    plan = Plan("square")
    plan.add_floor_from_boundary(boundaries)
    return plan


def test_serialization():
    """
    Test
    :return:
    """
    import libs.io.writer as writer

    plan = reader.create_plan_from_file(INPUT_FILES[0])

    new_plan = Plan("from_saved_data")

    for i in range(100):
        serialized_data = plan.serialize()
        writer.save_plan_as_json(serialized_data)

        new_serialized_data = reader.get_plan_from_json(serialized_data["name"] + ".json")
        new_plan = Plan("from_saved_data").deserialize(new_serialized_data)

    new_plan.plot()
    assert new_plan.check()


def test_multiple_floors():
    """
    Test a plan with multiple floors
    :return:
    """
    perimeter = [(0, 0), (1000, 0), (1000, 1000), (0, 1000)]
    perimeter_2 = [(0, 0), (800, 0), (800, 800), (0, 800)]
    plan = Plan("Multiple Floors")
    plan.add_floor_from_boundary(perimeter)
    plan.add_floor_from_boundary(perimeter_2, 1)

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

    duct_category = SPACE_CATEGORIES['duct']

    # add border duct
    plan = Plan()
    plan.add_floor_from_boundary(perimeter)
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


def test_add_touching_duct_to_space():
    """
    Test
    :return:
    """

    perimeter = [(0, 0), (1000, 0), (1000, 1000), (0, 1000)]
    duct_category = SPACE_CATEGORIES['duct']

    # add border duct
    plan = Plan()
    plan.add_floor_from_boundary(perimeter)

    # add touching duct
    touching_duct = [(0, 800), (200, 800), (200, 1000), (0, 1000)]
    plan.insert_space_from_boundary(touching_duct, duct_category)

    assert plan.check()


def test_remove_edge_from_space():
    """
    Test
    :return:
    """
    perimeter = [(0, 0), (1000, 0), (1000, 1000), (0, 1000)]

    # add border duct
    plan = Plan("Remove edge from space")
    plan.add_floor_from_boundary(perimeter)

    # add touching duct
    touching_duct = [(0, 800), (200, 800), (200, 1000), (0, 1000)]
    plan.empty_space.insert_face_from_boundary(touching_duct)

    plan.plot()

    edge = list(plan.mesh.faces[0].edges)[1]
    plan.empty_space.remove_internal_edge(edge)

    # plan.plot()

    assert plan.check()


def test_add_face():
    """
    Test. Create a new face, remove it, then add it again.
    :return:
    """
    perimeter = [(0, 0), (1000, 0), (1000, 1000), (0, 1000)]

    plan = Plan()
    plan.add_floor_from_boundary(perimeter)

    complex_face = [(700, 800), (1000, 700), (1000, 800), (800, 1000), (700, 1000)]
    plan.empty_space.insert_face_from_boundary(complex_face)

    face_to_remove = list(plan.empty_space.faces)[1]
    plan.empty_space.remove_face(face_to_remove)

    plan.empty_space.add_face(face_to_remove)

    plan.plot()

    assert plan.check()


def test_cut_to_inside_space():
    """
    Test a cut to a space inside another space.
    The cut should stop and not cut the internal space
    :return:
    """
    perimeter = [(0, 0), (1000, 0), (1000, 1000), (0, 1000)]
    plan = Plan()
    plan.add_floor_from_boundary(perimeter)
    duct = [(200, 200), (800, 200), (800, 800), (200, 800)]
    plan.insert_space_from_boundary(duct, SPACE_CATEGORIES['duct'])
    plan.empty_space.barycenter_cut(list(plan.empty_space.edges)[0])

    plan.plot()

    assert plan.check()


def test_add_overlapping_face():
    """
    Test. Create a new face, remove it, then add it again.
    :return:
    """
    perimeter = [(0, 0), (500, 0), (500, 500), (0, 500)]
    hole = [(200, 200), (300, 200), (300, 300), (200, 300)]
    hole_2 = [(50, 150), (150, 150), (150, 300), (50, 300)]

    plan = Plan()
    plan.add_floor_from_boundary(perimeter)

    plan.empty_space.insert_face_from_boundary(hole)
    face_to_remove = list(plan.empty_space.faces)[1]
    plan.empty_space.remove_face(face_to_remove)

    plan.empty_space.insert_face_from_boundary(hole_2)
    face_to_remove = list(plan.empty_space.faces)[1]
    plan.empty_space.remove_face(face_to_remove)

    plan.plot()

    assert plan.check()


def test_add_border_overlapping_face():
    """
    Test. Create a new face, remove it, then add it again.
    :return:
    """
    perimeter = [(0, 0), (500, 0), (500, 500), (0, 500)]
    hole = [(200, 200), (300, 200), (300, 300), (200, 300)]
    hole_2 = [(0, 150), (150, 150), (150, 300), (0, 300)]

    plan = Plan("test_add_border_overlapping_face")
    plan.add_floor_from_boundary(perimeter)

    plan.empty_space.insert_face_from_boundary(hole)
    face_to_remove = list(plan.empty_space.faces)[1]
    plan.empty_space.remove_face(face_to_remove)

    plan.empty_space.insert_face_from_boundary(hole_2)
    face_to_remove = list(plan.empty_space.faces)[1]
    plan.empty_space.remove_face(face_to_remove)

    plan.plot()

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

    plan = Plan("add_face_touching_internal_edge")
    plan.add_floor_from_boundary(perimeter)

    plan.empty_space.insert_face_from_boundary(hole)
    face_to_remove = list(plan.empty_space.faces)[1]
    plan.empty_space.remove_face(face_to_remove)

    plan.empty_space.insert_face_from_boundary(hole_2)
    face_to_remove = list(plan.empty_space.faces)[1]
    plan.empty_space.remove_face(face_to_remove)

    plan.empty_space.insert_face_from_boundary(hole_3)
    face_to_remove = list(plan.empty_space.faces)[1]
    plan.empty_space.remove_face(face_to_remove)

    plan.plot()

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

    plan = Plan()
    plan.add_floor_from_boundary(perimeter)

    plan.empty_space.insert_face_from_boundary(hole)
    face_to_remove = list(plan.empty_space.faces)[1]
    plan.empty_space.remove_face(face_to_remove)

    plan.empty_space.insert_face_from_boundary(hole_2)
    face_to_remove = list(plan.empty_space.faces)[0]
    plan.empty_space.remove_face(face_to_remove)

    plan.empty_space.insert_face_from_boundary(hole_3)
    face_to_remove = list(plan.empty_space.faces)[1]
    plan.empty_space.remove_face(face_to_remove)

    plan.plot()

    assert plan.check()


def test_insert_separating_wall():
    """
    Test
    :return:
    """
    perimeter = [(0, 0), (500, 0), (500, 500), (0, 500)]
    wall = [(250, 0), (300, 0), (300, 500), (250, 500)]
    plan = Plan('Plan_test_wall')
    plan.add_floor_from_boundary(perimeter)

    plan.insert_space_from_boundary(wall, category=SPACE_CATEGORIES['loadBearingWall'])

    plan.plot()

    assert plan.check()


def test_remove_middle_space():
    """
    Test
    :return:
    """
    perimeter = [(0, 0), (500, 0), (500, 500), (0, 500)]
    plan = Plan('my plan')
    plan.add_floor_from_boundary(perimeter)

    plan.empty_space.barycenter_cut(coeff=0.3)
    plan.empty_space.barycenter_cut()
    plan.empty_space.barycenter_cut()

    middle_face = list(plan.empty_space.faces)[1]

    plan.empty_space.remove_face(middle_face)
    plan.empty_space.add_face(middle_face)

    plan.plot()

    assert plan.check()


def test_remove_enclosing_space():
    """
    Test. Add various space inside an emptySpace.
    We test different cases such as an internal duct, a touching duct etc.
    :return:
    """

    perimeter = [(0, 0), (1000, 0), (1000, 1000), (0, 1000)]

    # add border duct
    plan = Plan()
    plan.add_floor_from_boundary(perimeter)

    # add single touching point
    point_duct = [(0, 600), (200, 500), (200, 700)]
    plan.empty_space.insert_face_from_boundary(point_duct)
    hole_face = plan.mesh.faces[0]

    plan.empty_space.remove_face(hole_face)
    plan.empty_space.add_face(hole_face)

    plan.plot()

    assert plan.check()


def test_remove_u_space():
    """
    Test.
    :return:
    """

    perimeter = [(0, 0), (1000, 0), (1000, 1000), (0, 1000)]

    # add border duct
    plan = Plan()
    plan.add_floor_from_boundary(perimeter)

    # add single touching point
    point_duct = [(0, 600), (200, 500), (200, 700), (0, 700)]
    plan.empty_space.insert_face_from_boundary(point_duct)
    hole_face = plan.mesh.faces[1]

    plan.empty_space.remove_face(hole_face)
    plan.empty_space.add_face(hole_face)

    plan.plot()

    assert plan.check()


def test_remove_middle_b_space():
    """
    Test
    :return:
    """
    perimeter = [(0, 0), (500, 0), (500, 500), (200, 500), (200, 200), (0, 200)]
    plan = Plan('my plan')
    plan.add_floor_from_boundary(perimeter)
    list(plan.empty_space.edges)[4].barycenter_cut(0)

    duct = [(200, 200), (300, 200), (300, 300)]
    plan.empty_space.insert_face_from_boundary(duct)

    plan.empty_space.remove_face(plan.mesh.faces[0])
    plan.empty_space.add_face(plan.mesh.faces[0])

    plan.plot()

    assert plan.check()


def test_remove_middle_u_space():
    """
    Test
    :return:
    """
    perimeter = [(0, 0), (500, 0), (500, 500), (200, 500), (200, 200), (0, 200)]
    plan = Plan('my plan')
    plan.add_floor_from_boundary(perimeter)
    list(plan.mesh.faces[0].edges)[4].barycenter_cut(0)

    duct = [(200, 300), (200, 150), (300, 150), (300, 300)]
    plan.empty_space.insert_face_from_boundary(duct)

    plan.empty_space.remove_face(plan.mesh.faces[0])
    plan.empty_space.add_face(plan.mesh.faces[0])

    plan.plot()

    assert plan.check()


def test_remove_middle_c_space():
    """
    Test
    :return:
    """
    perimeter = [(0, 0), (500, 0), (500, 500), (200, 500), (200, 200), (0, 200)]
    plan = Plan('my plan')
    plan.add_floor_from_boundary(perimeter)
    list(plan.mesh.faces[0].edges)[4].barycenter_cut(0)

    duct = [(200, 200), (400, 200), (400, 400), (200, 400)]
    plan.empty_space.insert_face_from_boundary(duct)

    plan.empty_space.remove_face(plan.mesh.faces[1])
    plan.empty_space.add_face(plan.mesh.faces[1])

    plan.plot()

    assert plan.check()


def test_remove_d_space():
    """
    Test
    :return:
    """
    perimeter = [(0, 0), (500, 0), (500, 500), (0, 500)]
    duct = [(0, 400), (100, 400), (100, 500), (50, 500)]

    plan = Plan('my plan')
    plan.add_floor_from_boundary(perimeter)
    plan.empty_space.insert_face_from_boundary(duct)

    plan.empty_space.remove_face(plan.mesh.faces[1])
    plan.empty_space.add_face(plan.mesh.faces[1])

    plan.plot()

    assert plan.check()


def test_remove_middle_e_space():
    """
    Test
    :return:
    """
    perimeter = [(0, 0), (800, 0), (800, 800), (0, 800)]
    duct = [(0, 0), (500, 0), (500, 300), (300, 300), (300, 500), (0, 500)]
    plan = Plan('remove_middle_e_space')
    plan.add_floor_from_boundary(perimeter)

    duct_2 = [(200, 100), (300, 100), (300, 300), (200, 300)]
    plan.empty_space.insert_face_from_boundary(duct)
    plan.empty_space.insert_face_from_boundary(duct_2)

    plan.empty_space.remove_face(plan.mesh.faces[0])
    plan.empty_space.add_face(plan.mesh.faces[0])

    assert plan.check()


def test_remove_face_between_holes():
    """
    Remove a face between two holes : creating a large holes.
    The reference edges of the space should be changed accordingly
    In this case the face is internal.
    Example :
    +-----------------------------------+
    |                SPACE              |
    |     +--------+       +-------+    |
    |     |        |       |       |    |
    |     |        +-------+       |    |
    |     | HOLE 1 | FACE  | HOLE 2|    |
    |     |        +-------+       |    |
    |     |        |       |       |    |
    |     +--------+       +-------+    |
    |                                   |
    +-----------------------------------+

    :return:
    """
    perimeter = [(0, 0), (800, 0), (800, 800), (0, 800)]
    hole_1 = [(100, 200), (300, 200), (300, 600), (100, 600)]
    hole_2 = [(500, 200), (700, 200), (700, 600), (500, 600)]
    face = [(300, 300), (500, 300), (500, 500), (300, 500)]
    plan = Plan('face_between_holes')
    plan.add_floor_from_boundary(perimeter)

    face_1 = plan.empty_space.insert_face_from_boundary(hole_1)
    plan.empty_space.remove_face(face_1)

    face_2 = plan.empty_space.insert_face_from_boundary(hole_2)
    plan.empty_space.remove_face(face_2)

    face_3 = plan.empty_space.insert_face_from_boundary(face)
    plan.empty_space.remove_face(face_3)

    assert len(list(plan.empty_space.holes_reference_edge)) == 1
    assert plan.check()


def test_remove_face_between_holes_2():
    """
    Remove a face between two holes : creating a large holes.
    The reference edges of the space should be changed accordingly.
    In this case the face shares a boundary with the space
    Example :
    +--------+----------------+------+
    |        |     FACE       |      |
    |  +-----+---+------+-----+---+  |
    |  |         |      |         |  |
    |  |         |      |         |  |
    |  |         |      |         |  |
    |  | HOLE 1  |      | HOLE 2  |  |
    |  |         |      |         |  |
    |  |         |      |         |  |
    |  |         |      |         |  |
    |  +---------+      +---------+  |
    |              SPACE             |
    +--------------------------------+

    :return:
    """
    perimeter = [(0, 0), (800, 0), (800, 800), (0, 800)]
    hole_1 = [(100, 200), (300, 200), (300, 600), (100, 600)]
    hole_2 = [(500, 200), (700, 200), (700, 600), (500, 600)]
    face = [(200, 600), (600, 600), (600, 800), (200, 800)]
    plan = Plan('face_between_holes')
    plan.add_floor_from_boundary(perimeter)

    face_1 = plan.empty_space.insert_face_from_boundary(hole_1)
    plan.empty_space.remove_face(face_1)

    face_2 = plan.empty_space.insert_face_from_boundary(hole_2)
    plan.empty_space.remove_face(face_2)

    face_3 = plan.empty_space.insert_face_from_boundary(face)
    plan.empty_space.remove_face(face_3)

    assert len(list(plan.empty_space.holes_reference_edge)) == 0
    assert plan.check()


def test_create_hole():
    """
    Test
    :return:
    """
    perimeter = [(0, 0), (800, 0), (800, 800), (0, 800)]
    duct = [(200, 200), (400, 200), (400, 400), (200, 400)]
    plan = Plan('create_hole')
    plan.add_floor_from_boundary(perimeter)
    plan.empty_space.insert_face_from_boundary(duct)

    plan.empty_space.remove_face(plan.mesh.faces[1])
    plan.empty_space.add_face(plan.mesh.faces[1])

    assert plan.check()


def test_bounding_box():
    """
    Test
    :return:
    """
    perimeter = [(100, 0), (150, 50), (400, 0), (600, 0), (500, 400), (400, 400), (400, 500),
                 (0, 500), (0, 400), (200, 400), (200, 200), (0, 200)]
    plan = Plan()
    plan.add_floor_from_boundary(perimeter)
    box = plan.empty_space.bounding_box((1, 0))
    assert box == (600.0, 500.0)


def test_remove_face_along_internal_edge():
    """
    Test
    :return:
    """
    perimeter = [(0, 0), (500, 0), (500, 500), (0, 500)]
    plan = Plan('my plan')
    plan.add_floor_from_boundary(perimeter)

    duct = [(150, 150), (300, 150), (300, 300), (150, 300)]
    plan.insert_space_from_boundary(duct)
    plan.empty_space.barycenter_cut(list(plan.mesh.faces[1].edges)[-1].pair, 1)
    my_space = plan.empty_space
    my_space.remove_face(plan.mesh.faces[0])
    my_space.add_face(plan.mesh.faces[0])

    plan.plot()

    assert plan.check()


def test_remove_encapsulating_face():
    """
    Test
    :return:
    """
    perimeter = [(0, 0), (500, 0), (500, 500), (0, 500)]
    plan = Plan('my plan')
    plan.add_floor_from_boundary(perimeter)

    duct = [(150, 150), (300, 150), (300, 300), (150, 300)]
    plan.empty_space.insert_face_from_boundary(duct)
    plan.empty_space.barycenter_cut(list(plan.mesh.faces[1].edges)[0].pair, 1)

    my_space = plan.empty_space
    my_space.remove_face(plan.mesh.faces[1])
    Space(plan, my_space.floor, plan.mesh.faces[1].edge)

    plan.plot()

    assert plan.check()


def test_merge_middle_b_space():
    """
    Test
    :return:
    """
    perimeter = [(0, 0), (500, 0), (500, 500), (200, 500), (200, 200), (0, 200)]
    plan = Plan('my plan')
    plan.add_floor_from_boundary(perimeter)

    edge = list(plan.mesh.faces[0].edges)[4]
    plan.empty_space.barycenter_cut(edge, 0)

    plan.plot()

    duct = [(200, 200), (300, 200), (300, 300)]
    plan.empty_space.insert_face_from_boundary(duct)

    plan.empty_space.remove_face(plan.mesh.faces[0])
    plan.empty_space.add_face(plan.mesh.faces[0])

    plan.spaces[1].merge(plan.spaces[0])

    plan.plot()

    assert plan.check()


def test_merge_u_space():
    """
    Test.
    :return:
    """

    perimeter = [(0, 0), (1000, 0), (1000, 1000), (0, 1000)]

    # add border duct
    plan = Plan()
    plan.add_floor_from_boundary(perimeter)

    # add single touching point
    point_duct = [(0, 800), (0, 500), (500, 500), (500, 1000), (200, 1000), (200, 800)]
    plan.empty_space.insert_face_from_boundary(point_duct)
    hole_face = plan.mesh.faces[1]

    plan.empty_space.remove_face(hole_face)
    plan.empty_space.add_face(hole_face)

    plan.spaces[0].merge(plan.spaces[1])

    plan.plot()

    assert plan.check()


def test_clone_plan():
    """

    :return:
    """
    perimeter = [(0, 0), (1000, 0), (1000, 1000), (0, 1000)]
    plan = Plan()
    plan.add_floor_from_boundary(perimeter)
    plan_2 = plan.clone()
    plan_2.empty_space.category = SPACE_CATEGORIES["duct"]
    plan.plot()
    plan_2.plot()
    space = plan.get_space_from_id(plan.spaces[0].id)
    assert space is plan.empty_space
    assert plan.spaces[0].id == plan_2.spaces[0].id


def test_clone_change_plan():
    """

    :return:
    """
    from libs.modelers.grid import GRIDS

    perimeter = [(0, 0), (1000, 0), (1000, 1000), (0, 1000)]
    duct = [(400, 400), (600, 400), (600, 600), (400, 600)]
    duct_2 = [(0, 0), (200, 0), (200, 200), (0, 200)]
    plan = Plan("first")
    plan.add_floor_from_boundary(perimeter)
    plan_2 = plan.clone("second")
    plan.insert_linear((200, 0), (600, 0), LINEAR_CATEGORIES["doorWindow"])
    plan.insert_space_from_boundary(duct, SPACE_CATEGORIES["duct"])
    plan_2.insert_space_from_boundary(duct_2, SPACE_CATEGORIES["duct"])
    GRIDS["finer_ortho_grid"].apply_to(plan_2)
    plan_2.plot()
    space = plan.get_space_from_id(plan.spaces[0].id)
    assert space is plan.empty_space
    assert not plan_2.empty_space.has_holes
    assert plan.spaces[0].id == plan_2.spaces[0].id
    plan.plot()


def test_deepcopy_change_plan():
    """
    :return:
    """
    from libs.modelers.grid import GRIDS
    import copy

    perimeter = [(0, 0), (1000, 0), (1000, 1000), (0, 1000)]
    duct = [(400, 400), (600, 400), (600, 600), (400, 600)]
    duct_2 = [(0, 0), (200, 0), (200, 200), (0, 200)]
    plan = Plan()
    plan.add_floor_from_boundary(perimeter)
    plan_2 = copy.deepcopy(plan)
    plan.insert_linear((200, 0), (600, 0), LINEAR_CATEGORIES["doorWindow"])
    plan.insert_space_from_boundary(duct, SPACE_CATEGORIES["duct"])
    plan_2.insert_space_from_boundary(duct_2, SPACE_CATEGORIES["duct"])
    GRIDS["finer_ortho_grid"].apply_to(plan_2)
    plan.plot()
    plan_2.plot()
    space = plan.get_space_from_id(plan.spaces[0].id)
    assert space is plan.empty_space
    assert not plan_2.empty_space.has_holes
    assert plan.spaces[0].id == plan_2.spaces[0].id


def test_pickling():
    from libs.modelers.grid import GRIDS
    import pickle

    plan = rectangular_plan(1000, 800)
    duct = [(400, 400), (600, 400), (600, 600), (400, 600)]
    plan.insert_linear((200, 0), (600, 0), LINEAR_CATEGORIES["doorWindow"])
    plan.insert_space_from_boundary(duct, SPACE_CATEGORIES["duct"])
    GRIDS["finer_ortho_grid"].apply_to(plan)
    data = pickle.dumps(plan)
    new_plan = pickle.loads(data)
    new_plan.plot()

    assert new_plan.check()


def test_insert_external_space():
    """
    Add a face outside the mesh. The face must be adjacent.
    +---------------+
    |               |
    |               +------+
    |    Mesh       | face |
    |               |      |
    |               +------+
    |               |
    +---------------+

    :return:
    """
    perimeter = [(0, 0), (500, 0), (500, 500), (0, 500)]
    face_perimeter = [(500, 200), (700, 200), (700, 400), (500, 400)]
    plan = Plan("apartment with balcony")
    floor = plan.add_floor_from_boundary(perimeter)
    plan.insert_space_from_boundary(face_perimeter, SPACE_CATEGORIES["balcony"], floor)
    plan.plot()

    assert plan.check()


def test_insert_complex_external_space():
    """
    Add a face outside the mesh. The face must be adjacent.
    +------------+
    |            |
    |            +--------+
    |   MESH     |   FACE |
    |            +---+    |
    |            |   |    |
    +-------+----+   |    |
            |  +-----+    |
            |             |
            +-------------+

    :return:
    """
    perimeter = [(0, 0), (500, 0), (500, 500), (0, 500)]
    face_perimeter = [(250, 0), (250, -200), (700, -200), (700, 400), (500, 400), (500, 200),
                      (600, 200), (600, -100), (400, -100), (400, 0)]
    plan = Plan("apartment with balcony")
    floor = plan.add_floor_from_boundary(perimeter)
    plan.insert_space_from_boundary(face_perimeter, SPACE_CATEGORIES["balcony"], floor)
    plan.plot()

    assert plan.check()


def test_maximum_adjacency_length():
    """
    Add a face outside the mesh. The face must be adjacent.
    +------------+
    |            |
    |            +--------+
    |   MESH     |   FACE |
    |            +---+    |
    |            |   |    |
    +-------+----+   |    |
            |  +-----+    |
            |             |
            +-------------+

    Adjacencies : 150 and 200
    Plan apartment with balcony:Spaces: empty / hole / balcony
    :return:
    """
    perimeter = [(0, 0), (500, 0), (500, 500), (0, 500)]
    face_perimeter = [(250, 0), (250, -200), (700, -200), (700, 400), (500, 400), (500, 200),
                      (600, 200), (600, -100), (400, -100), (400, 0)]
    plan = Plan("apartment with balcony")
    floor = plan.add_floor_from_boundary(perimeter)
    plan.insert_space_from_boundary(face_perimeter, SPACE_CATEGORIES["balcony"], floor)
    length = plan.spaces[0].maximum_adjacency_length(plan.spaces[2])

    assert length == 200, "test_maximum_adjacency_length"
    assert plan.spaces[0].adjacent_to(plan.spaces[2], 200)
    assert not plan.spaces[0].adjacent_to(plan.spaces[2], 201)


def test_contact_length():
    """
    Add a face outside the mesh. The face must be adjacent.
    +------------+
    |            |
    |            +--------+
    |   MESH     |   FACE |
    |            +---+    |
    |            |   |    |
    +-------+----+   |    |
            |  +-----+    |
            |             |
            +-------------+

    Adjacencies : 150 and 200
    Plan apartment with balcony:Spaces: empty / hole / balcony
    :return:
    """
    perimeter = [(0, 0), (500, 0), (500, 500), (0, 500)]
    face_perimeter = [(250, 0), (250, -200), (700, -200), (700, 400), (500, 400), (500, 200),
                      (600, 200), (600, -100), (400, -100), (400, 0)]
    plan = Plan("apartment with balcony")
    floor = plan.add_floor_from_boundary(perimeter)
    plan.insert_space_from_boundary(face_perimeter, SPACE_CATEGORIES["balcony"], floor)
    length = plan.spaces[0].contact_length(plan.spaces[2])

    assert length == 350, "test_contact_length"


def test_adjacent_spaces():
    """
    Add a face outside the mesh. The face must be adjacent.
    +---------------+
    |               |
    |               +------+
    |    Mesh       | face |
    |               |      |
    |               +------+
    |               |
    +---------------+

    :return:
    """
    perimeter = [(0, 0), (500, 0), (500, 500), (0, 500)]
    face_perimeter = [(500, 200), (700, 200), (700, 400), (500, 400)]
    plan = Plan("apartment with balcony")
    floor = plan.add_floor_from_boundary(perimeter)
    plan.insert_space_from_boundary(face_perimeter, SPACE_CATEGORIES["balcony"], floor)
    plan.insert_linear((500, 250), (500, 350), LINEAR_CATEGORIES["doorWindow"], floor)

    adjacent_spaces = plan.linears[0].adjacent_spaces()

    assert adjacent_spaces == plan.spaces, "adjacent_spaces"


@pytest.fixture
def l_plan() -> 'Plan':
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


def test_min_rotated_rectangle(l_plan):
    """
    Test the minimum_rotated_rectangle method
    :return:
    """
    assert l_plan.empty_space.minimum_rotated_rectangle() == [(1200.0, 0.0), (1200.0, 1200.0),
                                                              (0.0, 1200.0), (0.0, 0.0)]


def test_connected_spaces():
    """
    Add a doorWindow between the empty space and the balcony.
    The balcony is connected to the empty space
    +---------------+
    |               |
    |               +-------+
    |    empty      |balcony|
    |               |       |
    |               +-------+
    |               |
    +---------------+

    :return:
    """
    perimeter = [(0, 0), (500, 0), (500, 500), (0, 500)]
    face_perimeter = [(500, 200), (700, 200), (700, 400), (500, 400)]
    plan = Plan("apartment with balcony")
    floor = plan.add_floor_from_boundary(perimeter)
    plan.insert_space_from_boundary(face_perimeter, SPACE_CATEGORIES["balcony"], floor)
    plan.insert_linear((500, 250), (500, 350), LINEAR_CATEGORIES["doorWindow"], floor)

    assert plan.spaces[1] in plan.spaces[0].connected_spaces(), "connected_spaces"


def test_external_axes(l_plan):
    """
    Test the external axes method
    """
    assert l_plan.empty_space._external_axes() == [0.0, 16.0, 59.0, 22.0, 68.0]


def test_best_direction(l_plan):
    """
    Test the best direction
    """
    edge = l_plan.empty_space.edge
    assert l_plan.empty_space.best_direction(edge.normal) == (0, 1)


def test_centroid():
    """
    Exemple
    +------------+
    |            |
    |            +--------+
    |  KITCHEN   |        |
    |            +---+    |
    |            |   |    |
    +-------+----+   |    |
            |  +-----+    |
            |  BALCONY    |
            +-------------+
    """
    perimeter = [(0, 0), (500, 0), (500, 500), (0, 500)]
    face_perimeter = [(250, 0), (250, -200), (700, -200), (700, 400), (500, 400), (500, 200),
                      (600, 200), (600, -100), (400, -100), (400, 0)]
    plan = Plan("apartment with balcony")
    floor = plan.add_floor_from_boundary(perimeter)
    plan.insert_space_from_boundary(face_perimeter, SPACE_CATEGORIES["balcony"], floor)
    plan.insert_space_from_boundary(perimeter, SPACE_CATEGORIES["kitchen"], floor)
    plan.remove_null_spaces()
    for space in plan.spaces:
        centroid_2d = space.centroid()
        centroid_shapely = space.as_sp.centroid
        assert centroid_2d[0] == centroid_shapely.xy[0][0]
        assert centroid_2d[1] == centroid_shapely.xy[1][0]


def test_maximum_distance_to():
    """
    Exemple
    +------------+
    |            |
    |            |
    |  KITCHEN   |
    |            |
    |            |
    +------------+------------+
                 |            |
                 |            |
                 |  BALCONY   |
                 |            |
                 |            |
                 +------------+

    """
    perimeter = [(0, 0), (500, 0), (500, 500), (0, 500)]
    face_perimeter = [(500, 0), (500, -500), (1000, -500), (1000, 0)]
    plan = Plan("apartment with balcony")
    floor = plan.add_floor_from_boundary(perimeter)
    plan.insert_space_from_boundary(face_perimeter, SPACE_CATEGORIES["balcony"], floor)
    plan.insert_space_from_boundary(perimeter, SPACE_CATEGORIES["kitchen"], floor)
    plan.remove_null_spaces()
    assert plan.spaces[0].maximum_distance_to(plan.spaces[1]) == 1000*2**0.5

    """
    Exemple
    +------------+
    |            |
    |            +--------+
    |  KITCHEN   |        |
    |            +---+    |
    |            |   |    |
    +-------+----+   |    |
            |  +-----+    |
            |  BALCONY    |
            +-------------+
    """
    perimeter = [(0, 0), (500, 0), (500, 500), (0, 500)]
    face_perimeter = [(250, 0), (250, -200), (700, -200), (700, 400), (500, 400), (500, 200),
                      (600, 200), (600, -100), (400, -100), (400, 0)]
    plan = Plan("apartment with balcony")
    floor = plan.add_floor_from_boundary(perimeter)
    plan.insert_space_from_boundary(face_perimeter, SPACE_CATEGORIES["balcony"], floor)
    plan.insert_space_from_boundary(perimeter, SPACE_CATEGORIES["kitchen"], floor)
    plan.remove_null_spaces()
    # space 0 : hole
    assert plan.spaces[1].maximum_distance_to(plan.spaces[2]) == 700*2**0.5


def test_maximum_distance_to_2():
    import math
    from libs.modelers.grid import GRIDS
    plan = rectangular_plan(500, 500)
    GRIDS["finer_ortho_grid"].apply_to(plan)
    space_1 = plan.insert_space_from_boundary([(0, 0), (125, 0), (125, 125), (0, 125)])
    space_2 = plan.insert_space_from_boundary([(375, 375), (500, 375), (500, 500), (375, 500)])
    plan.plot()
    assert space_1.distance_to(space_2) == 500*math.sqrt(2)
    space_3 = plan.insert_space_from_boundary([(0, 375), (125, 375), (125, 500), (0, 500)])
    assert space_1.distance_to(space_3) == math.sqrt(500.0**2 + 125**2)


def test_space_area(l_plan):
    assert l_plan.empty_space.area == 915000.0
    perimeter = [(0, 0), (500, 0), (500, 500), (0, 500)]
    plan = Plan("test_perimeter")
    plan.add_floor_from_boundary(perimeter)
    assert plan.empty_space.area == 250000.0


def test_space_perimeter(l_plan):
    assert l_plan.empty_space.perimeter == 4488.139139839483
    perimeter = [(0, 0), (500, 0), (500, 500), (0, 500)]
    plan = Plan("test_perimeter")
    plan.add_floor_from_boundary(perimeter)
    assert plan.empty_space.perimeter == 2000.0


def test_perimeter_without_duct(l_plan):
    assert l_plan.empty_space.perimeter_without_duct == 4488.139139839483
    perimeter = [(0, 0), (500, 0), (500, 500), (0, 500)]
    plan = Plan("test_perimeter")
    plan.add_floor_from_boundary(perimeter)
    assert plan.empty_space.perimeter_without_duct == 2000.0
    face_perimeter = [(0, 100), (100, 200), (100, 200), (0, 200)]
    plan.insert_space_from_boundary(face_perimeter, SPACE_CATEGORIES["duct"])
    print(plan)
    assert plan.empty_space.perimeter_without_duct == (2000.0 - 100)


def test_number_corners(l_plan):
    """
    test the number of corners
    :param l_plan:
    :return:
    """
    assert l_plan.empty_space.number_of_corners() == 9


def test_number_corners_with_addition_space(l_plan):
    """
    test the number of corners
    :param l_plan:
    :return:
    """
    from libs.modelers.grid import GRIDS
    plan = rectangular_plan(500, 500)
    weird_space_boundary = [(0, 0), (250, 0), (250, 250), (125, 250), (125, 125), (0, 125)]
    plan.insert_space_from_boundary(weird_space_boundary)
    GRIDS["ortho_grid"].apply_to(plan)
    assert plan.spaces[0].number_of_corners(plan.spaces[1]) == 4


def test_number_corners_with_addition_face(l_plan):
    """
    test the number of corners
    :param l_plan:
    :return:
    """
    from libs.modelers.grid import GRIDS
    plan = rectangular_plan(500, 500)
    weird_space_boundary = [(0, 0), (250, 0), (250, 250), (125, 250), (125, 125), (0, 125)]
    plan.insert_space_from_boundary(weird_space_boundary)
    GRIDS["ortho_grid"].apply_to(plan)
    assert plan.spaces[0].number_of_corners(plan.mesh.faces[0]) == 6


def test_corner_stone():
    from libs.modelers.grid import GRIDS
    plan = rectangular_plan(500, 500)
    weird_space_boundary = [(0, 0), (250, 0), (250, 250), (125, 250), (125, 125), (0, 125)]
    plan.insert_space_from_boundary(weird_space_boundary)

    GRIDS["simple_grid"].apply_to(plan)
    plan.plot()

    faces_id = [365, 389]
    faces = list(map(lambda i: plan.mesh.get_face(i), faces_id))
    space = plan.spaces[1]

    assert space.corner_stone(*faces)

    faces_id += [383, 16]
    faces = list(map(lambda i: plan.mesh.get_face(i), faces_id))

    assert not space.corner_stone(*faces)


def test_boundary_polygon(l_plan):
    from libs.modelers.grid import GRIDS

    boundaries = [(0, 0), (500, 200), (1000, 0), (1000, 400), (1200, 400), (1200, 1200),
                  (500, 1000), (200, 500), (0, 500)]
    assert l_plan.empty_space.boundary_polygon() == boundaries
    GRIDS["simple_grid"].apply_to(l_plan)
    l_plan.plot()
    assert l_plan.empty_space.boundary_polygon() == boundaries


def test_cut_over_hole():
    plan = rectangular_plan(400, 800)
    duct = [(100, 200), (300, 200), (300, 600), (100, 600)]
    plan.insert_space_from_boundary(duct, SPACE_CATEGORIES["duct"])

    plan.empty_space.barycenter_cut(plan.empty_space.edge, vector=(0, 1))
    assert plan.check()


def test_cut_over_internal_edge():
    plan = rectangular_plan(400, 800)
    duct = [(100, 200), (300, 200), (300, 600), (100, 600)]
    plan.insert_space_from_boundary(duct, SPACE_CATEGORIES["duct"])

    plan.empty_space.barycenter_cut(list(plan.empty_space.edges)[3], coeff=0.1, vector=(0, -1))
    plan.plot()
    assert plan.check()


def test_cut_over_internal_edge_other_direction():
    plan = rectangular_plan(400, 800)
    duct = [(100, 200), (300, 200), (300, 600), (100, 600)]
    plan.insert_space_from_boundary(duct, SPACE_CATEGORIES["duct"])

    plan.empty_space.barycenter_cut(list(plan.empty_space.edges)[0], coeff=0.9, vector=(0, 1))
    plan.plot()
    assert plan.check()
