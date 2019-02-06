# coding=utf-8

"""
Test module for plan module
"""

import pytest

from libs.plan import Plan, Space
from libs.category import SPACE_CATEGORIES, LINEAR_CATEGORIES
from libs import reader, reader_test

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


def test_serialization():
    """
    Test
    :return:
    """
    import libs.writer as writer

    plan = reader.create_plan_from_file(INPUT_FILES[0])

    new_plan = Plan("from_saved_data")

    for i in range(100):
        serialized_data = plan.serialize()
        writer.save_plan_as_json(serialized_data)

        new_serialized_data = reader.get_plan_from_json(serialized_data["name"])
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

    plan.plot()

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

    plan.plot()

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
    from libs.grid import GRIDS

    perimeter = [(0, 0), (1000, 0), (1000, 1000), (0, 1000)]
    duct = [(400, 400), (600, 400), (600, 600), (400, 600)]
    duct_2 = [(0, 0), (200, 0), (200, 200), (0, 200)]
    plan = Plan()
    plan.add_floor_from_boundary(perimeter)
    plan_2 = plan.clone()
    plan.insert_linear((200, 0), (600, 0), LINEAR_CATEGORIES["doorWindow"])
    plan.insert_space_from_boundary(duct, SPACE_CATEGORIES["duct"])
    plan_2.insert_space_from_boundary(duct_2, SPACE_CATEGORIES["duct"])
    GRIDS["finer_ortho_grid"].apply_to(plan_2)
    plan.plot()
    plan_2.plot()
    space = plan.get_space_from_id(plan.spaces[0].id)
    assert space is plan.empty_space
    assert plan.spaces[0].id == plan_2.spaces[0].id


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
