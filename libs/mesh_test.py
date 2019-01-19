# coding=utf-8
"""
Mesh Testing module
"""
import pytest
from libs.mesh import Mesh, Vertex, Face, Edge


def test_simple_mesh():
    """
    Test
    :return:
    """
    perimeter = [(0, 0), (200, 0), (200, 200), (0, 200)]
    mesh = Mesh().from_boundary(perimeter)
    mesh.boundary_edge.pair.barycenter_cut(0)
    mesh.plot()
    assert mesh.check()


def test_outward_cut():
    """
    Test
    :return:
    """
    perimeter = [(0, 0), (200, 0), (200, 200), (0, 200), (100, 100), (0, 100)]
    mesh = Mesh().from_boundary(perimeter)

    for edge in mesh.boundary_edges:
        if edge.pair.next_is_outward:
            edge.pair.recursive_barycenter_cut(1)
            # edge.previous.pair.recursive_barycenter_cut(0)

    mesh.plot()

    assert mesh.check()


def test_serialization():
    """
    Test a mesh serialization
    :return:
    """
    import libs.writer as writer
    import libs.reader as reader

    perimeter = [(0, 0), (200, 0), (200, 200), (0, 200)]
    mesh = Mesh().from_boundary(perimeter)
    geometry = mesh.serialize()
    writer.save_mesh_as_json(geometry, "test")
    new_geometry = reader.get_mesh_from_json("test")
    mesh.deserialize(new_geometry)
    mesh.plot()
    assert mesh.check()


def test_remove_edge():
    """
    Test
    :return:
    """
    perimeter = [(0, 0), (200, 0), (200, 200), (100, 200), (100, 100), (0, 100)]
    mesh = Mesh().from_boundary(perimeter)

    edges = list(mesh.faces[0].edges)
    edges[0].recursive_barycenter_cut(0.5)
    edges[3].recursive_barycenter_cut(0.5)

    edges = list(mesh.faces[0].edges)
    edges[1].remove()

    edges = list(mesh.faces[1].edges)
    edges[0].remove()

    assert mesh.check()


def test_non_ortho_cut():
    """
    Test
    :return:
    """
    perimeter = [(0, 0), (500, 0), (500, 500), (250, 500), (250, 250), (0, 250)]
    mesh = Mesh().from_boundary(perimeter)
    edges = list(mesh.faces[0].edges)

    edges[0].recursive_barycenter_cut(0.5, 30)

    edges = list(mesh.faces[0].edges)
    edges[3].recursive_barycenter_cut(0.5, 100)

    edges = list(mesh.faces[2].edges)
    edges[4].recursive_barycenter_cut(0.8)

    assert mesh.check()


def test_cut():
    """
    Test a simple edge laser_cut
    :return:
    """
    perimeter = [(0, 0), (500, 0), (500, 500), (200, 500), (200, 200), (0, 200)]
    mesh = Mesh().from_boundary(perimeter)

    edges = list(mesh.boundary_edges)
    [edge.pair.recursive_barycenter_cut(0.6) for edge in edges]

    assert mesh.check()


def test_cut_to_vertex():
    """
    Test a simple edge laser_cut
    :return:
    """
    perimeter = [(0, 0), (500, 0), (500, 500), (200, 500), (200, 200), (0, 200)]
    mesh = Mesh().from_boundary(perimeter)

    edges = list(mesh.boundary_edges)
    edges[0].pair.recursive_barycenter_cut(0, 30.0)

    assert mesh.check()


def test_cut_to_start_vertex():
    """
    Test a simple edge laser_cut
    :return:
    """
    perimeter = [(0, 0), (500, 0), (500, 500), (200, 500), (200, 200), (0, 200)]
    mesh = Mesh().from_boundary(perimeter)

    edges = list(mesh.boundary_edges)
    edges[0].pair.recursive_barycenter_cut(0.4001)

    assert mesh.check()


def test_cut_inserted_face():
    """
    Test a simple edge laser_cut
    :return:
    """
    perimeter = [(0, 0), (500, 0), (500, 500), (200, 500), (200, 200), (0, 200)]
    mesh = Mesh().from_boundary(perimeter)

    perimeter_ = [(500, 300), (500, 500), (300, 500), (300, 300)]
    face = mesh.new_face_from_boundary(perimeter_)
    mesh.faces[0].insert_face(face)

    edges = list(mesh.faces[1].edges)
    edges[1].recursive_barycenter_cut(0.5)

    mesh.plot()

    assert mesh.check()


def test_snap_to_edge():
    """

    :return:
    """
    perimeter = [(0, 0), (200, 0), (200, 200), (0, 200)]
    mesh = Mesh().from_boundary(perimeter)
    edges = list(mesh.faces[0].edges)
    vertex = Vertex(mesh, 100.0, 0.0)
    for edge in edges:
        vertex.snap_to_edge(edge)

    assert mesh.check()


def test_add_face():
    """
    Test
    :return:
    """
    perimeter = [(0, 0), (500, 0), (500, 500), (250, 500), (250, 250), (0, 250)]
    mesh = Mesh()
    mesh.from_boundary(perimeter)
    # print(mesh)
    face = mesh.faces[0]

    """"# case one edge touching
    perimeter_2 = [(500, 250), (500, 450), (375, 375)]
    face = face.insert_face_from_boundary(perimeter_2)[0]

    # single point touching
    perimeter_3 = [(500, 200), (400, 200), (400, 175)]
    face_2 = mesh.new_face_from_boundary(perimeter_3)
    face = face.insert_face(face_2)[0]

    # two single points touching
    perimeter_4 = [(0, 125), (250, 0), (250, 125)]
    face_3 = mesh.new_face_from_boundary(perimeter_4)
    face = face.insert_face(face_3)[0]

    # two following edges touching
    perimeter_5 = [(0, 250), (0, 135), (250, 250)]
    face_4 = mesh.new_face_from_boundary(perimeter_5)
    face = face.insert_face(face_4)[0]

    # three following edges touching
    perimeter_6 = [(250, 250), (270, 250), (270, 475), (500, 475), (500, 500), (250, 500)]
    face_5 = mesh.new_face_from_boundary(perimeter_6)
    face = face.insert_face(face_5)[0]"""

    # enclosed face
    perimeter_7 = [(400, 25), (475, 25), (475, 100), (400, 100)]
    face_6 = mesh.new_face_from_boundary(perimeter_7)
    face.insert_face(face_6)
    mesh.plot()
    """
    for edge in mesh.boundary_edges:
        if edge.length > 50:
            edge.pair.recursive_barycenter_cut(0.5)"""

    assert mesh.check()


def test_add_and_cut_face():
    """
    Test
    :return:
    """

    perimeter = [(0, 0), (200, 0), (200, 200), (100, 200), (100, 100), (0, 100)]
    mesh = Mesh()
    mesh.from_boundary(perimeter)
    # print(mesh)
    face = mesh.faces[0]

    # two following edges touching
    perimeter_5 = [(0, 100), (0, 50), (100, 100)]
    face_2 = mesh.new_face_from_boundary(perimeter_5)
    face = face.insert_face(face_2)[0]

    # three following edges touching
    perimeter_6 = [(200, 200), (100, 200), (100, 100), (120, 100), (120, 180), (200, 180)]
    face_3 = mesh.new_face_from_boundary(perimeter_6)
    face = face.insert_face(face_3)[0]

    edges = list(face.edges)
    edges[2].recursive_barycenter_cut(0.8, 80.0)

    assert mesh.check()


def test_cut_snap():
    """
    Test
    :return:
    """

    perimeter = [(0, 0), (500, 0), (500, 500), (0, 500)]
    mesh = Mesh().from_boundary(perimeter)
    edges = list(mesh.boundary_edges)

    for edge in edges:
        edge.pair.recursive_barycenter_cut(0.5)

    edges[0].pair.recursive_barycenter_cut(0.5, 64)

    assert mesh.check()


def test_cut_inside_edge():
    """
    Test
    :return:
    """
    perimeter = [(0, 0), (1000, 0), (1000, 1000), (0, 1000)]
    mesh = Mesh().from_boundary(perimeter)

    hole = [(200, 200), (800, 200), (800, 800), (200, 800)]

    mesh.faces[0].insert_face_from_boundary(hole)

    edges = list(mesh.boundary_edges)

    edges[4].pair.previous.barycenter_cut()

    assert mesh.check()


def insert_touching_face():
    """
    test
    :return:
    """

    perimeter = [(0, 0), (200, 0), (200, 200), (400, 0), (1000, 0), (1000, 1000), (0, 1000)]
    duct = [(100, 200), (600, 200), (600, 400), (100, 400)]
    mesh = Mesh().from_boundary(perimeter)
    mesh.faces[0].insert_face_from_boundary(duct)

    assert mesh.check()


def cut_to_inside_edge():
    """
    Test
    :return:
    """
    perimeter = [(0, 0), (1000, 0), (1000, 1000), (0, 1000)]
    mesh = Mesh().from_boundary(perimeter)

    hole = [(200, 200), (800, 200), (800, 800), (200, 800)]

    mesh.faces[0].insert_face_from_boundary(hole)

    edges = list(mesh.boundary_edges)

    edges[3].pair.barycenter_cut(0.1)

    assert mesh.check()


def test_insert_identical_face():
    """
    Test
    :return:
    """
    perimeter = [(0, 0), (1000, 0), (1000, 1000), (0, 1000)]
    mesh = Mesh().from_boundary(perimeter)
    mesh.faces[0].insert_face_from_boundary(perimeter)
    assert mesh.check()


def test_remove_complex_edge():
    """
    Test
    :return:
    """
    perimeter = [(0, 0), (500, 0), (500, 500), (0, 500)]
    hole = [(100, 100), (400, 100), (400, 400), (100, 400)]

    mesh = Mesh().from_boundary(perimeter)
    mesh.faces[0].insert_face_from_boundary(hole)

    mesh.plot()

    mesh.faces[1].edge.remove()

    # mesh.plot()

    assert mesh.check()


def test_insert_complex_face_1():
    """
    Test
    :return:
    """
    perimeter = [(0, 0), (500, 0), (500, 500), (0, 500)]
    hole = [(200, 200), (300, 200), (300, 300), (200, 300)]

    mesh = Mesh().from_boundary(perimeter)
    mesh.faces[0].insert_face_from_boundary(hole)

    hole_2 = [(50, 150), (200, 150), (200, 300), (50, 300)]
    mesh.faces[0].insert_face_from_boundary(hole_2)
    mesh.plot()

    assert mesh.check()


def test_insert_complex_face_2():
    """
    Test
    :return:
    """
    perimeter = [(0, 0), (500, 0), (500, 500), (0, 500)]
    hole = [(200, 200), (300, 200), (300, 300), (200, 300)]

    mesh = Mesh().from_boundary(perimeter)
    mesh.faces[0].insert_face_from_boundary(hole)

    hole_2 = [(50, 150), (150, 150), (150, 300), (50, 300)]

    mesh.faces[0].insert_face_from_boundary(hole_2)

    assert mesh.check()


def test_insert_complex_face_3():
    """
    Test
    :return:
    """
    perimeter = [(0, 0), (500, 0), (500, 500), (0, 500)]
    hole = [(200, 200), (300, 200), (300, 300), (200, 300)]

    mesh = Mesh().from_boundary(perimeter)
    mesh.faces[0].insert_face_from_boundary(hole)

    hole_2 = [(50, 150), (150, 150), (150, 200), (50, 200)]

    mesh.faces[0].insert_face_from_boundary(hole_2)

    assert mesh.check()


def test_insert_complex_face_4():
    """
    Test
    :return:
    """
    perimeter = [(0, 0), (500, 0), (500, 500), (0, 500)]
    hole = [(200, 200), (300, 200), (300, 300), (200, 300)]

    mesh = Mesh().from_boundary(perimeter)
    mesh.faces[0].insert_face_from_boundary(hole)

    hole_2 = [(50, 200), (150, 200), (150, 300), (50, 300)]

    mesh.faces[0].insert_face_from_boundary(hole_2)

    assert mesh.check()


def test_insert_complex_face_5():
    """
    Test
    :return:
    """
    perimeter = [(0, 0), (500, 0), (500, 500), (0, 500)]
    hole = [(200, 200), (300, 200), (300, 300), (200, 300)]

    mesh = Mesh().from_boundary(perimeter)
    mesh.faces[0].insert_face_from_boundary(hole)

    hole_2 = [(0, 150), (150, 150), (150, 200), (0, 200)]

    mesh.faces[0].insert_face_from_boundary(hole_2)

    assert mesh.check()


def test_insert_complex_face_6():
    """
    Test
    :return:
    """
    perimeter = [(0, 0), (500, 0), (500, 500), (0, 500)]
    hole = [(200, 200), (300, 200), (300, 300), (200, 300)]

    mesh = Mesh().from_boundary(perimeter)
    mesh.faces[0].insert_face_from_boundary(hole)

    hole_2 = [(0, 150), (150, 150), (150, 300), (0, 300)]

    mesh.faces[0].insert_face_from_boundary(hole_2)

    assert mesh.check()


def test_insert_two_faces_on_internal_edge():
    """
    Test
    :return:
    """

    perimeter = [(0, 0), (500, 0), (500, 500), (0, 500)]
    hole = [(200, 200), (300, 200), (300, 300), (200, 300)]
    hole_2 = [(0, 150), (150, 150), (150, 200), (0, 200)]
    hole_3 = [(0, 200), (150, 200), (150, 300), (0, 300)]

    mesh = Mesh().from_boundary(perimeter)
    mesh.faces[0].insert_face_from_boundary(hole)
    mesh.faces[0].insert_face_from_boundary(hole_2)
    mesh.faces[2].insert_face_from_boundary(hole_3)

    assert mesh.check()


def test_insert_very_close_border_duct():
    """
    Test
    :return:
    """
    perimeter = [(0, 250), (250, 250), (250, 0), (500, 0), (500, 500), (0, 500)]
    hole = [(250, 251.01), (250, 200), (400, 200), (400, 251.01)]

    mesh = Mesh().from_boundary(perimeter)
    mesh.faces[0].insert_face_from_boundary(hole)

    assert mesh.check()


def test_insert_multiple_overlapping():
    """
    Test
    :return:
    """
    perimeter = [(0, 0), (500, 0), (500, 500), (0, 500)]
    hole = [(90, 300), (300, 300), (300, 400), (90, 400)]
    hole_2 = [(90, 100), (300, 100), (300, 200), (90, 200)]

    mesh = Mesh().from_boundary(perimeter)
    mesh.faces[0].insert_face_from_boundary(hole)
    mesh.faces[0].insert_face_from_boundary(hole_2)

    hole_3 = [(20, 50), (60, 50), (60, 450), (20, 450)]

    mesh.faces[0].insert_face_from_boundary(hole_3)

    assert mesh.check()


def test_insert_multiple_overlapping_closing():
    """
    Test
    :return:
    """
    perimeter = [(0, 0), (500, 0), (500, 500), (0, 500)]
    hole = [(90, 300), (300, 300), (300, 400), (90, 400)]
    hole_2 = [(90, 100), (300, 100), (300, 200), (90, 200)]

    mesh = Mesh().from_boundary(perimeter)
    mesh.faces[0].insert_face_from_boundary(hole)
    mesh.faces[0].insert_face_from_boundary(hole_2)

    hole_3 = [(20, 0), (60, 0), (60, 220), (500, 220), (500, 280),
              (60, 280), (60, 500), (20, 500)]

    mesh.faces[0].insert_face_from_boundary(hole_3)

    assert mesh.check()


def test_ortho_cut():
    """
    Plot a graph
    :return:
    """
    perimeter = [(0, 0), (200, 0), (200, 200), (100, 200), (100, 100), (0, 100)]
    mesh = Mesh().from_boundary(perimeter)
    edges = list(mesh.boundary_edges)
    for edge in edges:
        edge.pair.ortho_cut()

    assert mesh.check()


def test_face_clean():
    """
    Test
    :return:
    """

    perimeter = [(0, 0), (500, 0), (500, 500), (0, 500)]
    mesh = Mesh().from_boundary(perimeter)

    boundary_edge = mesh.boundary_edge.pair
    new_edge = Edge(mesh, boundary_edge.start, boundary_edge.next,
                    boundary_edge.pair, boundary_edge.face)
    new_edge.pair = Edge(mesh, new_edge.end,
                         boundary_edge.pair.next, new_edge, boundary_edge.pair.face)
    boundary_edge.face.edge = new_edge
    boundary_edge.previous.next = new_edge
    boundary_edge.next, new_edge.pair.next = new_edge.pair, boundary_edge
    new_face = Face(mesh, boundary_edge)
    boundary_edge.face = new_face
    new_edge.pair.face = new_face

    new_face.clean()
    assert mesh.check()


def test_simplify_mesh_triangle():
    """
    Test to collapse a small edge linking two triangle faces.
    This will delete both faces.
    :return:
    """
    perimeter = [(0, 0), (500, 0), (500, 500), (0, 500)]
    mesh = Mesh().from_boundary(perimeter)
    edge = mesh.boundary_edge.pair
    edge.barycenter_cut(0, 6)
    edge.barycenter_cut(0.01, 175)
    edge.next.barycenter_cut(1.0, 100.0)
    mesh.simplify()
    assert mesh.check()


def test_insert_external_face():
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
    face_perimeter = [(499, 200), (700, 200), (700, 400), (500, 400)]
    mesh = Mesh().from_boundary(perimeter)
    face = mesh.new_face_from_boundary(face_perimeter)
    mesh.insert_external_face(face)

    mesh.plot()

    assert mesh.check()


def test_insert_complex_external_face():
    """
    Add a face outside the mesh. The face must be adjacent.
    +------------+
    |            |
    |            +--------+
    |   MESH     |   FACE |
    |            +---+    |
    |            |   |    |
    +-------+--+-+   |    |
            |  |     |    |
            |  +-----+    |
            |             |
            +-------------+

    :return:
    """
    perimeter = [(0, 0), (500, 0), (500, 500), (0, 500)]
    face_perimeter = [(250, 0), (250, -200), (700, -200), (700, 400), (500, 400), (500, 200),
                      (600, 200), (600, -100), (400, -100), (400, 0)]
    mesh = Mesh().from_boundary(perimeter)
    face = mesh.new_face_from_boundary(face_perimeter)
    mesh.insert_external_face(face)

    mesh.plot()

    assert mesh.check()


def test_insert_overlapping_face():
    """
    Add a face outside the mesh. The face slightly overlaps the mesh (> EPSILON_COORDS).
    The method raises an OutsideFaceError
    :return:
    """
    from libs.utils.custom_exceptions import OutsideFaceError
    with pytest.raises(OutsideFaceError):
        perimeter = [(0, 0), (500, 0), (500, 500), (0, 500)]
        face_perimeter = [(498, 200), (700, 200), (700, 400), (500, 400)]
        mesh = Mesh().from_boundary(perimeter)
        face = mesh.new_face_from_boundary(face_perimeter)
        mesh.insert_external_face(face)
