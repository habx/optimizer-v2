# coding=utf-8
"""
Mesh Testing module
"""
from libs.mesh import Mesh, Vertex


def test_outward_cut():
    """
    Test
    :return:
    """
    perimeter = [(0, 0), (200, 0), (200, 200), (100, 200), (100, 100), (0, 100)]
    mesh = Mesh().from_boundary(perimeter)

    for edge in mesh.boundary_edges:
        if edge.pair.next_is_outward:
            edge.pair.laser_cut_at_barycenter(1)
            edge.previous.pair.laser_cut_at_barycenter(0)

    assert mesh.check()


def test_remove_edge():
    """
    Test
    :return:
    """
    perimeter = [(0, 0), (200, 0), (200, 200), (100, 200), (100, 100), (0, 100)]
    mesh = Mesh().from_boundary(perimeter)

    edges = list(mesh.faces[0].edges)
    edges[0].laser_cut_at_barycenter(0.5)
    edges[3].laser_cut_at_barycenter(0.5)

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

    edges[0].laser_cut_at_barycenter(0.5, 30)

    edges = list(mesh.faces[0].edges)
    edges[3].laser_cut_at_barycenter(0.5, 100)

    edges = list(mesh.faces[2].edges)
    edges[4].laser_cut_at_barycenter(0.8)

    assert mesh.check()


def test_cut():
    """
    Test a simple edge laser_cut
    :return:
    """
    perimeter = [(0, 0), (500, 0), (500, 500), (200, 500), (200, 200), (0, 200)]
    mesh = Mesh().from_boundary(perimeter)

    edges = list(mesh.boundary_edges)
    [edge.pair.laser_cut_at_barycenter(0.6) for edge in edges]

    assert mesh.check()


def test_cut_to_vertex():
    """
    Test a simple edge laser_cut
    :return:
    """
    perimeter = [(0, 0), (500, 0), (500, 500), (200, 500), (200, 200), (0, 200)]
    mesh = Mesh().from_boundary(perimeter)

    edges = list(mesh.boundary_edges)
    edges[0].pair.laser_cut_at_barycenter(0, 30.0)

    assert mesh.check()


def test_cut_to_start_vertex():
    """
    Test a simple edge laser_cut
    :return:
    """
    perimeter = [(0, 0), (500, 0), (500, 500), (200, 500), (200, 200), (0, 200)]
    mesh = Mesh().from_boundary(perimeter)

    edges = list(mesh.boundary_edges)
    edges[0].pair.laser_cut_at_barycenter(0.4001)

    assert mesh.check()


def test_cut_inserted_face():
    """
    Test a simple edge laser_cut
    :return:
    """
    perimeter = [(0, 0), (500, 0), (500, 500), (200, 500), (200, 200), (0, 200)]
    mesh = Mesh().from_boundary(perimeter)

    perimeter_ = [(500, 300), (500, 500), (300, 500), (300, 300)]
    mesh_ = Mesh().from_boundary(perimeter_)
    mesh.faces[0].insert_face(mesh_.faces[0])

    edges = list(mesh.faces[1].edges)
    edges[1].laser_cut_at_barycenter(0.5)

    assert mesh.check()


def test_snap_to_edge():
    """

    :return:
    """
    perimeter = [(0, 0), (200, 0), (200, 200), (0, 200)]
    mesh = Mesh().from_boundary(perimeter)
    edges = list(mesh.faces[0].edges)
    vertex = Vertex(100.0, 0.0)
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

    # case one edge touching
    perimeter_2 = [(500, 250), (500, 450), (375, 375)]
    face = face.insert_face_from_boundary(perimeter_2)[0]

    # single point touching
    perimeter_3 = [(500, 200), (400, 200), (400, 175)]
    mesh3 = Mesh().from_boundary(perimeter_3)
    face = face.insert_face(mesh3.faces[0])[0]

    # two single points touching
    perimeter_4 = [(0, 125), (250, 0), (250, 125)]
    mesh4 = Mesh().from_boundary(perimeter_4)
    face = face.insert_face(mesh4.faces[0])[0]

    # two following edges touching
    perimeter_5 = [(0, 250), (0, 135), (250, 250)]
    mesh5 = Mesh().from_boundary(perimeter_5)
    face = face.insert_face(mesh5.faces[0])[0]

    # three following edges touching
    perimeter_6 = [(250, 250), (270, 250), (270, 475), (500, 475), (500, 500), (250, 500)]
    mesh6 = Mesh().from_boundary(perimeter_6)
    face = face.insert_face(mesh6.faces[0])[0]

    # enclosed face
    perimeter_7 = [(400, 25), (475, 25), (475, 100), (400, 100)]
    mesh7 = Mesh().from_boundary(perimeter_7)
    face.insert_face(mesh7.faces[0])

    for edge in mesh.boundary_edges:
        if edge.length > 20:
            edge.pair.laser_cut_at_barycenter(0.5)

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
    mesh5 = Mesh().from_boundary(perimeter_5)
    face = face.insert_face(mesh5.faces[0])[0]

    # three following edges touching
    perimeter_6 = [(200, 200), (100, 200), (100, 100), (120, 100), (120, 180), (200, 180)]
    mesh6 = Mesh().from_boundary(perimeter_6)
    face = face.insert_face(mesh6.faces[0])[0]

    edges = list(face.edges)
    edges[2].laser_cut_at_barycenter(0.8, 80.0)

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
        edge.pair.laser_cut_at_barycenter(0.5)

    edges[0].pair.laser_cut_at_barycenter(0.5, 64)

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

    edges[4].pair.previous.cut_at_barycenter()

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

    edges[3].pair.cut_at_barycenter(0.1)

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

    mesh.plot(save=False)

    mesh.faces[1].edge.remove()

    assert mesh.check()
