"""for testing purposes"""
# TODO : replace this by cleaner unit tests
# (@florent: any idea on the best library and files structure for this ?)
import os

if 'MPLBACKEND' not in os.environ:
        os.environ['MPLBACKEND'] = 'svg'

import libs.mesh as m


def test_unit_vector():
    """
    Test
    """
    angles = [i * 45 for i in range(10)] + [-i * 45 for i in range(10)]
    for angle in angles:
        print('angle:', angle, '-', m.unit_vector(angle))
        print('verif: ', angle, m.ccw_angle(m.unit_vector(angle)))

# test_unit_vector()


def test_angle():
    """
    Test
    :return:
    """
    perimeter = [(0, 0), (200, 0), (200, 200), (100, 200), (100, 100), (0, 100)]
    mesh = m.Mesh().from_boundary(perimeter)

    for edge in mesh.faces[0].edges():
        print(edge.next_angle, edge.next_is_outward)

# test_angle()


def test_remove_edge():
    """
    Test
    :return:
    """
    perimeter = [(0, 0), (200, 0), (200, 200), (100, 200), (100, 100), (0, 100)]
    mesh = m.Mesh().from_boundary(perimeter)

    edges = list(mesh.faces[0].edges())
    edges[0].cut_at_barycenter(0.5)
    edges[3].cut_at_barycenter(0.5)

    edges = list(mesh.faces[0].edges())
    edges[1].remove()

    edges = list(mesh.faces[1].edges())
    edges[0].remove()

    mesh.check()
    mesh.plot()
    m.plt.show()

# test_remove_edge()


def test_non_ortho_cut():
    """
    Test
    :return:
    """
    perimeter = [(0, 0), (500, 0), (500, 500), (250, 500), (250, 250), (0, 250)]
    mesh = m.Mesh().from_boundary(perimeter)
    edges = list(mesh.faces[0].edges())

    edges[0].cut_at_barycenter(0.5, 30)

    edges = list(mesh.faces[0].edges())
    edges[3].cut_at_barycenter(0.5, 100)

    edges = list(mesh.faces[2].edges())
    edges[4].cut_at_barycenter(0.8)

    mesh.check()
    mesh.plot()
    m.plt.show()

# test_non_ortho_cut()


def test_cut():
    """
    Test a simple edge cut
    :return:
    """
    perimeter = [(0, 0), (500, 0), (500, 500), (200, 500), (200, 200), (0, 200)]
    mesh = m.Mesh().from_boundary(perimeter)

    edges = list(mesh.boundary_edges())
    [edge.pair.cut_at_barycenter(0.6) for edge in edges]

    mesh.check()
    mesh.plot()
    m.plt.show()

# test_cut()


def test_cut_to_vertex():
    """
    Test a simple edge cut
    :return:
    """
    perimeter = [(0, 0), (500, 0), (500, 500), (200, 500), (200, 200), (0, 200)]
    mesh = m.Mesh().from_boundary(perimeter)

    edges = list(mesh.boundary_edges())
    edges[0].pair.cut_at_barycenter(0, 30.0)

    mesh.check()
    mesh.plot()
    m.plt.show()

test_cut_to_vertex()


def test_cut_to_start_vertex():
    """
    Test a simple edge cut
    :return:
    """
    perimeter = [(0, 0), (500, 0), (500, 500), (200, 500), (200, 200), (0, 200)]
    mesh = m.Mesh().from_boundary(perimeter)

    edges = list(mesh.boundary_edges())
    edges[0].pair.cut_at_barycenter(0.4001)

    mesh.check()
    mesh.plot()
    m.plt.show()

test_cut_to_start_vertex()


def test_cut_inserted_face():
    """
    Test a simple edge cut
    :return:
    """
    perimeter = [(0, 0), (500, 0), (500, 500), (200, 500), (200, 200), (0, 200)]
    mesh = m.Mesh().from_boundary(perimeter)

    perimeter_ = [(500, 300), (500, 500), (300, 500), (300, 300)]
    mesh_ = m.Mesh().from_boundary(perimeter_)
    mesh.faces[0].insert_face(mesh_.faces[0])

    edges = list(mesh.faces[1].edges())
    edges[1].cut_at_barycenter(0.5)

    mesh.check()
    mesh.plot()
    m.plt.show()

test_cut_inserted_face()


def test_snap_to_edge():
    """

    :return:
    """
    perimeter = [(0, 0), (200, 0), (200, 200), (0, 200)]
    mesh = m.Mesh().from_boundary(perimeter)
    edges = list(mesh.faces[0].edges())
    vertex = m.Vertex(100.0, 0.0)
    for edge in edges:
        vertex.snap_to_edge(edge)

    mesh.check()
    mesh.plot()
    m.plt.show()

test_snap_to_edge()


def test_add_face():
    """
    Test
    :return:
    """
    perimeter = [(0, 0), (500, 0), (500, 500), (250, 500), (250, 250), (0, 250)]
    mesh = m.Mesh()
    mesh.from_boundary(perimeter)
    # print(mesh)
    face = mesh.faces[0]

    # case one edge touching
    perimeter_2 = [(500, 250), (500, 450), (375, 375)]
    mesh2 = m.Mesh()
    mesh2.from_boundary(perimeter_2)
    face = face.insert_face(mesh2.faces[0])

    # single point touching
    perimeter_3 = [(500, 200), (400, 200), (400, 175)]
    mesh3 = m.Mesh().from_boundary(perimeter_3)
    face = face.insert_face(mesh3.faces[0])

    # two single points touching
    perimeter_4 = [(0, 125), (250, 0), (250, 125)]
    mesh4 = m.Mesh().from_boundary(perimeter_4)
    face = face.insert_face(mesh4.faces[0])

    # two following edges touching
    perimeter_5 = [(0, 250), (0, 135), (250, 250)]
    mesh5 = m.Mesh().from_boundary(perimeter_5)
    face = face.insert_face(mesh5.faces[0])

    # three following edges touching
    perimeter_6 = [(250, 250), (270, 250), (270, 475), (500, 475), (500, 500), (250, 500)]
    mesh6 = m.Mesh().from_boundary(perimeter_6)
    face = face.insert_face(mesh6.faces[0])

    # enclosed face
    perimeter_7 = [(400, 25), (475, 25), (475, 100), (400, 100)]
    mesh7 = m.Mesh().from_boundary(perimeter_7)
    face.insert_face(mesh7.faces[0])

    for edge in mesh.boundary_edges():
        if edge.length > 20:
            edge.pair.cut_at_barycenter(0.5)

    mesh.check()
    mesh.plot()
    m.plt.show()

test_add_face()


def test_add_and_cut_face():
    """
    Test
    :return:
    """

    perimeter = [(0, 0), (200, 0), (200, 200), (100, 200), (100, 100), (0, 100)]
    mesh = m.Mesh()
    mesh.from_boundary(perimeter)
    # print(mesh)
    face = mesh.faces[0]

    # two following edges touching
    perimeter_5 = [(0, 100), (0, 50), (100, 100)]
    mesh5 = m.Mesh().from_boundary(perimeter_5)
    face = face.insert_face(mesh5.faces[0])

    # three following edges touching
    perimeter_6 = [(200, 200), (100, 200), (100, 100), (120, 100), (120, 180), (200, 180)]
    mesh6 = m.Mesh().from_boundary(perimeter_6)
    face = face.insert_face(mesh6.faces[0])

    edges = list(face.edges())
    edges[2].cut_at_barycenter(0.8, 80.0)

    mesh.check()
    mesh.plot()
    m.plt.show()

test_add_and_cut_face()


def test_cut_snap():
    """
    Test
    :return:
    """

    perimeter = [(0, 0), (500, 0), (500, 500), (0, 500)]
    mesh = m.Mesh().from_boundary(perimeter)
    edges = list(mesh.boundary_edges())

    for edge in edges:
        edge.pair.cut_at_barycenter(0.5)

    edges[0].pair.cut_at_barycenter(0.5, 64)

    mesh.check()
    mesh.plot()
    m.plt.show()


test_cut_snap()
