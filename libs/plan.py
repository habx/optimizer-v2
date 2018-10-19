# coding=utf-8
"""
Plan Module
We define a grid as a mesh with specific functionalities

Ideas :
- a face should have a type (space, object, reserved_space) and a nature from type:
"""
from typing import Optional, List, Sequence, Tuple
import logging
import matplotlib.pyplot as plt

from libs.mesh import Mesh, Face, Edge, Vertex


SPACE_COLORS = {
    'duct': 'k',
    'window': '#A3A3A3',
    'entrance': 'r',
    'empty': 'b',
    'space': 'b'
}


class Plan:
    """
    Main class containing the floor plan of the apartment
    • mesh
    • spaces
    • walls
    • objects
    • json_input
    """
    def __init__(self, mesh: Optional[Mesh] = None,
                 spaces: Optional[List['Space']] = None,
                 walls: Optional[List['Wall']] = None):
        self.mesh = mesh
        self.spaces = spaces or []
        self.walls = walls or []

    def from_boundary(self, boundary):
        """
        Creates a plan from a list of points
        1. create the mesh
        2. Add an empty space
        :param boundary:
        :return:
        """
        self.mesh = Mesh().from_boundary(boundary)
        empty_space = EmptySpace(self, self.mesh.faces[0])
        self.add_space(empty_space)

        return self

    def add_space(self, space):
        """
        Add a space in the plan
        :param space:
        :return:
        """
        self.spaces.append(space)

    def get_spaces(self, category: Optional[str] = None):
        """
        Returns an iterator of the spaces contained in the place
        :param category:
        :return:
        """
        if category is not None:
            return (space for space in self.spaces if space.category == category)

        return (space for space in self.spaces)

    @property
    def empty_space(self):
        """
        The first empty space of the plan
        Note : the empty space is only used for the creation of the plan
        :return:
        """
        return list(self.get_spaces(category='empty'))[0]

    def plot(self, ax=None):
        """
        Plots a plan
        :return:
        """

        if ax is None:
            fig, ax = plt.subplots()

        ax.set_aspect('equal')

        for space in self.spaces:
            space.plot(ax)

        plt.show()

    def check(self):
        """
        Used to verify plan consistency
        NOTE : To be completed
        :return:
        """
        logging.info('-'*12 + ' checking plan: ')
        is_valid = True

        for space in self.spaces:
            is_valid = space.check()

        if is_valid:
            logging.info('0K')
        else:
            logging.info('NOT OK')

        return is_valid


class Space:
    """
    Space Class
    """
    category = 'space'

    def __init__(self, plan: Plan, face: Face, boundary_edge: Optional[Edge] = None):
        self.plan = plan
        self._face = face  # one face belonging to the space
        self._edge = boundary_edge or face.edge  # one edge on the boundary of the space
        # set the circular reference
        face.space = self
        # set the boundary of the Space
        for edge in self._edge.siblings:
            edge.space_next = edge.next

    @property
    def face(self) -> Face:
        """
        property
        One of the face of the space
        :return:
        """
        return self._face

    @face.setter
    def face(self, value: Face):
        self._face = value
        # set the circular reference
        value.space = self
        # if the space has no edge it means it's empty so we can add an edge
        if self.edge is None:
            self.edge = value.edge

    @property
    def edge(self) -> Edge:
        """
        property
        :return: on edge of the space
        """
        return self._edge

    @edge.setter
    def edge(self, value: Edge):
        """
        property
        """
        # TODO check if value is on border
        self._edge = value

    @property
    def faces(self):
        """
        The faces included in the Space. Returns an iterator.
        :return:
        """
        seen = [self.face]
        yield self.face

        def get_adjacent_faces(face):
            """
            Recursive function to retrieve all the faces of the space
            :param face:
            :return:
            """
            for edge in face.edges:
                new_face = edge.pair.face
                if new_face and new_face.space is self and new_face not in seen:
                    seen.append(new_face)
                    yield new_face
                    yield from get_adjacent_faces(new_face)

        yield from get_adjacent_faces(self.face)

    @property
    def edges(self):
        """
        The boundary edges of the space
        :return:
        """
        if self.edge is None:
            return None
        return self.edge.space_siblings

    def is_boundary(self, edge):
        """
        Returns True if the edge is on the boundary of the space.
        :param edge:
        :return:
        """
        if self.is_outside(edge):
            return False

        return edge.is_space_boundary

    def starts_from_boundary(self, edge) -> Optional[Edge]:
        """
        Returns True if the edge belongs to the Space and starts from a boundary edge
        :param edge:
        :return:
        """

        if self.is_boundary(edge):
            return edge

        if not self.is_internal(edge):
            return None

        # check if one the edge starting from the same vertex belongs to the space boundary
        current_edge = edge.cw
        while current_edge is not edge:
            if self.is_boundary(current_edge):
                return current_edge
            current_edge = current_edge.cw

        return None

    def is_internal(self, edge):
        """
        Returns True if the edge is internal of the space
        :param edge:
        :return:
        """
        if self.is_outside(edge):
            return False

        return not self.is_boundary(edge)

    def is_outside(self, edge):
        """
        Return True if the edge is outside of the space (not on the boundary or inside)
        :param edge:
        :return:
        """
        return edge.face is None or edge.face.space is not self

    def add_face(self, face: Face):
        """
        Adds a face to the space and adjust the edges list accordingly
        If the added face belongs to another space we first need to remove it from the space
        We do not enable to add a face inside a hole in the face
        TODO : test this
        :param face: face to add to space
        """
        # if the space has no faces yet just add the face as the reference for the Space
        if self.face is None:
            self.face = face
            return

        # check if the face is already inside the space
        if face in self.faces:
            raise ValueError('Cannot add a face that already ' +
                             'belongs to the space: {0}'.format(face))

        # else we make sure the new face is adjacent to at least one of the space faces
        shared_edges = []
        for edge in face.edges:
            if self.is_boundary(edge.pair):
                shared_edges.append(edge)
            else:
                if shared_edges:
                    break
        else:
            raise ValueError('Cannot add a face that is not adjacent' +
                             ' to the space:{0}'.format(face))

        # check for previously shared edges:
        edge = shared_edges[0].previous
        while self.is_boundary(edge.pair):
            if edge in shared_edges:
                raise ValueError('Cannot add a face that is completely enclosed in the Space')
            shared_edges.insert(0, edge)
            edge = edge.previous

        end_edge = shared_edges[-1]
        start_edge = shared_edges[0]

        end_edge.pair.space_previous.space_next = end_edge.next
        start_edge.previous.space_next = start_edge.pair.space_next

        for edge in shared_edges:
            edge.pair.space_next = None

        for edge in end_edge.next.siblings:
            if edge.next is start_edge:
                break
            edge.space_next = edge.next

        # finish by adding the space reference in the face object
        face.space = self

    def remove_face(self, face: Face):
        """
        Remove a face from the space and adjust the edges list accordingly
        :param face: face to remove from space
        """
        if face not in self.faces:
            raise ValueError('Cannot remove a face' +
                             ' that does not belong to the space:{0}'.format(face))

        # 1 : check if the face is the reference stored in the Space
        if self.face is face:
            # verify that the face is not the only face in the Space
            # if another face is found, it becomes the face reference of the Space
            for other_face in self.faces:
                if other_face is not face:
                    self.face = other_face
                    break
            else:
                self.face = None
                self.edge = None
                logging.info('INFO: removing only face left in the Space: {0}'.format(self))
                return

        # 2 : check if the space edge belongs to the face:
        if self.edge in face.edges:
            for boundary_edge in self.edge.space_siblings:
                if boundary_edge is not self.edge and boundary_edge.face is not face:
                    self.edge = boundary_edge
                    break
            else:
                raise ValueError('Something is wrong with this space structure !')

        # 2 : find a touching edge or an enclosed face
        same_face = True
        exit_edge = None
        enclosed_face = None
        for edge in face.edges:
            if self.is_boundary(edge) or self.starts_from_boundary(edge):
                break
            # check for enclosed face (an enclosed face has only one external face)
            # and find the exit edge
            same_face = same_face and (edge.pair.face is edge.next.pair.face)
            if edge.next.pair.next.pair is not edge:
                exit_edge = edge.next.pair.next
        else:
            if not same_face:
                raise ValueError('Can not remove from the space' +
                                 ' a face that is not on the boundary:{0}'.format(face))
            enclosed_face = True

        # CASE 1 : enclosed face
        if enclosed_face:
            logging.info('Found enclosed face')
            exit_edge.space_next = exit_edge.next
            exit_edge.pair.previous.space_next = exit_edge.pair

            # loop around the enclosed face
            edge = exit_edge.pair
            while edge is not exit_edge:
                edge.space_next = edge.next
                edge = edge.next
            return

        # CASE 2 : touching face
        # we will be temporarily breaking the space_next references
        # so we need to store them in a list (no stitching algorithm here)
        edges = []
        for edge in face.edges:
            space_edge = self.starts_from_boundary(edge)
            ante_space_edge = space_edge.space_previous if space_edge else None
            edges.append((edge, space_edge, ante_space_edge))

        previous_edge, previous_space_edge, previous_ante_space_edge = edges[-1]
        for edge_tuple in edges:

            edge, space_edge, ante_space_edge = edge_tuple

            if space_edge is not None:
                if space_edge is not edge:
                    edge.pair.space_next = space_edge
                if previous_space_edge is previous_edge:
                    previous_space_edge.space_next = None
                else:
                    ante_space_edge.space_next = previous_edge.pair
            else:
                edge.pair.space_next = previous_edge.pair

            previous_edge, previous_space_edge, previous_ante_space_edge = edge_tuple

        # check for separated faces
        # store the faces of the space before we remove the face
        space_faces = list(self.faces)
        # remove the face from the Space
        face.space = None
        # check if we still find every face
        for face in space_faces:
            if not face.is_linked_to_space():
                self.plan.add_space(EmptySpace(self.plan, face))

    def cut(self, edge: Edge, vertex: Vertex, angle: float = 90.0, traverse: str = 'absolute'):
        """
        Cuts the space at the corresponding edge
        Adjust the self.faces and self.edges list accordingly
        :param edge:
        :param vertex:
        :param angle:
        :param traverse:
        :return:
        """
        if not self.is_boundary(edge):
            # Important : this prevent the cut of internal space boundary (for space with holes)
            logging.warning('WARNING: Cannot cut an edge that is not' +
                            ' on the boundary of the space:{0}'.format(edge))
            return

        def callback(new_edges: Optional[Tuple[Edge, Edge]]):
            """
            Callback to insure space consistency
            :param new_edges: Tuple of the new edges created by the cut
            """
            start_edge, end_edge = new_edges
            return self.is_boundary(end_edge)

        edge.laser_cut(vertex, angle, callback=callback, traverse=traverse)

    def cut_at_barycenter(self, edge: Optional[Edge] = None, coeff: float = 0.5,
                          angle: float = 90.0, traverse: str = 'absolute'):
        """
        Convenience method
        :param edge:
        :param coeff:
        :param angle:
        :param traverse:
        :return:
        """
        edge = edge or self.edge
        vertex = Vertex().barycenter(edge.start, edge.end, coeff)
        return self.cut(edge, vertex, angle, traverse)

    def plot(self, ax=None):
        """
        plot the space
        """

        # if ax is not provided it means we want to plot just this face
        # so we create a matplot fig and show it
        show_plot = False
        if ax is None:
            fig, ax = plt.subplots()
            show_plot = True

        color = SPACE_COLORS[self.category]
        for face in self.faces:
            face.plot(ax, color=color)
        for edge in self.edges:
            edge.plot_half_edge(ax, color=color)

        if show_plot:
            plt.show()

    def check(self):
        """
        Check consistency of space
        :return:
        """
        is_valid = True
        # check if edges are correct
        if ((self.edge is None) + (self.face is None)) == 1:
            is_valid = False
            logging.error('Error in space: only one of edge or face is None')
            return is_valid

        faces = list(self.faces)

        for edge in self.edges:
            if edge.face not in faces:
                logging.error('Error in space: boundary edge with wrong face')
                is_valid = False

        return is_valid


class EmptySpace(Space):
    """
    Empty Space Class : Used for the empty initial plan
    Has only one face
    """

    category = 'empty'

    def add_fixed_space(self, boundary, category: str) -> 'FixedSpace':
        """
        Adds a fixed item inside the first face of the mesh
        1. Create the corresponding face in the mesh
        2. Create the fixedItem instance of the proper type
        :param boundary:
        :param category:
        :return:
        """
        fixed_space_categories = {
            'frontDoor': EntranceSpace,
            'duct': DuctSpace,
            'window': WindowSpace,
            'doorWindow': WindowSpace
        }

        # create the mesh of the fixed space
        fixed_space_mesh = Mesh().from_boundary(boundary)
        face_of_fixed_item = fixed_space_mesh.faces[0]
        container_face = self.face

        # insert the face in the emptySpace
        new_faces = container_face.insert_face(face_of_fixed_item)
        self.face = new_faces[0]  # per convention the ref. face of the Space is the biggest one

        # remove the face of the fixed_item from the empty space
        self.remove_face(face_of_fixed_item)

        # create the fixedSpace and add it to the plan
        fixed_space = fixed_space_categories[category](self.plan, face_of_fixed_item)
        self.plan.add_space(fixed_space)

        return fixed_space


class FixedSpace(Space):
    """
    Fixed Space Class : Duct,
    """
    pass


class WindowSpace(FixedSpace):
    """
    Window Space Class : Window, entrance
    """

    category = 'window'


class DuctSpace(FixedSpace):
    """
    Fixed Space Class : Duct,
    """
    category = 'duct'


class EntranceSpace(FixedSpace):
    """
    Fixed Space Class : Entrance
    """
    category = 'entrance'


class Wall:
    """
    Wall Class
    """
    def __init__(self, edges: Sequence[Edge]):
        self.edges = edges


class LoadBearingWall(Wall):
    """
    LoadbearingWall Class
    """
    pass


class BoundaryWall(Wall):
    """
    BoundaryWall Class
    """
    pass


class Transformation:
    """
    Transformation class
    Describes a transformation of a mesh
    A transformation queries the edges of the mesh and modify them as prescribed
    • a query
    • a transformation
    """

    pass


class Factory:
    """
    Grid Factory Class
    Creates a mesh according to transformations, boundaries and fixed-items json_input
    and specific rules
    """
    pass


class Query:
    """
    Queries a mesh and returns a generator with the required edges
    """
