# coding=utf-8
"""
Plan Module
Creates the following classes:
• Plan : contains the description of a blue print
• Space : a 2D space in an apartment blueprint : can be a room, or a pillar, or a duct.
• Linear : a 1D object in an apartment. For example : a window, a door or a wall.
"""
from typing import Optional, List, Tuple, Sequence, Generator
import logging
import matplotlib.pyplot as plt

from libs.mesh import Mesh, Face, Edge, Vertex
from libs.category import space_categories, LinearCategory, SpaceCategory
from libs.plot import plot_save, plot_edge
from libs.utils.custom_types import Coords2d
from libs.utils.custom_exceptions import OutsideFaceError, OutsideVertexError
from libs.utils.decorator_timer import DecoratorTimer


class Plan:
    """
    Main class containing the floor plan of the apartment
    • mesh : the corresponding geometric mesh
    • spaces : rooms or ducts or pillars etc.
    • linears : windows, doors, walls etc.
    """
    def __init__(self, name: str = 'unnamed_plan', mesh: Optional[Mesh] = None,
                 spaces: Optional[List['Space']] = None, linears: Optional[List['Linear']] = None):
        self.name = name
        self.mesh = mesh
        self.spaces = spaces or []
        self.linears = linears or []

    def from_boundary(self, boundary: Sequence[Coords2d]):
        """
        Creates a plan from a list of points
        1. create the mesh
        2. Add an empty space
        :param boundary:
        :return:
        """
        self.mesh = Mesh().from_boundary(boundary)
        empty_space = Space(self, self.mesh.faces[0])
        self.add_space(empty_space)

        return self

    def add_space(self, space: 'Space'):
        """
        Add a space in the plan
        :param space:
        :return:
        """
        self.spaces.append(space)

    def add_linear(self, linear: 'Linear'):
        """
        Add a linear in the plan
        :param linear:
        :return:
        """
        self.linears.append(linear)

    def get_spaces(self, category: Optional[str] = None) -> Generator['Space', None, None]:
        """
        Returns an iterator of the spaces contained in the place
        :param category:
        :return:
        """
        if category is not None:
            return (space for space in self.spaces if space.category.name == category)

        return (space for space in self.spaces)

    @property
    def empty_spaces(self) -> Generator['Space', None, None]:
        """
        The first empty space of the plan
        Note : the empty space is only used for the creation of the plan
        :return:
        """
        return self.get_spaces(category='empty')

    @property
    def empty_space(self) -> Optional['Space']:
        """
        The first empty space of the plan
        Note : the empty space is only used for the creation of the plan
        :return:
        """
        return list(self.empty_spaces)[0]

    @property
    def directions(self) -> Sequence[Tuple[float, float]]:
        """
        Returns the main directions of the mesh of the plan
        :return:
        """
        return self.mesh.directions

    def insert_space_from_boundary(self,
                                   boundary: Sequence[Tuple[Coords2d]],
                                   category: SpaceCategory = space_categories['empty']):
        """
        Inserts a new space inside the reference face of the space.
        By design, will not insert a space overlapping several faces of the receiving space.
        The new space is designed from the boundary. By default, the category is empty.
        :param boundary
        :param category
        """
        for empty_space in self.empty_spaces:
            try:
                empty_space.insert_space(boundary, category)
                break
            except OutsideFaceError:
                continue
        else:
            # this should probably raise an exception but too many input blueprints are
            # incorrect due to wrong load bearing walls geometry, it would fail too many tests
            logging.error('Could not insert the space in the plan because it overlaps other non' +
                          ' empty spaces: {0}, {1}'.format(boundary, category))

    def insert_linear(self, point_1: Coords2d, point_2: Coords2d, category: LinearCategory):
        """
        Inserts a face in an empty face
        :param point_1
        :param point_2
        :param category
        :return:
        """
        for empty_space in self.empty_spaces:
            try:
                empty_space.insert_linear(point_1, point_2, category)
                break
            except OutsideVertexError:
                pass
        else:
            raise ValueError('Could not insert the linear in the plan:' +
                             '[{0},{1}] - {2}'.format(point_1, point_2, category))

    def plot(self, ax=None, show: bool = False, save: bool = True):
        """
        Plots a plan
        :return:
        """
        for space in self.spaces:
            ax = space.plot(ax, save=False)

        for linear in self.linears:
            ax = linear.plot(ax, save=False)

        ax.set_title(self.name)

        plot_save(save, show)

    def check(self) -> bool:
        """
        Used to verify plan consistency
        NOTE : To be completed
        :return:
        """
        is_valid = self.mesh.check()

        for space in self.spaces:
            is_valid = is_valid and space.check()

        if is_valid:
            logging.info('Checking plan: ' + '✅ 0K')
        else:
            logging.info('Checking plan: ' + '🔴 NOT OK')

        return is_valid


class Space:
    """
    Space Class
    """
    def __init__(self,
                 plan: Plan,
                 face: Face,
                 boundary_edge: Optional[Edge] = None,
                 category: SpaceCategory = space_categories['empty']):
        self.plan = plan
        self.category = category
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
        self._edge = value

    @property
    def faces(self) -> Generator[Face, None, None]:
        """
        The faces included in the Space. Returns an iterator.
        :return:
        """
        seen = [self.face]
        yield self.face

        def get_adjacent_faces(face: Face) -> Generator[Face, None, None]:
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
    def edges(self) -> Generator[Edge, None, None]:
        """
        The boundary edges of the space
        :return:
        """
        if self.edge is None:
            return
        yield from self.edge.space_siblings

    @property
    def area(self) -> float:
        """
        Returns the area of the Space
        :return:
        """
        _area = 0.0
        for face in self.faces:
            _area += face.area

        return _area

    @property
    def perimeter(self) -> float:
        """
        Returns the length of the Space perimeter
        :return:
        """
        _perimeter = 0.0

        for edge in self.edges:
            _perimeter += edge.length

        return _perimeter

    def is_boundary(self, edge: Edge) -> bool:
        """
        Returns True if the edge is on the boundary of the space.
        :param edge:
        :return:
        """
        if self.is_outside(edge):
            return False

        return edge.is_space_boundary

    def starts_from_boundary(self, edge: Edge) -> Optional[Edge]:
        """
        Returns the boundary edge if the edge belongs to the Space and starts from a boundary edge
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

    def is_internal(self, edge: Edge) -> bool:
        """
        Returns True if the edge is internal of the space
        :param edge:
        :return:
        """
        if self.is_outside(edge):
            return False

        return not self.is_boundary(edge)

    def is_outside(self, edge: Edge) -> bool:
        """
        Return True if the edge is outside of the space (not on the boundary or inside)
        :param edge:
        :return:
        """
        return edge.face is None or edge.face.space is not self

    def add_face(self, face: Face, start_from: Optional[Edge] = None):
        """
        Adds a face to the space and adjust the edges list accordingly
        If the added face belongs to another space we first need to remove it from the space
        We do not enable to add a face inside a hole in the face (enclosed face)
        TODO : if the face belongs to another Space we should also correct the other Space boundary
        :param face: face to add to space
        :param start_from: an edge to start the adjacency search from
        """
        # if the space has no faces yet just add the face as the reference for the Space
        if self.face is None:
            self.face = face
            return

        # check if the face is already inside the space
        if face in self.faces:
            raise ValueError('Cannot add a face that already ' +
                             'belongs to the space: {0}'.format(face))

        # we start the search for a boundary edge from the start_from edge or the face edge
        start_from = start_from or face.edge

        # else we make sure the new face is adjacent to at least one of the space faces
        space_edges = []
        for edge in start_from.siblings:
            if not space_edges:
                # first boundary edge found
                if self.is_boundary(edge.pair):
                    space_edges.append(edge)
            else:
                # note : we only keep "connected" space boundaries
                if self.is_boundary(edge.pair) and edge.pair.space_next is space_edges[-1].pair:
                    space_edges.append(edge)
                else:
                    break
        else:
            if not space_edges:
                raise ValueError('Cannot add a face that is not adjacent' +
                                 ' to the space:{0}'.format(face))
            else:
                raise ValueError('Cannot add a face that is completely enclosed in the Space')

        # check for previous space boundaries:
        edge = space_edges[0].pair.space_next.pair
        while edge.face is face:
            if edge in space_edges:
                raise ValueError('Cannot add a face that is completely enclosed in the Space')
            space_edges.insert(0, edge)
            edge = edge.pair.space_next.pair

        end_edge = space_edges[-1]
        start_edge = space_edges[0]

        end_edge.pair.space_previous.space_next = end_edge.next
        start_edge.previous.space_next = start_edge.pair.space_next

        # remove the old space references
        for edge in space_edges:
            edge.pair.space_next = None

        # add the new space references inside the added face
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
        if face.space is not self or face not in self.faces:
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
                logging.info('Removing only face left in the Space: {0}'.format(self))
                return

        # 2 : check if the space edge belongs to the face and assign another to
        #     preserve space edge reference
        if self.edge in face.edges:
            for boundary_edge in self.edge.space_siblings:
                if boundary_edge is not self.edge and boundary_edge.face is not face:
                    self.edge = boundary_edge
                    break
            else:
                raise ValueError('Something is wrong with this space structure !')

        # 2 : find a boundary edge or an enclosed face
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
            face.space = None
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
                # if the boundary edge of the face belong to a disconnected face
                # we must change it (TODO : we could find a cleaner way...)
                if self.edge.face is face:
                    for other_face in space_faces:
                        if other_face is face:
                            continue
                        if other_face.space is self:
                            for edge in other_face.edges:
                                if self.is_boundary(edge):
                                    self.edge = edge
                                    break
                            if self.edge.face is not face:
                                break
                # create a new space of the same category
                self.plan.add_space(Space(self.plan, face, category=self.category))

    def insert_space(self,
                     boundary: Sequence[Coords2d],
                     category: SpaceCategory = space_categories['empty']) ->'Space':
        """
        Adds a new space inside the first face of the space
        :param boundary:
        :param category:
        :return: the new space
        """
        # create the mesh of the fixed space
        space_mesh = Mesh().from_boundary(boundary)
        face_of_space = space_mesh.faces[0]
        container_face = self.face

        # insert the face in the emptySpace
        new_faces = container_face.insert_face(face_of_space)
        self.face = new_faces[0]  # per convention the ref. face of the Space is the biggest one

        # remove the face of the fixed_item from the empty space
        self.remove_face(face_of_space)

        # create the space and add it to the plan
        space = Space(self.plan, face_of_space, category=category)
        self.plan.add_space(space)

        return space

    def insert_linear(self,
                      point_1: Coords2d,
                      point_2: Coords2d,
                      category: LinearCategory) -> 'Linear':
        """
        Inserts a linear inside the Space boundary given a
        :return: a linear
        """
        vertex_1 = Vertex(*point_1)
        vertex_2 = Vertex(*point_2)
        new_edge = self.face.insert_edge(vertex_1, vertex_2)
        new_linear = Linear(new_edge, category)
        self.plan.add_linear(new_linear)

        return new_linear

    def cut(self,
            edge: Edge,
            vertex: Vertex,
            angle: float = 90.0,
            traverse: str = 'absolute',
            max_length: Optional[float] = None):
        """
        Cuts the space at the corresponding edge
        Adjust the self.faces and self.edges list accordingly
        :param edge:
        :param vertex:
        :param angle:
        :param traverse:
        :param max_length
        :return:
        """
        if not self.is_boundary(edge):
            # Important : this prevent the cut of internal space boundary (for space with holes)
            logging.warning('WARNING: Cannot cut an edge that is not' +
                            ' on the boundary of the space:{0}'.format(edge))
            return

        # TODO : not sure about this. Does not seem like the best approach.
        # probably best to slice non rectilinear space into smaller simple space,
        # than apply a grid generation to these spaces
        # max_length = max_length if max_length is not None else edge.max_length

        def callback(new_edges: Optional[Tuple[Edge, Edge]]) -> bool:
            """
            Callback to insure space consistency
            :param new_edges: Tuple of the new edges created by the cut
            """
            start_edge, end_edge, new_face = new_edges
            return end_edge.pair.space is not self

        edge.laser_cut(vertex, angle, traverse=traverse, callback=callback,
                       max_length=max_length)

    def cut_at_barycenter(self, edge: Optional[Edge] = None, coeff: float = 0.5,
                          angle: float = 90.0, traverse: str = 'absolute',
                          max_length: Optional[float] = None):
        """
        Convenience method
        :param edge:
        :param coeff:
        :param angle:
        :param traverse:
        :param max_length:
        :return:
        """
        edge = edge or self.edge
        vertex = Vertex().barycenter(edge.start, edge.end, coeff)
        return self.cut(edge, vertex, angle, traverse, max_length=max_length)

    def plot(self, ax=None, save: Optional[bool] = None):
        """
        plot the space
        """
        color = self.category.color
        for face in self.faces:
            ax = face.plot(ax, color=color, save=save)
        for edge in self.edges:
            edge.plot_half_edge(ax, color=color, save=save)

        return ax

    def check(self) -> bool:
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
            if edge.space is not self:
                logging.error('Error in edge: boundary edge with wrong space')
                is_valid = False

        return is_valid


class Linear:
    """
    Linear Class
    A linear is an object composed of one or several contiguous edges localized on the boundary
    of a space object
    """
    def __init__(self, edge: Edge, category: LinearCategory):
        """
        Init
        :param edge: one of the edge of the Linear. A linear can have more than one edge.
        """
        if edge.space_next is None:
            raise ValueError('cannot create a linear that is not on the boundary of a space')

        self._edge = edge
        # set the circular reference
        edge.linear = self
        self.category = category

    @property
    def edge(self) -> Edge:
        """
        property
        :return: the reference edge of the linear
        """
        return self._edge

    @edge.setter
    def edge(self, value: Edge):
        if value.space_next is None:
            raise ValueError('cannot create a linear that is not on the boundary of a space')
        self._edge = value
        value.linear = self

    @property
    def edges(self) -> Generator[Edge, None, None]:
        """
        All the edges of the Linear
        :return:
        """
        return (edge for edge in self.edge.space_siblings if edge.linear is self)

    def add_edge(self, edge: Edge):
        """
        Add an edge to the linear
        :return:
        """
        if edge.space_next is None:
            raise ValueError('cannot add an edge to a linear' +
                             ' that is not on the boundary of a space')

        if self.edge is None:
            self.edge = Edge
            return

        if self in (edge.space_next, edge.space_previous):
            edge.linear = self
        else:
            raise ValueError('Cannot add an edge that is not connected to the linear' +
                             ' on a space boundary')

    def plot(self, ax=None, save: bool = None):
        """
        Plots the linear object
        :return:
        """
        for edge in self.edges:
            x_coords, y_coords = zip(*edge.as_sp.coords)
            ax = plot_edge(x_coords, y_coords, ax,
                           color=self.category.color,
                           width=self.category.width, alpha=0.6, save=save)
        return ax

    def check(self) -> bool:
        """
        Check if the linear is valid.
        A linear is valid if all its edges are connected.
        :return:
        """
        is_valid = True
        for edge in self.edges:
            if edge is self.edge:
                continue
            if self not in (edge.space_next, edge.space_previous):
                is_valid = False

        return is_valid


if __name__ == '__main__':

    import libs.reader as reader
    logging.getLogger().setLevel(logging.INFO)

    @DecoratorTimer()
    def floor_plan():
        """
        Test the creation of a specific blueprint
        :return:
        """
        input_file = "Bussy_Regis.json"
        plan = reader.create_plan_from_file(input_file)

        for empty_space in plan.empty_spaces:
            boundary_edges = list(empty_space.edges)

            for edge in boundary_edges:
                if edge.length > 30:
                    empty_space.cut_at_barycenter(edge, 0)
                    empty_space.cut_at_barycenter(edge, 1)

        plan.plot(save=False)
        plt.show()

        assert plan.check()

    floor_plan()
