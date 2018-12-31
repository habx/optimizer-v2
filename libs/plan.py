# coding=utf-8
"""
Plan Module
Creates the following classes:
• Plan : contains the description of a blue print
• Space : a 2D space in an apartment blueprint : can be a room, or a pillar, or a duct.
• Linear : a 1D object in an apartment. For example : a window, a door or a wall.
TODO : remove infinity loops checks in production
TODO : replace raise ValueError with assertions
"""
from typing import TYPE_CHECKING, Optional, List, Tuple, Sequence, Generator, Union
import logging
import uuid

import matplotlib.pyplot as plt
from shapely.geometry import Polygon, LineString

from libs.mesh import Mesh, Face, Edge, Vertex
from libs.category import LinearCategory, SpaceCategory, SPACE_CATEGORIES
from libs.plot import plot_save, plot_edge, plot_polygon
import libs.transformation as transformation
from libs.size import Size
from libs.utils.custom_types import Coords2d, TwoEdgesAndAFace, Vector2d
from libs.utils.custom_exceptions import OutsideFaceError, OutsideVertexError
from libs.utils.decorator_timer import DecoratorTimer
from libs.utils.geometry import dot_product, normal_vector

if TYPE_CHECKING:
    from libs.seed import Seed


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

    def __repr__(self):
        output = 'Plan ' + self.name + ':  \n'
        for space in self.spaces:
            output += space.__repr__() + ' - \n'
        return output

    def from_boundary(self, boundary: Sequence[Coords2d]) -> 'Plan':
        """
        Creates a plan from a list of points
        1. create the mesh
        2. Add an empty space
        :param boundary:
        :return:
        """
        self.mesh = Mesh().from_boundary(boundary)
        empty_space = Space(self, self.mesh.faces[0].edge)
        self.add_space(empty_space)
        return self

    def add_space(self, space: 'Space'):
        """
        Add a space in the plan
        :param space:
        :return:
        """
        self.spaces.append(space)

    def remove_space(self, space: 'Space'):
        """
        Removes a space from the plan
        :param space:
        :return:
        """
        if space not in self.spaces:
            raise ValueError('Cannot remove from the plan a space that does not belong to it: {0}'
                             .format(space))

        self.spaces.remove(space)

    def add_linear(self, linear: 'Linear'):
        """
        Add a linear in the plan
        :param linear:
        :return:
        """
        linear.plan = self
        self.linears.append(linear)

    def get_component(self,
                      cat_name: Optional[str] = None) -> Generator['PlanComponent', None, None]:
        """
        Returns an iterator of the spaces contained in the place
        :param cat_name: the name of the category
        :return:
        """
        for space in self.spaces:
            if cat_name is not None:
                if space.category.name == cat_name:
                    yield space
            else:
                yield space

        for linear in self.linears:
            if cat_name is not None:
                if linear.category.name == cat_name:
                    yield linear
            else:
                yield linear

    def get_spaces(self, category_name: Optional[str] = None) -> Generator['Space', None, None]:
        """
        Returns an iterator of the spaces contained in the place
        :param category_name:
        :return:
        """
        if category_name is not None:
            return (space for space in self.spaces if space.category.name == category_name)

        return (space for space in self.spaces)

    def is_space_edge(self, edge: 'Edge') -> bool:
        """
        Returns True if the edge is on the boundary of a space
        :param edge:
        :return:
        """
        for space in self.spaces:
            if space.is_boundary(edge):
                return True

        return False

    def is_mutable(self, edge: 'Edge') -> bool:
        """
        Returns True if the edge or its pair does not belong to an immutable linear
        :param edge:
        :return:
        """
        for linear in self.linears:
            if linear.has_edge(edge):
                return linear.category.mutable
        return True

    @property
    def empty_spaces(self) -> Generator['Space', None, None]:
        """
        The first empty space of the plan
        Note : the empty space is only used for the creation of the plan
        :return:
        """
        return self.get_spaces(category_name='empty')

    @property
    def empty_space(self) -> Optional['Space']:
        """
        The largest empty space of the plan
        :return:
        """
        return max(self.empty_spaces, key=lambda space: space.area)

    @property
    def directions(self) -> Sequence[Tuple[float, float]]:
        """
        Returns the main directions of the mesh of the plan
        :return:
        """
        return self.mesh.directions

    @property
    def is_empty(self):
        """
        Returns False if the plan contains mutable space other than empty spaces
        :return:
        """
        for space in self.spaces:
            if space.category.name != 'empty' and space.category.mutable:
                return False
        return True

    def insert_space_from_boundary(self,
                                   boundary: Sequence[Coords2d],
                                   category: SpaceCategory = SPACE_CATEGORIES('empty')):
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
        Inserts a linear object in the plan at the given points
        Will try to insert it in every empty space.
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
                logging.debug("Failed to insert the linear")
                pass
        else:
            raise ValueError('Could not insert the linear in the plan:' +
                             '[{0},{1}] - {2}'.format(point_1, point_2, category))

    @property
    def boundary_as_sp(self) -> Optional[LineString]:
        """
        Returns the boundary of the plan as a LineString
        """
        vertices = []
        edge = None
        for edge in self.mesh.boundary_edges:
            vertices.append(edge.start.coords)
        if edge is None:
            return None
        vertices.append(edge.end.coords)
        return LineString(vertices)

    def plot(self, ax=None, show: bool = False, save: bool = True,
             options: Tuple = ('face', 'edge', 'half-edge', 'border')):
        """
        Plots a plan
        :return:
        """
        for space in self.spaces:
            ax = space.plot(ax, save=False, options=options)

        for linear in self.linears:
            ax = linear.plot(ax, save=False)

        ax.set_title(self.name)

        plot_save(save, show)

        return ax

    def check(self) -> bool:
        """
        Used to verify plan consistency
        NOTE : To be completed
        :return:
        """
        is_valid = self.mesh.check()

        for space in self.spaces:
            is_valid = is_valid and space.check()
            for face_id in space._faces_id:
                for other_space in self.spaces:
                    if other_space is space:
                        continue
                    if face_id in other_space._faces_id:
                        logging.debug("A face is in multiple space: %s", face_id)
                        is_valid = False

        if is_valid:
            logging.info('Checking plan: ' + '✅ OK')
        else:
            logging.info('Checking plan: ' + '🔴 NOT OK')

        return is_valid

    def remove_null_spaces(self):
        """
        Remove from the plan spaces with no edge reference
        :return:
        """
        space_to_remove = []
        for space in self.spaces:
            if space.edge is None:
                space_to_remove.append(space)
        for space in space_to_remove:
            self.remove_space(space)

    def make_space_seedable(self, category):
        """
        Make seedable spaces with specified category name
        :return:
        """
        for space in self.spaces:
            if space.category.name == category:
                space.category.seedable = True

    def count_category_spaces(self, category) -> int:
        """
        count the number of spaces with given category
        :return:
        """
        return sum(space.category.name == category for space in self.spaces)

    def count_mutable_spaces(self) -> int:
        """
        count the number of mutable spaces
        :return:
        """
        return sum(space.category.mutable for space in self.spaces)


class PlanComponent:
    """
    A component of a plan. Can be a linear (1D) or a space (2D)
    """

    def __init__(self, plan: Plan):
        self.plan = plan
        self.category: Union[SpaceCategory, LinearCategory] = None


class Space(PlanComponent):
    """
    Space Class
    """

    def __init__(self, plan: 'Plan', edge: 'Edge',
                 category: SpaceCategory = SPACE_CATEGORIES('empty')):
        super().__init__(plan)
        self._edges_id = [edge.id] if edge else []
        # the new data structure is very simple : a space is a set of face id.
        # + a boundary edge reference
        self._faces_id = [edge.face._id] if edge.face else []
        self.category = category

    def __repr__(self):
        output = 'Space: ' + self.category.name + ' - ' + str(id(self))
        return output

    @property
    def face(self) -> Face:
        """
        property
        The face of the reference edge of the space
        :return:
        """
        return self.edge.face if self.edge else None

    def has_face(self, face: 'Face') -> bool:
        """
        returns True if the face belongs to the space
        :param face:
        :return:
        """

        if face is None:
            return False

        return face._id in self._faces_id

    @property
    def faces(self) -> Generator[Face, None, None]:
        """
        The faces included in the Space. Returns an iterator.
        :return:
        """
        return (self.plan.mesh.get_face(face_id) for face_id in self._faces_id)

    def add_face_id(self, value: uuid.UUID):
        """
        Adds a face_id if possible
        :param value:
        :return:
        """
        if value not in self._faces_id:
            self._faces_id.append(value)

    @property
    def reference_edges(self) -> Generator['Edge', None, None]:
        """
        Yields the reference edge of the space
        :return:
        """
        for edge_id in self._edges_id:
            yield self.plan.mesh.get_edge(edge_id)

    @property
    def edge_is_none(self) -> bool:
        """
        Property
        Returns True if the reference edge of the space is not set
        :return:
        """
        return len(self._edges_id) == 0

    @property
    def edge(self) -> Optional['Edge']:
        """
        Returns the first reference edge.
        Per convention, the first reference edge is on the outside boundary of the space
        :return:
        """
        if self.edge_is_none:
            return None

        return self.plan.mesh.get_edge(self._edges_id[0])

    @edge.setter
    def edge(self, value: 'Edge'):
        """
        Sets the first reference edge
        :param value:
        :return:
        """
        if not self._edges_id:
            self._edges_id = [value.id]
        else:
            self._edges_id[0] = value.id

    def next_edge(self, edge: 'Edge') -> 'Edge':
        """
        Returns the next boundary edge of the space
        :param edge:
        :return:
        """
        next_edge = edge.next
        seen = []
        while not self.is_boundary(next_edge):
            if next_edge in seen:
                raise Exception("The mesh is badly formed for space: %s", self)
            seen.append(next_edge)
            next_edge = next_edge.cw

        return next_edge

    def previous_edge(self, edge: 'Edge') -> Edge:
        """
        Returns the previous boundary edge of the space
        :param edge:
        :return:
        """
        previous_edge = edge.previous
        seen = []
        while not self.is_boundary(previous_edge):
            if previous_edge in seen:
                raise Exception("The mesh is badly formed for space: %s", self)
            seen.append(previous_edge)
            previous_edge = previous_edge.pair.previous

        return previous_edge

    def siblings(self, edge: 'Edge') -> Generator[Edge, None, None]:
        """
        Returns the boundary edges linked to the specified edge
        :param edge:
        :return:
        """
        yield edge
        seen = [edge]
        current_edge = self.next_edge(edge)
        while current_edge is not edge:
            if current_edge in seen:
                raise Exception("The space is badly formed %s at edge %s", self, edge)
            yield current_edge
            seen.append(current_edge)
            current_edge = self.next_edge(current_edge)

    @property
    def edges(self) -> Generator[Edge, None, None]:
        """
        The boundary edges of the space
        :return: an iterator
        """
        for reference_edge in self.reference_edges:
            yield from self.siblings(reference_edge)

    @property
    def exterior_edges(self) -> Generator[Edge, None, None]:
        """
        Returns the exterior perimeter of the space
        :return:
        """
        yield from self.siblings(self.edge)

    @property
    def hole_edges(self) -> [Edge]:
        """
        Returns the internal reference edges
        :return:
        """
        if not self.has_holes:
            return []
        return [self.plan.mesh.get_edge(edge_id) for edge_id in self._edges_id[1:]]

    @property
    def has_holes(self):
        """
        Returns True if the space has internal holes
        :return:
        """
        return len(self._edges_id) > 0

    @property
    def area(self) -> float:
        """
        Returns the area of the Space.
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

    @property
    def as_sp(self) -> Optional[Polygon]:
        """
        Returns a shapely polygon
        :return:
        """
        if self.edge is None:
            return

        list_vertices = [edge.start.coords for edge in self.exterior_edges]
        list_vertices.append(list_vertices[0])

        holes = []
        for hole_edge in self.hole_edges:
            _vertices = [edge.start.coords for edge in self.siblings(hole_edge)]
            _vertices.append(_vertices[0])
            holes.append(_vertices)

        return Polygon(list_vertices, holes)

    def bounding_box(self, vector: Vector2d = None) -> Tuple[float, float]:
        """
        Returns the bounding rectangular box of the space according to the direction vector
        :param vector:
        :return:
        """
        if self.edge_is_none:
            return 0.0, 0.0

        vector = vector or self.edge.unit_vector
        total_x = 0
        max_x = 0
        min_x = 0
        total_y = 0
        max_y = 0
        min_y = 0

        for space_edge in self.edges:
            total_x += dot_product(space_edge.vector, vector)
            max_x = max(total_x, max_x)
            min_x = min(total_x, min_x)
            total_y += dot_product(space_edge.vector, normal_vector(vector))
            max_y = max(total_y, max_y)
            min_y = min(total_y, min_y)

        return max_x - min_x, max_y - min_y

    @property
    def size(self, edge: Optional[Edge] = None) -> Size:
        """
        Returns the size of the space
        :return:
        """
        vector = edge.unit_vector if edge else None
        return Size(self.area, *self.bounding_box(vector))

    @property
    def mutable(self):
        """
        Returns True if the space can be modified
        :return:
        """
        return self.category.mutable

    def is_boundary(self, edge: Edge) -> bool:
        """
        Returns True if the edge is on the boundary of the space.
        :param edge:
        :return:
        """
        return (not self.is_outside(edge)) and self.is_outside(edge.pair)

    def is_internal(self, edge: Edge) -> bool:
        """
        Returns True if the edge is internal of the space
        :param edge:
        :return:
        """
        return not self.is_outside(edge) and not self.is_boundary(edge)

    def is_outside(self, edge: Edge) -> bool:
        """
        Return True if the edge is outside of the space (not on the boundary or inside)
        :param edge:
        :return:
        """
        # per convention
        if edge is None or edge.face is None:
            return True

        return not self.has_face(edge.face)

    def add_first_face(self, face: Face):
        """
        Adds the first face of the space
        :param face:
        :return:
        """
        if self.face is not None:
            raise ValueError('the space already has a face:' +
                             ' {0} - {1}'.format(self, face))
        logging.debug('Adding the first face of the Space: {0}'.format(self))
        self._edges_id = [face.edge.id]
        self._faces_id.append(face._id)

    def add_face(self, face: Face):
        """
        Adds a face to the space
        :param face: face to add to space
        """
        if self.face is None:
            return self.add_first_face(face)
        # preserve edges references
        forbidden_edges = [edge.pair for edge in face.edges]
        self.change_reference_edges(forbidden_edges)

        self._faces_id.append(face._id)

    def remove_face(self, face: Face) -> Sequence['Space']:
        """
        Remove a face from the space and adjust the edges list accordingly
        from the first space and add it to the second one in the same time)
        Note : the biggest challenge of this method is to verify whether the removal
        of the specified face will split the space into several disconnected components.
        A new space must be created for each new disconnected component.
        :param face: face to remove from space
        """
        logging.debug("Removing a face %s, from space %s", face, self)

        # case 1 : the only face of the space
        if len(self._faces_id) == 1:
            return self.remove_only_face(face)

        # case 2 : full enclosed face which will create a hole
        for edge in face.edges:
            if self.is_outside(edge.pair):
                break
        else:
            logging.debug("Removing a full enclosed face. A hole is created")
            self._faces_id.remove(face.id)
            self._edges_id.append(face.edge.pair.id)
            return [self]

        # case 3 : standard case
        forbidden_edges = list(face.edges)
        self.change_reference_edges(forbidden_edges)

        # case 2 : the space has more than 1 face. We must check if we are creating an
        # unconnected space
        adjacent_faces = list(self.adjacent_faces(face))

        # if there is only one adjacent face to the removed one
        # no need to check for connectivity
        if len(adjacent_faces) == 1:
            self._faces_id.remove(face.id)
            return [self]

        remaining_faces = adjacent_faces[:]
        space_connected_components = []
        created_spaces = [self]

        self._faces_id.remove(face._id)

        # we must check to see if we split the space by removing the face
        # for each adjacent face inside the space check if they are still connected
        self_boundary_face = None

        while remaining_faces:

            adjacent_face = remaining_faces[0]
            connected_faces = [adjacent_face]

            for connected_face in self.connected_faces(adjacent_face):
                # try to reach the other adjacent faces
                if connected_face in remaining_faces:
                    remaining_faces.remove(connected_face)
                connected_faces.append(connected_face)

            remaining_faces.remove(adjacent_face)

            if len(remaining_faces) != 0:
                logging.debug("Found a disconnected component")
                space_connected_components.append(connected_faces)
            else:
                logging.debug("Last component found")
                self_boundary_face = adjacent_face
                break

        if len(space_connected_components) == 0:
            return created_spaces

        # we must create a new space per newly created space components
        for component in space_connected_components:
            # create a new space with the disconnected faces and add it to the plan
            for _edge in component[0].edges:
                if _edge.pair.face is face:
                    boundary_edge = _edge
                    break
            else:
                raise Exception("We should have found a boundary edge")

            new_space = Space(self.plan, boundary_edge, self.category)
            # remove the disconnected faces from the initial space
            # and add them to the new space
            for component_face in component:
                self._faces_id.remove(component_face._id)
                new_space.add_face_id(component_face._id)

            # transfer internal edge reference from self to new spaces
            for internal_reference_edge in self.hole_edges:
                if not new_space.is_outside(internal_reference_edge):
                    new_space._edges_id.append(internal_reference_edge.id)
                    self._edges_id.remove(internal_reference_edge.id)

            self.plan.add_space(new_space)
            created_spaces.append(new_space)

        # preserve self edge reference
        if self.is_outside(self.edge):
            for _edge in self_boundary_face.edges:
                if _edge.pair.face is face:
                    boundary_edge = _edge
                    break
            else:
                raise Exception("We should have found a boundary edge")
            self._edges_id[0] = boundary_edge.id

        return created_spaces

    def remove_only_face(self, face: Face) -> Sequence['Space']:
        """
        Removes the only face of the space
        :param face:
        :return: the modified space
        TODO : should we remove the empty space from the plan ?
        """
        logging.debug('Removing only face left in the Space: %s', self)
        self._edges_id = []
        self._faces_id.remove(face._id)
        return [self]

    def set_edges(self):
        """
        Sets the reference edges of the space.
        We need one edge for the exterior boundary, and one edge per hole inside the space
        NOTE : Per convention the edge of the exterior is stored as the first element of the
        _edges_id array.
        :return:
        """
        space_edges = []
        self._edges_id = []
        max_perimeter = 0.0
        for face in self.faces:
            for edge in face.edges:
                if self.is_boundary(edge) and edge not in space_edges:
                    # in order to determine which edge is the exterior one we have to
                    # measure its perimeter
                    perimeter = sum(_edge.length for _edge in self.siblings(edge))
                    if perimeter > max_perimeter:
                        max_perimeter = perimeter
                        self._edges_id = [edge.id] + self._edges_id
                    else:
                        self._edges_id.append(edge.id)

                    space_edges = list(self.edges)

        assert len(self._edges_id), "The space is badly shaped: {}".format(self)

    def change_reference_edges(self, forbidden_edges: ['Edge']):
        """
        Changes the edge reference and returns True if succeeds
        :param forbidden_edges:
        :return:
        """
        i = 0
        for edge in self.reference_edges:
            if edge not in forbidden_edges:
                continue
            for other_edge in self.siblings(edge):
                if other_edge not in forbidden_edges:
                    self._edges_id[i] = other_edge.id
                    break
            else:
                self._edges_id.remove(edge.id)
                i -= 1
            i += 1

    def connected_faces(self, face: Face) -> Generator[Face, None, None]:
        """
        Returns the faces of the space connected to the provided face
        Note: the face provided must belong to the space
        :param face:
        :return:
        """
        assert self.has_face(face), "The face must belong to the space"

        def _propagate(current_face: Face) -> Generator[Face, None, None]:
            for adjacent_face in self.adjacent_faces(current_face):
                if adjacent_face not in seen:
                    seen.append(adjacent_face)
                    yield adjacent_face
                    yield from _propagate(adjacent_face)

        seen = [face]
        return _propagate(face)

    def adjacent_faces(self, face: Face) -> Generator[Face, None, None]:
        """
        Returns the adjacent faces in the space of the face
        :param face:
        :return:
        """
        assert self.has_face(face), "The face must belong to the space"

        seen = [face]
        for edge in face.edges:
            if self.has_face(edge.pair.face) and edge.pair.face not in seen:
                yield edge.pair.face
                seen.append(edge.pair.face)

    def merge(self, *spaces: 'Space') -> 'Space':
        """
        Merge the space with all the other provided spaces.
        Returns the merged space.
        :param spaces:
        :return: self
        """
        for space in spaces:
            self._merge(space)
        return self

    def _merge(self, space: 'Space') -> 'Space':
        """
        Merges two spaces together and return the remaining space
        :param space:
        :return:
        """
        self._faces_id |= space._faces_id
        return self

    def insert_face(self, face: 'Face'):
        """
        Insert a face inside the space reference face
        :param face:
        :return:
        """
        container_face = self.face
        created_faces = container_face.insert_face(face)
        self.add_face(face)
        # we need to add to the space the new faces eventually created by the insertion
        for face in created_faces:
            if face is container_face:
                continue
            self.add_face(face)
        # sometimes the container_face can be deleted by the insertion
        # so we need to check this and remove the deleted face from the space if needed
        if container_face not in created_faces:
            self._faces_id.remove(container_face.id)
        # we must set the boundary in case the reference edge is no longer part of the space
        self.set_edges()

    def insert_face_from_boundary(self, perimeter: Sequence[Coords2d]) -> 'Face':
        """
        Inserts a face inside the space reference face from the given coordinates
        :param perimeter:
        :return:
        """
        face_to_insert = self.plan.mesh.new_face_from_boundary(perimeter)
        try:
            self.insert_face(face_to_insert)
            return face_to_insert
        except OutsideFaceError:
            self.plan.mesh.remove_face_fully(face_to_insert)
            raise

    def insert_space(self,
                     boundary: Sequence[Coords2d],
                     category: SpaceCategory = SPACE_CATEGORIES['empty']) -> 'Space':
        """
        Adds a new space inside the first face of the space
        Used to insert specific space such as duct or load bearing wall
        :param boundary:
        :param category:
        :return: the new space
        """
        face_of_space = self.insert_face_from_boundary(boundary)
        self.remove_face(face_of_space)
        # create the space and add it to the plan
        space = Space(self.plan, face_of_space.edge, category=category)
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
        # TODO : we should not create vertices directly but go trough a face interface
        vertex_1 = Vertex(self.plan.mesh, *point_1)
        vertex_2 = Vertex(self.plan.mesh, *point_2)
        new_edge = self.face.insert_edge(vertex_1, vertex_2)
        new_linear = Linear(self.plan, self, new_edge, category)
        self.plan.add_linear(new_linear)

        return new_linear

    def cut(self,
            edge: Edge,
            vertex: Vertex,
            angle: float = 90.0,
            traverse: str = 'absolute',
            max_length: Optional[float] = None) -> TwoEdgesAndAFace:
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
                            ' on the boundary of the space: %s', edge)
            return None

        def callback(new_edges: Optional[Tuple[Edge, Edge]]) -> bool:
            """
            Callback to insure space consistency
            :param new_edges: Tuple of the new edges created by the cut
            """
            start_edge, end_edge, new_face = new_edges
            # add the created face to the space
            if new_face is not None:
                self.add_face(new_face)

            return self.is_outside(end_edge.pair)

        return edge.recursive_cut(vertex, angle, traverse=traverse, callback=callback,
                                  max_length=max_length)

    def barycenter_cut(self, edge: Optional[Edge] = None, coeff: float = 0.5,
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
        vertex = (transformation.get['barycenter']
                  .config(vertex=edge.end, coeff=coeff)
                  .apply_to(edge.start))
        return self.cut(edge, vertex, angle, traverse, max_length=max_length)

    def plot(self, ax=None,
             save: Optional[bool] = None,
             options: Tuple['str'] = ('face', 'border', 'half-edge')):
        """
        plot the space
        """
        # do not try to plot an empty space
        if self.edge is None:
            return ax

        color = self.category.color
        x, y = self.as_sp.exterior.xy
        ax = plot_polygon(ax, x, y, options, color, save)

        if 'face' in options:
            for face in self.faces:
                if face is None:
                    continue
                ax = face.plot(ax, color=color, save=save, options=('fill', 'border', 'dash'))

        if 'half-edge' in options:
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
            logging.error('Error in space: only one of edge or face is None: {0}'.format(self.edge))
            return is_valid

        faces = list(self.faces)

        for edge in self.edges:
            if edge.face not in faces:
                logging.error('Error in space: boundary edge face not in space faces: ' +
                              '{0} - {1}'.format(edge, edge.face))
                is_valid = False

        return is_valid

    def immutable_components(self) -> ['PlanComponent']:
        """
        Return the components associated to the space
        :return: [PlanComponent]
        """
        immutable_associated = []

        for linear in self.plan.linears:
            if linear.space is self and not linear.category.mutable:
                immutable_associated.append(linear)

        for space in self.plan.spaces:
            if not space.category.mutable and space.adjacent_to(self):
                immutable_associated.append(space)

        return immutable_associated

    def components_category_associated(self) -> [str]:
        """
        Return the name of the components associated to the space
        :return: [Plan Component name]
        """
        return [component.category.name for component in self.immutable_components()]

    def neighboring_mutable_spaces(self) -> ['Space']:
        """
        Return the neighboring mutable spaces
        :return: ['Space']
        """
        neighboring_spaces = []
        for edge in self.edges:
            if edge.pair.face is not None and edge.pair.face.space.category.mutable is True:
                if not (edge.pair.face.space in neighboring_spaces):
                    neighboring_spaces.append(edge.pair.face.space)
        return neighboring_spaces

    def adjacent_to(self, other: 'Space') -> bool:
        """
        Check the adjacency with an other space
        :return:
        """
        for edge in self.edges:
            if other.has_face(edge.pair.face):
                return True
        return False


class Linear(PlanComponent):
    """
    Linear Class
    A linear is an object composed of one or several contiguous edges localized on the boundary
    of a space object
    """

    def __init__(self, plan: Plan, space: 'Space', edge: Edge, category: LinearCategory):

        if not plan.is_space_edge(edge):
            raise ValueError('cannot create a linear that is not on the boundary of a space')

        if space.plan is not plan:
            raise ValueError('cannot create a linear of a space not belonging to the same plan')

        super().__init__(plan)
        self.category = category
        self.space = space
        self._edges_id = [edge.id]

    def __repr__(self):
        return 'Linear: ' + self.category.__repr__() + ' - ' + str(id(self))

    @property
    def edges(self) -> Generator[Edge, None, None]:
        """
        All the edges of the Linear
        :return:
        """
        return (self.plan.mesh.get_edge(edge_id) for edge_id in self._edges_id)

    def add_edge(self, edge: Edge):
        """
        Add an edge to the linear
        :return:
        """
        if not self.plan.is_space_edge(edge):
            raise ValueError('cannot add an edge to a linear' +
                             ' that is not on the boundary of a space')
        if edge.id not in self._edges_id:
            self._edges_id.append(edge.id)

    def has_edge(self, edge: 'Edge') -> bool:
        """
        Returns True if the edge belongs to the linear
        :param edge:
        :return:
        """
        return edge.id in self._edges_id

    @property
    def as_sp(self) -> Optional[LineString]:
        """
        Returns a shapely LineString
        :return:
        """
        vertices = []
        edge = None
        for edge in self.edges:
            vertices.append(edge.start.coords)
        if edge is None:
            return None
        vertices.append(edge.end.coords)
        return LineString(vertices)

    @property
    def length(self) -> float:
        """
        Returns the length of the Linear.
        :return:
        """
        _length = 0.0
        for edge in self.edges:
            _length += edge.length

        return _length

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
        if len(list(self.edges)) == 1:
            return is_valid

        for edge in self.edges:
            if (self.space.next_edge(edge) not in self.edges
                    or self.space.previous_edge(edge) not in self.edges):
                return False

        return is_valid


class SeedSpace(Space):
    """"
    A space use to seed a plan
    """

    def __init__(self, plan: Plan, edge: Edge, seed: 'Seed'):
        super().__init__(plan, edge, SPACE_CATEGORIES['seed'])
        self.seed = seed

    def face_component(self, face: 'Face') -> bool:
        """
        Returns True if the face is linked to a component of the Space
        :param face:
        :return:
        """
        for edge in face.edges:
            if edge in self.seed.edges:
                return True
        else:
            return False


if __name__ == '__main__':
    import libs.reader as reader

    logging.getLogger().setLevel(logging.DEBUG)


    @DecoratorTimer()
    def floor_plan():
        """
        Test the creation of a specific blueprint
        :return:
        """
        input_file = "Paris18_A501.json"
        plan = reader.create_plan_from_file(input_file)

        plan.plot(save=False)
        plt.show()

        for empty_space in plan.empty_spaces:
            boundary_edges = list(empty_space.edges)

            for edge in boundary_edges:
                if edge.length > 30:
                    empty_space.barycenter_cut(edge, 0)
                    empty_space.barycenter_cut(edge, 1)

        plan.plot(save=False)
        plt.show()

        assert plan.check()

    floor_plan()

    def add_two_face_touching_internal_edge_and_border():
        """
        Test. Create a new face, remove it, then add it again.
        :return:
        """
        perimeter = [(0, 0), (500, 0), (500, 500), (0, 500)]
        hole = [(200, 200), (300, 200), (300, 300), (200, 300)]
        hole_2 = [(0, 150), (150, 150), (150, 200), (0, 200)]
        hole_3 = [(0, 200), (150, 200), (150, 300), (0, 300)]

        plan = Plan().from_boundary(perimeter)

        plan.empty_space.insert_face_from_boundary(hole)
        face_to_remove = list(plan.empty_space.faces)[1]
        plan.empty_space.remove_face(face_to_remove)

        plan.plot(save=False)
        plt.show()

        plan.empty_space.insert_face_from_boundary(hole_2)
        face_to_remove = list(plan.empty_space.faces)[0]
        plan.plot(save=False)
        plt.show()
        plan.empty_space.remove_face(face_to_remove)

        plan.plot(save=False)
        plt.show()

        plan.empty_space.insert_face_from_boundary(hole_3)
        face_to_remove = list(plan.empty_space.faces)[1]
        plan.empty_space.remove_face(face_to_remove)

        assert plan.check()

    # add_two_face_touching_internal_edge_and_border()
