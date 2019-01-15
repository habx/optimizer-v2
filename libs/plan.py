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
from typing import Optional, List, Tuple, Sequence, Generator, Union
import logging
import uuid

import matplotlib.pyplot as plt
from shapely.geometry import Polygon, LineString, LinearRing

from libs.mesh import Mesh, Face, Edge, Vertex, MeshOps
from libs.category import LinearCategory, SpaceCategory, SPACE_CATEGORIES
from libs.plot import plot_save, plot_edge, plot_polygon
import libs.transformation as transformation
from libs.size import Size
from libs.utils.custom_types import Coords2d, TwoEdgesAndAFace, Vector2d
from libs.utils.custom_exceptions import OutsideFaceError, OutsideVertexError
from libs.utils.decorator_timer import DecoratorTimer
from libs.utils.geometry import dot_product, normal_vector, ccw_angle


class PlanComponent:
    """
    A component of a plan. Can be a linear (1D) or a space (2D)
    """

    def __init__(self,
                 plan: 'Plan',
                 floor: 'Floor',
                 _id: Optional['uuid.UUID'] = None):

        assert floor.id in plan.floors, "PlanComponent: The floor is not in the plan!"

        self.id = _id or uuid.uuid4()
        self.plan = plan
        self.category: Union[SpaceCategory, LinearCategory] = None
        self.floor = floor

        # add the component to the plan
        self.add()

    @property
    def edge(self) -> 'Edge':
        """
        Returns the reference edge of the plan component
        :return:
        """
        raise NotImplementedError

    @property
    def mesh(self) -> Optional['Mesh']:
        """
        Returns the mesh of the plan component
        :return:
        """
        return self.floor.mesh

    @property
    def edges(self) -> Generator['Edge', None, None]:
        """
        Returns the edges of the plan component
        :return:
        """
        raise NotImplementedError

    def remove(self):
        """
        remove from the plan
        :return:
        """
        self.plan.remove(self)

    def add(self):
        """
        Add the element to the plan
        :return:
        """
        self.plan.add(self)


class Space(PlanComponent):
    """
    Space Class
    A very simple data structure : a space is :
    • a list of the id of the faces composing the space
    • a list of the id of the reference edges of the space (an edge is a reference if is localised
    on the boundary of the space). We store one an only one reference edge per boundary.
    (ex. a space with no hole has one reference edge, a space with a hole has two reference edges :
    one for the exterior boundary, one for the hole boundary).
    Per convention, the first element of the edges id list is the reference edge of the exterior
    boundary.
    • a category
    • a ref to its parent plan
    """

    def __init__(self,
                 plan: 'Plan',
                 floor: 'Floor',
                 edge: Optional['Edge'],
                 category: SpaceCategory = SPACE_CATEGORIES['empty'],
                 _id: Optional[uuid.UUID] = None):
        super().__init__(plan, floor, _id=_id)
        self._edges_id = [edge.id] if edge else []
        self._faces_id = [edge.face._id] if edge and edge.face else []
        self.category = category

    def __repr__(self):
        output = 'Space: ' + self.category.name + ' - ' + str(id(self))
        return output

    def clone(self, plan: 'Plan') -> 'Space':
        """
        Creates a clone of the space
        The plan and the category are passed by reference
        the edges and faces id list are shallow copied (as they only contain immutable uuid).
        :return:
        """
        new_space = Space(plan, self.floor, None, category=self.category, _id=self.id)
        new_space._faces_id = self._faces_id[:]
        new_space._edges_id = self._edges_id[:]
        return new_space

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

    def has_edge(self, edge: 'Edge') -> bool:
        """
        Returns True if the edge belongs to the space
        :param edge:
        :return:
        """
        return self.has_face(edge.face)

    def has_linear(self, linear: 'Linear') -> bool:
        """
        Returns True if the linear is on the space boundary
        :param linear:
        :return:
        """
        return linear.edge in self.edges

    @property
    def faces(self) -> Generator[Face, None, None]:
        """
        The faces included in the Space. Returns an iterator.
        :return:
        """
        return (self.mesh.get_face(face_id) for face_id in self._faces_id)

    def add_face_id(self, face: 'Face'):
        """
        Adds a face_id if possible
        :param face:
        :return:
        """
        if face.id not in self._faces_id:
            self._faces_id.append(face.id)

    def remove_face_id(self, face: 'Face'):
        """
        Removes a face_id
        :param face:
        :return:
        """
        self._faces_id.remove(face.id)

    @property
    def reference_edges(self) -> Generator['Edge', None, None]:
        """
        Yields the reference edge of the space
        :return:
        """
        for edge_id in self._edges_id:
            yield self.mesh.get_edge(edge_id)

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

        return self.mesh.get_edge(self._edges_id[0])

    @edge.setter
    def edge(self, value: 'Edge'):
        """
        Sets the first reference edge
        :param value:
        :return:
        """
        if not self._edges_id:
            self._edges_id = [value.id]
        self._edges_id[0] = value.id

    def add_edge(self, edge: 'Edge'):
        """
        Adds a reference edge
        :param edge:
        :return:
        """
        if edge.id in self._edges_id:
            return
        self._edges_id.append(edge.id)

    def remove_edge(self, edge: 'Edge'):
        """
        Removes a reference edge
        :param edge:
        :return:
        """
        assert self._edges_id[0] != edge.id, "Cannot remove the exterior reference edge"
        self._edges_id.remove(edge.id)

    def next_edge(self, edge: 'Edge') -> 'Edge':
        """
        Returns the next boundary edge of the space
        :param edge:
        :return:
        """
        assert self.is_boundary(edge), ("The edge has to be a boundary "
                                        "edge: {0} of space: {1}".format(edge, self))
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
        assert self.is_boundary(edge), "The edge has to be a boundary edge: {}".format(edge)
        previous_edge = edge.previous
        seen = []
        while not self.is_boundary(previous_edge):
            if previous_edge in seen:
                raise Exception("The mesh is badly formed for space: %s", self)
            seen.append(previous_edge)
            previous_edge = previous_edge.pair.previous

        return previous_edge

    def next_angle(self, edge: 'Edge') -> float:
        """
        Returns the angle betwen the edge and the next edge on the boundary
        :param edge:
        :return:
        """
        assert self.is_boundary(edge), "The edge has to be a boundary edge: {}".format(edge)
        return ccw_angle(self.next_edge(edge).vector, edge.opposite_vector)

    def previous_angle(self, edge: 'Edge') -> float:
        """
        Returns the angle between the edge and the prev
        :param edge:
        :return:
        """
        assert self.is_boundary(edge), "The edge has to be a boundary edge: {}".format(edge)
        return ccw_angle(edge.vector, self.previous_edge(edge).opposite_vector)

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
                raise Exception("A reference edge is wrong for space %s at edge %s", self, edge)
            yield current_edge
            seen.append(current_edge)
            current_edge = self.next_edge(current_edge)

    @property
    def edges(self) -> Generator[Edge, None, None]:
        """
        The boundary edges of the space
        :return: an iterator
        """
        if self.edge:
            seen = []
            for reference_edge in self.reference_edges:
                for edge in self.siblings(reference_edge):
                    assert edge not in seen, "The space reference edges are wrong: {}".format(self)
                    seen.append(edge)
                    yield edge

    @property
    def exterior_edges(self) -> Generator[Edge, None, None]:
        """
        Returns the exterior perimeter of the space
        :return:
        """
        if not self.edge:
            return
        yield from self.siblings(self.edge)

    @property
    def hole_edges(self) -> Generator[Edge, None, None]:
        """
        Returns the internal reference edges
        :return:
        """
        return (self.mesh.get_edge(edge_id) for edge_id in self._edges_id[1:])

    @property
    def has_holes(self):
        """
        Returns True if the space has internal holes
        :return:
        """
        return len(self._edges_id) > 1

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
        if self.edge is None:
            return 0.0, 0.0

        vector = vector or self.edge.unit_vector
        total_x = 0
        max_x = 0
        min_x = 0
        total_y = 0
        max_y = 0
        min_y = 0

        for space_edge in self.exterior_edges:
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
        Per convention the bounding box is used to estimate a width and a depth
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

    def add_face(self, face: Face):
        """
        Adds a face to the space
        :param face: face to add to space
        We have to check for the edge case where we create a hole in the space by adding
        a "U" shaped face
        +------------+
        |    Face    |
        +--+------+--+
        |  | Hole |  |
        |  +------+  |
        |    Space   |
        +------------+
        """
        logging.debug("Space: Adding a face %s, to space %s", face, self)

        # case 1: adding the first face
        if self.face is None:
            logging.debug('Space: Adding the first face of the Space: %s', self)
            self.edge = face.edge
            self.add_face_id(face)
            return

        # case 2: adding an enclosing face
        #     +-------------------+
        #     |                   |
        #     |       Face        |
        #     |                   |
        #     +----+---------+    |
        #     |    |         |    |
        #     |    |  Space  |    |
        #     |    |         |    |
        #     |    +---------+    |
        #     |                   |
        #     +-------------------+

        face_edges = list(face.edges)
        for edge in self.exterior_edges:
            if edge.pair not in face_edges:
                break
            face_edges.remove(edge.pair)
        else:
            logging.debug("Space: Adding a fully enclosing face")
            # make sure we are not selecting an internal edge
            for edge in face_edges:
                if edge.pair not in face_edges:
                    self.edge = edge
                    break
            else:
                raise Exception("This should never happen !")

            self.add_face_id(face)

        # case 3 : standard case
        # sadly we have to check if a hole has been created
        check_for_hole = False
        shared_edges = []
        boundary_edges = list(self.edges)
        for edge in face.edges:
            if edge.pair in boundary_edges:
                shared_edge = edge.pair
                if (shared_edges and
                        (self.next_edge(shared_edge) not in shared_edges
                         and self.next_edge(shared_edges[0]) is not shared_edge)):
                    logging.debug("Space: Found a discontinuity border")
                    check_for_hole = True
                    break
                shared_edges.append(shared_edge)

        # preserve edges references
        forbidden_edges = [edge.pair for edge in face.edges]
        self.change_reference_edges(forbidden_edges)
        self.add_face_id(face)

        if check_for_hole:
            self.set_edges()

    def _clean_hole_disappearance(self):
        """
        Check if the removal of a face has linked an internal hole with the exterior of the space.
        If it is the case : we must remove the corresponding edge reference
        Example of a case where a hole is removed from the initial space :
         +------------------------+
         |                        |
         |       +-------+        |
         |       |       +--------+
         | space | hole  |  face  | exterior
         |       |       +--------+
         |       +-------+        |
         |                        |
         |                        |
           +------------------------+
        :return:
        """
        if not self.has_holes:
            return
        exterior_edges = list(self.exterior_edges)
        for edge in self.hole_edges:
            if edge in exterior_edges:
                self.remove_edge(edge)

    def _check_edges_references(self) -> bool:
        """
        Returns True if the edges are ok
        :return:
        """
        # check for duplicates
        if len(self._edges_id) != len(list(set(self._edges_id))):
            logging.warning("Space: Found duplicates in edges list: %s", self)
            return False

        # check for disconnectivity:
        for edge in self.hole_edges:
            if edge in self.exterior_edges:
                logging.warning("Space: Found connected reference edges: %s", self)
                return False

    def remove_face(self, face: Face) -> [[Optional['Space']]]:
        """
        Remove a face from the space

        Note : the biggest challenge of this method is to verify whether the removal
        of the specified face will split the space into several disconnected components.
        A new space must be created for each new disconnected component.

        :param face: face to remove from space
        :returns the modified spaces (including the created spaces)
        """
        logging.debug("Space: Removing a face %s, from space %s", face, self)

        assert self.has_face(face), "Cannot remove a face not belonging to the space"

        # case 1 : the only face of the space
        if len(self._faces_id) == 1:
            self.remove_only_face(face)
            return [self]

        # case 2 : fully enclosed face which will create a hole
        #     +-------------------+
        #     |                   |
        #     |      Space        |
        #     |                   |
        #     +----+---------+    |
        #     |    |         |    |
        #     |    |  Face   |    |
        #     |    |         |    |
        #     |    +---------+    |
        #     |                   |
        #     +-------------------+

        for edge in face.edges:
            if self.is_outside(edge.pair):
                break
        else:
            logging.debug("Space: Removing a full enclosed face. A hole is created")
            self.remove_face_id(face)
            self.add_edge(face.edge.pair)
            self._clean_hole_disappearance()
            return [self]

        # case 3 : fully enclosing face
        #     +-------------------+
        #     |                   |
        #     |       Face        |
        #     |                   |
        #     +----+---------+    |
        #     |    |         |    |
        #     |    |  Space  |    |
        #     |    |         |    |
        #     |    +---------+    |
        #     |                   |
        #     +-------------------+
        # check if we are removing an enclosing face (this means that the removed face contains
        # all the edges boundary (this is no good)

        face_edges = list(face.edges)
        for edge in self.exterior_edges:
            if edge not in face_edges:
                break
            face_edges.remove(edge)
        else:
            logging.debug("Space: Removing a fully enclosing face")
            # make sure we are not selecting an internal edge
            for edge in face_edges:
                if edge.pair not in face_edges:
                    self.edge = edge.pair
                    break
            else:
                raise Exception("This should never happen!!")
            self.remove_face_id(face)
            self._clean_hole_disappearance()
            return [self]

        # case 4 : standard case
        forbidden_edges = list(face.edges)
        self.change_reference_edges(forbidden_edges)

        # We must check if we are creating one or several
        # unconnected spaces
        adjacent_faces = list(self.adjacent_faces(face))

        # if there is only one adjacent face to the removed one
        # no need to check for connectivity
        if len(adjacent_faces) == 1:
            logging.debug("Space: Removing a face with only one adjacent edge")
            self.remove_face_id(face)
            self._clean_hole_disappearance()
            return [self]

        remaining_faces = adjacent_faces[:]
        space_connected_components = []
        created_spaces = [self]

        self.remove_face_id(face)

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
                logging.debug("Space: Found a disconnected component")
                space_connected_components.append(connected_faces)
            else:
                self_boundary_face = adjacent_face
                break

        if len(space_connected_components) == 0:
            logging.debug("Space: Removing a face without splitting the space")
            self._clean_hole_disappearance()
            return created_spaces

        logging.debug("Space: The removal of a face split the space in disconnected components: %s",
                      self)

        # we must create a new space per newly created space components
        for component in space_connected_components:
            # create a new space with the disconnected faces and add it to the plan
            for _edge in component[0].edges:
                if _edge.pair.face is face:
                    boundary_edge = _edge
                    break
            else:
                raise Exception("Space: We should have found a boundary edge")

            new_space = Space(self.plan, self.floor, boundary_edge, self.category)
            # remove the disconnected faces from the initial space
            # and add them to the new space
            for component_face in component:
                self.remove_face_id(component_face)
                new_space.add_face_id(component_face)

            # transfer internal edge reference from self to new spaces
            for internal_reference_edge in self.hole_edges:
                if new_space.has_edge(internal_reference_edge):
                    self.remove_edge(internal_reference_edge)
                    new_space.add_edge(internal_reference_edge)

            new_space._clean_hole_disappearance()
            created_spaces.append(new_space)

        # preserve self edge reference
        if self.is_outside(self.edge):
            for _edge in self_boundary_face.edges:
                if _edge.pair.face is face:
                    boundary_edge = _edge
                    break
            else:
                raise Exception("We should have found a boundary edge")
            self.edge = boundary_edge

        self._clean_hole_disappearance()

        return created_spaces

    def remove_only_face(self, face: Face):
        """
        Removes the only face of the space.
        Note : this does not remove the space from the plan
        (to enable us to reverse the transformation)
        :param face:
        :return:
        """
        logging.debug("Space: Removing only face left in the Space: %s", self)

        self._edges_id = []
        self.remove_face_id(face)
        # self.remove()

    def set_edges(self):
        """
        Sets the reference edges of the space.
        We need one edge for the exterior boundary, and one edge per hole inside the space
        NOTE : Per convention the edge of the exterior is stored as the first element of the
        _edges_id array.
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
                        self.add_edge(edge)

                    space_edges = list(self.edges)

        assert len(self._edges_id), "The space is badly shaped: {}".format(self)

    def change_reference_edges(self, forbidden_edges: Sequence['Edge']):
        """
        Changes the edge references of the space
        If all the edges of the boundary are in the forbidden list, the reference is simply
        removed from the list of edges id. It means that a hole is filled.
        :param forbidden_edges: a list of edges that cannot be used as reference
        (typically because they will cease to be on the boundary of the space)
        """
        assert len(self._edges_id) == len(list(set(self._edges_id))), "Duplicate in edges !"

        for edge in self.reference_edges:
            if edge not in forbidden_edges:
                continue
            for other_edge in self.siblings(edge):
                if other_edge not in forbidden_edges:
                    assert other_edge.id not in self._edges_id, "The edge cannot already be a ref"
                    i = self._edges_id.index(edge.id)
                    # we replace the edge id in place to preserve the list order
                    self._edges_id[i] = other_edge.id
                    break
            else:
                self.remove_edge(edge)

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
        self_edges = list(self.edges)
        forbidden_edges = [edge.pair for edge in space.exterior_edges if edge.pair in self_edges]
        self.change_reference_edges(forbidden_edges)
        self._faces_id += space._faces_id
        space._faces_id = []
        space._edges_id = []
        space.remove()
        return self

    def insert_face(self, face: 'Face', container_face: Optional['Face'] = None):
        """
        Insert a face inside the space reference face
        :param face:
        :param container_face
        :return:
        """
        container_face = container_face or self.face
        created_faces = container_face.insert_face(face)
        self.add_face_id(face)
        # we need to add to the space the new faces eventually created by the insertion
        for face in created_faces:
            if face is container_face:
                continue
            self.add_face_id(face)
        # sometimes the container_face can be deleted by the insertion
        # so we need to check this and remove the deleted face from the space if needed
        if container_face not in created_faces:
            self.remove_face_id(container_face)
        # we must set the boundary in case the reference edge is no longer part of the space
        self.set_edges()

        # propagate the changes of the mesh
        self.plan.update_from_mesh()

    def insert_face_from_boundary(self, perimeter: Sequence[Coords2d]) -> 'Face':
        """
        Inserts a face inside the space reference face from the given coordinates
        :param perimeter:
        :return:
        """
        face_to_insert = self.mesh.new_face_from_boundary(perimeter)
        for face in self.faces:
            try:
                self.insert_face(face_to_insert, face)
                return face_to_insert
            except OutsideFaceError:
                continue

        self.mesh.remove_face_and_children(face_to_insert)
        raise OutsideFaceError

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
        self.add_face_id(face_of_space)
        self.remove_face(face_of_space)
        # create the space and add it to the plan
        space = Space(self.plan, self.floor, face_of_space.edge, category=category)
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
        vertex_1 = Vertex(self.mesh, *point_1)
        vertex_2 = Vertex(self.mesh, *point_2)
        new_edge = self.face.insert_edge(vertex_1, vertex_2)
        new_linear = Linear(self.plan, self.floor, new_edge, category)

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
        Returns True if the cut was successful
        :param edge:
        :param vertex:
        :param angle:
        :param traverse:
        :param max_length
        :return:
        """
        assert not self.is_outside(edge), "The edge must belong to the space"

        def immutable(_edge: Edge) -> bool:
            """
            Returns true if the edge is immutable
            :param _edge:
            :return:
            """
            return not self.plan.is_mutable(_edge)

        def callback(new_edges: Optional[Tuple[Edge, Edge]]) -> bool:
            """
            Callback to insure space consistency
            Will stop the cut if it returns True
            :param new_edges: Tuple of the new edges created by the cut
            """
            start_edge, end_edge, new_face = new_edges
            # add the created face to the space
            if new_face is not None:
                self.add_face_id(new_face)
            if self.is_outside(end_edge.pair):
                return True
            return False

        return edge.recursive_cut(vertex, angle, traverse=traverse, callback=callback,
                                  max_length=max_length, immutable=immutable)

    def barycenter_cut(self, edge: Optional[Edge] = None, coeff: float = 0.5,
                       angle: float = 90.0, traverse: str = 'absolute',
                       max_length: Optional[float] = None) -> TwoEdgesAndAFace:
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

    def ortho_cut(self, edge: 'Edge') -> bool:
        """
        Ortho cuts the specified edge and adds the created face to the space
        Returns True if the cut was successful
        :param edge:
        :return:
        """

        def _immutable(_edge: Edge) -> bool:
            return not self.plan.is_mutable(_edge)

        if _immutable(edge):
            logging.debug("Space: Cannot remove an immutable edge: %s in space: %s", edge, self)
            return False

        cut_data = edge.ortho_cut(_immutable)

        if not cut_data:
            return False

        initial_edge, split_edge, new_face = cut_data

        if new_face is not None:
            self.add_face_id(new_face)
            return True

        return False

    def remove_internal_edge(self, edge: 'Edge') -> bool:
        """
        Removes an edge of the space
        :param edge:
        :return:
        """
        assert self.is_internal(edge), "The edge must an internal edge of the space"

        if not self.plan.is_mutable(edge):
            logging.debug("Space: Cannot remove an immutable edge: %s in space: %s", edge, self)
            return False
        # remove the edge and remove the deleted face from the space
        self.remove_face_id(edge.face)
        edge.remove()

        # we need to update the plan from the mesh because of cleanups
        self.plan.update_from_mesh()

        return True

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
            logging.error('Space: Error in space: only one of edge or face is None: %s', self.edge)

        # check that they are not duplicates in the faces list id
        if len(self._faces_id) != len(list(self._faces_id)):
            logging.error("Space: Duplicate faces id in space %s", self)
            is_valid = False

        # check that they are not duplicates in the edges list id
        if len(self._edges_id) != len(list(self._edges_id)):
            logging.error("Space: Duplicate edges id in space %s", self)
            is_valid = False

        faces = list(self.faces)

        for edge in self.edges:
            if edge.face not in faces:
                logging.error('Space: boundary edge face not in space faces: %s - %s',
                              edge, edge.face)
                is_valid = False

        return is_valid

    def immutable_components(self) -> ['PlanComponent']:
        """
        Return the components associated to the space
        :return: [PlanComponent]
        """
        immutable_associated = []

        for linear in self.plan.linears:
            if self.has_linear(linear) and not linear.category.mutable:
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

    def adjacent_to(self, other: Union['Space', 'Face']) -> bool:
        """
        Check the adjacency with an other space or face
        :return:
        """
        for edge in other.edges:
            if self.has_edge(edge.pair):
                return True
        return False

    def count_ducts(self) -> float:
        """
        counts the number of ducts the space is adjacent to
        :return: float
        """

        number_ducts = sum(
            space is not self and space.adjacent_to(self) and space.category
            and space.category.name is 'duct' for space in
            self.plan.spaces)

        return number_ducts

    def count_windows(self) -> float:
        """
        counts the number of linear of type window in the space
        :return: float
        """

        number_windows = sum(
            self.has_linear(component) and component.category.window_type for component in
            self.plan.linears)

        return number_windows


class Linear(PlanComponent):
    """
    Linear Class
    A linear is an object composed of one or several contiguous edges localized on the boundary
    of a space object
    """

    def __init__(self,
                 plan: 'Plan',
                 floor: 'Floor',
                 edge: Optional[Edge],
                 category: LinearCategory,
                 _id: Optional[uuid.UUID] = None):

        if edge and not plan.is_space_boundary(edge):
            raise ValueError('cannot create a linear that is not on the boundary of a space')

        super().__init__(plan, floor, _id)
        self.category = category
        self._edges_id = [edge.id] if edge else []

    def __repr__(self):
        return 'Linear: ' + self.category.__repr__() + ' - ' + str(id(self))

    def clone(self, plan: 'Plan') -> 'Linear':
        """
        Returns a copy of the linear
        :return:
        """
        new_linear = Linear(plan, self.floor,  None, self.category, _id=self.id)
        new_linear._edges_id = self._edges_id[:]
        return new_linear

    @property
    def edge(self) -> Optional['Edge']:
        """
        The first edge of the linear
        :return:
        """
        if not self._edges_id:
            return None
        return self.mesh.get_edge(self._edges_id[0])

    @property
    def edges(self) -> Generator[Edge, None, None]:
        """
        All the edges of the Linear
        :return:
        """
        return (self.mesh.get_edge(edge_id) for edge_id in self._edges_id)

    def add_edge_id(self, edge: 'Edge'):
        """
        Adds the edge id
        :param edge:
        :return:
        """
        if edge.id in self._edges_id:
            return
        self._edges_id.append(edge.id)

    def remove_edge_id(self, edge: 'Edge'):
        """
        Adds the edge id
        :param edge:
        :return:
        """
        self._edges_id.remove(edge.id)

    def add_edge(self, edge: Edge):
        """
        Add an edge to the linear
        TODO : we should check that the edge is contiguous to the other linear edges
        :return:
        """
        if not self.plan.is_space_boundary(edge):
            raise ValueError('cannot add an edge to a linear' +
                             ' that is not on the boundary of a space')
        self.add_edge_id(edge)

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

        return is_valid


class Floor:
    """
    A class to describe a floor of a plan.
    The level correspond to the stacking order of the floor.
    The meta dict could be use to store properties of the floor such as : level, height etc.
    """
    def __init__(self,
                 mesh: 'Mesh',
                 level: Optional[int] = None,
                 meta: Optional[dict] = None):
        self.id = uuid.uuid4()
        self.mesh = mesh
        self.level = level
        self.meta = meta

    def __repr__(self):
        return "Floor: {}".format(self.id)


class Plan:
    """
    Main class containing the floor plan of the apartment
    • mesh : the corresponding geometric mesh
    • spaces : rooms or ducts or pillars etc.
    • linears : windows, doors, walls etc.
    • floors : floors of the plan stored in a dict
    """

    def __init__(self,
                 name: str = 'unnamed_plan',
                 mesh: Optional[Mesh] = None,
                 floor_level: int = 0,
                 floor_meta: Optional[int] = None,
                 spaces: Optional[List['Space']] = None,
                 linears: Optional[List['Linear']] = None):
        self.name = name
        self.spaces = spaces or []
        self.linears = linears or []
        self.floors = {}
        # add a floor
        if mesh:
            new_floor = Floor(mesh, floor_level, floor_meta)
            self.floors[new_floor.id] = new_floor
            mesh.add_watcher(lambda modifications:  self._watcher(modifications))

    def __repr__(self):
        output = 'Plan ' + self.name + ':'
        for space in self.spaces:
            output += space.__repr__() + ' | '
        return output

    def _watcher(self, modifications):
        """
        A watcher for mesh modification. The watcher must be manually called from the plan.
        ex: by calling self.mesh.watch()
        :param modifications
        :return:
        """
        logging.debug("Plan: Watcher called")

        inserted_faces = (modification for _id, modification in modifications.items()
                          if modification[0] == MeshOps.INSERT and type(modification[1]) == Face)

        removed_edges = (modification for _id, modification in modifications.items()
                         if modification[0] == MeshOps.REMOVE and type(modification[1]) == Edge)

        removed_faces = (modification for _id, modification in modifications.items()
                         if modification[0] == MeshOps.REMOVE and type(modification[1]) == Face)

        # add inserted face to the space of the receiving face
        # this must be done before removals
        for face_add in inserted_faces:
            assert type(face_add[2]) == Face, ("Plan: an insertion op "
                                               "should indicate the receiving face")
            face = face_add[1]
            # check if the face was not already added to the mesh
            face_space = self.get_space_of_face(face)
            if face_space:
                logging.debug("Plan: Adding face from mesh "
                              "update %s buf face is already in a space", face)
                continue

            space = self.get_space_of_face(face_add[2])
            if space:
                logging.debug("Plan: Adding face from mesh update %s", face)
                space.add_face_id(face)

        # remove edges
        for remove_edge in removed_edges:
            edge = remove_edge[1]
            space = self.get_space_from_reference(edge)
            if space:
                space.set_edges()
            linear = self.get_linear(edge)
            if linear:
                logging.debug("Plan: Removing edge from linear from mesh update %s", edge)
                linear.remove_edge_id(edge)

        # remove faces
        for remove_face in removed_faces:
            face = remove_face[1]
            space = self.get_space_of_face(face)
            if space:
                logging.debug("Plan: Removing face from space from mesh update %s", face)
                space.remove_face_id(face)

    def update_from_mesh(self):
        """
        Updates the plan from the mesh, and also updates all linked plan.
        :return:
        """
        self.mesh.watch()

    def clone(self, name: str = "") -> 'Plan':
        """
        Returns a copy of the plan
        :param name: the name of the cloned plan
        :return:
        """
        name = name or self.name + '_copy'
        new_plan = Plan(name, self.mesh)
        # make a shallow copy of the floors
        new_plan.floors = self.floors.copy()
        # clone spaces and linears
        new_plan.spaces = [space.clone(new_plan) for space in self.spaces]
        new_plan.linears = [linear.clone(new_plan) for linear in self.linears]

        return new_plan

    @property
    def mesh(self) -> Optional['Mesh']:
        """
        Property
        Returns the only mesh of the plan
        Note : a plan can have multiple meshes
        :return:
        """
        if not self.floor:
            return None
        return self.floor.mesh

    @property
    def floor(self) -> Optional['Floor']:
        """
        Returns any floor of the plan
        :return:
        """
        if not self.floors:
            return None
        return next(iter(self.floors.values()))

    def add_floor(self, new_floor: 'Floor'):
        """
        Adds a mesh to the plan. Used for example for multiple floors.
        :param new_floor:
        :return:
        """
        self.floors[new_floor.id] = new_floor

    @property
    def floor_count(self) -> int:
        """
        Return the number of floors of the plan
        :return:
        """
        return len(self.floors)

    @property
    def has_multiple_floors(self):
        """
        Property
        Returns True if the plan has multiple meshes
        :return:
        """
        return self.floor_count > 1

    def get_mesh(self, floor_id: uuid.UUID) -> Optional['Mesh']:
        """
        Returns the mesh of the floor_id
        :param floor_id:
        :return:
        """
        return self.floors.get(floor_id, None)

    def get_from_id(self, _id: uuid.UUID) -> Optional['PlanComponent']:
        """
        returns the component of the given id
        :param _id: a uuid
        :return: a component
        """
        # Start by searching in spaces
        for space in self.spaces:
            if space.id == _id:
                return space
        # then search in linears
        for linear in self.linears:
            if linear.id == _id:
                return linear
        logging.debug("Plan: component not found for this id %s", _id)
        return None

    def get_space_from_id(self, _id: uuid.UUID) -> Optional['Space']:
        """
        returns the component of the given id
        :param _id: a uuid
        :return: a component
        """
        # Start by searching in spaces
        for space in self.spaces:
            if space.id == _id:
                return space
        logging.debug("Plan: Space not found for this id %s", _id)
        return None

    def get_linear_from_id(self, _id: uuid.UUID) -> Optional['Linear']:
        """
        returns the component of the given id
        :param _id: a uuid
        :return: a component
        """
        for linear in self.linears:
            if linear.id == _id:
                return linear
        logging.debug("Plan: Linear not found for this id %s", _id)
        return None

    def add_floor_from_boundary(self,
                                boundary: Sequence[Coords2d],
                                floor_level: Optional[int] = None,
                                floor_meta: Optional[dict] = None) -> 'Floor':
        """
        Creates a plan from a list of points
        1. create the mesh
        2. Add an empty space
        :param boundary:
        :param floor_level:
        :param floor_meta:
        :return:
        """
        mesh = Mesh().from_boundary(boundary)
        mesh.add_watcher(lambda modifications: self._watcher(modifications))
        new_floor = Floor(mesh, floor_level, floor_meta)
        self.add_floor(new_floor)
        Space(self, new_floor, mesh.faces[0].edge)
        return new_floor

    def add(self, plan_component):
        """
        Adds a component to the plan
        :param plan_component:
        :return:
        """
        if type(plan_component) == Space:
            self._add_space(plan_component)

        if type(plan_component) == Linear:
            self._add_linear(plan_component)

    def remove(self, plan_component):
        """
        Adds a component to the plan
        :param plan_component:
        :return:
        """
        if type(plan_component) == Space:
            self._remove_space(plan_component)

        if type(plan_component) == Linear:
            self._remove_linear(plan_component)

    def _add_space(self, space: 'Space'):
        """
        Add a space in the plan
        :param space:
        :return:
        """
        if space in self.spaces:
            logging.debug("Plan : trying to add a space that is already in the plan %s", space)
        self.spaces.append(space)

    def _remove_space(self, space: 'Space'):
        """
        Removes a space from the plan
        :param space:
        :return:
        """
        self.spaces.remove(space)

    def _add_linear(self, linear: 'Linear'):
        """
        Add a linear in the plan
        :param linear:
        :return:
        """
        if linear in self.linears:
            logging.debug("Plan : trying to add a linear that is already in the plan %s", linear)
        self.linears.append(linear)

    def _remove_linear(self, linear: 'Linear'):
        """
        Removes a linear from the plan
        :param linear:
        :return:
        """
        self.linears.remove(linear)

    def get_components(self,
                       cat_name: Optional[str] = None) -> Generator['PlanComponent', None, None]:
        """
        Returns an iterator of the components contained in the plan.
        Can be filtered according to a category name
        :param cat_name: the name of the category
        :return:
        """
        yield from self.get_spaces(cat_name)
        yield from self.get_linears(cat_name)

    def get_spaces(self,
                   category_name: Optional[str] = None,
                   floor: Optional['Floor'] = None,
                   ) -> Generator['Space', None, None]:
        """
        Returns an iterator of the spaces contained in the place
        :param category_name:
        :param floor:
        :return:
        """
        assert floor is None or floor.id in self.floors, (
            "The floor specified does not exist in the plan floors: {}".format(floor, self.floors))

        if category_name is not None:
            return (space for space in self.spaces
                    if space.category.name == category_name
                    and (floor is None or space.floor is floor))
        else:
            return (space for space in self.spaces
                    if (floor is None or space.floor is floor))

    def get_space_of_face(self, face: Face) -> Optional['Space']:
        """
        Retrieves the space to which the face belongs.
        Returns None if no space is found
        :param face:
        :return:
        """
        for space in self.spaces:
            if space.has_face(face):
                return space
        return None

    def get_space_of_edge(self, edge: Edge) -> Optional['Space']:
        """
        Retrieves the space to which the face belongs.
        Returns None if no space is found
        :param edge:
        :return:
        """
        if edge.face is None:
            return None
        return self.get_space_of_face(edge.face)

    def get_space_from_reference(self, edge: 'Edge') -> Optional['Space']:
        """
        Returns the space that has the specified edge as a reference.
        Returns None if the edge is not a reference of a space.
        :param edge:
        :return:
        """
        for space in self.spaces:
            if edge.id in space._edges_id:
                return space

        return None

    def get_linear(self, edge: Edge) -> Optional['Linear']:
        """
        Retrieves the linear to which the edge belongs.
        Returns None if no linear is found
        :param edge:
        :return:
        """
        for linear in self.linears:
            if linear.has_edge(edge):
                return linear
        return None

    def get_linears(self, category_name: Optional[str] = None) -> Generator['Linear', None, None]:
        """
        Returns an iterator of the linears contained in the place
        :param category_name:
        :return:
        """
        if category_name is not None:
            return (linear for linear in self.linears if linear.category.name == category_name)

        return (linear for linear in self.linears)

    def is_space_boundary(self, edge: 'Edge') -> bool:
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

    def category_edges(self, *cat: str) -> List['Edge']:
        """
        Returns the list of edges belonging to a space of given category
        :return List['Edge']:
        """

        list_edges = []

        cat_spaces = (space for space in self.spaces if space.category.name in cat)
        for space in cat_spaces:
            for edge in space.edges:
                list_edges.append(edge)

        cat_linears = (linear for linear in self.linears if linear.category.name in cat)
        for linear in cat_linears:
            for edge in linear.edges:
                list_edges.append(edge)

        return list_edges

    @property
    def empty_spaces(self) -> Generator['Space', None, None]:
        """
        The empty spaces of the plan
        :return:
        """
        return self.get_spaces(category_name='empty')

    def empty_spaces_of_floor(self, floor: 'Floor') -> Generator['Space', None, None]:
        """
        The empty spaces of the floor
        :param floor:
        :return:
        """
        return self.get_spaces(category_name="empty", floor=floor)

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

    @property
    def area(self) -> float:
        """
        Returns the area of the plan.
        :return:
        """
        _area = 0.0
        for space in self.spaces:
            _area += space.area
        return _area

    def insert_space_from_boundary(self,
                                   boundary: Sequence[Coords2d],
                                   category: SpaceCategory = SPACE_CATEGORIES['empty'],
                                   floor: Optional['Floor'] = None) -> 'Space':
        """
        Inserts a new space inside the empty spaces of the plan.
        By design, will not insert a space overlapping several faces of the receiving spaces.
        The new space is designed from the boundary. By default, the category is empty.
        :param boundary
        :param category
        :param floor
        """
        floor = floor or self.floor
        for empty_space in self.empty_spaces_of_floor(floor):
            try:
                new_space = empty_space.insert_space(boundary, category)
                self.update_from_mesh()
                return new_space
            except OutsideFaceError:
                continue
        else:
            # TODO: this should probably raise an exception but too many input blueprints are
            # incorrect due to wrong load bearing walls geometry, it would fail too many tests
            logging.error('Plan: Could not insert the space in the plan because '
                          'it overlaps other non empty spaces: %s, %s', boundary, category)

    def insert_linear(self,
                      point_1: Coords2d,
                      point_2: Coords2d,
                      category: LinearCategory,
                      floor: Optional['Floor'] = None):
        """
        Inserts a linear object in the plan at the given points
        Will try to insert it in every empty space.
        :param point_1
        :param point_2
        :param category
        :param floor
        :return:
        """
        floor = floor or self.floor
        for empty_space in self.empty_spaces_of_floor(floor):
            try:
                empty_space.insert_linear(point_1, point_2, category)
                self.mesh.watch()
                break
            except OutsideVertexError:
                continue
        else:
            raise ValueError('Could not insert the linear in the plan:' +
                             '[{0},{1}] - {2}'.format(point_1, point_2, category))

    @property
    def boundary_as_sp(self) -> Optional[LinearRing]:
        """
        Returns the boundary of the plan as a LineString
        """
        return self.mesh.boundary_as_sp if self.mesh else None

    def plot(self,
             show: bool = False,
             save: bool = True,
             options: Tuple = ('face', 'edge', 'half-edge', 'border'),
             floor: Optional[Floor] = None):
        """
        Plots a plan. If a plan has more than one floor, the desired floor_id must be
        specified.
        :return:
        """
        assert floor is None or floor.id in self.floors, (
            "The floor id specified does not exist in the plan floors")

        n_rows = self.floor_count
        fig, ax = plt.subplots(n_rows)
        fig.subplots_adjust(hspace=0.4)  # needed to prevent overlapping of subplots title

        for i, floor in enumerate(self.floors.values()):
            _ax = ax[i] if n_rows > 1 else ax
            _ax.set_aspect('equal')

            for space in self.spaces:
                if space.floor is not floor:
                    continue
                space.plot(_ax, save=False, options=options)

            for linear in self.linears:
                if linear.floor is not floor:
                    continue
                linear.plot(_ax, save=False)

            _ax.set_title(self.name + " - floor id:{}".format(floor.id))

        plot_save(save, show)

    def check(self) -> bool:
        """
        Used to verify plan consistency
        NOTE : To be completed
        :return:
        """
        is_valid = True

        for floor in self.floors.values():
            mesh = floor.mesh
            is_valid = is_valid and mesh.check()

        for space in self.spaces:
            is_valid = is_valid and space.check()
            # check that a face only belongs to one space and one space only
            for face_id in space._faces_id:
                for other_space in self.spaces:
                    if other_space is space:
                        continue
                    if face_id in other_space._faces_id:
                        logging.debug("Plan: A face is in multiple space: %s", face_id)
                        is_valid = False

        if is_valid:
            logging.info('Plan: Checking plan: ' + '✅ OK')
        else:
            logging.info('Plan: Checking plan: ' + '🔴 NOT OK')

        return is_valid

    def remove_null_spaces(self):
        """
        Remove from the plan spaces with no edge reference
        :return:
        """
        logging.debug("Plan: removing null spaces of plan %s", self)
        space_to_remove = (space for space in self.spaces if space.edge is None)
        for space in space_to_remove:
            space.remove()

    def count_category_spaces(self, category_name: str) -> int:
        """
        count the number of spaces with the given category name
        :return:
        """
        return sum(space.category.name == category_name for space in self.spaces)

    def count_mutable_spaces(self) -> int:
        """
        count the number of mutable spaces
        :return:
        """
        return sum(space.category.mutable for space in self.spaces)

    def mutable_spaces(self) -> Generator['Space', None, None]:
        """
        Returns an iterator on mutable spaces
        :return:
        """
        yield from (space for space in self.spaces if space.mutable)

    def circulation_spaces(self) -> Generator['Space', None, None]:
        """
        Returns an iterator on mutable spaces
        :return:
        """

        yield from (space for space in self.spaces if space.category.circulation)



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

    # floor_plan()


    def clone_and_change_plan():
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
        plan.insert_space_from_boundary(duct, SPACE_CATEGORIES["duct"])
        plan_2.insert_space_from_boundary(duct_2, SPACE_CATEGORIES["duct"])
        GRIDS["finer_ortho_grid"].apply_to(plan_2)

        plan.mesh.check()
        plan.plot()
        plan_2.plot()
        space = plan.get_space_from_id(plan.spaces[0].id)
        assert space is plan.empty_space
        assert plan.spaces[0].id == plan_2.spaces[0].id


    clone_and_change_plan()
