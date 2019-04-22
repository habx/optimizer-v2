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
from typing import (
    TYPE_CHECKING,
    Optional,
    List,
    Tuple,
    Sequence,
    Generator,
    Union,
    Dict,
    Any,
    Iterable
)
import logging
import uuid

import matplotlib.pyplot as plt
from shapely.geometry import Polygon, LineString, LinearRing

from libs.mesh.mesh import Mesh, Face, Edge, Vertex, MeshOps, MeshComponentType
from libs.plan.category import LinearCategory, SpaceCategory, SPACE_CATEGORIES, LINEAR_CATEGORIES
from libs.io.plot import plot_save, plot_edge, plot_polygon
import libs.mesh.transformation as transformation
from libs.specification.size import Size
from libs.utils.custom_types import Coords2d, TwoEdgesAndAFace, Vector2d
from libs.utils.custom_exceptions import OutsideFaceError, OutsideVertexError
from libs.utils.decorator_timer import DecoratorTimer
from libs.utils.geometry import (
    dot_product,
    normal_vector,
    opposite_vector,
    ccw_angle,
    pseudo_equal,
    unit_vector
)

if TYPE_CHECKING:
    from libs.mesh.mesh import MeshModification

ANGLE_EPSILON = 1.0  # value to check if an angle has a specific value


class PlanComponent:
    """
    A component of a plan. Can be a linear (1D) or a space (2D)
    """
    __slots__ = 'id', 'plan', 'category', 'floor'

    def __init__(self,
                 plan: 'Plan',
                 floor: 'Floor',
                 _id: Optional[int] = None):

        assert floor.id in plan.floors, "PlanComponent: The floor is not in the plan!"

        self.id = _id
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
        self.plan = None

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

    __slots__ = '_edges_id', '_faces_id'

    def __init__(self,
                 plan: 'Plan',
                 floor: 'Floor',
                 edge: Optional['Edge'] = None,
                 category: SpaceCategory = SPACE_CATEGORIES['empty'],
                 _id: Optional[int] = None):
        super().__init__(plan, floor, _id=_id)
        self._edges_id = [edge.id] if edge else []
        self._faces_id = [edge.face.id] if edge and edge.face else []
        self.category = category

    def serialize(self) -> Dict:
        """
        Returns a serialize version of the space
        :return:
        """
        output = {
            "id": self.id,
            "floor": str(self.floor.id),
            "edges": self._edges_id,
            "faces": self._faces_id,
            "category": self.category.name
        }

        return output

    def deserialize(self, value: Dict) -> 'Space':
        """
        Fills the space with the specified serialized data.
        The plan and floor data is already filled.
        :return:
        """
        self._edges_id = list(map(lambda x: int(x), value["edges"]))
        self._faces_id = list(map(lambda x: int(x), value["faces"]))
        self.category = SPACE_CATEGORIES[value["category"]]
        return self

    def __repr__(self):
        output = 'Space: {} - id:{}'.format(self.category.name, self.id)
        return output

    def clone(self, plan: 'Plan') -> 'Space':
        """
        Creates a clone of the space
        The plan and the category are passed by reference
        the edges and faces id list are shallow copied (as they only contain id).
        :return:
        """
        new_floor = plan.floors[self.floor.id]
        new_space = Space(plan, new_floor, category=self.category, _id=self.id)
        new_space._faces_id = self._faces_id[:]
        new_space._edges_id = self._edges_id[:]
        return new_space

    def copy(self, other_space: 'Space'):
        """
        Copies the properties of the other_space into the space
        :param other_space:
        :return:
        """
        self._faces_id = other_space._faces_id[:]
        self._edges_id = other_space._edges_id[:]
        self.category = other_space.category
        self.floor = other_space.floor

    @property
    def face(self) -> Face:
        """
        property
        The face of the reference edge of the space
        :return:
        """
        return self.edge.face if self.edge else None

    @property
    def largest_face(self) -> Face:
        """
        property
        The face of the reference edge of the space
        :return:
        """
        return max(list(self.faces), key=lambda face: face.area)

    def has_face(self, face: 'Face') -> bool:
        """
        returns True if the face belongs to the space
        :param face:
        :return:
        """

        if face is None or face.mesh is None:
            return False

        return self.has_face_id(face.id, face.mesh.id)

    def has_face_id(self, face_id: int, mesh_id: uuid.UUID) -> bool:
        """
        returns True if the face_id belongs to the space
        :param face_id:
        :param mesh_id:
        :return:
        """
        return face_id in self._faces_id and mesh_id == self.floor.mesh.id

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

    @property
    def number_of_faces(self) -> int:
        """
        Returns the number of face of the space
        :return:
        """
        return len(self._faces_id)

    def add_face_id(self, face_id: int):
        """
        Adds a face_id if possible
        :param face_id:
        :return:
        """
        if face_id not in self._faces_id:
            self._faces_id.append(face_id)

    def remove_face_id(self, face_id: int):
        """
        Removes a face_id
        :param face_id:
        :return:
        """
        self._faces_id.remove(face_id)

    @property
    def reference_edges(self) -> Generator['Edge', None, None]:
        """
        Yields the reference edge of the space
        :return:
        """
        for edge_id in self._edges_id:
            yield self.mesh.get_edge(edge_id)

    def has_reference(self, edge: 'Edge') -> bool:
        """
        Checks if an edge is a reference edge of the space
        :param edge:
        :return: True if the edge is a reference, False otherwise
        """
        if not edge or not edge.mesh:
            return False
        return self.has_reference_id(edge.id, edge.mesh.id)

    def has_reference_id(self, edge_id: int, mesh_id: 'uuid.UUID') -> bool:
        """
        Checks if an edge is a reference edge of the space
        :param edge_id:
        :param mesh_id:
        :return: True if the edge is a reference, False otherwise
        """
        return self.floor.mesh.id == mesh_id and edge_id in self._edges_id

    @property
    def edge(self) -> Optional['Edge']:
        """
        Returns the first reference edge.
        Per convention, the first reference edge is on the outside boundary of the space
        :return:
        """
        if len(self._edges_id) == 0:
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
        if not self.is_boundary(edge):
            raise ValueError("The edge has to be a boundary "
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

    def previous_is_aligned(self, edge: 'Edge') -> bool:
        """
        Indicates if the previous edge is approximately aligned with this one,
        using a pseudo equality on the angle
        :return: boolean
        """
        if not self.is_boundary(edge):
            raise ValueError("Space: The edge must belong to the boundary %s", edge)

        is_aligned = pseudo_equal(self.previous_angle(edge), 180, ANGLE_EPSILON)
        return is_aligned

    def next_is_aligned(self, edge: 'Edge') -> bool:
        """
        Indicates if the next edge is approximately aligned with this one,
        using a pseudo equality on the angle
        :return: boolean
        """
        if not self.is_boundary(edge):
            raise ValueError("Space: The edge must belong to the boundary %s", edge)

        is_aligned = pseudo_equal(self.next_angle(edge), 180, ANGLE_EPSILON)
        return is_aligned

    def next_aligned_siblings(self, edge: Edge) -> Generator['Edge', 'Edge', None]:
        """
        Returns the edges that are aligned with edge, follows it and contiguous
        Starts with the edge itself, then all the next ones
        :return:
        """
        if not self.is_boundary(edge):
            raise ValueError("Space: The edge must belong to the boundary %s", edge)

        yield edge
        # forward check

        aligned = True
        while aligned:
            if self.next_is_aligned(edge):
                yield self.next_edge(edge)
                edge = self.next_edge(edge)
            else:
                aligned = False

    def aligned_siblings(self, edge: 'Edge') -> Generator['Edge', 'Edge', None]:
        """
        Returns all the edge on the space boundary that are aligned with the edge
        :param edge:
        :return:
        """
        if not self.is_boundary(edge):
            raise ValueError("Space: The edge must belong to the boundary %s", edge)

        yield edge

        # forward check
        current = edge
        while self.next_is_aligned(current):
            current = self.next_edge(current)
            yield current

        # backward check
        current = edge
        while self.previous_is_aligned(current):
            current = self.previous_edge(current)
            yield current

    def line(self, edge: 'Edge', mesh_line: Optional[List['Edge']] = None) -> ['Edge']:
        """
        Returns the internal edges of the space that are aligned with the specified edge.
        :param edge:
        :param mesh_line: for performance purpose the already computed mesh line of the edge
        :return:
        """
        # retrieve all the edges of the mesh aligned with the edge
        # and search for the continuous segment of edges belonging to the space
        line = mesh_line or edge.line
        if not line:
            return
        temp_line = [line[0]]
        # we skip the first edge because it can be set on the boundary per convention
        for _edge in line[1::]:
            if self.is_outside(_edge) or self.is_boundary(_edge):
                if edge in temp_line:
                    return temp_line
                temp_line = []
            else:
                temp_line.append(_edge)
        return temp_line

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
                    if edge in seen:
                        raise ValueError("The space reference edges are wrong: {}".format(self))
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

    def _external_axes(self, face: Optional['Face'] = None) -> [float]:
        """
        Returns the external axes of the space.
        For every edge of the space adjacent to an external or null space we store
        the angle to the x axis (defined by the vector (1, 0) modulo 90.0 to account
        for both orthogonal directions.
        :param face: an optional face. When specified, only check the axes of this specific face
        :return:
        """
        output = {}
        # retrieve all the edges of the space that are adjacent to the outside
        boundary_edges = []
        edges_to_search = self.edges if not face else face.edges
        for _edge in edges_to_search:
            adjacent_space = self.plan.get_space_of_edge(_edge.pair)
            if adjacent_space is not self and (adjacent_space is None
                                               or adjacent_space.category.external):
                boundary_edges.append(_edge)

        if not boundary_edges:
            return []

        # check for the angle of each edge
        for _edge in boundary_edges:
            angle = ccw_angle((1, 0), _edge.vector) % 90.0

            if angle in output:
                output[angle] += _edge.length
            else:
                output[angle] = _edge.length

        return sorted(output.keys(), key=lambda k: output[k], reverse=True)

    def _directions(self, face: Optional['Face'] = None):
        if not self._external_axes(face):
            return None

        x = unit_vector(self._external_axes(face)[0])
        y = normal_vector(x)
        return x, y, opposite_vector(x), opposite_vector(y)

    @property
    def directions(self) -> Optional[Tuple[Vector2d, Vector2d, Vector2d, Vector2d]]:
        """
        Returns the 4 authorized directions for the given space
        :return:
        """
        return self._directions()

    def face_directions(self, face: 'Face') -> Optional[Tuple[Vector2d, Vector2d,
                                                              Vector2d, Vector2d]]:
        """
        Returns the main direction of a specific face of the space
        :param face:
        :return:
        """
        return self._directions(face)

    def best_direction(self, vector: Vector2d) -> Vector2d:
        """
        Returns the closest direction of the space to the specified vector
        :param vector:
        :return:
        """
        return max(self.directions, key=lambda d: dot_product(d, vector))

    @property
    def area(self) -> float:
        """
        Returns the area of the Space.
        :return:
        """
        return sum(map(lambda f: f.area, self.faces))

    def cached_area(self) -> float:
        """
        Returns the cached area of the space
        :return:
        """
        return sum(map(lambda f: f.cached_area, self.faces))

    @property
    def perimeter(self) -> float:
        """
        Returns the length of the Space perimeter
        Note: this will count the perimeter of each holes of the space
        :return:
        """
        return sum(map(lambda e: e.length, self.edges))

    @property
    def perimeter_without_duct(self) -> float:
        """
        Returns the length of the Space perimeter without duct adjacencies
        :return:
        """
        output = 0
        for edge in self.edges:
            space = self.plan.get_space_of_edge(edge.pair)
            output += 0 if space and space.category.name == "duct" else edge.length
        return output

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

    def minimum_rotated_rectangle(self) -> Optional[Tuple[Coords2d, Coords2d, Coords2d, Coords2d]]:
        """
        Returns the smallest minimum rotated rectangle
        We rely on shapely minimum_rotated_rectangle method
        :return:
        """
        if not self.edge:
            return None

        output = self.as_sp.minimum_rotated_rectangle.exterior.coords[:]
        output.pop()

        return output

    def distance_to(self, other: 'Space', kind: str = "max") -> float:
        """
        Returns the max or the min distance to the other space
        :param other:
        :param kind: whether to return the max or the min distance
        :return:
        """
        choices = {
            "min": min,
            "max": max
        }
        return choices[kind]((e1.start.distance_to(e2.start)
                              for e1 in self.exterior_edges for e2 in other.exterior_edges))

    def adjacency_to(self, other: 'Space') -> float:
        """
        Returns the length of the boundary between two spaces
        :param other:
        :return:
        """
        shared_edges = (e for e in self.edges for other_e in other.edges if e.pair is other_e)
        return sum(map(lambda e: e.length, shared_edges))

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
            self.add_face_id(face.id)
            self.set_edges()  # needed in case the face has a hole !
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

            self.add_face_id(face.id)

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
        self.add_face_id(face.id)

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

    def remove_face(self, face: Face) -> List[Optional['Space']]:
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
            self.remove_face_id(face.id)
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
            self.remove_face_id(face.id)
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
            self.remove_face_id(face.id)
            self._clean_hole_disappearance()
            return [self]

        remaining_faces = adjacent_faces[:]
        space_connected_components = []
        created_spaces = [self]

        self.remove_face_id(face.id)

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
                self.remove_face_id(component_face.id)
                new_space.add_face_id(component_face.id)

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
        self.remove_face_id(face.id)
        # self.remove()

    def set_edges(self):
        """
        Sets the reference edges of the space.
        We need one edge for the exterior boundary, and one edge per hole inside the space
        NOTE : Per convention the edge of the exterior is stored as the first element of the
        _edges_id array.
        """
        if not self.number_of_faces:
            return
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

        if not len(self._edges_id):
            raise ValueError("The space is badly shaped: {}".format(self))

    def change_reference_edges(self, forbidden_edges: Sequence['Edge'],
                               boundary_edge: Optional['Edge'] = None):
        """
        Changes the edge references of the space
        If all the edges of the boundary are in the forbidden list, the reference is simply
        removed from the list of edges id. It means that a hole is filled.
        :param forbidden_edges: a list of edges that cannot be used as reference
        (typically because they will cease to be on the boundary of the space)
        :param boundary_edge
        """
        assert len(self._edges_id) == len(list(set(self._edges_id))), "Duplicate in edges !"

        for edge in self.reference_edges:
            if edge not in forbidden_edges:
                continue
            i = self._edges_id.index(edge.id)
            for other_edge in self.siblings(edge):
                if other_edge not in forbidden_edges:
                    assert other_edge.id not in self._edges_id, "The edge cannot already be a ref"
                    # we replace the edge id in place to preserve the list order
                    self._edges_id[i] = other_edge.id
                    break
            else:
                if i == 0:
                    logging.warning("Space: removing the first reference edge: %s", edge)
                    if not boundary_edge:
                        raise ValueError("Space: changing reference edges, you should have"
                                         "specified a boundary edge !")
                    self._edges_id[0] = boundary_edge.id
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

    def face_is_adjacent(self, face: Face) -> bool:
        """
        Returns True if the face is adjacent to the space
        :param face:
        :return: bool
        """
        if [edge for edge in face.edges if self.has_edge(edge.pair)]:
            return True
        return False

    def corner_stone(self, face: 'Face') -> bool:
        """
        Returns True if the removal of this face will split the space
        into several disconnected parts
        :return:
        """
        if not face:
            return False

        # case 1 : the only face of the space
        if len(self._faces_id) == 1:
            return True

        # case 2 : fully enclosing face
        face_edges = list(face.edges)
        for edge in self.exterior_edges:
            if edge not in face_edges:
                break
            face_edges.remove(edge)
        else:
            return False

        # case 4 : standard case
        forbidden_edges = list(face.edges)
        self.change_reference_edges(forbidden_edges)
        adjacent_faces = list(self.adjacent_faces(face))

        if len(adjacent_faces) == 1:
            return False

        remaining_faces = adjacent_faces[:]

        # temporarily remove the face_id from the other_space
        self.remove_face_id(face.id)

        # we must check to see if we split the other_space by removing the face
        # for each adjacent face inside the other_space check if they are still connected
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
                self.add_face_id(face.id)
                return True
            else:
                break

        self.add_face_id(face.id)
        return False

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
        self._faces_id += space._faces_id
        self._edges_id += space._edges_id[1:]  # preserve the holes
        space._faces_id = []
        space._edges_id = []
        space.remove()
        self.set_edges()
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
        self.add_face_id(face.id)
        # we need to add to the space the new faces eventually created by the insertion
        for face in created_faces:
            if face is container_face:
                continue
            self.add_face_id(face.id)
        # sometimes the container_face can be deleted by the insertion
        # so we need to check this and remove the deleted face from the space if needed
        if container_face not in created_faces:
            self.remove_face_id(container_face.id)
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
        self.add_face_id(face_of_space.id)
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
        vertex_1 = Vertex(self.mesh, *point_1, mutable=False)
        vertex_2 = Vertex(self.mesh, *point_2, mutable=False)
        new_edge = self.face.insert_edge(vertex_1, vertex_2)
        new_linear = Linear(self.plan, self.floor, new_edge, category)

        return new_linear

    def cut(self,
            edge: Edge,
            vertex: Vertex,
            angle: float = 90.0,
            vector: Optional[Vector2d] = None,
            traverse: str = 'absolute',
            max_length: Optional[float] = None) -> TwoEdgesAndAFace:
        """
        Cuts the space at the corresponding edge
        Adjust the self.faces and self.edges list accordingly
        Returns True if the cut was successful
        :param edge:
        :param vertex:
        :param angle:
        :param vector:
        :param traverse:
        :param max_length
        :return:
        """
        assert not self.is_outside(edge), "The edge must belong to the space"

        def callback(new_edges: Optional[Tuple[Edge, Edge]]) -> bool:
            """
            Callback to insure space consistency
            Will stop the cut if it returns True
            :param new_edges: Tuple of the new edges created by the cut
            """
            start_edge, end_edge, new_face = new_edges
            # add the created face to the space
            if new_face is not None:
                self.add_face_id(new_face.id)
            if self.is_outside(end_edge.pair):
                return True
            return False

        return edge.recursive_cut(vertex, angle, vector=vector, traverse=traverse,
                                  callback=callback, max_length=max_length)

    def barycenter_cut(self,
                       edge: Optional[Edge] = None,
                       coeff: float = 0.5,
                       angle: float = 90.0,
                       vector: Optional[Vector2d] = None,
                       traverse: str = 'absolute',
                       max_length: Optional[float] = None) -> TwoEdgesAndAFace:
        """
        Convenience method
        :param edge:
        :param coeff:
        :param angle:
        :param vector:
        :param traverse:
        :param max_length:
        :return:
        """
        edge = edge or self.edge
        vertex = (transformation.get['barycenter']
                  .config(vertex=edge.end, coeff=coeff)
                  .apply_to(edge.start))

        cut_data = self.cut(edge, vertex, angle, vector, traverse, max_length)

        # clean vertex in mesh structure
        if not cut_data and vertex.edge is None and vertex.mesh:
            vertex.remove_from_mesh()

        return cut_data

    def ortho_cut(self, edge: 'Edge') -> bool:
        """
        Ortho cuts the specified edge and adds the created face to the space
        Returns True if the cut was successful
        :param edge:
        :return:
        """
        cut_data = edge.ortho_cut()

        if not cut_data:
            return False

        initial_edge, split_edge, new_face = cut_data

        if new_face is not None:
            self.add_face_id(new_face.id)
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
        self.remove_face_id(edge.face.id)
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

    def adjacent_to(self, other: Union['Space', 'Face'], length: int = None) -> bool:
        """
        Check the adjacency with an other space or face
        with constraint of adjacency length
        :return:
        """
        if length is None:
            for edge in other.edges:
                if self.has_edge(edge.pair):
                    return True
            return False
        else:
            if self.maximum_adjacency_length(other) >= length:
                return True
            else:
                return False

    def contact_length(self, space: 'Space') -> float:
        """
        Returns the border's length between two spaces
        :return: float
        """
        border_length = 0
        for edge in self.edges:
            if space.has_edge(edge.pair):
                border_length += edge.length
        return border_length

    def count_t_edges(self) -> int:
        """
        Returns the number of T-edge of the space
        an edge is defined as a T-edge if the edge in continuity is not on the boundary of its space
        :return: float
        """

        def _is_t_edge(edge: 'Edge') -> bool:
            continuous_edge = edge.continuous_edge
            if continuous_edge:
                space_continuous = self.plan.get_space_of_edge(continuous_edge)
                if space_continuous and continuous_edge not in space_continuous.edges:
                    return True
            return False

        corner_min_angle = 20
        number_of_t_edge = 0

        list_corner_edges = [edge for edge in self.exterior_edges if
                             not edge.is_mesh_boundary and ccw_angle(edge.vector, self.next_edge(
                                 edge).vector) >= corner_min_angle]

        for edge in list_corner_edges:
            number_of_t_edge += _is_t_edge(edge)
            number_of_t_edge += _is_t_edge(self.next_edge(
                edge).pair)

        return number_of_t_edge

    def adjacent_spaces(self, length: int = None) -> List['Space']:
        """
        Gets the list of spaces adjacent to a given one
        :return: List['Space']
        """
        spaces_list = []
        for edge in self.edges:
            if edge.pair:
                adjacent_space = self.plan.get_space_of_edge(edge.pair)
                if adjacent_space and adjacent_space not in spaces_list and self.adjacent_to(
                        adjacent_space, length):
                    spaces_list.append(adjacent_space)
        return spaces_list

    def maximum_adjacency_length(self, other: Union['Space', 'Face']) -> float:
        """
        Returns the maximum adjacency length with an other space or face
        with constraint of adjacency length
        :return: float : length
        """
        adjacency_length = []
        previous_edge = False
        number_of_adjacencies = 0
        for edge in other.edges:
            if self.has_edge(edge.pair):
                if not previous_edge:
                    adjacency_length.append(edge.length)
                    number_of_adjacencies += 1
                    previous_edge = True
                else:
                    adjacency_length[number_of_adjacencies - 1] += edge.length
            else:
                previous_edge = False

        if adjacency_length:
            return max(adjacency_length)
        else:
            return 0

    def aspect_ratio(self, added_faces: Optional[Iterable['Face']] = None) -> float:
        """
        Returns the aspect ratio of the space, calculated as : perimeter ** 2 / area
        If a list of added faces is provided: returns the
        aspect ratio of the space with the added face.
        :param added_faces:
        :return:
        """
        area = self.area
        perimeter = self.perimeter
        added_faces = list(added_faces or [])
        faces_edges = [e for f in added_faces for e in f.edges]
        for face in added_faces:
            shared_edges = [e.length for e in face.edges
                            if self.is_boundary(e.pair) or e.pair in faces_edges]
            shared_perimeter = sum(shared_edges)
            area += face.area
            perimeter += face.perimeter - 2 * shared_perimeter

        return perimeter ** 2 / area

    def number_of_corners(self, other: Optional[Union['Space', 'Face']] = None) -> int:
        """
        Returns the number of corner of the space.
        If a space or a face is specified, returns the number of corner with the added face
        or space
        :param other:
        :return:
        """
        corner_min_angle = 20.0
        num_corners = 0
        for edge in self.exterior_edges:
            angle = ccw_angle(edge.opposite_vector, self.next_edge(edge).vector)
            if not pseudo_equal(angle, 180.0, corner_min_angle):
                num_corners += 1

        if not other:
            return num_corners

        num_corners += other.number_of_corners()

        # find an edge outside
        for edge in other.edges:
            if not self.is_boundary(edge.pair):
                outside_edge = edge
                break
        else:
            raise ValueError("Space: the face or space must be adjacent to the space but not"
                             "included inside it")

        outside = True
        entry_edges = []
        exit_edges = []
        shared_edges = []
        previous_edge = outside_edge
        if isinstance(other, Face):
            siblings = outside_edge.siblings
        elif isinstance(other, Space):
            siblings = other.siblings(outside_edge)
        else:
            raise ValueError("Space: other should be a Face or a Space instance")

        for edge in siblings:
            if self.is_boundary(edge.pair) and outside:
                entry_edges.append((previous_edge, edge))
                outside = False
            elif not self.is_boundary(edge.pair) and not outside:
                exit_edges.append((previous_edge, edge))
                outside = True
            elif not outside:
                shared_edges.append((previous_edge, edge))

            previous_edge = edge

        # check the initial outside edge
        if self.is_boundary(previous_edge.pair) and not outside:
            exit_edges.append((previous_edge, outside_edge))

        for prev, nxt in entry_edges:
            angle = ccw_angle(prev.opposite_vector, self.next_edge(nxt.pair).vector)
            if pseudo_equal(angle, 180, corner_min_angle):
                num_corners -= 2

        for prev, nxt in exit_edges:
            angle = ccw_angle(nxt.opposite_vector, self.previous_edge(prev.pair).vector)
            if pseudo_equal(angle, 180, corner_min_angle):
                num_corners -= 2

        for prev, nxt in shared_edges:
            angle = ccw_angle(prev.opposite_vector, nxt.vector)
            if not pseudo_equal(angle, 180, corner_min_angle):
                num_corners -= 2

        return num_corners

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

    def openings(self) -> ['Linear']:
        """
        Returns the associated openings
        :return: ['Linear']
        """
        openings_list = []
        for component in self.immutable_components():
            if component.category in LINEAR_CATEGORIES.values() and component.category.aperture:
                openings_list.append(component)
        return openings_list

    def connected_spaces(self) -> ['Space']:
        """
        Returns the connected spaces
        :return: ['Space']
        """
        connected_spaces = []
        for door in self.openings():
            for space in door.adjacent_spaces():
                if space is not self and space not in connected_spaces:
                    connected_spaces.append(space)
        return connected_spaces

    def centroid(self) -> Coords2d:
        """
        Returns the centroid coords of the space
        :return: ['Space']
        """

        centroid_x = 0
        centroid_y = 0
        for edge in self.edges:
            centroid_x += (edge.start.x + edge.next.start.x) * (
                    edge.start.x * edge.next.start.y - edge.next.start.x * edge.start.y)
            centroid_y += (edge.start.y + edge.next.start.y) * (
                    edge.start.x * edge.next.start.y - edge.next.start.x * edge.start.y)

        centroid_x = centroid_x * 1 / (6 * self.area)
        centroid_y = centroid_y * 1 / (6 * self.area)

        return [centroid_x, centroid_y]

    def maximum_distance_to(self, other: Union['Space', 'Face']) -> float:
        """
        Returns the maximum distance with an other space or face
        :return: float : length
        """
        max_distance = max(e.start.distance_to(o.start)
                           for e in self.exterior_edges for o in other.exterior_edges)
        return max_distance


class Linear(PlanComponent):
    """
    Linear Class
    A linear is an object composed of one or several contiguous edges localized on the boundary
    of a space object
    """

    __slots__ = '_edges_id'

    def __init__(self,
                 plan: 'Plan',
                 floor: 'Floor',
                 edge: Optional[Edge] = None,
                 category: Optional[LinearCategory] = None,
                 _id: Optional[int] = None):

        if edge and not plan.is_space_boundary(edge):
            raise ValueError('cannot create a linear that is not on the boundary of a space')

        super().__init__(plan, floor, _id)
        self.category = category
        self._edges_id = [edge.id] if edge else []

    def __repr__(self):
        return 'Linear: ' + self.category.__repr__() + ' - ' + str(id(self))

    def serialize(self) -> Dict:
        """
        Returns a serialize version of the space
        :return:
        """
        output = {
            "id": self.id,
            "floor": str(self.floor.id),
            "edges": list(map(str, self._edges_id)),
            "category": self.category.name
        }

        return output

    def deserialize(self, value: Dict) -> 'Linear':
        """
        Fills the linear properties from the serialized value
        :param value:
        :return:
        """
        self._edges_id = list(map(lambda x: int(x), value["edges"]))
        self.category = LINEAR_CATEGORIES[value["category"]]
        return self

    def clone(self, plan: 'Plan') -> 'Linear':
        """
        Returns a copy of the linear
        :return:
        """
        new_floor = plan.floors[self.floor.id]
        new_linear = Linear(plan, new_floor, category=self.category, _id=self.id)
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

    def add_edge_id(self, edge_id: int):
        """
        Adds the edge id
        :param edge_id:
        :return:
        """
        if edge_id in self._edges_id:
            return
        self._edges_id.append(edge_id)

    def remove_edge_id(self, edge_id: int):
        """
        Adds the edge id
        :param edge_id:
        :return:
        """
        self._edges_id.remove(edge_id)

    def add_edge(self, edge: Edge):
        """
        Add an edge to the linear
        TODO : we should check that the edge is contiguous to the other linear edges
        :return:
        """
        if not self.plan.is_space_boundary(edge):
            raise ValueError('cannot add an edge to a linear' +
                             ' that is not on the boundary of a space')
        self.add_edge_id(edge.id)

    def has_edge(self, edge: 'Edge') -> bool:
        """
        Returns True if the edge belongs to the linear
        :param edge:
        :return:
        """
        if not edge or not edge.mesh:
            return False

        return self.has_edge_id(edge.id, edge.mesh.id)

    def has_edge_id(self, edge_id: int, mesh_id: 'uuid.UUID') -> bool:
        """
        Returns True if the edge belongs to the linear
        :param edge_id:
        :param mesh_id:
        :return:
        """
        return self.floor.mesh.id == mesh_id and edge_id in self._edges_id

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

    def adjacent_spaces(self) -> ['Space']:
        """
        Returns the adjacent spaces
        :return: ['Space']
        """
        spaces_list = []
        for edge in self.edges:
            if self.plan.get_space_of_edge(edge) not in spaces_list:
                spaces_list.append(self.plan.get_space_of_edge(edge))
            if self.plan.get_space_of_edge(edge.pair) not in spaces_list:
                spaces_list.append(self.plan.get_space_of_edge(edge.pair))
        return spaces_list


class Floor:
    """
    A class to describe a floor of a plan.
    The level correspond to the stacking order of the floor.
    The meta dict could be use to store properties of the floor such as : level, height etc.
    """

    __slots__ = 'plan', 'id', 'mesh', 'level', 'meta'

    def __init__(self,
                 plan: 'Plan',
                 mesh: Optional['Mesh'] = None,
                 level: Optional[int] = 0,
                 meta: Optional[dict] = None,
                 _id: Optional[int] = None):
        self.plan = plan
        self.id = _id
        self.mesh = mesh
        self.level = level
        self.meta = meta

        if self.mesh:
            self.mesh.add_watcher(self.plan.watcher)

    def __repr__(self):
        return "Floor: {}".format(self.id)

    def clone(self, new_plan: 'Plan') -> 'Floor':
        """
        Creates a copy of the floor
        :param new_plan:
        :return:
        """
        new_floor = Floor(new_plan, self.mesh, self.level, self.meta, _id=self.id)
        return new_floor

    def store_mesh_globally(self):
        """
        Stores the mesh in a global variable named MESHES
        :return:
        """
        if "MESHES" not in globals():
            globals().setdefault("MESHES", {self.mesh.id: self.mesh})
        else:
            globals()["MESHES"][self.mesh.id] = self.mesh

    def get_mesh_from_global(self, mesh_id: uuid.UUID):
        """
        Sets the mesh of the floor by retrieving it from the global MESHES
        according to the specified mesh id
        :return:
        """
        self.mesh = globals().get("MESHES")[mesh_id]

    def serialize(self, embedded_mesh: bool = True) -> Dict:
        """
        Returns a serialized version of the floor
        :param embedded_mesh: whether to serialize or not the linked mesh. If the mesh is not
        serialized, it is expected that the mesh will be found in a global variables named
        meshes which contains a dict : { mesh_id: mesh_object }.
        The global variable will then be used when deserializing the mesh.
        :return:
        """
        output = {
            "id": self.id,
            "mesh": self.mesh.serialize() if embedded_mesh else str(self.mesh.id),
            "level": self.level,
            "meta": self.meta
        }

        if not embedded_mesh:
            # make sure the mesh is stored in a global variable
            self.store_mesh_globally()

        return output

    def deserialize(self, data: Dict, embedded_mesh: bool = False) -> 'Floor':
        """
        Creates a floor from serialized data
        :param data: the dictionary containing the serialized data
        :param embedded_mesh: whether the mesh is embedded or not inside the serialize data
        :return:
        """
        # add new deserialized mesh and corresponding watcher
        if embedded_mesh:
            self.mesh = Mesh().deserialize(data["mesh"])
        else:
            self.get_mesh_from_global(uuid.UUID(data["mesh"]))

        self.mesh.add_watcher(self.plan.watcher)

        self.level = int(data["level"])
        self.meta = data["meta"]
        return self

    @property
    def boundary_as_sp(self) -> Optional[LinearRing]:
        """
        Returns the boundary of the plan as a LineString
        """
        return self.mesh.boundary_as_sp if self.mesh else None


class Plan:
    """
    Main class containing the floor plan of the apartment
    • mesh : the corresponding geometric mesh
    • spaces : rooms or ducts or pillars etc.
    • linears : windows, doors, walls etc.
    • floors : floors of the plan stored in a dict
    """

    __slots__ = 'name', 'spaces', 'linears', 'floors', 'id', '_counter'

    def __init__(self,
                 name: str = 'unnamed_plan',
                 mesh: Optional[Mesh] = None,
                 floor_level: int = 0,
                 floor_meta: Optional[int] = None,
                 spaces: Optional[List['Space']] = None,
                 linears: Optional[List['Linear']] = None):
        self.id = uuid.uuid4()
        self.name = name
        self.spaces = spaces or []
        self.linears = linears or []
        self.floors: Dict[int, 'Floor'] = {}
        self._counter = 0

        # add a floor if a mesh is specified in the init (per convenience)
        if mesh:
            new_floor = Floor(self, mesh, floor_level, floor_meta)
            self.floors[new_floor.id] = new_floor

    def __repr__(self):
        output = 'Plan ' + self.name + ':'
        for space in self.spaces:
            output += space.__repr__() + ' | '
        return output

    def get_id(self) -> int:
        """
        Returns an incremental id
        :return:
        """
        self._counter += 1
        return self._counter

    def _reset_counter(self):
        """
        Reset the id counter to a proper value. Needed when deserializing a plan.
        :return:
        """
        spaces_id = set(map(lambda s: s.id, self.spaces))
        linears_id = set(map(lambda l: l.id, self.linears))
        floors_id = set(map(lambda f: f.id, self.floors.values()))
        self._counter = max(spaces_id | linears_id | floors_id)

    def clear(self):
        """
        Clears the data of the plan
        :return:
        """
        self.name = ""
        self.spaces = []
        self.linears = []
        self.floors = {}

    def store_meshes_globally(self):
        """
        Store the meshes of the plan in a global variable named MESHES.
        This is needed for multiprocessing.
        :return:
        """
        for floor in self.floors.values():
            floor.store_mesh_globally()

    def serialize(self, embedded_mesh: bool = True) -> Dict[str, Any]:
        """
        Returns a serialize version of the plan
        :param embedded_mesh: whether to embed the mesh in the serialized data
        :return:
        """
        output = {
            "name": self.name,
            "spaces": [space.serialize() for space in self.spaces],
            "linears": [linear.serialize() for linear in self.linears],
            "floors": [floor.serialize(embedded_mesh) for floor in self.floors.values()]
        }

        return output

    def deserialize(self, data: Dict, embedded_mesh: bool = True) -> 'Plan':
        """
        Adds plan data from serialized input value
        :param data: the serialized data
        :param embedded_mesh: whether to expect the mesh to be embedded in the data
        :return: a plan
        """
        self.clear()
        self.name = data["name"]

        # add floors
        for floor in data["floors"]:
            self.add_floor(Floor(self, _id=floor["id"]).deserialize(floor, embedded_mesh))

        # add spaces
        for space in data["spaces"]:
            floor_id = int(space["floor"])
            floor = self.floors[floor_id]
            Space(self, floor, _id=int(space["id"])).deserialize(space)

        # add linears
        for linear in data["linears"]:
            floor_id = int(linear["floor"])
            floor = self.floors[floor_id]
            Linear(self, floor, _id=linear["id"]).deserialize(linear)

        self._reset_counter()

        return self

    def __getstate__(self) -> Dict:
        """
        Used to replace pickling method.
        This is needed due to the circular references in the mesh that makes it inefficient
        for the standard pickle protocol.
        For performance purposes : we do not pickle the mesh
        """
        return self.serialize(embedded_mesh=False)

    def __setstate__(self, state: Dict):
        """ Used to replace pickling method. """
        self.deserialize(state, embedded_mesh=False)

    def watcher(self, modifications: Dict[int, 'MeshModification'], mesh_id: uuid.UUID):
        """
        A watcher for mesh modification. The watcher must be manually called from the plan.
        ex: by calling self.mesh.watch()
        :param modifications: a dictionary containing the mesh modification
        :param mesh_id: the id of the corresponding mesh (a plan can have several meshes,
        one for each floor)
        :return:
        """
        logging.debug("Plan: Updating plan from mesh watcher")

        inserted_faces = (modification for _id, modification in modifications.items()
                          if modification[0] == MeshOps.INSERT
                          and modification[1][0] == MeshComponentType.FACE)

        removed_edges = (modification for _id, modification in modifications.items()
                         if modification[0] == MeshOps.REMOVE
                         and modification[1][0] == MeshComponentType.EDGE)

        inserted_edges = (modification for _id, modification in modifications.items()
                          if modification[0] == MeshOps.INSERT
                          and modification[1][0] == MeshComponentType.EDGE)

        removed_faces = (modification for _id, modification in modifications.items()
                         if modification[0] == MeshOps.REMOVE
                         and modification[1][0] == MeshComponentType.FACE)

        # add inserted face to the space of the receiving face
        # this must be done before removals
        for face_add in inserted_faces:
            assert face_add[2][0] == MeshComponentType.FACE, ("Plan: an insertion op of a face "
                                                              "should indicate the receiving face")
            # check if the face was not already added to the mesh
            face_space = self.get_space_from_face_id(face_add[1][1], mesh_id)
            if face_space:
                logging.debug("Plan: Adding face from mesh "
                              "update %s buf face is already in a space", face_space)
                continue

            space = self.get_space_from_face_id(face_add[2][1], mesh_id)
            if space:
                logging.debug("Plan: Adding face from mesh update %s", space)
                space.add_face_id(face_add[1][1])

        # add inserted edge to the linear of the receiving face
        for edge_add in inserted_edges:
            assert (edge_add[2][0] == MeshComponentType.EDGE), ("Plan: an insertion "
                                                                "op of an edge should indicate "
                                                                "the receiving edge")
            linear = self.get_linear_from_edge_id(edge_add[2][1], mesh_id)
            if linear is not None:
                logging.debug("Plan: Adding Edge to linear from mesh update %s", edge_add[1])
                linear.add_edge_id(edge_add[1][1])

        # remove faces
        for remove_face in removed_faces:
            space = self.get_space_from_face_id(remove_face[1][1], mesh_id)
            if space:
                logging.debug("Plan: Removing face from space from mesh update %s", space)
                space.remove_face_id(remove_face[1][1])
                space.set_edges()

        # remove edges
        for remove_edge in removed_edges:
            space = self.get_space_from_reference_id(remove_edge[1][1], mesh_id)
            if space:
                space.set_edges()
            linear = self.get_linear_from_edge_id(remove_edge[1][1], mesh_id)
            if linear:
                logging.debug("Plan: Removing edge from linear from mesh update %s", linear)
                linear.remove_edge_id(remove_edge[1][1])

    def update_from_mesh(self):
        """
        Updates the plan from the mesh, and also updates all linked plan.
        :return:
        """
        for floor in self.floors.values():
            floor.mesh.watch()

    def simplify(self):
        """
        Simplifies the meshes of the plan
        :return:
        """
        for floor in self.floors.values():
            floor.mesh.simplify()

    def clone(self, name: str = "") -> 'Plan':
        """
        Returns a copy of the plan
        :param name: the name of the cloned plan
        :return:
        """
        name = name or self.name
        new_plan = Plan(name)
        # clone floors, spaces and linears
        new_plan.floors = {floor.id: floor.clone(new_plan) for floor in self.floors.values()}
        new_plan.spaces = [space.clone(new_plan) for space in self.spaces]
        new_plan.linears = [linear.clone(new_plan) for linear in self.linears]
        new_plan._counter = self._counter

        return new_plan

    def copy(self, plan: 'Plan') -> 'Plan':
        """
        Copies the data from an another plan
        :param plan:
        :return:
        """
        self.name = plan.name
        self.floors = {floor.id: floor.clone(self) for floor in plan.floors.values()}
        self.spaces = [space.clone(self) for space in plan.spaces]
        self.linears = [linear.clone(self) for linear in plan.linears]
        self._counter = plan._counter
        return self

    def __deepcopy__(self, memo) -> 'Plan':
        """
        We overload deepcopy in order to be able to clone a plan using the
        following copy.deepcopy(plan). This is needed for proper interface with
        other libraries such as deap.
        :param memo: needed for the deepcopy overloading (useless in our case)
        :return: a clone of the plan
        """
        return self.clone()

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
        if not new_floor.id:
            new_floor.id = self.get_id()

        self.floors[new_floor.id] = new_floor

    def get_floor_from_id(self, _id: int) -> Optional['Floor']:
        """
        Returns the floor with the specified id. Returns None if no floor is found.
        :param _id:
        :return:
        """
        return self.floors.get(_id, None)

    @property
    def floor_count(self) -> int:
        """
        Return the number of floors of the plan
        :return:
        """
        return len(self.floors)

    def floor_of_given_level(self, level: int) -> Optional['Floor']:
        """
        Returns the floor of the given level
        :return:
        """
        for floor in self.floors.values():
            if floor.level == level:
                return floor
        logging.info("Plan: floor_of_given_level: No floor at this level")
        return None

    @property
    def has_multiple_floors(self):
        """
        Property
        Returns True if the plan has multiple meshes
        :return:
        """
        return self.floor_count > 1

    @property
    def first_level(self) -> int:
        """
        Property
        Returns the first level of the plan
        :return:
        """
        return min(floor.level for floor in self.floors.values())

    @property
    def list_level(self) -> Generator[int, None, None]:
        """
        Property
        Returns the generator on levels of the plan
        :return:
        """
        return (floor.level for floor in self.floors.values())

    def get_mesh(self, floor_id: int) -> Optional['Mesh']:
        """
        Returns the mesh of the floor_id
        :param floor_id:
        :return:
        """
        return self.floors.get(floor_id, None)

    def get_from_id(self, _id: int) -> Optional['PlanComponent']:
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

    def get_space_from_id(self, _id: int) -> Optional['Space']:
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

    def get_linear_from_id(self, _id: int) -> Optional['Linear']:
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

    def get_linear_from_edge_id(self, edge_id: int, mesh_id: uuid.UUID) -> Optional['Linear']:
        """
        Returns the linear that owns the edge if found, None otherwise
        :param edge_id:
        :param mesh_id:
        :return:
        """
        for linear in self.linears:
            if linear.has_edge_id(edge_id, mesh_id):
                return linear
        return None

    def get_linear_from_edge(self, edge: 'Edge') -> Optional['Linear']:
        """
        Returns the linear that owns the edge if found, None otherwise
        :param edge:
        :return:
        """
        if not edge or not edge.mesh:
            return None

        return self.get_linear_from_edge_id(edge.id, edge.mesh.id)

    def add_floor_from_boundary(self,
                                boundary: Sequence[Coords2d],
                                floor_level: Optional[int] = 0,
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
        mesh.add_watcher(self.watcher)
        new_floor = Floor(self, mesh, floor_level, floor_meta)
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
        Remove a component to the plan
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
        if space.id is None:
            space.id = self.get_id()

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
        if linear.id is None:
            linear.id = self.get_id()

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
                   floor: Optional['Floor'] = None) -> Generator['Space', None, None]:
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

    def get_space_from_face_id(self, face_id: int, mesh_id: 'uuid.UUID') -> Optional['Space']:
        """
        Returns the space that contains the face with the given id and mesh_id
        :param face_id:
        :param mesh_id:
        :return the space
        """
        for space in self.spaces:
            if space.has_face_id(face_id, mesh_id):
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
            if space.has_reference(edge):
                return space

        return None

    def get_space_from_reference_id(self, edge_id: int, mesh_id: 'uuid.UUID') -> Optional['Space']:
        """
        Returns the space with the edge of the given id and mesh id in its boundary
        :param edge_id:
        :param mesh_id:
        :return:
        """
        for space in self.spaces:
            if space.has_reference_id(edge_id, mesh_id):
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

    def is_external(self, edge: 'Edge') -> bool:
        """
        Returns True if the edge is on the exterior of the apartment (meaning its face is none or
        its space is external)
        :param edge:
        :return: bool
        """
        space = self.get_space_of_edge(edge)
        return space is None or space.category.external

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

    @property
    def indoor_area(self) -> float:
        """
        Returns the indoor area of the plan.
        :return:
        """
        _area = 0.0
        for space in self.spaces:
            if space.category.external is False:
                _area += space.area
        return _area

    @property
    def indoor_perimeter(self) -> float:
        """
        Returns the perimeter of the plan.
        :return:
        """
        _perimeter = 0.0
        for space in self.spaces:
            for edge in space.edges:
                if (edge.pair.face is None or
                    edge.pair in list(space.edge
                                      for space in self.spaces if space.category.external is True)):
                    _perimeter += edge.length
        return _perimeter

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
        face_to_insert = None
        for empty_space in self.empty_spaces_of_floor(floor):
            try:
                new_space = empty_space.insert_space(boundary, category)
                self.update_from_mesh()
                return new_space
            except OutsideFaceError:
                continue
        else:
            try:
                face_to_insert = floor.mesh.new_face_from_boundary(boundary)
                new_exterior_faces = floor.mesh.insert_external_face(face_to_insert)
                # add the eventually created holes
                for face in new_exterior_faces:
                    Space(self, floor, face.edge, SPACE_CATEGORIES["hole"])
                # create the new space
                new_space = Space(self, floor, face_to_insert.edge, category)
                return new_space

            except OutsideFaceError:
                if face_to_insert:
                    floor.mesh.remove_face_and_children(face_to_insert)

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
        Plots a plan.
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

        return ax

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
            for face in space.faces:
                for other_space in self.spaces:
                    if other_space is space:
                        continue
                    if other_space.has_face(face):
                        logging.error("Plan: A face is in multiple space: %s, %s - %s",
                                      face, other_space, space)
                        is_valid = False

        if is_valid:
            logging.info('Plan: Checking Plan: ' + '✅ OK')
        else:
            logging.info('Plan: Checking Plan: ' + '🔴 NOT OK')

        return is_valid

    def remove_null_spaces(self):
        """
        Remove from the plan spaces with no edge reference
        :return:
        """
        logging.debug("Plan: removing null spaces of plan %s", self)
        space_to_remove = [space for space in self.spaces if space.edge is None]
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

    def front_door(self) -> Optional['Linear']:
        """
        Returns the front door
        :return:
        """
        for linear in self.linears:
            if linear.category.name == 'frontDoor':
                return linear
        return None


if __name__ == '__main__':
    import libs.io.reader as reader

    logging.getLogger().setLevel(logging.DEBUG)


    @DecoratorTimer()
    def floor_plan():
        """
        Test the creation of a specific blueprint
        :return:
        """
        input_file = "011.json"
        plan = reader.create_plan_from_file(input_file)

        plan.plot(save=False)
        plt.show()

        plan.check()


    floor_plan()


    def clone_and_change_plan():
        """
        :return:
        """
        from libs.modelers.grid import GRIDS

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

    # clone_and_change_plan()
