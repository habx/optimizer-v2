# coding=utf-8
"""
Plan Module
Creates the following classes:
• Plan : contains the description of a blue print
• Space : a 2D space in an apartment blueprint : can be a room, or a pillar, or a duct.
• Linear : a 1D object in an apartment. For example : a window, a door or a wall.
"""
from typing import TYPE_CHECKING, Optional, List, Tuple, Sequence, Generator, Union
import logging
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
from libs.utils.geometry import dot_product, normal_vector, pseudo_equal

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

    def from_boundary(self, boundary: Sequence[Coords2d]):
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
             options: Tuple = ('face', 'edge', 'half-edge', 'fill', 'border')):
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
        num = 0
        for space in self.spaces:
            if space.category.name == category:
                num += 1
        return num

    def count_mutable_spaces(self) -> int:
        """
        count the number of mutable spaces
        :return:
        """
        num = 0
        for space in self.spaces:
            if space.category.mutable:
                num += 1
        return num

    def mutable_spaces(self) -> Generator['Space', None, None]:
        """
        Returns an iterator on mutable spaces
        :return:
        """

        for space in self.spaces:
            if space.mutable:
                yield space

    def circulation_spaces(self) -> Generator['Space', None, None]:
        """
        Returns an iterator on mutable spaces
        :return:
        """

        for space in self.spaces:
            if space.category.circulation:
                yield space


class PlanComponent:
    """
    A component of a plan. Can be a linear (1D) or a space (2D)
    """

    def __init__(self, plan: Plan, edge: Edge):
        self.plan = plan
        self.edge = edge
        self.category: Union[SpaceCategory, LinearCategory] = None


class Space(PlanComponent):
    """
    Space Class
    """

    def __init__(self, plan: Plan, edge: Edge,
                 category: SpaceCategory = SPACE_CATEGORIES('empty')):
        super().__init__(plan, edge)
        self.category = category
        # set the circular reference
        edge.face.space = self
        # set the boundary of the Space if the edge has not already a boundary
        if not self.edge.is_space_boundary:
            for _edge in self.edge.siblings:
                _edge.space_next = _edge.next

    def __repr__(self):
        output = 'Space: ' + self.category.name + ' - ' + str(id(self))
        return output

    @property
    def face(self) -> Face:
        """
        property
        One of the face of the space
        :return:
        """
        return self.edge.face if self.edge else None

    @property
    def faces(self) -> Generator[Face, None, None]:
        """
        The faces included in the Space. Returns an iterator.
        :return:
        """
        return self.get_adjacent_faces(self.face)

    def get_adjacent_faces(self, face: Face) -> Generator[Face, None, None]:
        """
        Returns the adjacent faces of the specified face in the space,
        per convention we start by returning the provided face
        Needed for example when a space has been divided in disconnected pieces
        :param face:
        :return:
        """
        if face is None:
            return

        seen = [face]
        yield face

        def _get_adjacent_faces(_face: Face) -> Generator[Face, None, None]:
            """
                Recursive function to retrieve all the faces of the space
                :param _face:
                :return:
                """
            for edge in _face.edges:
                # if the edge is a boundary of the space do not propagate
                if edge.is_space_boundary:
                    continue
                new_face = edge.pair.face
                if new_face and new_face.space is self and new_face not in seen:
                    seen.append(new_face)
                    yield new_face
                    yield from _get_adjacent_faces(new_face)

        yield from _get_adjacent_faces(face)

    @property
    def edges(self) -> Generator[Edge, Edge, None]:
        """
        The boundary edges of the space
        :return: an iterator
        """
        if self.edge is None:
            return
        yield from self.edge.space_siblings

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
            return None
        list_vertices = [edge.start.coords for edge in self.edges]
        list_vertices.append(list_vertices[0])
        return Polygon(list_vertices)

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
        while current_edge.face is not edge.face:
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

    def _remove_lone_space_edge(self, edge: Edge):
        """
        Remove a "lone edge" defined as having its pair as a space_next
        :param edge:
        :return:
        """
        if edge is edge.pair.space_next:
            return self._remove_lone_space_edge(edge.pair)

        if edge.pair is not edge.space_next:
            return

        edge_space_previous = edge.space_previous
        edge_space_previous.space_next = edge.pair.space_next
        edge.space_next = None
        edge.pair.space_next = None

        # recursive check the next edge
        return self._remove_lone_space_edge(edge_space_previous)

    def _add_enclosed_face(self, face):
        """
        Inserts a fully enclosed face in the space
        :param face:
        :return:
        """
        logging.debug('Adding an enclosed face: {0} - {1}'.format(face, self))
        # find the space boundary edges linking the enclosed face to the rest of the space
        touch_edges = []
        change_reference_edge = False
        for edge in face.edges:
            if edge.pair is self.edge:
                change_reference_edge = True
            if edge.pair.space_previous is not edge.next.pair:
                touch_edges.append((edge.pair.space_previous, edge.next.pair))

        # if need be we change the space reference edge
        if change_reference_edge:
            self.edge = touch_edges[0][0]

        for edges in touch_edges:
            edge_pair_space_previous, edge_next_pair = edges
            edge_pair_space_previous.space_next = edge_next_pair.space_next
            # check if we created a lone space edge
            self._remove_lone_space_edge(edge_pair_space_previous)

        # remove space next references
        for edge in face.edges:
            edge.pair.space_next = None
            edge.space_next = None

        face.space = self

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
        self.edge = face.edge
        face.space = self
        for edge in face.edges:
            edge.space_next = edge.next

    def add_face(self, face: Face, start_from: Optional[Edge] = None):
        """
        Adds a face to the space and adjust the edges list accordingly
        If the added face belongs to another space we first need to remove it from the space
        We do not enable to add a face inside a hole in the face (enclosed face)
        :param face: face to add to space
        :param start_from: an edge to start the adjacency search from
        """
        if face.space:
            raise ValueError('Cannot add a face that already ' +
                             'belongs to another space : {0}'.format(face))

        # if the space has no faces yet just add the face as the reference for the Space
        if self.edge is None:
            return self.add_first_face(face)

        # we start the search for a boundary edge from the start_from edge or the face edge
        start_from = start_from or face.edge

        # we make sure the new face is adjacent to at least one of the space faces
        adjacent_edges = []
        for edge in start_from.siblings:
            if not adjacent_edges:
                # first boundary edge found
                if self.is_boundary(edge.pair):
                    adjacent_edges.append(edge)
            else:
                # note : we only search for "contiguous" space boundaries
                if (self.is_boundary(edge.pair) and edge is adjacent_edges[-1].next
                        and edge.pair.space_next is adjacent_edges[-1].pair):
                    adjacent_edges.append(edge)
                else:
                    break

        if not adjacent_edges:
            raise ValueError('Cannot add a face that is not adjacent' +
                             ' to the space:{0} - {1}'.format(face, self))

        # check for other shared space boundaries that are localised before the first edge in
        # space_edges : (we go backward)
        edge = adjacent_edges[0].pair.space_next.pair
        while edge.face is face and edge.next is adjacent_edges[0] and edge not in adjacent_edges:
            adjacent_edges.insert(0, edge)
            edge = edge.pair.space_next.pair

        if len(adjacent_edges) == len(list(face.edges)):
            # the face is completely enclosed in the space
            return self._add_enclosed_face(face)

        end_edge = adjacent_edges[-1]
        start_edge = adjacent_edges[0]

        end_edge.pair.space_previous.space_next = end_edge.next
        start_edge.previous.space_next = start_edge.pair.space_next

        # preserve space edge reference
        # (if the reference edge of the space belongs to the boundary with
        # the added face)
        if self.edge.pair in adjacent_edges:
            self.edge = end_edge.next  # per convention

        # remove the old space references
        for edge in adjacent_edges:
            edge.pair.space_next = None

        # add the new space references inside the added face
        for edge in end_edge.next.siblings:
            if edge.next is start_edge:
                break
            edge.space_next = edge.next

        # finish by adding the space reference in the face object
        face.space = self

    def remove_only_face(self, face: Face) -> Sequence['Space']:
        """
        Removes the only face of the space
        :param face:
        :return: the modified space
        """
        if self.face is not face:
            raise ValueError('the face is not the reference face of the space:' +
                             ' {0} - {1}'.format(self, face))
        logging.debug('Removing only face left in the Space: {0}'.format(self))
        self.edge = None
        face.space = None
        # remove space_next references inside the face
        for edge in face.edges:
            edge.space_next = None
        return [self]

    def remove_encapsulating_face(self, face: Face) -> Sequence['Space']:
        """
        Removes a face that encapsulates other faces
        This means that the face contains all the space boundary edges but is not the
        only face in the space
        example:
        • - - - - •
        |   •     |
        | /  \    |
        •    •    |
        | \ /     |
        |  •      |
        • - - - - •

        :param face:
        :return:
        """
        logging.debug('Removing an encapsulating face from a space:' +
                      '{0} - {1}'.format(face, self))

        entry_edge = None
        previous_edge_is_boundary = self.is_boundary(face.edge)
        for edge in face.edge.next.siblings:
            if self.is_internal(edge):
                edge_is_boundary = False
            else:
                edge_is_boundary = True
            if not edge_is_boundary and previous_edge_is_boundary:
                entry_edge = edge
                break
            previous_edge_is_boundary = edge_is_boundary

        if not entry_edge:
            raise ValueError('This face is not encapsulating: {0} - {1}'.format(face, self))

        previous_edge = entry_edge
        for edge in entry_edge.next.siblings:
            if self.is_boundary(edge):
                break
            edge.pair.space_next = previous_edge.pair
            previous_edge = edge

        entry_edge.pair.space_next = previous_edge.pair

        # change the space edge reference
        self.edge = entry_edge.pair

        # remove the space reference of the face
        for edge in face.edges:
            edge.space_next = None
        face.space = None

        return [self]

    def change_reference(self, face: Face) -> bool:
        """
        Changes the face reference of the space.
        Returns True if the change is possible or unnecessary,
        False if all the space edges belong to the face
        :return:
        """
        # if the face does not include the edge reference of the space
        # do nothing
        if self.face is not face:
            return True
        # find an edge from another face in the Space boundary
        # if another face is found, it becomes the face reference of the Space
        for sibling in self.edges:
            if sibling.face is not face:
                self.edge = sibling
                return True

        return False

    def remove_face(self, face: Face) -> Sequence['Space']:
        """
        Remove a face from the space and adjust the edges list accordingly
        from the first space and add it to the second one in the same time)
        :param face: face to remove from space
        """
        # TODO : we should remove the face not in self.faces check for performance purposes
        if face.space is not self or face not in self.faces:
            raise ValueError('Cannot remove a face' +
                             ' that does not belong to the space:{0}'.format(face))

        # 1 : check if the face includes the reference edge of the Space and change it
        reference_has_changed = self.change_reference(face)
        if not reference_has_changed:
            # only one face in the space
            if list(self.faces) == [self.face]:
                return self.remove_only_face(face)
            # very specific case of an encapsulating face
            return self.remove_encapsulating_face(face)

        # 2 : find the edges of the face that are in contact with the space
        same_face = True
        exit_edge = None
        enclosed_face = None
        for edge in face.edges:
            if self.starts_from_boundary(edge):
                break
            # check for enclosed face (an enclosed face has only one external face)
            # and find the exit edge
            same_face = same_face and (edge.pair.face is edge.next.pair.face)
            if edge.next.pair.next.pair is not edge:
                exit_edge = edge.next.pair.next
        else:
            if not same_face or not exit_edge.is_internal:
                raise ValueError('Can not remove from the space' +
                                 ' a face that is not on the boundary:{0}'.format(face))
            enclosed_face = True

        # CASE 1 : enclosed face
        if enclosed_face:
            logging.debug('Removing and enclosed face from the space: ' +
                          '{0} - {1}'.format(face, self))
            exit_edge.space_next = exit_edge.next
            exit_edge.pair.previous.space_next = exit_edge.pair

            # loop around the enclosed face
            edge = exit_edge.pair
            while edge is not exit_edge:
                edge.space_next = edge.next
                edge = edge.next
            face.space = None
            return [self]

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
                # we need to treat the special case of a "bowtie space"
                bowtie_edge = edge.bowtie
                if bowtie_edge:
                    edge.pair.space_next = bowtie_edge.previous.pair
                else:
                    edge.pair.space_next = previous_edge.pair
            previous_edge, previous_space_edge, previous_ante_space_edge = edge_tuple

        # remove the face from the Space
        face.space = None

        # check for separated faces amongst the modified faces
        seen = [face, None]
        modified_spaces = [self]
        for face_edge in face.edges:
            remaining_face = face_edge.pair.face
            if remaining_face in seen or remaining_face.space is not self:
                continue
            seen.append(remaining_face)
            if not remaining_face.is_linked_to_space():
                # create a new space of the same category
                # find a boundary edge
                boundary_edge = None
                for edge in remaining_face.edges:
                    if edge.is_space_boundary:
                        boundary_edge = edge
                        break

                if boundary_edge:
                    new_space = Space(self.plan, boundary_edge, category=self.category)
                    for _face in self.get_adjacent_faces(remaining_face):
                        _face.space = new_space
                    self.plan.add_space(new_space)
                    modified_spaces.append(new_space)

        return modified_spaces

    def merge(self, *spaces: 'Space') -> 'Space':
        """
        Merge the space with all the other provided spaces
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
        # check it both space belong to the same plan
        if self.plan is not space.plan:
            raise ValueError('Cannot merge two spaces not belonging to the same plan')
        # if the other space has no faces do nothing
        if space.edge is None:
            return self
        # if the space has no faces yet just swap edge references
        if self.edge is None:
            self.edge = space.edge
            space.edge = None
            return self

        logging.debug('Merging spaces: {0} - {1}'.format(self, space))

        # we make sure the new face is adjacent to at least one of the space faces
        adjacent_edges = []
        for edge in space.edges:
            if not adjacent_edges:
                # first boundary edge found
                if self.is_boundary(edge.pair):
                    adjacent_edges.append(edge)
            else:
                # note : we only search for "contiguous" space boundaries
                if (self.is_boundary(edge.pair) and edge is adjacent_edges[-1].next
                        and edge.pair.space_next is adjacent_edges[-1].pair):
                    adjacent_edges.append(edge)
                else:
                    break

        if not adjacent_edges:
            raise ValueError('Cannot merge two non adjacents spaces : {0} - {1}'
                             .format(space, self))

        # check for other shared space boundaries that are localised before the first edge in
        # space_edges : (we go backward)
        edge = adjacent_edges[0].pair.space_next.pair
        while (edge.space is space and edge.space_next is adjacent_edges[0]
               and edge not in adjacent_edges):
            adjacent_edges.insert(0, edge)
            edge = edge.pair.space_next.pair

        if len(adjacent_edges) == len(list(space.edges)):
            # the face is completely enclosed in the space
            return self._merge_enclosed_space(space)

        end_edge = adjacent_edges[-1]
        start_edge = adjacent_edges[0]

        end_edge.pair.space_previous.space_next = end_edge.space_next
        start_edge.space_previous.space_next = start_edge.pair.space_next

        # preserve space edge reference
        # (if the reference edge of the space belongs to the boundary with
        # the added face)
        if self.edge.pair in adjacent_edges:
            self.edge = end_edge.space_next  # per convention

        # remove the old space references
        for edge in adjacent_edges:
            edge.pair.space_next = None
            edge.space_next = None

        # add the new space references inside the added space
        for face in space.faces:
            face.space = self

        # finish by nulling the references of the merged space
        space.edge = None

        return self

    def _merge_enclosed_space(self, space: 'Space') -> 'Space':

        logging.debug('Merging an enclosed space: {0} - {1}'.format(space, self))
        # find the space boundary edges linking the enclosed face to the rest of the space
        touch_edges = []
        change_reference_edge = False
        for edge in space.edges:
            if edge.pair is self.edge:
                change_reference_edge = True
            if edge.pair.space_previous is not edge.space_next.pair:
                touch_edges.append((edge.pair.space_previous, edge.space_next.pair))

        # if need be we change the space reference edge
        if change_reference_edge:
            logging.debug('Changing space reference edge: {0}'.format(self))
            self.edge = touch_edges[0][0]

        for edges in touch_edges:
            edge_pair_space_previous, edge_next_pair = edges
            edge_pair_space_previous.space_next = edge_next_pair.space_next
            # check if we created a lone space edge
            self._remove_lone_space_edge(edge_pair_space_previous)

        for face in space.faces:
            face.space = self
        # remove space next references
        for edge in list(space.edges):
            edge.space_next = None
            edge.pair.space_next = None

        space.edge = None
        return self

    def insert_space(self,
                     boundary: Sequence[Coords2d],
                     category: SpaceCategory = SPACE_CATEGORIES['empty']) -> 'Space':
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
        container_face.insert_face(face_of_space)

        # remove the face of the fixed_item from the empty space
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
        vertex_1 = Vertex(*point_1)
        vertex_2 = Vertex(*point_2)
        new_edge = self.face.insert_edge(vertex_1, vertex_2)
        new_linear = Linear(self.plan, new_edge, category)
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
                            ' on the boundary of the space:{0}'.format(edge))
            return None

        # TODO : not sure about this. Does not seem like the best approach.
        # probably best to slice non rectilinear space into smaller simpler spaces,
        # than apply a grid generation to these spaces
        # max_length = max_length if max_length is not None else edge.max_length

        def callback(new_edges: Optional[Tuple[Edge, Edge]]) -> bool:
            """
            Callback to insure space consistency
            :param new_edges: Tuple of the new edges created by the cut
            """
            start_edge, end_edge, new_face = new_edges
            return end_edge.pair.space is not self

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
             options: Tuple['str'] = ('face', 'fill', 'border', 'half-edge')):
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
                ax = face.plot(ax, color=color, save=save, options=('border', 'dash'))

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

        # check if the boundary is correct
        if self.face and not pseudo_equal(self.area, self.as_sp.area, epsilon=0.01):
            logging.error('Error in space: the boundary does not contain all the space faces: ' +
                          '{0} != {1}, {2}'.format(self.area, self.as_sp.area, self.as_sp))
            is_valid = False

        for edge in self.edges:
            if edge.face not in faces:
                logging.error('Error in space: boundary edge face not in space faces: ' +
                              '{0} - {1}'.format(edge, edge.face))
                is_valid = False
            if edge.space is not self:
                logging.error('Error in edge: boundary edge with wrong space: ' +
                              '{0} - {1}'.format(edge, edge.space))
                is_valid = False

        return is_valid

    def components_associated(self) -> ['PlanComponent']:
        """
        Return the components associated to the space
        :return: [PlanComponent]
        """
        immutable_associated = []
        for edge in self.edges:
            if edge.linear is not None:
                if not (edge.linear.category.name in immutable_associated):
                    immutable_associated.append(edge.linear)
            if edge.pair.face is not None and edge.pair.face.space.category.mutable is False:
                if not (edge.pair.face.space.category.name in immutable_associated):
                    immutable_associated.append(edge.pair.face.space)
        return immutable_associated

    def components_category_associated(self) -> [str]:
        """
        Return the name of the components associated to the space
        :return: [Plan Component name]
        """
        return [component.category.name for component in self.components_associated()]

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
            if edge.pair.space is other:
                return True
        return False

    def count_ducts(self) -> float:
        """
        counts the number of ducts the space is adjacent to
        :return: float
        """
        number_ducts = 0
        for edge in self.edges:
            if edge.pair is not None and edge.pair.space is not None and edge.pair.space.category \
                    is not None \
                    and edge.pair.space.category.name is 'duct':
                number_ducts += 1
        return number_ducts

    def count_windows(self) -> float:
        """
        counts the number of linear of type window in the space
        :return: float
        """
        number_windows = 0
        for edge in self.edges:
            if edge.linear is not None and edge.linear.category is not None \
                    and edge.linear.category.window_type:
                number_windows += 1
        return number_windows


class Linear(PlanComponent):
    """
    Linear Class
    A linear is an object composed of one or several contiguous edges localized on the boundary
    of a space object
    """

    def __init__(self, plan: Plan, edge: Edge, category: LinearCategory):
        """
        Init
        :param edge: one of the edge of the Linear. A linear can have more than one edge.
        """
        if edge.space_next is None:
            raise ValueError('cannot create a linear that is not on the boundary of a space')
        super().__init__(plan, edge)
        self.category = category
        # set the circular reference
        edge.linear = self

    def __repr__(self):
        return 'Linear: ' + self.category.__repr__() + ' - ' + str(id(self))

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
        input_file = "Groslay_A-00-01_oldformat.json"
        plan = reader.create_plan_from_file(input_file)

        plan.plot(save=False)

        plt.show()

        assert plan.check()


    floor_plan()
