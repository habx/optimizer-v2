# coding=utf-8
"""
Mutation module
A mutation modifies a space
The module exports a catalog containing various mutations
TODO : we should distinguish mesh mutations from spaces mutations
"""
from typing import Callable, Sequence, Optional, TYPE_CHECKING
import logging

import libs.transformation as transformation
from libs.plan import Plan
import libs.utils.geometry as geometry
from libs.utils.custom_exceptions import OutsideFaceError
from libs.utils.custom_types import Vector2d

if TYPE_CHECKING:
    from libs.mesh import Edge
    from libs.plan import Space
    from libs.utils.custom_types import TwoEdgesAndAFace

EdgeMutation = Callable[['Edge', 'Space'], Sequence['Space']]
ReverseEdgeMutation = EdgeMutation
MutationFactoryFunction = Callable[..., EdgeMutation]


class Mutation:
    """
    Mutation class
    Will mutate a face and return the modified spaces
    """

    def __init__(self,
                 mutation: EdgeMutation,
                 spaces_modified: Optional[EdgeMutation] = None,
                 reversible: bool = True,
                 name: str = ''):
        self.name = name
        self._mutation = mutation
        self._spaces_modified = spaces_modified
        self._initial_state = Plan("storage")
        self.reversible = reversible

    def __repr__(self):
        return 'Mutation: {0}'.format(self.name)

    def apply_to(self, edge: 'Edge', space: 'Space') -> Sequence['Space']:
        """
        Applies the mutation to the edge
        :param edge:
        :param space:
        :return: the list of modified spaces
        """
        self._store_initial_sate(self.spaces_modified(edge, space))
        output = self._mutation(edge, space)
        # update the plan if need be
        space.plan.update_from_mesh()
        return output

    def reverse(self, modified_spaces: Sequence['Space']):
        """
        Reverse the mutation
        :param: the modified spaces including newly created spaces
        :return:
        """
        logging.debug("Mutation: Reversing the mutation: %s", self)

        if not self.reversible:
            logging.warning("Mutation: Trying to reverse an irreversible mutation: %s", self)
            return

        if len(modified_spaces) == 0:
            logging.debug("Mutation: Reversing but no spaces were modified: %s", self)
            return

        # copy back the initial state into the plan
        for space in modified_spaces:
            initial_space = self._initial_state.get_space_from_id(space.id)
            if not initial_space:
                space.remove()
            else:
                space.copy(initial_space)

        # remove all the spaces from the storage plan
        self._initial_state.clear()

    def _store_initial_sate(self, spaces_modified: Sequence['Space']):
        """
        Keeps an initial state of the modified spaces
        :param spaces_modified: the spaces that will be modified by the mutation
        :return: void
        """
        if len(spaces_modified) == 0:
            return
        self._initial_state.clear()
        # add the plan floors to the temp plan
        self._initial_state.floors = spaces_modified[0].plan.floors
        # we store the cloned spaces in a temp plan structure
        # we only clone the modified spaces for performance purposes
        for space in spaces_modified:
            space.clone(self._initial_state)

    def spaces_modified(self, edge: 'Edge', space: 'Space') -> Sequence['Space']:
        """
        Returns the spaces that the mutation will modify
        A mutation can implement its own method or per convention the spaces modified will
        be the space of the pair of the specified edge and all the other specified spaces
        This method is used in particular to reverse the mutation
        :return:
        """
        if self._spaces_modified:
            return self._spaces_modified(edge, space)

        # per default we return the space and the other adjacent space according
        # to the specific edge

        # if the edge pair is a boundary of the mesh : no space will be modified
        if not edge.pair.face:
            return []

        other_space = space.plan.get_space_of_face(edge.pair.face)
        return [space, other_space] if other_space else [space]


class MutationFactory:
    """
    Mutation Factory class
    Returns a Mutation instance when called
    """

    def __init__(self, factory: MutationFactoryFunction, name: str = '', reversible: bool = True):
        self.name = name or factory.__name__
        self.factory = factory
        self.reversible = reversible

    def __call__(self, *args, **kwargs) -> Mutation:
        name = self.name
        for arg in args:
            name += ', {0}'.format(arg)
        for key, value in kwargs.items():
            name += ', ' + key + ':{0}'.format(value)

        return Mutation(self.factory(*args, **kwargs), name=name, reversible=self.reversible)


# Mutation Catalog

# Face Mutations

def swap_face(edge: 'Edge', space: 'Space') -> Sequence['Space']:
    """
    Swaps the edge pair face: by adding it from the specified space and adding it to the other space
    Eventually merge the second space with all other specified spaces
    Returns a list of space :
    • the merged space
    • the newly created spaces by the removal of the face
    The mutation is reversible : swap_face(edge.pair, swap_face(edge, spaces))
    :param edge:
    :param space:
    :return: a list of space
    """
    assert space.has_edge(edge), "Mutation: The edge must belong to the space"

    plan = space.plan
    face = edge.pair.face

    if not face:
        logging.debug("Mutation: Trying to swap a face on the boundary of the mesh: %s", edge)
        return []

    other_space = plan.get_space_of_edge(edge.pair)

    logging.debug("Mutation: Swapping face %s of space %s to space %s", face, space, other_space)

    # only remove the face if it belongs to a space
    created_spaces = other_space.remove_face(face) if other_space else []
    space.add_face(face)

    return [space] + list(created_spaces)


def swap_aligned_face(edge: 'Edge', space: 'Space') -> Sequence['Space']:
    """
    Adds all the faces of the aligned edges
    • checks if the edge is just after a corner
    • gather all the next aligned edges
    • for each edge add the corresponding faces in an order avoiding space cutting
    :return:
    """

    assert space.has_face(edge.face), "Mutation: The edge must belong to the first space"

    plan = space.plan
    other_space = plan.get_space_of_edge(edge.pair)

    if other_space.face:
        assert (other_space.adjacent_to(edge.face),
                "Mutation: The edge face must be adjacent to the second space")

    # list of aligned edges
    aligned_edges = []
    for aligned in space.next_aligned_siblings(edge):
        if not other_space.has_edge(aligned.pair):
            break
        aligned_edges.append(aligned)

    list_face_aligned = []
    for aligned in aligned_edges:
        if aligned.face not in list_face_aligned:
            list_face_aligned.append(aligned.face)

    count_exchanged_face = 0

    face_removed = True
    while face_removed:
        face_removed = False
        for aligned_face in list_face_aligned:
            # no space removal
            if len(space._faces_id) <= 1:
                break

            if space.corner_stone(aligned_face):
                logging.debug("graph not connected CORNER STONE")
                break

            space.remove_face(aligned_face)
            list_face_aligned.remove(aligned_face)
            face_removed = True

            other_space.add_face(aligned_face)

            count_exchanged_face += 1

        if count_exchanged_face == 0:
            # no operation performed in the process
            return []

    return [space, other_space]


def remove_edge(edge: 'Edge', space: 'Space') -> Sequence['Space']:
    """
    Removes an edge from a space.
    Not reversible
    :param edge:
    :param space:
    :return:
    """
    removed = False
    if space.is_internal(edge) and not edge.is_internal:
        removed = space.remove_internal_edge(edge)
    return [space] if removed else []


def remove_line(edge: 'Edge', space: 'Space') -> Sequence['Space']:
    """
    Removes an edge an all the aligned edges to it
    Not reversible
    :param edge:
    :param space:
    :return: the modified spaces
    Example:
             ^           ^          ^         ^
             |           |          |         |
             |           |          |         |
    *------->*---------->*--------->*-------->*--...
     aligned |    Edge   | aligned  | aligned |
     Edge    |           | Edge     | Edge    |
             v           v          v         v

    """
    # find all aligned edges
    # we check forward with the edge and backward with the edge pair
    line_edges = [edge]
    for current_edge in (edge, edge.pair):
        keep_going = True
        while keep_going:
            keep_going = False
            for _edge in current_edge.end.edges:
                angle = geometry.ccw_angle(_edge.vector, current_edge.opposite_vector)
                if geometry.pseudo_equal(angle, 180, geometry.ANGLE_EPSILON):
                    # if we encounter a space edge we stop
                    if space.is_boundary(edge):
                        break
                    line_edges.append(_edge)
                    current_edge = _edge
                    keep_going = True
                    break

    removed = False
    for line_edge in line_edges:
        if space.is_internal(line_edge) and not line_edge.is_internal:
            removed += space.remove_internal_edge(line_edge)

    return [space] if removed else []

# Cuts Mutation


def _space_modified_by_cut(cut_data: 'TwoEdgesAndAFace') -> bool:
    return cut_data and cut_data[2]


def ortho_cut(edge: 'Edge', space: 'Space') -> Sequence['Space']:
    """
    Cutting an edge of a space orthogonally to the intersected edge
    :param edge:
    :param space:
    :return:
    """
    space_modified = space.ortho_cut(edge)
    return [space] if space_modified else []


def barycenter_cut(coeff: float,
                   angle: float = 90.0,
                   traverse: str = 'relative') -> EdgeMutation:
    """
    Action Factory
    :param coeff:
    :param angle:
    :param traverse:
    :return: an action function
    """

    def _action(edge: 'Edge', space: 'Space') -> Sequence['Space']:
        cut_data = space.barycenter_cut(edge, coeff, angle, traverse)
        return [space] if _space_modified_by_cut(cut_data) else []

    return _action


def translation_cut(dist: float,
                    angle: float = 90.0,
                    traverse: str = 'relative') -> EdgeMutation:
    """
    Action Factory
    :param dist:
    :param angle:
    :param traverse:
    :return: an action function
    """

    def _action(edge: 'Edge', space: 'Space') -> Sequence['Space']:
        # check the length of the edge. It cannot be inferior to the translation distance
        if edge.length < dist:
            return []
        # create a translated vertex
        vertex = (transformation.get['translation']
                  .config(vector=edge.unit_vector, coeff=dist)
                  .apply_to(edge.start))

        cut_data = space.cut(edge, vertex, angle=angle, traverse=traverse)

        return [space] if _space_modified_by_cut(cut_data) else []

    return _action


def insert_aligned_rectangle(height: float,
                             width: Optional[float] = None,
                             absolute_offset: float = 0,
                             relative_offset: Optional[float] = None) -> EdgeMutation:
    """
    Inserts a rectangular face aligned with the edge
    :param width:
    :param height:
    :param absolute_offset:
    :param relative_offset:
    :return:
    """

    def _action(edge: 'Edge', space: 'Space') -> Sequence['Space']:
        _width = width or edge.length
        offset = (relative_offset * edge.length) if relative_offset else absolute_offset
        face = space.largest_face
        rectangle = geometry.rectangle(edge.start.coords, edge.vector, _width, height, offset)
        try:
            return face.insert_crop_face_from_boundary(rectangle)
        except OutsideFaceError:
            logging.debug("Mutation: Trying to insert rectangular face outside of face: %s",
                          face)
            return []

    return _action


def slice_cut(offset: float,
              padding: float = 20,
              vector: Optional[Vector2d] = None) -> EdgeMutation:
    """
    Cuts the face with a slice
    :param offset:
    :param padding:
    :param vector:
    :return:
    """

    def _action(edge: 'Edge', space: 'Space') -> Sequence['Space']:
        face = edge.face

        # check depth of the face, if the face is not deep enough do not slice
        deep_enough = (edge.depth - padding) >= offset
        if not deep_enough:
            return []
        _vector = vector or edge.vector
        # slice the face
        created_faces = edge.slice(offset, _vector)
        return [space] if len(created_faces) > 1 else []

    return _action


MUTATION_FACTORIES = {
    "translation_cut": MutationFactory(translation_cut, reversible=False),
    "barycenter_cut": MutationFactory(barycenter_cut, reversible=False),
    "rectangle_cut": MutationFactory(insert_aligned_rectangle, reversible=False),
    "slice_cut": MutationFactory(slice_cut, reversible=False)
}

MUTATIONS = {
    "swap_face": Mutation(swap_face),
    "swap_aligned_face": Mutation(swap_aligned_face),
    "ortho_projection_cut": Mutation(ortho_cut, reversible=False),
    "remove_edge": Mutation(remove_edge, reversible=False),
    "remove_line": Mutation(remove_line, reversible=False)
}
