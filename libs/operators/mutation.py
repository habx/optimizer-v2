# coding=utf-8
"""
Mutation module
A mutation modifies a space
The module exports a catalog containing various mutations
TODO : we should distinguish mesh mutations from spaces mutations
"""
from typing import Callable, Sequence, Optional, TYPE_CHECKING
import logging

from libs.plan.plan import Plan, Space
import libs.utils.geometry as geometry
from libs.utils.custom_exceptions import OutsideFaceError

if TYPE_CHECKING:
    from libs.mesh.mesh import Edge
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
        if output:
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
    Swaps the edge pair face: by adding it to the specified space and removing it to the other space
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


def remove_face(edge: 'Edge', space: 'Space') -> Sequence['Space']:
    """
    remove the edge face: by removing to the specified space and adding it to the other space
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
    face = edge.face
    other_space = plan.get_space_of_edge(edge.pair)
    if not other_space:
        return []

    logging.debug("Mutation: Swapping face %s of space %s to space %s", face, space, other_space)

    # only remove the face if it belongs to a space
    created_spaces = space.remove_face(face)
    other_space.add_face(face)

    return [other_space] + list(created_spaces)


def swap_aligned_face(edge: 'Edge', space: 'Space') -> Sequence['Space']:
    """
    Removes all the faces of the aligned edges
    • checks if the edge is just after a corner
    • gather all the next aligned edges
    • for each edge add the corresponding faces in an order avoiding space cutting
    :return:
    """

    assert space.has_face(edge.face), "Mutation: The edge must belong to the first space"

    plan = space.plan
    other_space = plan.get_space_of_edge(edge.pair)

    assert other_space.adjacent_to(edge.face), ("Mutation: The edge face must "
                                                "be adjacent to the second space")

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

    for face in list_face_aligned:
        # no space removal
        if space.number_of_faces <= 1:
            break

        if space.corner_stone(face):
            logging.debug("Mutation: aligned edges trying to remove a corner stone")
            break

        space.remove_face(face)
        list_face_aligned.remove(face)
        other_space.add_face(face)
    else:
        return []

    return [space, other_space]


def add_aligned_face(edge: 'Edge', space: 'Space') -> Sequence['Space']:
    """
    Adds to space all the faces of the aligned edges
    • checks if the edge is just after a corner
    • gather all the next aligned edges
    • for each edge add the corresponding faces
    :return:
    """
    assert space.has_face(edge.face), "Mutation: The edge must belong to the first space"

    plan = space.plan
    other_space = plan.get_space_of_edge(edge.pair)

    if other_space.face:
        assert other_space.adjacent_to(
            edge.face), "Mutation: The edge face must be adjacent to the second space"

    aligned_edges = []
    for aligned in space.next_aligned_siblings(edge):
        if not other_space.has_edge(aligned.pair):
            break
        aligned_edges.append(aligned)

    list_face_aligned = []
    for aligned in aligned_edges:
        if aligned.face not in list_face_aligned:
            list_face_aligned.append(aligned.pair.face)

    # adds faces of list_face_aligned to space.
    list_created_spaces = [other_space]
    while list_face_aligned:
        aligned_face = list_face_aligned[0]
        sp_to_remove = None
        for sp in space.plan.spaces:
            if sp.has_face(aligned_face):
                sp_to_remove = sp
                break
        if sp_to_remove:
            created_spaces = sp_to_remove.remove_face(aligned_face)
            space.add_face(aligned_face)

            list_face_aligned.remove(aligned_face)
            for sp in created_spaces:
                if sp not in list_created_spaces:
                    list_created_spaces.append(sp)
        else:
            break

    return [space] + list_created_spaces


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
    line = space.line(edge)
    removed = 0
    for line_edge in line:
        # check if the line_edge still belongs to the mesh
        # this is needed because sometimes the edge will have already be removed if
        # the removal of a previous line_edge caused the edge.remove() method to clean the mesh
        # of single vertex (edge case but it can happened and it has)
        if not line_edge.mesh:
            continue
        # if we encounter a boundary edge : we stop
        if not space.is_internal(line_edge):
            # if the first edge of the line is on the boundary we do not need to break
            if line_edge is line[0]:
                continue
            break
        # we must cannot remove an internal edge of face as it would break the mesh
        if not line_edge.is_internal:
            removed += space.remove_internal_edge(line_edge)

    return [space] if removed else []


def merge_spaces(edge: 'Edge', space: 'Space') -> Sequence['Space']:
    """
    Merge the two spaces separated by the edge
    :param edge:
    :param space:
    :return:
    """
    other = space.plan.get_space_of_edge(edge.pair)

    if not other or other is space:
        return []

    space.merge(other)

    return [space, other]


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


def barycenter_cut(coeff: float, traverse: str = "absolute") -> EdgeMutation:
    """
    Action Factory
    :param coeff:
    :param traverse:
    :return: an action function
    """

    def _action(edge: 'Edge', space: 'Space') -> Sequence['Space']:
        vector = space.best_direction(edge.normal)
        cut_data = space.barycenter_cut(edge, coeff, vector=vector, traverse=traverse)
        return [space] if _space_modified_by_cut(cut_data) else []

    return _action


def translation_cut(offset: float,
                    reference_point: str = "start") -> EdgeMutation:
    """
    Mutation Factory
    Cuts the edge after a specified offset from the start vertex
    :param offset:
    :param reference_point: the end or the start of the edge
    :return: an action function
    """

    def _action(edge: 'Edge', space: 'Space') -> Sequence['Space']:
        # check the length of the edge. It cannot be inferior to the translation distance
        if edge.length < offset:
            return []

        _coeff = offset / edge.length
        _coeff = (1 - _coeff) if reference_point == "end" else _coeff

        return barycenter_cut(_coeff)(edge, space)

    return _action


def section_cut(coeff: float,
                traverse: str = "no",
                min_area: float = 40000,
                min_angle: float = 10) -> EdgeMutation:
    """
    Action Factory. Cuts the space and creates a new space if the face has a different main axis.
    The difference is predicated according to the specified min_angle.
    The face will not be separated into a new space if its area is inferior to the min_area
    specified
    :param coeff:
    :param traverse:
    :param min_area:
    :param min_angle:
    :return:
    """

    def _action(edge: 'Edge', space: 'Space') -> Sequence['Space']:
        vector = space.best_direction(edge.normal)
        initial_face = edge.face
        cut_data = space.barycenter_cut(edge, coeff, vector=vector, traverse=traverse)
        # check the new face direction
        if not (cut_data and cut_data[2]):
            return []

        new_face = cut_data[2]
        for _face in (new_face, initial_face):
            if _face.area <= min_area:
                continue
            # check if the new face has a different direction
            new_directions = space.face_directions(_face)
            if not new_directions:
                continue
            angle_1 = geometry.ccw_angle((0, 1), new_directions[0])
            angle_2 = geometry.ccw_angle((0, 1), space.directions[0])
            delta = abs(angle_1 - angle_2) > min_angle

            if delta:
                new_space = Space(space.plan, space.floor, category=space.category)
                space.remove_face(_face)
                new_space.add_face(_face)
                return [space, new_space]

        return [space]

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


def slice_cut(offset: float, padding: float = 0) -> EdgeMutation:
    """
    Cuts the face with a slice
    :param offset:
    :param padding:
    :return:
    """

    def _action(edge: 'Edge', space: 'Space') -> Sequence['Space']:
        # check depth of the face, if the face is not deep enough do not slice
        # try recursively to slice the next adjacent face
        # note:  we're making the assumption that the mesh is a quad grid
        # we're also limiting the recursion to three times only to prevent possible infinite loops
        max_iteration = 3
        deep_enough = (edge.depth - padding) >= offset
        if not deep_enough:
            new_edge = edge.next.next.pair
            if max_iteration == 0 or space.plan.get_space_of_edge(new_edge) is not space:
                return []
            else:
                max_iteration -= 1
                return _action(new_edge, space)
        # note we need to select the opposite vector of the edge vector because
        # of the order in which shapely will return the vertices of the
        # slice intersection (see the edge slice method)
        _vector = space.best_direction(geometry.opposite_vector(edge.vector))
        # slice the face
        created_faces = edge.slice(offset, _vector)
        return [space] if len(created_faces) > 1 else []

    return _action


MUTATION_FACTORIES = {
    "translation_cut": MutationFactory(translation_cut, reversible=False),
    "barycenter_cut": MutationFactory(barycenter_cut, reversible=False),
    "rectangle_cut": MutationFactory(insert_aligned_rectangle, reversible=False),
    "slice_cut": MutationFactory(slice_cut, reversible=False),
    "section_cut": MutationFactory(section_cut, reversible=False)
}

MUTATIONS = {
    "swap_face": Mutation(swap_face),
    "remove_face": Mutation(remove_face),
    "swap_aligned_face": Mutation(swap_aligned_face),
    "add_aligned_face": Mutation(add_aligned_face),
    "ortho_projection_cut": Mutation(ortho_cut, reversible=False),
    "remove_edge": Mutation(remove_edge, reversible=False),
    "remove_line": Mutation(remove_line, reversible=False),
    "merge_spaces": Mutation(merge_spaces)
}
