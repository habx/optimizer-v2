# coding=utf-8
"""
Mutation module
A mutation modifies a space
The module exports a catalog containing various mutations
"""
from typing import Callable, Sequence, Optional, TYPE_CHECKING
import logging

import libs.transformation as transformation

if TYPE_CHECKING:
    from libs.mesh import Edge
    from libs.plan import Space
    from libs.utils.custom_types import TwoEdgesAndAFace

EdgeMutation = Callable[['Edge', Sequence['Space']], Sequence['Space']]
ReverseEdgeMutation = Callable[['Edge', Sequence['Space']], Sequence['Space']]
MutationFactoryFunction = Callable[..., EdgeMutation]


class Mutation:
    """
    Mutation class
    Will mutate a face and return the modified spaces
    """
    def __init__(self, mutation: EdgeMutation,
                 reverse_mutation: Optional[ReverseEdgeMutation] = None,
                 spaces_modified: Optional[EdgeMutation] = None,
                 name: str = ''):
        self.name = name
        self._mutation = mutation
        self._reverse_mutation = reverse_mutation
        self._spaces_modified = spaces_modified

    def __repr__(self):
        return 'Mutation: {0}'.format(self.name)

    def apply_to(self, edge: 'Edge', spaces: Sequence['Space']) -> Sequence['Space']:
        """
        Applies the mutation to the edge
        :param edge:
        :param spaces:
        :return: the list of modified spaces
        """
        return self._mutation(edge, spaces)

    def reverse(self, edge_pair: 'Edge', spaces: Sequence['Space']):
        """
        Reverse the mutation.
        Note : some mutation cannot be reversed perfectly (for example when a space is split into
        several spaces by the removal of a face)
        :param edge_pair:
        :param spaces
        :return:
        """
        assert self._reverse_mutation, 'This mutation can not be reversed ! : {0}'.format(self)
        return self._reverse_mutation(edge_pair, spaces)

    def spaces_modified(self, edge: 'Edge', spaces: ['Space']) -> Sequence['Space']:
        """
        Returns the spaces that the mutation will modify
        A mutation can implement its own method or per convention the spaces modified will
        be the space of the pair of the specified edge and all the other specified spaces
        :return:
        """
        if self._spaces_modified:
            return self._spaces_modified(edge, spaces)

        modified_spaces = [spaces[0].plan.get_space_of_face(edge.face)] + spaces

        return modified_spaces


class MutationFactory:
    """
    Mutation Factory class
    Returns a Mutation instance when called
    """
    def __init__(self, factory: MutationFactoryFunction, name: str = ''):
        self.name = name or factory.__name__
        self.factory = factory

    def __call__(self, *args, **kwargs) -> Mutation:
        name = self.name
        for arg in args:
            name += ', {0}'.format(arg)
        for key, value in kwargs.items():
            name += ', ' + key + ':{0}'.format(value)

        return Mutation(self.factory(*args, **kwargs), name=name)


# Mutation Catalog

# Face Mutations

def swap_face(edge: 'Edge', spaces: [Optional['Space']]) -> Sequence['Space']:
    """
    Swaps the edge face: by removing it from the first space and adding it to the second space
    Eventually merge the second space with all other specified spaces
    Returns a list of space :
    • the merged space
    • the newly created spaces by the removal of the face
    The mutation is reversable : swap_face(edge.pair, swap_face(edge, spaces))
    :param edge:
    :param spaces:
    :return: a list of space
    """
    assert len(spaces) >= 2, "At least two spaces must be provided"
    assert spaces[0].has_face(edge.face), "The edge must belong to the first space"

    if spaces[1].face:
        assert (spaces[1].adjacent_to(edge.face),
                "The edge face must be adjacent to the second space")

    face = edge.face

    logging.debug("Mutation: swapping face %s of space %s to space %s", face, spaces[0], spaces[1])

    created_spaces = spaces[0].remove_face(face)
    spaces[1].add_face(face)

    if len(spaces) > 2:
        logging.debug("Mutation: swap face, merging spaces")
        spaces[1].merge(*spaces[2:])

    return [spaces[1]] + list(created_spaces)


def add_aligned_face(_: 'Edge') -> Sequence['Space']:
    """
    Adds all the faces of the aligned edges
    • checks if the edge is just after a corner
    • gather all the next aligned edges
    • for each edge add the corresponding faces
    :return:
    """
    pass


def remove_edge(edge: 'Edge', spaces: Sequence['Space']) -> Sequence['Space']:
    """
    Removes an edge from a space.
    Not reversable
    :param edge:
    :param spaces:
    :return:
    """
    assert len(spaces) == 1, "You must provide one space"
    spaces[0].remove_internal_edge(edge)
    return spaces

# Cuts Mutation


def _space_modified_by_cut(cut_data: 'TwoEdgesAndAFace') -> bool:
    return cut_data and cut_data[2]


def ortho_cut(edge: 'Edge', spaces: Sequence['Space']) -> Sequence['Space']:
    """
    Cutting an edge of a space orthogonally to the intersected edge
    :param edge:
    :param spaces:
    :return:
    """
    assert len(spaces) == 1, "You must provide exactly one space"

    space_modified = spaces[0].ortho_cut(edge)
    return spaces if space_modified else []


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
    def _action(edge: 'Edge', spaces: Sequence['Space']) -> Sequence['Space']:

        assert len(spaces) == 1, "You must provide exactly one space"

        cut_data = spaces[0].barycenter_cut(edge, coeff, angle, traverse)
        return spaces if _space_modified_by_cut(cut_data) else []

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
    def _action(edge: 'Edge', spaces: Sequence['Space']) -> Sequence['Space']:

        assert len(spaces) == 1, "You must provide exactly one space"

        # check the length of the edge. It cannot be inferior to the translation distance
        if edge.length < dist:
            return []
        # create a translated vertex
        vertex = (transformation.get['translation']
                  .config(vector=edge.unit_vector, coeff=dist)
                  .apply_to(edge.start))

        cut_data = spaces[0].cut(edge, vertex, angle=angle, traverse=traverse)

        return spaces if _space_modified_by_cut(cut_data) else []

    return _action


MUTATION_FACTORIES = {
    "translation_cut": MutationFactory(translation_cut),
    "barycenter_cut": MutationFactory(barycenter_cut)
}

MUTATIONS = {
    "swap_face": Mutation(swap_face, swap_face),
    "ortho_projection_cut": Mutation(ortho_cut),
    "remove_edge": Mutation(remove_edge)
}
