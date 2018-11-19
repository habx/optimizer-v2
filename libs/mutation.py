# coding=utf-8
"""
Mutation module
A mutation modifies a space
The module exports a catalog containing various mutations
"""
from typing import Callable, Sequence, Optional, TYPE_CHECKING

from libs.utils.catalog import Catalog
import libs.transformation as transformation

if TYPE_CHECKING:
    from libs.mesh import Edge
    from libs.plan import Space
    from libs.utils.custom_types import TwoEdgesAndAFace

EdgeTransformation = Callable[['Edge'], Sequence['Space']]
ReverseEdgeTransformation = Callable[['Edge', Sequence['Space']], Sequence['Space']]
MutationFactoryFunction = Callable[..., EdgeTransformation]


MUTATIONS = Catalog('mutations')


class Mutation:
    """
    Mutation class
    Will mutate a face and return the modified spaces
    """
    def __init__(self, action: EdgeTransformation,
                 reverse_action: Optional[ReverseEdgeTransformation] = None, name: str = ''):
        self.name = name
        self._action = action
        self._reverse_action = reverse_action

    def __repr__(self):
        return 'Mutation: {0}'.format(self.name)

    def apply_to(self, edge: 'Edge'):
        """
        Applies the mutation to the edge
        :param edge:
        :return: the list of modified spaces
        """
        return self._action(edge)

    def reverse(self, edge: 'Edge', spaces: Sequence['Space']):
        """
        Reverse the mutation.
        Note : some mutation cannot be reversed perfectly (for example when a space is split into
        several spaces by the removal of a face)
        :param edge:
        :param spaces
        :return:
        """
        if self._reverse_action is None:
            raise Exception('This mutation can not be reversed ! : {0}'.format(self))
        return self._reverse_action(edge, spaces)


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

        return Mutation(self.factory(*args, **kwargs), name)


# Mutation Catalog

# Face Mutations

def add_face(edge: 'Edge') -> Sequence['Space']:
    """
    Swaps a face from two spaces
    :param edge:
    :return:
    """
    if edge.pair.face is None:
        return ()

    if not edge.pair.space.mutable:
        return ()

    space = edge.space
    other_space = edge.pair.space
    face = edge.pair.face

    if space is None:
        raise ValueError('The edge has to belong to the boundary of a space')

    if other_space is None:
        raise ValueError('The pair edge has to belong to the boundary of a space')

    other_space.remove_face(face)
    space.add_face(face)

    return space, other_space


def remove_face(edge: 'Edge', spaces: Sequence['Space']) -> Sequence['Space']:
    """
    Swaps a face from two spaces
    :param edge:
    :param spaces:
    :return:
    """
    space = spaces[0]
    other_space = spaces[1]
    face = edge.pair.face

    space.remove_face(face)
    other_space.add_face(face)

    return space, other_space


MUTATIONS.add(Mutation(add_face, remove_face, 'add_face'))


# Cuts Mutation


def _space_modified_by_cut(cut_data: 'TwoEdgesAndAFace') -> Sequence['Space']:
    """
    Used to check if the cut has changed the mesh (meaning a new face has been created)
    :param cut_data:
    :return:
    """
    return [cut_data[2].space] if (cut_data and cut_data[2]) else None


def _remove_edge(edge: 'Edge') -> Sequence['Space']:
    remaining_face = edge.remove()
    return [remaining_face.space] if remaining_face else []


MUTATIONS.add(Mutation(_remove_edge, name='remove_edge'))


def _ortho_cut(edge: 'Edge') -> Sequence['Space']:
    cut_data = edge.ortho_cut()
    return _space_modified_by_cut(cut_data)


MUTATIONS.add(Mutation(_ortho_cut, name='ortho_projection_cut'))


def barycenter_cut(coeff: float,
                   angle: float = 90.0,
                   traverse: str = 'relative') -> EdgeTransformation:
    """
    Action Factory
    :param coeff:
    :param angle:
    :param traverse:
    :return: an action function
    """
    def _action(edge: 'Edge') -> Sequence['Space']:
        cut_data = edge.recursive_barycenter_cut(coeff, angle, traverse)
        return _space_modified_by_cut(cut_data)

    return _action


MUTATIONS.add_factory(MutationFactory(barycenter_cut, 'barycenter_cut'))


def translation_cut(dist: float,
                    angle: float = 90.0,
                    traverse: str = 'relative') -> EdgeTransformation:
    """
    Action Factory
    :param dist:
    :param angle:
    :param traverse:
    :return: an action function
    """
    def _action(edge: 'Edge') -> Sequence['Space']:
        # check the length of the edge. It cannot be inferior to the translation distance
        if edge.length < dist:
            return []
        # create a translated vertex
        vertex = (transformation.get['translation']
                  .config(vector=edge.unit_vector, coeff=dist)
                  .apply_to(edge.start))
        cut_data = edge.recursive_cut(vertex, angle=angle, traverse=traverse)
        return _space_modified_by_cut(cut_data)

    return _action


MUTATIONS.add_factory(MutationFactory(translation_cut, 'translation_cut'))
