# coding=utf-8
"""
Genetic Algorithm Crossover module
A cross-over takes two individuals as input and returns two blended individuals
The individuals are modified in place
"""
import logging
from typing import TYPE_CHECKING, Optional

from libs.utils.custom_exceptions import SpaceShapeError

if TYPE_CHECKING:
    from libs.refiner.core import Individual
    from libs.plan.plan import Space


def best_spaces(ind_1: 'Individual', ind_2: 'Individual'):
    """
    Blends the two plans.
    For each floor :
    1. Finds the best space of each individual relative to the other.
    2. Copies the best space of each individual into the other
    3. If an error is raised while copying the space, return the initial individual
    :param ind_1:
    :param ind_2:
    :return: a tuple of the crossed over individuals
    """

    differences = [(i, ind_1.fitness.sp_wvalue[i] - ind_2.fitness.sp_wvalue[i])
                   for i in ind_1.fitness.sp_wvalue]

    differences.sort(key=lambda t: t[1])

    best_1 = differences[0][0]
    best_2 = differences[-1][0]

    try:
        output_1 = copy_space(ind_2, ind_1, best_2)
    except SpaceShapeError:
        logging.info("Refiner: Failed to cross over individual 1: %s - %s", ind_1,
                     ind_1.get_space_from_id(best_2))
        output_1 = ind_1

    try:
        output_2 = copy_space(ind_1, ind_2, best_1)
    except SpaceShapeError:
        logging.info("Refiner: Failed to cross over individual 2: %s - %s", ind_2,
                     ind_2.get_space_from_id(best_1))
        output_2 = ind_2

    return output_1, output_2


def copy_space(from_ind: 'Individual', to_ind: 'Individual', space_id: int) -> 'Individual':
    """
    Copies a space from one individual to another, meaning the space will contain the exact
    same faces as the copied individual.
    In order to `copy` the space, we must:
    •  remove the missing faces from their current space and
       add them to the desired space;
    •  remove the extraneous faces from the desired spaces and add them to the space they belong
       to in the copied individual or if the space is not adjacent to any other adjacent spaces;
    •  check if the plan has been broken by splitting a space (in which case the set_edges method
       will throw an exception
    :param from_ind:
    :param to_ind:
    :param space_id:
    :return:
    """
    # we must clone the individuals to enable rollback if the plan gets broken
    copied_ind = from_ind.clone()
    modified_ind = to_ind.clone()
    copied_space = copied_ind.get_space_from_id(space_id)
    modified_space = modified_ind.get_space_from_id(space_id)

    copied_space_faces_id = set(copied_space._faces_id)
    modified_space_faces_id = set(modified_space._faces_id)

    shared_faces_id = copied_space_faces_id & modified_space_faces_id
    missing_faces_id = copied_space_faces_id - shared_faces_id
    extraneous_faces_id = modified_space_faces_id - shared_faces_id

    modified_spaces = [modified_space]

    for face_id in missing_faces_id:
        other_space = modified_ind.get_space_from_face_id(face_id, modified_space.floor.mesh.id)
        other_space.remove_face_id(face_id)
        modified_space.add_face_id(face_id)
        modified_spaces.append(other_space)

    for face_id in extraneous_faces_id:
        modified_space.remove_face_id(face_id)
        other = get_adjacent_mutable_space(face_id, modified_space)
        if not other:
            raise SpaceShapeError("No adjacent mutable space found !")
        other.add_face_id(face_id)
        modified_spaces.append(other)

    # set the new reference edges of each modified spaces
    # this might raise a SpaceShapeError Exception if we've split a space in half
    for space in modified_spaces:
        space.set_edges()

    return modified_ind


def get_adjacent_mutable_space(face_id: int, space: 'Space')-> Optional['Space']:
    """
    Returns a mutable space adjacent to the face which `face_id` is specified and
    different from the specified `space`
    :param face_id:
    :param space: the space of the face
    :return:
    """
    mesh = space.floor.mesh
    face = mesh.get_face(face_id)

    for edge in face.edges:
        # we look for edge pointing to another space
        if not edge.pair.face or not space.has_face(edge.pair.face):
            continue
        # we check if the space is mutable
        other = space.plan.get_space_from_face_id(edge.pair.face.id, mesh.id)
        if not other or not other.mutable:
            continue
        return other

    return None


__all__ = ['best_spaces']
