# coding=utf-8
"""
Genetic Algorithm Crossover module
A cross-over takes two individuals as input and returns two blended individuals
The individuals are modified in place
"""
import logging
import math
import random

from typing import TYPE_CHECKING, Optional, Tuple

from libs.utils.custom_exceptions import SpaceShapeError
from libs.utils.geometry import cross_product

MIN_ADJACENCY_EDGE_LENGTH = None

if TYPE_CHECKING:
    from libs.refiner.core import Individual
    from libs.plan.plan import Space


def best_spaces(ind_1: 'Individual', ind_2: 'Individual') -> Tuple['Individual', 'Individual']:
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

    spaces_id = [i for i in ind_1.fitness.sp_wvalue]
    differences = [math.fabs(ind_1.fitness.sp_wvalue[i] - ind_2.fitness.sp_wvalue[i])
                   for i in spaces_id]

    # check if the plans are different
    if max(differences) == 0.0:
        logging.debug("Refiner: Crossover: The individuals are the same")
        return ind_1, ind_1

    # select a random space
    random_space_id = random.choices(spaces_id, differences, k=1)[0]

    # copy the best version of the selected space into the other individual
    if ind_1.fitness.sp_wvalue[random_space_id] > ind_2.fitness.sp_wvalue[random_space_id]:
        output_1 = ind_1
        try:
            output_2 = copy_space(ind_1, ind_2, random_space_id)
        except SpaceShapeError:
            logging.debug("Refiner: Failed to cross over individual 2: %s - %s", ind_2,
                          ind_2.get_space_from_id(random_space_id))
            output_2 = ind_2
    else:
        output_2 = ind_2
        try:
            output_1 = copy_space(ind_2, ind_1, random_space_id)
        except SpaceShapeError:
            logging.debug("Refiner: Failed to cross over individual 1: %s - %s", ind_1,
                          ind_1.get_space_from_id(random_space_id))
            output_1 = ind_1

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

    copied_space_faces_id = copied_space.faces_id
    modified_space_faces_id = modified_space.faces_id

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
        space.set_edges_check()

    modified_ind.modified_spaces |= {s.id for s in modified_spaces}

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


def connected_differences(ind_1: 'Individual', ind_2: 'Individual'):
    """
    Blends the two plans.
    For each floor :
    1. Finds every face that is different between the two individuals.
    2. Pick a random face amongst them
    3. Select all connected different faces
    4. Swaps their corresponding spaces between the two plans
    NOTE : we must check for "corner stone" situation where the removal of the faces will
    split the spaces in half... If a corner stone is found, we do nothing.
    :param ind_1:
    :param ind_2:
    :return: a tuple of the blended individual
    """
    # Note : this should never happen
    if not ind_1 or not ind_2:
        return ind_1, ind_2

    differences = {}

    for floor in ind_1.floors.values():

        for f in floor.mesh.faces:
            space_1 = ind_1.get_space_of_face(f)
            space_2 = ind_2.get_space_of_face(f)
            if space_1.id == space_2.id or not space_1.mutable or not space_2.mutable:
                continue
            differences[f] = (space_1, space_2)

        if len(differences) <= 1:
            # nothing to do
            continue

        # pick a random face and find all different faces connected to it
        # we apply the difference between the fitness value of the corresponding spaces to
        # select the face to select
        faces = list(differences.keys())
        weights = (math.fabs(ind_1.fitness.sp_wvalue[differences[f][0].id]
                             - ind_2.fitness.sp_wvalue[differences[f][1].id])
                   for f in faces)
        seed_face = random.choices(faces, weights=weights)[0]
        space_dict = differences.copy()
        connected_faces = {seed_face}
        del differences[seed_face]
        while True:
            connections = set([f for f in differences.keys()
                               for o in connected_faces if o.is_adjacent(f)])
            for f in connections:
                del differences[f]
                connected_faces.add(f)
            if not connections:
                break

        connected_faces = list(connected_faces)
        impacted_spaces_ind_1 = [space_dict[f][0] for f in connected_faces]
        impacted_spaces_ind_2 = [space_dict[f][1] for f in connected_faces]

        for space in set(impacted_spaces_ind_1) | set(impacted_spaces_ind_2):
            faces = list(filter(lambda _f: space.has_face(_f), connected_faces))
            if space.corner_stone(*faces, min_adjacency_length=MIN_ADJACENCY_EDGE_LENGTH):
                logging.debug("Crossover: No crossover possible")
                return ind_1, ind_2

        logging.debug("Refiner: Crossover: Mating %s - %s", ind_1, ind_2)
        modified_spaces_ind_1 = set()
        modified_spaces_ind_2 = set()
        for i, face in enumerate(connected_faces):
            space_1 = impacted_spaces_ind_1[i]
            space_2 = impacted_spaces_ind_2[i]
            other_1 = ind_1.get_space_from_id(space_2.id)
            other_2 = ind_2.get_space_from_id(space_1.id)

            space_1.remove_face_id(face.id)
            other_1.add_face_id(face.id)

            space_2.remove_face_id(face.id)
            other_2.add_face_id(face.id)

            modified_spaces_ind_1 |= {space_1, other_1}
            modified_spaces_ind_2 |= {space_2, other_2}

        # make sure the plan structure is correct
        for space in modified_spaces_ind_1:
            space.set_edges()
            ind_1.modified_spaces.add(space.id)

        for space in modified_spaces_ind_2:
            space.set_edges()
            ind_2.modified_spaces.add(space.id)

        return ind_1, ind_2


__all__ = ['best_spaces', 'connected_differences']
