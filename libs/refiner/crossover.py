# coding=utf-8
"""
Genetic Algorithm Crossover module
A cross-over takes two individuals as input and returns two blended individuals
The individuals are modified in place
"""
import random
from typing import TYPE_CHECKING
import logging
import math

from libs.refiner.mutation import MIN_ADJACENCY_EDGE_LENGTH

if TYPE_CHECKING:
    from libs.refiner.core import Individual


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


__all__ = ['connected_differences']
