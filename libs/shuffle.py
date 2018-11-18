# coding=utf-8
"""
Shuffle module
"""
from typing import Callable, Sequence, TYPE_CHECKING, Generator
import matplotlib.pyplot as plt

from libs.growth import GROWTH_METHODS

if TYPE_CHECKING:
    from libs.plan import Space
    from libs.mesh import Face
    from libs.seeder import Seeder
    from libs.growth import GrowthMethod


class Shuffle:
    """
    Shuffle class
    Will change a plan according to selectors, mutations, and constraints
    • selectors : return the edge to mutate
    • mutations : modify the face and return the two modified faces
    • constraints : score the new spaces
    """
    def __init__(self, name, selector: 'GrowthMethod', mutation: 'Mutation', score: 'Score'):
        self.name = name
        self.selector = selector
        self.mutation = mutation
        self.score = score

    def apply_to(self, seeder: 'Seeder'):
        """
        Runs the shuffle
        :param seeder:
        :return:
        """
        while True:
            modified_spaces = []
            for seed in seeder.seeds:
                for face in self.selector(seed):
                    new_spaces = self.modify(face, seed.space, self.mutation)
                    modified_spaces += new_spaces
                    #
                    if new_spaces:
                        face.space.plan.plot(save=False, options=('fill', 'border', 'face'))
                        plt.show()
                    #
            if not modified_spaces:
                break

    def modify(self, face: 'Face', space: 'Space',
               mutation: 'Mutation') -> Sequence['Space']:
        """
        Modifies the plan and check if the mutation has improved
        :param face:
        :param space:
        :param mutation:
        :return:
        """
        initial_spaces = space, face.space
        old_score = self.score.compute(*initial_spaces)
        new_spaces = self.mutation.apply_to(face, space)
        new_score = self.score.compute(*new_spaces)

        if old_score <= new_score:
            mutation.revert(face, initial_spaces[1])
            return ()

        return new_spaces


class Mutation:
    """
    Mutation class
    Will mutate a face and return the modified spaces
    """
    def __init__(self, name: str, action: Callable, revert_action: Callable):
        self.name = name
        self._action = action
        self._revert_action = revert_action

    def apply_to(self, face: 'Face', space: 'Space'):
        """
        Applies the mutation
        :param face:
        :param space:
        :return:
        """
        return self._action(face, space)

    def revert(self, face: 'Face', space: 'Space'):
        """
        :param face:
        :param space
        :return:
        """
        return self._revert_action(face, space)


class Score:
    """
    Score class
    """
    def __init__(self, name, score_function):
        self.name = name
        self._score_function = score_function

    def compute(self, *spaces) -> float:
        """
        Computes the score
        :param spaces:
        :return:
        """
        output = 0
        for space in spaces:
            output += self._score_function(space)

        return output


def swap_action(face: 'Face', space: 'Space') -> Sequence['Space']:
    """
    Swaps a face from two spaces
    :param face:
    :param space:
    :return:
    """
    if face is None:
        return ()

    if face.space is None:
        raise ValueError('The edge has to belong to the boundary of a space')

    face.space.remove_face(face)
    space.add_face(face)

    return space, face.space


def reverse_swap_action(face, space) -> Sequence['Space']:
    """
    Reverts a swap
    :param face:
    :param space:
    :return:
    """
    if face.space is None:
        raise ValueError('The edge pair has to belong to the boundary of a space')

    face.space.remove_face(face)
    space.add_face(face)

    return ()


def corner_score(space: 'Space') -> float:
    """
    Scores a space by counting the number of corners
    :param space:
    :return:
    """
    number_of_corners = 0
    for edge in space.edges:
        if not edge.next_is_aligned:
            number_of_corners += 1

    return number_of_corners


def any_faces(seed: 'Seed') -> Generator['Face', None, None]:
    """
    Returns any face around the seed space
    :param seed:
    :return:
    """
    for edge in list(seed.space.edges):
        if edge.pair.face and edge.pair.face.space is not seed.space:
            yield edge.pair.face


swap_mutation = Mutation('swap', swap_action, reverse_swap_action)
number_corners_score = Score('corners', corner_score)
simple_shuffle = Shuffle('simple', any_faces, swap_mutation,
                         number_corners_score)


if __name__ == '__main__':
    from libs.plan import Plan

    def score_try():
        """
        Test
        :return:
        """
        perimeter = [(0, 0), (200, 200), (500, 0), (500, 500), (0, 500)]
        my_plan = Plan('plan').from_boundary(perimeter)
        print(number_corners_score.compute(my_plan.empty_space))
        my_plan.plot(save=False)
        plt.show()

    score_try()

