# coding=utf-8
"""
Shuffle Module
"""

import sys
import os
sys.path.append(os.path.abspath('../'))

from typing import TYPE_CHECKING, Optional, Sequence, Any
from libs.plot import Plot

import matplotlib.pyplot as plt

from libs.mutation import MUTATIONS
from libs.selector import SELECTORS
from libs.constraint import CONSTRAINTS
from libs.action import Action
from libs.utils.catalog import Catalog

SHUFFLES = Catalog('shuffle')

if TYPE_CHECKING:
    from libs.action import Action
    from libs.constraint import Constraint
    from libs.plan import Plan


class Shuffle:
    """
    Shuffle class
    """
    def __init__(self,
                 name: str,
                 actions: Sequence['Action'],
                 selector_args: Sequence[Sequence[Any]],
                 constraints: Sequence['Constraint']):
        self.name = name
        self.actions = actions
        self.selectors_args = selector_args
        self.constraints = constraints
        # pseudo private
        self._action_index = 0
        self._plot = None

    def run(self, plan: 'Plan', show: bool = False):
        """
        Runs the shuffle on the provided plan
        :param plan: the plan to modify
        :param show: whether to show a live plotting of the plan
        :return:
        """

        for action in self.actions:
            action.flush()

        if show:
            self._plot = Plot()
            plt.ion()
            self._plot.draw(plan)
            plt.show()
            plt.pause(0.0001)

        self._action_index = 0

        while True:

            all_modified_spaces = []

            for space in plan.spaces:
                modified_spaces = self.current_action.apply_to(space, self.current_selector_args,
                                                               constraints=self.constraints)
                if modified_spaces and show:
                    self._plot.update(modified_spaces)

                all_modified_spaces += modified_spaces

            if not all_modified_spaces:
                self._action_index += 1
                if not self.current_action:
                    break

    @property
    def current_action(self) -> Optional['Action']:
        """
        Returns the current action
        :return:
        """
        if self._action_index >= len(self.actions):
            return None
        return self.actions[self._action_index]

    @property
    def current_selector_args(self) -> Sequence[Any]:
        """
        Returns the current selector arguments for the current action
        :return:
        """
        if self._action_index >= len(self.selectors_args):
            return []
        return self.selectors_args[self._action_index]


swap_action = Action(SELECTORS['other_space_boundary'], MUTATIONS['add_face'])

few_corner_shuffle = Shuffle('few_corners', (swap_action,), (), (CONSTRAINTS['few_corners'],))
square_shape_shuffle = Shuffle('square_shape', (swap_action,), (), (CONSTRAINTS['square_shape'],
                                                                    CONSTRAINTS['few_corners']))

SHUFFLES.add(few_corner_shuffle, square_shape_shuffle)
