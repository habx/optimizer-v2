# coding=utf-8
"""
Shuffle Module
"""
from typing import TYPE_CHECKING, List, Optional, Sequence, Any

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
                 actions: List['Action'],
                 selector_args: Sequence[Any],
                 constraints: List['Constraint']):
        self.name = name
        self.actions = actions
        self.selectors_args = selector_args
        self.constraints = constraints
        self._action_index = 0

    def run(self, plan: 'Plan'):
        """
        Runs the shuffle on the provided plan
        :return:
        """
        while True:
            all_modified_spaces = []
            for space in plan.spaces:
                modified_spaces = self.current_action().apply_to(space, (),
                                                                 constraints=self.constraints)
                all_modified_spaces += modified_spaces
            if not all_modified_spaces:
                self._action_index += 1
                if not self.current_action():
                    break

    def current_action(self) -> Optional[Action]:
        """
        Returns the current action
        :return:
        """
        if self._action_index >= len(self.actions):
            return None
        return self.actions[self._action_index]
