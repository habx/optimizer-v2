# coding=utf-8
"""
Shuffle Module
"""
from typing import TYPE_CHECKING, Optional, Sequence, Any
from libs.plot import Plot

import matplotlib.pyplot as plt
import logging

from libs.mutation import MUTATIONS
from libs.selector import SELECTORS
from libs.constraint import CONSTRAINTS
from libs.action import Action

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

    def run(self, plan: 'Plan',
            selector_args: Optional[Sequence[Any]] = None,
            show: bool = False,
            plot=None):
        """
        Runs the shuffle on the provided plan
        :param plan: the plan to modify
        :param selector_args: arguments for the selector if we need them at runtime
        :param show: whether to show a live plotting of the plan
        :param plot: a plot, in order to draw the shuffle on top (for example in a seed sequence)
        :return:
        """
        logging.debug("Shuffle: Running for plan %s", plan)

        for action in self.actions:
            action.flush()

        if show:
            if not plot:
                self._plot = Plot()
                plt.ion()
                self._plot.draw(plan)
                plt.show()
                plt.pause(1)
            else:
                self._plot = plot

        self._action_index = 0
        slct_args = selector_args if selector_args else self.current_selector_args

        while True:

            all_modified_spaces = []

            for space in plan.spaces:
                modified_spaces = self.current_action.apply_to(space, slct_args, self.constraints)
                if modified_spaces and show:
                    self._plot.update(modified_spaces)

                all_modified_spaces += modified_spaces

            if not all_modified_spaces:
                self._action_index += 1
                if not self.current_action:
                    break

        plan.remove_null_spaces()

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


swap_seed_action = Action(SELECTORS['other_seed_space'], MUTATIONS['swap_face'])
swap_action = Action(SELECTORS["space_boundary"], MUTATIONS["swap_face"])
swap_aligned_action = Action(SELECTORS["swap_aligned"], MUTATIONS["swap_aligned_face"])

simple_shuffle = Shuffle('simple', [swap_action], (), [CONSTRAINTS['square_shape']])
simple_shuffle_min_size = Shuffle('simple_min_size', [swap_action], (),
                                  [CONSTRAINTS['square_shape'],
                                   CONSTRAINTS["min_size"],
                                   CONSTRAINTS['few_corners']])

few_corner_shuffle = Shuffle('few_corners', [swap_seed_action], (), [CONSTRAINTS['few_corners']])
square_shape_shuffle = Shuffle('square_shape', [swap_seed_action], (), [CONSTRAINTS['square_shape'],
                                                                        CONSTRAINTS['few_corners']])

square_shape_shuffle_component = Shuffle('square_shape', [swap_seed_action], (),
                                         [CONSTRAINTS['square_shape'],
                                          CONSTRAINTS['few_corners'],
                                          CONSTRAINTS['component_surface_objective'],
                                          CONSTRAINTS['component_aspect_constraint'],
                                          CONSTRAINTS['alignments']])


square_shape_shuffle_component_aligned = Shuffle('swap_aligned',
                                                 [swap_aligned_action],
                                                 (),
                                                 [CONSTRAINTS['square_shape'],
                                                  CONSTRAINTS['few_corners'],
                                                  CONSTRAINTS['component_surface_objective'],
                                                  CONSTRAINTS['number_of_components']])

SHUFFLES = {
    "seed_few_corner": few_corner_shuffle,
    "seed_square_shape": square_shape_shuffle,
    "seed_square_shape_component": square_shape_shuffle_component,
    "seed_square_shape_component_aligned": square_shape_shuffle_component_aligned,
    "simple_shuffle": simple_shuffle,
    "simple_shuffle_min_size": simple_shuffle_min_size
}

if __name__ == '__main__':
    import matplotlib

    from libs.grid import GRIDS
    from libs.plan import Plan
    from libs.category import SPACE_CATEGORIES

    matplotlib.use('TkAgg')

    logging.getLogger().setLevel(logging.DEBUG)


    def rectangular_plan(width: float, depth: float) -> Plan:
        """
        a simple rectangular plan

       0, depth   width, depth
         +------------+
         |            |
         |            |
         |            |
         +------------+
        0, 0     width, 0

        :return:
        """
        boundaries = [(0, 0), (width, 0), (width, depth), (0, depth)]
        plan = Plan("square")
        plan.add_floor_from_boundary(boundaries)
        return plan


    def seed_square_shape():
        """
        Test
        :return:
        """
        simple_grid = GRIDS["simple_grid"]
        plan = rectangular_plan(500, 500)
        plan = simple_grid.apply_to(plan)

        plan.plot(save=False)
        plt.show()

        new_space_boundary = [(62.5, 0), (62.5, 62.5), (0, 62.5), (0, 0)]
        seed = plan.insert_space_from_boundary(new_space_boundary, SPACE_CATEGORIES["seed"])

        MUTATIONS["swap_face"].apply_to(seed.edge.next, seed)
        MUTATIONS["swap_face"].apply_to(seed.edge, seed)

        plan.plot()
        plt.show()

        plan.empty_space.category = SPACE_CATEGORIES["seed"]

        SHUFFLES["simple_shuffle"].run(plan, show=True)

        plan.plot()
        plan.check()


    seed_square_shape()
