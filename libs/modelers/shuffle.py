# coding=utf-8
"""
Shuffle Module
"""
from typing import TYPE_CHECKING, Optional, Sequence, Any
from libs.io.plot import Plot

import matplotlib.pyplot as plt
import logging

from libs.operators.mutation import MUTATIONS
from libs.operators.selector import SELECTORS, SELECTOR_FACTORIES
from libs.operators.constraint import CONSTRAINTS
from libs.operators.action import Action

if TYPE_CHECKING:
    from libs.operators.action import Action
    from libs.operators.constraint import Constraint
    from libs.plan.plan import Plan


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
                self._plot = Plot(plan)
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
swap_action_room = Action(SELECTORS["polish"], MUTATIONS["swap_face"])
swap_aligned_action = Action(SELECTORS["swap_aligned"], MUTATIONS["swap_aligned_face"])

simple_shuffle = Shuffle('simple', [swap_action], (), [CONSTRAINTS['square_shape']])
simple_shuffle_min_size = Shuffle('simple_min_size', [swap_action], (),
                                  [CONSTRAINTS['square_shape'],
                                   CONSTRAINTS["min_size"],
                                   CONSTRAINTS['few_corners']])

few_corner_shuffle = Shuffle('few_corners', [swap_seed_action], (), [CONSTRAINTS['few_corners']])

bedroom_corner_shuffle = Shuffle("bedroom_corners",
                                 [Action(SELECTOR_FACTORIES["category"](["bedroom"]),
                                         MUTATIONS['swap_face'])],
                                 (), [CONSTRAINTS['few_corners']])

square_shape_shuffle = Shuffle('square_shape', [swap_seed_action], (), [CONSTRAINTS['square_shape'],
                                                                        CONSTRAINTS['few_corners']])


square_shape_shuffle_rooms = Shuffle('square_shape', [swap_action_room], (),
                                     [CONSTRAINTS['few_corners']])

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
    "simple_shuffle_min_size": simple_shuffle_min_size,
    "square_shape_shuffle_rooms": square_shape_shuffle_rooms,
    "bedrooms_corner": bedroom_corner_shuffle
}

if __name__ == '__main__':

    logging.getLogger().setLevel(logging.DEBUG)

    def try_plan():
        """Tries to shuffle a plan"""
        import libs.io.reader as reader
        import libs.io.writer as writer
        from libs.modelers.grid import GRIDS
        from libs.modelers.seed import Seeder, GROWTH_METHODS
        from libs.space_planner.space_planner import SpacePlanner
        from libs.plan.plan import Plan

        plan_name = "grenoble_201"

        try:
            new_serialized_data = reader.get_plan_from_json(plan_name + "_solution")
            plan = Plan(plan_name).deserialize(new_serialized_data)
        except FileNotFoundError:
            plan = reader.create_plan_from_file(plan_name + ".json")
            GRIDS['optimal_grid'].apply_to(plan)
            seeder = Seeder(plan, GROWTH_METHODS).add_condition(SELECTORS['seed_duct'], 'duct')
            (seeder.plant()
             .grow(show=True)
             .fill(show=True)
             .merge_small_cells(min_cell_area=10000, show=True))
            spec = reader.create_specification_from_file(plan_name + "_setup0.json")
            spec.plan = plan

            space_planner = SpacePlanner("test", spec)
            best_solutions = space_planner.solution_research()
            if best_solutions:
                solution = best_solutions[0]
                plan = solution.plan
                writer.save_plan_as_json(plan.serialize(), plan_name + "_solution")
            else:
                logging.info("No solution for this plan")
                return

        SHUFFLES["bedrooms_corner"].run(plan, show=True)

    try_plan()

