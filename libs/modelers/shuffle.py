# coding=utf-8
"""
Shuffle Module
"""
from typing import TYPE_CHECKING, Optional, Sequence, Any

import logging

from libs.operators.mutation import MUTATIONS
from libs.operators.selector import SELECTORS
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

    def apply_to(self, plan: 'Plan',
                 selector_args: Optional[Sequence[Any]] = None):
        """
        Runs the shuffle on the provided plan
        :param plan: the plan to modify
        :param selector_args: arguments for the selector if we need them at runtime
        :return:
        """
        logging.debug("Shuffle: Running for plan %s", plan)

        for action in self.actions:
            action.flush()

        self._action_index = 0
        slct_args = selector_args if selector_args else self.current_selector_args

        while True:

            all_modified_spaces = []

            for space in plan.spaces:
                modified_spaces = self.current_action.apply_to(space, slct_args, self.constraints)

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
                                 [
                                     Action(SELECTORS["bedroom_small_faces"],
                                            MUTATIONS["remove_face"]),
                                     Action(SELECTORS["bedroom_small_faces_pair"],
                                            MUTATIONS['swap_face']),
                                  ],
                                 (), [CONSTRAINTS["few_corners_bedroom"]])

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
        import time
        import libs.io.reader as reader
        import libs.io.writer as writer
        from libs.modelers.grid import GRIDS
        from libs.modelers.seed import SEEDERS
        from libs.space_planner.space_planner import SpacePlanner
        from libs.plan.plan import Plan

        plan_name = "001"
        specification_number = "0"
        solution_number = 1

        try:
            new_serialized_data = reader.get_plan_from_json(plan_name + "_solution_"
                                                            + str(solution_number) + ".json")
            plan = Plan(plan_name).deserialize(new_serialized_data)
        except FileNotFoundError:
            plan = reader.create_plan_from_file(plan_name + ".json")
            GRIDS['optimal_grid'].apply_to(plan)
            SEEDERS["simple_seeder"].apply_to(plan)
            spec = reader.create_specification_from_file(plan_name + "_setup"
                                                         + specification_number +".json")
            spec.plan = plan
            space_planner = SpacePlanner("test", spec)
            best_solutions = space_planner.solution_research()
            if best_solutions:
                solution = best_solutions[solution_number]
                plan = solution.plan
                writer.save_plan_as_json(plan.serialize(), plan_name + "_solution_"
                                         + str(solution_number))
            else:
                logging.info("No solution for this plan")
                return

        time.sleep(0.5)
        SHUFFLES["bedrooms_corner"].apply_to(plan)

    try_plan()

