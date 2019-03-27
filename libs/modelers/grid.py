# coding=utf-8
"""
Grid module
"""

from typing import Tuple, Sequence, Optional, TYPE_CHECKING
import logging
import matplotlib.pyplot as plt

from libs.io import reader
from libs.operators.mutation import MUTATIONS, MUTATION_FACTORIES
from libs.operators.selector import SELECTORS, SELECTOR_FACTORIES
from libs.io.plot import Plot


if TYPE_CHECKING:
    from libs.operators.mutation import Mutation
    from libs.operators.selector import Selector
    from libs.plan.plan import Plan, Space, Edge


class Grid:
    """
    Creates a grid inside a plan.
    TODO : for clarity purpose we should probably replace the operator concept by an Action instance
    """

    def __init__(self, name: str, operators: Sequence[Tuple['Selector', 'Mutation', bool]]):
        self.name = name
        self.operators = operators or []
        self._plot: Optional['Plot'] = None
        self._seen: ['Edge'] = []  # use to modify an edge only once

    def clone(self, name: str = "") -> 'Grid':
        """
        Returns a shallow clone of the grid.
        :param name of the new grid
        :return:
        """
        name = name or self.name + "__copy"
        new_grid = Grid(name, self.operators[:])
        return new_grid

    def apply_to(self, plan: 'Plan', show: bool = False) -> 'Plan':
        """
        Returns the modified plan with the created grid
        :param plan:
        :param show: display the modifications
        :return: the plan with the created grid
        """
        logging.debug("Grid: Applying Grid %s to plan %s", self.name, plan)

        if show:
            self._initialize_plot(plan)

        for operator in self.operators:
            self._seen = []
            self._apply_operator(plan, operator, show)
            # we simplify the mesh between each application of an operator
            plan.simplify()
            plan.update_from_mesh()

        if show:
            self._destroy_plot()

        return plan

    def _apply_operator(self, plan: 'Plan',
                        operator: Tuple['Selector', 'Mutation', bool],
                        show: bool = False):
        """
        Apply operation to the empty spaces of the plan
        :param plan:
        :param operator:
        :param show:
        :return:
        """
        for empty_space in plan.empty_spaces:
            mesh_has_changed = self._select_and_slice(empty_space, operator, show)
            if mesh_has_changed:
                return self._apply_operator(plan, operator, show)
        return

    def _select_and_slice(self, space: 'Space',
                          operator: Tuple['Selector', 'Mutation', bool], show: bool) -> bool:
        """
        Selects the correct edges and applies the slice transformation to them
        :param space:
        :param operator:
        :return:
        """
        _selector, _mutation, apply_once = operator
        for edge in _selector.yield_from(space):
            if edge in self._seen:
                continue
            logging.debug("Grid: Applying cut %s to edge %s of space %s", _mutation, edge, space)
            mesh_has_changed = _mutation.apply_to(edge, space)
            if show:
                self._plot.update_faces([space])
            if apply_once:
                self._seen.append(edge)
            if mesh_has_changed:
                return True
        return False

    def _initialize_plot(self, plan: 'Plan', plot: Optional['Plot'] = None):
        """
        Creates a plot
        :return:
        """
        # if the grid has already a plot : do nothing
        if self._plot:
            return

        if not plot:
            self._plot = Plot(plan)
            plt.ion()
            self._plot.draw(plan)
            plt.show()
            plt.pause(0.0001)
        else:
            self._plot = plot

    def _destroy_plot(self):
        """
        destroy plot on exit
        :return:
        """
        if self._plot:
            plt.ioff()
            plt.close()

    def extend(self, name: str = "", *operators: Tuple['Selector', 'Mutation', bool]) -> 'Grid':
        """
        Adds one or several operators to the grid, returns a new grid.
        Can be used to extend grids for example.
        :param name: name of the new grid
        :param operators:
        :return: a new grid
        """
        new_grid = self.clone(name)
        for operator in operators:
            new_grid.operators.append(operator)
        return new_grid

    def __add__(self, other: 'Grid') -> 'Grid':
        """
        Returns a grid that is the concatenation of the two grids
        :param other: the other grid
        :return: a new grid
        """
        new_grid = self.clone("{} + {}".format(self.name, other.name))
        for operator in other.operators:
            new_grid.operators.append(operator)
        return new_grid


# grid
simple_grid = Grid("simple_grid", [
    (
        SELECTOR_FACTORIES["edges_length"]([], [[100]]),
        MUTATION_FACTORIES["barycenter_cut"](0.5), False
    )
])

sequence_grid = Grid('sequence_grid', [
    (
        SELECTORS["previous_angle_salient_non_ortho"],
        MUTATIONS['ortho_projection_cut'], False
    ),
    (
        SELECTORS["next_angle_salient_non_ortho"],
        MUTATIONS['ortho_projection_cut'], False
    ),
    (
        SELECTORS["previous_angle_convex"],
        MUTATIONS['ortho_projection_cut'], False
    ),
    (
        SELECTORS["next_angle_convex"],
        MUTATIONS['ortho_projection_cut'], False
    ),
    (
        SELECTORS["previous_angle_salient_ortho"],
        MUTATIONS['ortho_projection_cut'], False
    ),
    (
        SELECTORS["close_to_window"],
        MUTATIONS['remove_edge'], False
    ),
    (
        SELECTORS["between_windows"],
        MUTATION_FACTORIES['barycenter_cut'](0.5), False
    ),
    (
        SELECTORS["between_edges_between_windows"],
        MUTATION_FACTORIES['barycenter_cut'](0.5), False
    ),
    (
        SELECTORS["aligned_edges"],
        MUTATION_FACTORIES['barycenter_cut'](1.0), False
    ),
    (
        SELECTORS["edge_min_150"],
        MUTATION_FACTORIES['barycenter_cut'](0.5), False
    ),
    (
        SELECTORS["close_to_window"],
        MUTATIONS['remove_edge'], False

    )
])

ortho_grid = Grid('ortho_grid', [
    (
        SELECTORS["previous_angle_salient_non_ortho"],
        MUTATIONS['ortho_projection_cut'], False
    ),
    (
        SELECTORS["next_angle_salient_non_ortho"],
        MUTATIONS['ortho_projection_cut'], False
    ),
    (
        SELECTORS["previous_angle_convex"],
        MUTATIONS['ortho_projection_cut'], False
    ),
    (
        SELECTORS["next_angle_convex"],
        MUTATIONS['ortho_projection_cut'], False
    ),
    (
        SELECTORS["previous_angle_salient_ortho"],
        MUTATION_FACTORIES['barycenter_cut'](0), False
    ),
    (
        SELECTORS["next_angle_salient_ortho"],
        MUTATION_FACTORIES['barycenter_cut'](1.0), False
    ),
    (
        SELECTORS["edge_min_500"],
        MUTATION_FACTORIES['barycenter_cut'](0.5), False
    ),
    (
        SELECTORS["between_windows"],
        MUTATION_FACTORIES['barycenter_cut'](0.5), False
    ),
    (
        SELECTORS["between_edges_between_windows"],
        MUTATION_FACTORIES['barycenter_cut'](0.5), False
    ),
    (
        SELECTORS["edge_min_300"],
        MUTATION_FACTORIES['barycenter_cut'](0.5), False
    ),
    (
        SELECTORS["aligned_edges"],
        MUTATION_FACTORIES['barycenter_cut'](1.0), False
    )
])

finer_ortho_grid = ortho_grid.extend("finer_ortho_grid",
                                     (SELECTORS["edge_min_150"],
                                      MUTATION_FACTORIES['barycenter_cut'](0.5), False))

rectangle_grid = Grid("rectangle", [
    (SELECTORS["duct_edge_min_10"], MUTATION_FACTORIES["rectangle_cut"](180), True),
    (SELECTORS["duct_edge_min_10"], MUTATION_FACTORIES["rectangle_cut"](180, 180,
                                                                        relative_offset=1.0), True)
])

section_grid = Grid("section", [
    (SELECTORS["next_concave_non_ortho"], MUTATION_FACTORIES["section_cut"](1), True),
    (SELECTORS["previous_concave_non_ortho"], MUTATION_FACTORIES["section_cut"](0), True),
    (SELECTORS["previous_convex_non_ortho"], MUTATION_FACTORIES["section_cut"](0), True),
    (SELECTORS["next_convex_non_ortho"], MUTATION_FACTORIES["section_cut"](1), True),
])

corner_grid = Grid("corner", [
    (SELECTORS["previous_angle_salient"], MUTATION_FACTORIES["barycenter_cut"](0), True),
    (SELECTORS["next_angle_salient"], MUTATION_FACTORIES["barycenter_cut"](1), True)
])

load_bearing_wall_grid = Grid("load_bearing_wall", [
    (SELECTORS["adjacent_to_load_bearing_wall"],
     MUTATION_FACTORIES["barycenter_cut"](0), True),
    (SELECTORS["adjacent_to_load_bearing_wall"],
     MUTATION_FACTORIES["barycenter_cut"](1), True)
])

window_grid = Grid("window", [
    (SELECTORS["between_windows"], MUTATION_FACTORIES["barycenter_cut"](0.5), True),
    (SELECTORS["between_edges_between_windows"], MUTATION_FACTORIES["barycenter_cut"](0.5), True),
    (SELECTORS["window_doorWindow"], MUTATION_FACTORIES["slice_cut"](300), True),
    (SELECTORS["before_window"],
     MUTATION_FACTORIES["translation_cut"](10, reference_point="end"), True),
    (SELECTORS["after_window"], MUTATION_FACTORIES["translation_cut"](10), True)
])

duct_grid = Grid("duct", [
    (SELECTORS["duct_edge_min_10"], MUTATION_FACTORIES["slice_cut"](180, padding=60), True),
    (SELECTORS["duct_edge_min_10"], MUTATION_FACTORIES["slice_cut"](100, padding=60), True),
    (SELECTORS["duct_edge_not_touching_wall"], MUTATION_FACTORIES["barycenter_cut"](0), True),
    (SELECTORS["duct_edge_not_touching_wall"], MUTATION_FACTORIES["barycenter_cut"](1), True),
    (SELECTORS["corner_duct_first_edge"], MUTATION_FACTORIES["barycenter_cut"](1), True),
    (SELECTORS["corner_duct_second_edge"], MUTATION_FACTORIES["barycenter_cut"](0), True),
    (SELECTORS["duct_edge_min_160"], MUTATION_FACTORIES["barycenter_cut"](0.5),
     True),
])

entrance_grid = Grid("front_door", [
    (SELECTORS["front_door"], MUTATION_FACTORIES["slice_cut"](130, padding=20), True),
    (SELECTORS["before_front_door"],
     MUTATION_FACTORIES["translation_cut"](5, reference_point="end"), True),
    (SELECTORS["after_front_door"], MUTATION_FACTORIES["translation_cut"](5), True)
])

stair_grid = Grid("starting_step", [
    (SELECTORS["starting_step"], MUTATION_FACTORIES["slice_cut"](130, padding=20), True),
    (SELECTORS["before_starting_step"],
     MUTATION_FACTORIES["translation_cut"](5, reference_point="end"), True),
    (SELECTORS["after_starting_step"], MUTATION_FACTORIES["translation_cut"](5), True)
])

completion_grid = Grid("completion", [
    (SELECTORS["wrong_direction"], MUTATIONS["remove_line"], True),
    # (SELECTORS["edge_min_150"], MUTATION_FACTORIES["barycenter_cut"](0.5), False),
    (SELECTORS["all_aligned_edges"], MUTATION_FACTORIES['barycenter_cut'](1.0), False)
])

cleanup_grid = Grid("cleanup", [
    (SELECTORS["adjacent_to_empty_space"], MUTATIONS["merge_spaces"], True),
    (SELECTORS["cuts_linear"], MUTATIONS["remove_edge"], True),
    (SELECTORS["close_to_wall"], MUTATIONS["remove_edge"], False),
    (SELECTORS["close_to_window"], MUTATIONS["remove_edge"], False),
    (SELECTORS["close_to_front_door"], MUTATIONS["remove_edge"], False),
    (SELECTORS["corner_face"], MUTATIONS["remove_edge"], False),
    (SELECTOR_FACTORIES["tight_lines"]([60]), MUTATIONS["remove_line"], False),
    (SELECTORS["h_edge"], MUTATIONS["remove_edge"], False),
    (SELECTORS["face_min_area"], MUTATIONS["remove_edge"], False)
])


GRIDS = {
    "ortho_grid": ortho_grid,
    "sequence_grid": sequence_grid,
    "simple_grid": simple_grid,
    "finer_ortho_grid": finer_ortho_grid,
    "rectangle_grid": rectangle_grid,
    "duct": duct_grid,
    "optimal_grid": (section_grid + corner_grid + duct_grid + load_bearing_wall_grid + window_grid +
                     entrance_grid + stair_grid + completion_grid + cleanup_grid),
    "test_grid_temp": section_grid
}

if __name__ == '__main__':

    logging.getLogger().setLevel(logging.DEBUG)

    def create_a_grid():
        """
        Test
        :return:
        """
        plan = reader.create_plan_from_file("saint-maur-raspail_H03.json")
        plan.check()
        new_plan = GRIDS["optimal_grid"].apply_to(plan, show=True)
        new_plan.check()
        new_plan.plot(save=False)
        plt.show()
        print(len(new_plan.mesh.faces))


    create_a_grid()
