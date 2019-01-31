# coding=utf-8
"""
Grid module
"""

from typing import Tuple, Sequence, TYPE_CHECKING
import logging
import matplotlib.pyplot as plt

from libs import reader
from libs.mutation import MUTATIONS, MUTATION_FACTORIES
from libs.selector import SELECTORS, SELECTOR_FACTORIES

if TYPE_CHECKING:
    from libs.mutation import Mutation
    from libs.selector import Selector
    from libs.plan import Plan, Space, Edge


class Grid:
    """
    Creates a grid inside a plan.
    TODO : for clarity purpose we should probably replace the operator concept by an Action instance
    """

    def __init__(self, name: str, operators: Sequence[Tuple['Selector', 'Mutation', bool]]):
        self.name = name
        self.operators = operators or []
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

    def apply_to(self, plan: 'Plan') -> 'Plan':
        """
        Returns the modified plan with the created grid
        :param plan:
        :return: the plan with the created grid
        """
        logging.debug("Grid: Applying Grid %s to plan %s", self.name, plan)

        for operator in self.operators:
            self._seen = []
            self._apply_operator(plan, operator)
            # we simplify the mesh between each operator
            plan.mesh.simplify()
            plan.update_from_mesh()

        return plan

    def _apply_operator(self, plan: 'Plan', operator: Tuple['Selector', 'Mutation', bool]):
        """
        Apply operation to the empty spaces of the plan
        :param plan:
        :param operator:
        :return:
        """
        for empty_space in plan.empty_spaces:
            mesh_has_changed = self._select_and_slice(empty_space, operator)
            if mesh_has_changed:
                return self._apply_operator(plan, operator)
        return

    def _select_and_slice(self, space: 'Space',
                          operator: Tuple['Selector', 'Mutation', bool]) -> bool:
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
            if apply_once:
                self._seen.append(edge)
            if mesh_has_changed:
                return True
        return False

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
        SELECTORS["previous_angle_convex_non_ortho"],
        MUTATIONS['ortho_projection_cut'], False
    ),
    (
        SELECTORS["next_angle_convex_non_ortho"],
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
        SELECTORS["previous_angle_convex_non_ortho"],
        MUTATIONS['ortho_projection_cut'], False
    ),
    (
        SELECTORS["next_angle_convex_non_ortho"],
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

corner_grid = Grid("corner", [
    (SELECTORS["previous_angle_salient"], MUTATIONS['ortho_projection_cut'], True),
    (SELECTORS["next_angle_salient"], MUTATIONS['ortho_projection_cut'], True)
])

duct_grid = Grid("duct", [
    (SELECTORS["duct_edge_min_10"], MUTATION_FACTORIES["slice_cut"](180, padding=60), True),
    (SELECTORS["duct_edge_min_10"], MUTATION_FACTORIES["barycenter_cut"](0, traverse="no"), True),
    (SELECTORS["duct_edge_min_10"], MUTATION_FACTORIES["barycenter_cut"](1, traverse="no"), True),
    (SELECTORS["all_aligned_edges"], MUTATION_FACTORIES['barycenter_cut'](1.0), True)
])

load_bearing_wall_grid = Grid("load_bearing_wall", [
    (SELECTORS["adjacent_to_load_bearing_wall"],
     MUTATION_FACTORIES["barycenter_cut"](0, traverse="no"), True)
])

window_grid = Grid("window", [
    (SELECTORS["window_doorWindow"], MUTATION_FACTORIES["slice_cut"](300), True),
    (SELECTORS["between_windows"], MUTATION_FACTORIES['barycenter_cut'](0.5), True),
    (SELECTORS["between_edges_between_windows"], MUTATION_FACTORIES['barycenter_cut'](0.5), True)
])

cleanup_grid = Grid("small_faces", [
    (SELECTORS["close_to_window"], MUTATIONS["remove_edge"], False),
    (SELECTORS["cuts_linear"], MUTATIONS["remove_edge"], True),
    (SELECTOR_FACTORIES["min_depth"]([40]), MUTATIONS["remove_line"], False),
    (SELECTORS["close_to_external_wall"], MUTATIONS["remove_edge"], False),
    (SELECTORS["h_edge"], MUTATIONS["remove_edge"], True)
])

GRIDS = {
    "ortho_grid": ortho_grid,
    "sequence_grid": sequence_grid,
    "simple_grid": simple_grid,
    "finer_ortho_grid": finer_ortho_grid,
    "rectangle_grid": rectangle_grid,
    "duct": duct_grid,
    "test_grid": corner_grid + load_bearing_wall_grid + window_grid + duct_grid + cleanup_grid
}

if __name__ == '__main__':

    logging.getLogger().setLevel(logging.DEBUG)

    def create_a_grid():
        """
        Test
        :return:
        """
        plan = reader.create_plan_from_file("Sartrouville_Rdc.json")
        new_plan = (corner_grid +
                    load_bearing_wall_grid +
                    window_grid +
                    duct_grid +
                    cleanup_grid).apply_to(plan)
        new_plan.check()
        new_plan.plot(save=False)
        plt.show()


    create_a_grid()
