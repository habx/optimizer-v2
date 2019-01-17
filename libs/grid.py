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
    from libs.plan import Plan, Space


class Grid:
    """
    Creates a grid inside a plan.
    TODO : for clarity purpose we should probably replace the operator concept by an Action instance
    """

    def __init__(self, name: str, operators: Sequence[Tuple['Selector', 'Mutation']]):
        self.name = name
        self.operators = operators or []

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
            self.apply_operator(plan, operator)
            # we simplify the mesh between each operator
            plan.mesh.simplify()
            plan.update_from_mesh()

        return plan

    def apply_operator(self, plan: 'Plan', operator: Tuple['Selector', 'Mutation']):
        """
        Apply operation to the empty spaces of the plan
        :param plan:
        :param operator:
        :return:
        """
        for empty_space in plan.empty_spaces:
            mesh_has_changed = self.select_and_slice(empty_space, operator)
            if mesh_has_changed:
                return self.apply_operator(plan, operator)
        return

    @staticmethod
    def select_and_slice(space: 'Space',
                         operator: Tuple['Selector', 'Mutation']) -> bool:
        """
        Selects the correct edges and applies the slice transformation to them
        :param space:
        :param operator:
        :return:
        """
        _selector, _mutation = operator
        for edge in _selector.yield_from(space):
            logging.debug("Grid: Applying cut %s to edge %s of space %s", _mutation, edge, space)
            mesh_has_changed = _mutation.apply_to(edge, [space])
            if mesh_has_changed:
                return True
        return False

    def extend(self, name: str = "", *operators: Tuple['Selector', 'Mutation']) -> 'Grid':
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


# grid
simple_grid = Grid("simple_grid", [
    (
        SELECTOR_FACTORIES["edges_length"]([], [[100]]),
        MUTATION_FACTORIES["barycenter_cut"](0.5)
    )
])

sequence_grid = Grid('sequence_grid', [
    (
        SELECTORS["previous_angle_salient_non_ortho"],
        MUTATIONS['ortho_projection_cut']
    ),
    (
        SELECTORS["next_angle_salient_non_ortho"],
        MUTATIONS['ortho_projection_cut']
    ),
    (
        SELECTORS["previous_angle_convex_non_ortho"],
        MUTATIONS['ortho_projection_cut']
    ),
    (
        SELECTORS["next_angle_convex_non_ortho"],
        MUTATIONS['ortho_projection_cut']
    ),
    (
        SELECTORS["previous_angle_salient_ortho"],
        MUTATIONS['ortho_projection_cut']
    ),
    (
        SELECTORS["close_to_window"],
        MUTATIONS['remove_edge']
    ),
    (
        SELECTORS["between_windows"],
        MUTATION_FACTORIES['barycenter_cut'](0.5)
    ),
    (
        SELECTORS["between_edges_between_windows"],
        MUTATION_FACTORIES['barycenter_cut'](0.5)
    ),
    (
        SELECTORS["aligned_edges"],
        MUTATION_FACTORIES['barycenter_cut'](1.0)
    ),
    (
        SELECTORS["edge_min_150"],
        MUTATION_FACTORIES['barycenter_cut'](0.5)
    ),
    (
        SELECTORS["close_to_window"],
        MUTATIONS['remove_edge']

    )
])

ortho_grid = Grid('ortho_grid', [
    (
        SELECTORS["previous_angle_salient_non_ortho"],
        MUTATIONS['ortho_projection_cut']
    ),
    (
        SELECTORS["next_angle_salient_non_ortho"],
        MUTATIONS['ortho_projection_cut']
    ),
    (
        SELECTORS["previous_angle_convex_non_ortho"],
        MUTATIONS['ortho_projection_cut']
    ),
    (
        SELECTORS["next_angle_convex_non_ortho"],
        MUTATIONS['ortho_projection_cut']
    ),
    (
        SELECTORS["previous_angle_salient_ortho"],
        MUTATION_FACTORIES['barycenter_cut'](0)
    ),
    (
        SELECTORS["next_angle_salient_ortho"],
        MUTATION_FACTORIES['barycenter_cut'](1.0)
    ),
    (
        SELECTORS["edge_min_500"],
        MUTATION_FACTORIES['barycenter_cut'](0.5)
    ),
    (
        SELECTORS["between_windows"],
        MUTATION_FACTORIES['barycenter_cut'](0.5)
    ),
    (
        SELECTORS["between_edges_between_windows"],
        MUTATION_FACTORIES['barycenter_cut'](0.5)
    ),
    (
        SELECTORS["edge_min_300"],
        MUTATION_FACTORIES['barycenter_cut'](0.5)
    ),
    (
        SELECTORS["aligned_edges"],
        MUTATION_FACTORIES['barycenter_cut'](1.0)
    )
])

finer_ortho_grid = ortho_grid.extend("finer_ortho_grid",
                                     (SELECTORS["edge_min_150"],
                                      MUTATION_FACTORIES['barycenter_cut'](0.5)))

GRIDS = {
    "ortho_grid": ortho_grid,
    "sequence_grid": sequence_grid,
    "simple_grid": simple_grid,
    "finer_ortho_grid": finer_ortho_grid
}

if __name__ == '__main__':

    logging.getLogger().setLevel(logging.DEBUG)

    def create_a_grid():
        """
        Test
        :return:
        """
        plan = reader.create_plan_from_file("Massy_C303.json")
        new_plan = finer_ortho_grid.apply_to(plan)
        new_plan.check()

        new_plan.plot(save=False)
        plt.show()


    create_a_grid()
