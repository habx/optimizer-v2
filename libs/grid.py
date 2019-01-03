# coding=utf-8
"""
Grid module
"""

from typing import Tuple, Sequence, TYPE_CHECKING
import logging
import matplotlib.pyplot as plt

from libs import reader
from libs.mutation import MUTATIONS, MUTATION_FACTORIES
from libs.selector import SELECTORS

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

    def apply_to(self, plan: 'Plan') -> 'Plan':
        """
        Returns the modified plan with the created grid
        :param plan:
        :return: the plan with the created grid
        """
        logging.debug("Applying Grid %s to plan %s", self.name, plan)

        for operator in self.operators:
            self.apply_operator(plan, operator)
            # we simplify the mesh between each operator
            # plan.mesh.simplify()

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
            logging.debug("Applying mutation %s to edge %s of space %s", _mutation, edge, space)
            mesh_has_changed = _mutation.apply_to(edge, [space])
            if mesh_has_changed:
                return True
        return False


# grid


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

GRIDS = {
    "ortho_grid": ortho_grid,
    "sequence_grid": sequence_grid
}

if __name__ == '__main__':

    logging.getLogger().setLevel(logging.DEBUG)

    def create_a_grid():
        """
        Test
        :return:
        """
        plan = reader.create_plan_from_file("Antony_A22.json")
        new_plan = ortho_grid.apply_to(plan)
        new_plan.check()

        new_plan.plot(save=False)
        plt.show()


    create_a_grid()
