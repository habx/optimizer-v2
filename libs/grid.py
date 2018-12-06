# coding=utf-8
"""
Grid module
"""
from typing import Tuple, Sequence, Union, TYPE_CHECKING
import logging
import matplotlib.pyplot as plt

from libs import reader
from libs.utils.catalog import Catalog
from libs.mutation import MUTATIONS
from libs.selector import SELECTORS

if TYPE_CHECKING:
    from libs.mutation import Mutation
    from libs.selector import Selector
    from libs.plan import Plan, Space
    from libs.mesh import Face

GRIDS = Catalog('grids')


class Grid:
    """
    Creates a grid inside a plan.
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
        for operator in self.operators:
            self.iterate(plan, operator)
            # we simplify the mesh between each operator
            plan.mesh.simplify()

        return plan

    def iterate(self, plan: 'Plan', operator: Tuple['Selector', 'Mutation']):
        """
        Apply operation to each face of the empty spaces of the plan
        TODO: this is brutal as we try each face each time a cut happens
        a better way could be found !!
        :param plan:
        :param operator:
        :return:
        """
        for empty_space in plan.empty_spaces:
            for face in empty_space.faces:
                mesh_has_changed = self.select_and_slice(face, operator)
                if mesh_has_changed:
                    return self.iterate(plan, operator)
        return

    @staticmethod
    def select_and_slice(face: Union['Space', 'Face'],
                         operator: Tuple['Selector', 'Mutation']) -> bool:
        """
        Selects the correct edges and applies the slice transformation to them
        :param face:
        :param operator:
        :return:
        """
        _selector, _slicer = operator
        for edge in _selector.yield_from(face):
            mesh_has_changed = _slicer.apply_to(edge)
            if mesh_has_changed:
                return True
        return False

# grid


sequence_grid = Grid('sequence_grid', [
    (
        SELECTORS['previous_angle_salient_non_ortho'],
        MUTATIONS['ortho_projection_cut']
    ),
    (
        SELECTORS['next_angle_salient_non_ortho'],
        MUTATIONS['ortho_projection_cut']
    ),
    (
        SELECTORS['previous_angle_convex_non_ortho'],
        MUTATIONS['ortho_projection_cut']
    ),
    (
        SELECTORS['next_angle_convex_non_ortho'],
        MUTATIONS['ortho_projection_cut']
    ),
    (
       SELECTORS['previous_angle_salient_ortho'],
       MUTATIONS['ortho_projection_cut']
    ),
    (
        SELECTORS['close_to_window'],
        MUTATIONS['remove_edge']
    ),
    (
        SELECTORS['between_windows'],
        MUTATIONS.factory['barycenter_cut'](0.5)
    ),
    (
        SELECTORS['between_edges_between_windows'],
        MUTATIONS.factory['barycenter_cut'](0.5)
    ),
    (
        SELECTORS['aligned_edges'],
        MUTATIONS.factory['barycenter_cut'](1.0)
    ),
    (
        SELECTORS['edge_min_150'],
        MUTATIONS.factory['barycenter_cut'](0.5)
    ),
    (
        SELECTORS['close_to_window'],
        MUTATIONS['remove_edge']

    )
])

GRIDS.add(sequence_grid)


ortho_grid = Grid('ortho_grid', [
    (
        SELECTORS['previous_angle_salient_non_ortho'],
        MUTATIONS['ortho_projection_cut']
    ),
    (
        SELECTORS['next_angle_salient_non_ortho'],
        MUTATIONS['ortho_projection_cut']
    ),
    (
        SELECTORS['previous_angle_convex_non_ortho'],
        MUTATIONS['ortho_projection_cut']
    ),
    (
        SELECTORS['next_angle_convex_non_ortho'],
        MUTATIONS['ortho_projection_cut']
    ),
    (
       SELECTORS['previous_angle_salient_ortho'],
       MUTATIONS.factory['barycenter_cut'](0)
    ),
    (
        SELECTORS['next_angle_salient_ortho'],
        MUTATIONS.factory['barycenter_cut'](1.0)
    ),
    (
        SELECTORS['edge_min_500'],
        MUTATIONS.factory['barycenter_cut'](0.5)
    ),
    (
        SELECTORS['between_windows'],
        MUTATIONS.factory['barycenter_cut'](0.5)
    ),
    (
        SELECTORS['between_edges_between_windows'],
        MUTATIONS.factory['barycenter_cut'](0.5)
    ),
    (
        SELECTORS['edge_min_300'],
        MUTATIONS.factory['barycenter_cut'](0.5)
    ),
    (
        SELECTORS['aligned_edges'],
        MUTATIONS.factory['barycenter_cut'](1.0)
    )
])

GRIDS.add(ortho_grid)

if __name__ == '__main__':

    logging.getLogger().setLevel(logging.INFO)
    from libs import reader_test

    def create_a_grid():
        """
        Test
        :return:
        """
        input_file = reader.BLUEPRINT_INPUT_FILES[5]
        # 16: Edison_10 6: Antony_A22 9: Antony_B22
        #
        plan = reader.create_plan_from_file(input_file)

        new_plan = ortho_grid.apply_to(plan)
        new_plan.check()

        new_plan.plot(save=False)
        plt.show()

    create_a_grid()
