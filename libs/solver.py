# coding=utf-8
"""
Solver module (Quick implementation of custom solver as a test)
1. create the cells
2. create the adjacency matrix
3. parametrize the solver
4. run
"""

import logging

from typing import TYPE_CHECKING, Dict

import libs.reader as reader
from libs.seed import (
    Seeder,
    GROWTH_METHODS,
    FILL_METHODS
)
from libs.selector import SELECTORS
from libs.grid import GRIDS
from libs.shuffle import SHUFFLES


if TYPE_CHECKING:
    from libs.plan import Plan
    from libs.category import SpaceCategory
    from libs.specification import Specification
    from libs.cpsolver.solver import Solver
    from libs.cpsolver.variables import Cell
    import uuid


COMPONENTS_REQUIRED = {
    "living": {"doorWindow": 1},
    "bedroom": {"window": 1},
    "entrance": {"frontDoor": 1},
    "kitchen": {"duct": 1},
    "bathroom": {"duct": 1},
    "wc": {"duct": 1},
    "dressing": {"window": -1}
}

MIN_AREA_COEFF = 0
MAX_AREA_COEFF = 3.0


def adjacency_matrix(plan: 'Plan') -> [[bool]]:
    """
    Returns an adjacency matrix
    :param plan:
    :return:
    """
    seed_spaces = list(plan.get_spaces("seed"))
    size = len(seed_spaces)
    matrix = [[0 for _ in range(size)] for _ in range(size)]
    for i, space in enumerate(seed_spaces):
        for j, other_space in enumerate(seed_spaces):
            matrix[i][j] = space.adjacent_to(other_space)

    return matrix


def add_cells(plan: 'Plan', solver: 'Solver') -> Dict[int, 'uuid.UUID']:
    """
    Returns an array of cell
    :param plan:
    :param solver:
    :return:
    """
    cells_to_spaces = {}
    for ix, space in enumerate(plan.get_spaces("seed")):
        if space.area == 0:  # only add non empty space
            continue
        props = {
            "area": space.area,
            "components": tuple(space.components_category_associated())
        }
        solver.add_cell(solver.domain(), props, ix)
        cells_to_spaces[ix] = space.id

    logging.debug("SOLVER : Added cells %s", solver.cells)
    logging.debug("SOLVER : Added cells props %s", solver.cells_props)

    return cells_to_spaces


def add_values(spec: 'Specification', solver: 'Solver') -> Dict[int, 'SpaceCategory']:
    """
    Creates the values
    :param spec:
    :param solver
    :return:
    """
    values_category = {}

    for ix, item in enumerate(spec.items):
        solver.add_value(ix)
        solver.add_area_constraint(ix,
                                   item.min_size.area * MIN_AREA_COEFF,
                                   item.max_size.area * MAX_AREA_COEFF)
        solver.add_connectivity_constraint(ix, solver.adjacency)
        solver.add_component_constraint(ix, COMPONENTS_REQUIRED.get(item.category.name, {}))
        values_category[ix] = item.category

    logging.debug("SOLVER : Added values %s", solver.values)

    return values_category


def create_solution(solution: ['Cell'],
                    cells_to_spaces: Dict[int, 'uuid.UUID'],
                    values_category: Dict[int, 'SpaceCategory'],
                    plan: 'Plan'):
    """
    Display a solution
    :param solution:
    :param cells_to_spaces
    :param values_category
    :param plan
    :return:
    """
    new_plan = plan.clone()
    for cell in solution:
        space = new_plan.get_space_from_id(cells_to_spaces[cell.ix])
        if not space:
            continue
        space.category = values_category[cell.value_ix()]
        for other_cell in solution:
            if cell.ix >= other_cell.ix or cell.value_ix() != other_cell.value_ix():
                continue
            other_space = new_plan.get_space_from_id(cells_to_spaces[other_cell.ix])
            if not other_space:
                continue
            space.merge(other_space)

    new_plan.plot()


if __name__ == '__main__':
    from libs.cpsolver.solver import Solver

    logging.getLogger().setLevel(logging.DEBUG)


    def compose():
        """
        Test
        :return:
        """
        ################
        # Create a plan
        input_file = 'Antony_A22.json'  # 5 Levallois_Letourneur / Antony_A22
        plan = reader.create_plan_from_file(input_file)
        GRIDS['finer_ortho_grid'].apply_to(plan)

        plan.plot()

        seeder = Seeder(plan, GROWTH_METHODS).add_condition(SELECTORS['seed_duct'], 'duct')
        (seeder.plant()
         .grow()
         .shuffle(SHUFFLES['seed_square_shape'])
         .fill(FILL_METHODS, (SELECTORS["farthest_couple_middle_space_area_min_100000"],
                              "empty"))
         .fill(FILL_METHODS, (SELECTORS["single_edge"], "empty"), recursive=True)
         .simplify(SELECTORS["fuse_small_cell"])
         .shuffle(SHUFFLES['seed_square_shape']))

        plan.plot()

        ########################
        # create solver
        input_file = 'Antony_A22_setup.json'
        spec = reader.create_specification_from_file(input_file)
        spec.plan = plan

        logging.debug("**************************")
        logging.debug("*                        *")
        logging.debug("*         SOLVER         *")
        logging.debug("*                        *")
        logging.debug("**************************")

        my_solver = Solver(adjacency_matrix(plan), {"num_solutions": 10,
                                                    "num_fails": 1500,
                                                    "num_restarts": 36})
        values_categories = add_values(spec, my_solver)
        cells_to_spaces = add_cells(plan, my_solver)

        my_solver.solve()

        for solution in my_solver.solutions:
            create_solution(solution, cells_to_spaces, values_categories, plan)
    compose()
