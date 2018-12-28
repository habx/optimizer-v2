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

import matplotlib.pyplot as plt

import libs.reader as reader
from libs.seed import (
    Seeder,
    Filler,
    GROWTH_METHODS,
    GROWTH_METHODS_FILL,
    GROWTH_METHODS_SMALL_SPACE_FILL
)
from libs.selector import SELECTORS
from libs.grid import GRIDS
from libs.shuffle import SHUFFLES


if TYPE_CHECKING:
    from libs.plan import Plan, Space
    from libs.specification import Specification
    from libs.cpsolver.solver import Solver
    from libs.cpsolver.variables import Cell


COMPONENTS_REQUIRED = {
    "living": {"window": 2, "doorWindow": 1},
    "bedroom": {"window": 1},
    "entrance": {"frontDoor": 1},
    "kitchen": {"duct": 1, "window": 1},
    "bathroom": {"duct": 1},
    "wc": {"duct": 1},
    "dressing": {"window": -1}
}

MIN_AREA_COEFF = 0.5
MAX_AREA_COEFF = 1.5


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


def add_cells(plan: 'Plan', solver: 'Solver') -> Dict[int, 'Space']:
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
        cells_to_spaces[ix] = space

    logging.debug("SOLVER : Added cells %s", solver.cells)
    logging.debug("SOLVER : Added cells props %s", solver.cells_props)

    return cells_to_spaces


def add_values(spec: 'Specification', solver: 'Solver'):
    """
    Creates the values
    :param spec:
    :param solver
    :return:
    """

    for ix, item in enumerate(spec.items):
        solver.add_value(ix)
        solver.add_area_constraint(ix,
                                   item.min_size.area * MIN_AREA_COEFF,
                                   item.max_size.area * MAX_AREA_COEFF)
        solver.add_connectivity_constraint(ix, solver.adjacency)
        solver.add_component_constraint(ix, COMPONENTS_REQUIRED.get(item.category.name, {}))

    logging.debug("SOLVER : Added values %s", solver.values)


def create_solution(solution: ['Cell'], cells_to_spaces: Dict[int, 'Space']):
    """
    Display a solution
    :param solution:
    :param cells_to_spaces
    :return:
    """
    for cell in solution:
        space = cells_to_spaces[cell.ix]
        for other_cell in solution:
            if cell is other_cell or cell.value_ix() != other_cell.value_ix():
                continue
            other_space = cells_to_spaces[other_cell.ix]
            space.merge(other_space)


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
        input_file = 'Bussy_Regis.json'  # 5 Levallois_Letourneur / Antony_A22
        plan = reader.create_plan_from_file(input_file)

        seeder = Seeder(plan, GROWTH_METHODS)
        seeder.add_condition(SELECTORS['seed_duct'], 'duct')
        GRIDS['ortho_grid'].apply_to(plan)

        seeder.plant()
        seeder.grow()
        SHUFFLES['square_shape'].run(plan)

        plan.plot(show=True)
        plt.show()

        seed_empty_furthest_couple_middle = SELECTORS[
            'seed_empty_furthest_couple_middle_space_area_min_100000']
        seed_empty_area_max_100000 = SELECTORS['area_max=100000']
        seed_methods = [(seed_empty_furthest_couple_middle, GROWTH_METHODS_FILL, "empty"),
                        (seed_empty_area_max_100000, GROWTH_METHODS_SMALL_SPACE_FILL, "empty")]

        filler = Filler(plan, seed_methods)
        filler.apply_to(plan)
        plan.remove_null_spaces()
        fuse_selector = SELECTORS['fuse_small_cell']
        filler.fusion(fuse_selector)

        SHUFFLES['square_shape'].run(plan, show=True)

        ########################
        # create solver
        input_file = 'Bussy_Regis_setup.json'
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
        add_values(spec, my_solver)
        cells_to_spaces = add_cells(plan, my_solver)

        my_solver.solve()
        solution = my_solver.solutions[5]
        create_solution(solution, cells_to_spaces)

        plan.plot(show=True)
        plt.show()

    compose()
