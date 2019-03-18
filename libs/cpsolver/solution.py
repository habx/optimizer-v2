# coding=utf-8
"""
Temporary Module
Try of a naive implementation of the custom solver directly on a simple mesh
1. we create a simple grid
2. we compute : • the adjacency matrix
3. we add each cell to the solver
3. We create a manual specification for the apartment
"""
from typing import List, Dict, TYPE_CHECKING
import logging
import math
import uuid

from libs.category import SPACE_CATEGORIES
import libs.reader as reader
from libs.grid import GRIDS
from libs.plan import Space
from libs.cpsolver.solver import Solver, Cell

if TYPE_CHECKING:
    from libs.plan import Plan

# the perimeter must be < sqrt(Area) * PERIMETER_RATIO
PERIMETER_RATIO = 4.8
# the max distance between to cells must be < sqrt(space.area) * MAX_SIZE_RATIO
MAX_SIZE_RATIO = 1.6

SQM = 10000


def import_test_plan(input_file: str = "grenoble_101.json") -> 'Plan':
    """
    Imports and add grid to a plan
    :return:
    """
    plan = reader.create_plan_from_file(input_file)
    new_plan = GRIDS["optimal_grid"].apply_to(plan)
    new_plan.plot()
    create_seeds(new_plan)
    return new_plan


def create_seeds(plan: 'Plan'):
    """
    Creates a seed for each empty face of the plan
    :param plan:
    :return:
    """
    empty_space = plan.empty_space
    for face in list(empty_space.faces):
        Space(plan, empty_space.floor, face.edge, SPACE_CATEGORIES["seed"])

    empty_space.remove()


def get_adjacency_matrix(plan: 'Plan') -> List[List[float]]:
    """
    Returns the adjacency matrix
    :param plan:
    :return:
    """
    seeds = list(plan.get_spaces("seed"))
    output = [[s.adjacency_to(other_s) for s in seeds] for other_s in seeds]
    return output


def get_distance_matrix(plan: 'Plan') -> List[List[float]]:
    """
    Returns the distance matrix
    :param plan:
    :return:
    """
    seeds = list(plan.get_spaces("seed"))
    output = [[s.distance_to(other_s) for s in seeds] for other_s in seeds]
    return output


def add_values(spec: List[Dict], solver: 'Solver') -> Dict[int, str]:
    """
    Creates the values
    :param spec:
    :param solver
    :return:
    """
    values_category = {}

    for ix, item in enumerate(spec):
        solver.add_value(ix)
        solver.add_component_constraint(ix, item["components"])
        solver.add_max_size_constraint(ix, math.sqrt(item["max_area"]) * MAX_SIZE_RATIO)
        solver.add_area_constraint(ix, item["min_area"], item["max_area"])
        solver.add_connectivity_constraint(ix, solver.adjacency)
        solver.add_max_perimeter_constraint(ix, math.sqrt(item["max_area"]) * PERIMETER_RATIO)

        values_category[ix] = item["category_name"]

    logging.debug("SOLVER : Added values %s", solver.values)

    return values_category


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
            "perimeter": space.perimeter,
            "components": tuple(space.components_category_associated())
        }
        solver.add_cell(solver.domain(), props, ix)
        cells_to_spaces[ix] = space.id

    logging.debug("SOLVER : Added cells %s", solver.cells)
    logging.debug("SOLVER : Added cells props %s", solver.cells_props)

    return cells_to_spaces


def create_solution(solution: ['Cell'],
                    cells_to_spaces: Dict[int, 'uuid.UUID'],
                    values_category: Dict[int, str],
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
        space.category = SPACE_CATEGORIES[values_category[cell.value_ix()]]
        for other_cell in solution:
            if cell.ix >= other_cell.ix or cell.value_ix() != other_cell.value_ix():
                continue
            other_space = new_plan.get_space_from_id(cells_to_spaces[other_cell.ix])
            if not other_space:
                continue
            space.merge(other_space)

    new_plan.plot()


def exec_solver():
    """
    Parametrize the solver
    :return:
    """
    params = {
        "num_solutions": 10,
        "num_fails": 300000,
        "num_restarts": 0
    }
    spec = [
        {
            "category_name": "entrance",
            "min_area": 6 * SQM,
            "max_area": 8 * SQM,
            "components": {
                "frontDoor": 1,
            }
        },
        {
            "category_name": "living",
            "min_area": 25 * SQM,
            "max_area": 30 * SQM,
            "components": {
                ("window", "doorWindow"): 1,
                "duct": 1
            }
        },
        {
            "category_name": "bedroom",
            "min_area": 14 * SQM,
            "max_area": 16 * SQM,
            "components": {
                ("window", "doorWindow"): 1
            }
        },
        {
            "category_name": "bedroom",
            "min_area": 10 * SQM,
            "max_area": 12 * SQM,
            "components": {
                ("window", "doorWindow"): 1
            }
        },
        {
            "category_name": "wc",
            "min_area": 1.5 * SQM,
            "max_area": 3 * SQM,
            "components": {
                "duct": 1,
                "window": -1,
                "doorWindow": -1,
            }
        },
        {
            "category_name": "bathroom",
            "min_area": 4 * SQM,
            "max_area": 7 * SQM,
            "components": {
                "duct": 1,
            }
        },
    ]

    plan = import_test_plan()
    solver = Solver(get_adjacency_matrix(plan), get_distance_matrix(plan), params)
    values_categories = add_values(spec, solver)
    cells_to_spaces = add_cells(plan, solver)

    solver.solve()

    for solution in solver.solutions:
        create_solution(solution, cells_to_spaces, values_categories, plan)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)

    exec_solver()
