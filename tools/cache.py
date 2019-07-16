"""
Cache Module
Get a previously cached plan or compute it
"""
from typing import Optional
import logging

import libs.io.writer as writer
from libs.modelers.grid import GRIDS
from libs.modelers.seed import SEEDERS
from libs.space_planner.space_planner import SPACE_PLANNERS
import libs.io.reader as reader
from libs.modelers.corridor import Corridor, CORRIDOR_BUILDING_RULES
from libs.space_planner.solution import Solution


def get_solution(plan_name: str = "001",
                 spec_name: str = "0",
                 solution_number: int = 0,
                 grid: str = "optimal_grid",
                 seeder: str = "simple_seeder",
                 do_circulation: bool = False,
                 max_nb_solutions: int = 3) -> 'Solution':
    """
    Returns a specification and the corresponding solution plan
    :param plan_name: The name of the file of the plan blueprint source
    :param spec_name: The number of the setup of the corresponding plan
    :param solution_number: The solution number (note if the solution number is higher than the
    total number of solutions found, it returns the last solution)
    :param grid: the name of the grid to use
    :param seeder: the name of the seeder to use
    :param do_circulation: whether to create a circulation in the plan
    :param max_nb_solutions
    """

    spec_file_name = plan_name + "_setup" + spec_name + ".json"
    solution_file_name = (plan_name + "_solution_" + str(solution_number)
                          + "_" + grid + "_" + seeder +
                          ("_circulation" if do_circulation else "") + ".json")
    try:
        return _retrieve_solution_from_cache(solution_file_name)

    except FileNotFoundError:

        return _compute_solution_from_start(plan_name, spec_file_name,
                                            solution_number, grid, seeder, do_circulation,
                                            max_nb_solutions)


def _retrieve_solution_from_cache(solution_file_name: str) -> 'Solution':
    """
    Retrieves a specification and a plan instances from cached serialized data
    :param solution_file_name:
    :return:
    """
    new_serialized_data = reader.get_plan_from_json(solution_file_name)
    solution = Solution.deserialize(new_serialized_data)
    return solution


def _compute_solution_from_start(plan_name: str,
                                 spec_file_name: str,
                                 solution_number: int,
                                 grid: str,
                                 seeder: str,
                                 do_circulation: bool,
                                 max_nb_solutions: int) -> Optional['Solution']:
    """
    Computes the plan and the spec file directly from the input json
    :param plan_name:
    :param spec_file_name:
    :param solution_number:
    :param grid:
    :param seeder:
    :param do_circulation:
    :return:
    """
    corridor_building_rule = CORRIDOR_BUILDING_RULES["no_cut"]

    solution_file_name = (plan_name + "_solution_" + str(solution_number)
                          + "_" + grid + "_" + seeder +
                          ("_circulation" if do_circulation else "") + ".json")

    plan = reader.create_plan_from_file(plan_name + ".json")
    spec = reader.create_specification_from_file(spec_file_name)

    GRIDS[grid].apply_to(plan)
    SEEDERS[seeder].apply_to(plan)
    spec.plan = plan
    space_planner = SPACE_PLANNERS["standard_space_planner"]
    best_solutions = space_planner.apply_to(spec, max_nb_solutions)

    if best_solutions:
        # make sure the solution number is correct
        num_solutions = len(best_solutions)
        if solution_number >= num_solutions:
            logging.info("Cache: Get Plan : no solution for the specified number, "
                         "retrieving solution n°%i", num_solutions - 1)
            solution_number = num_solutions - 1
        solution = best_solutions[solution_number]
        if do_circulation:
            Corridor(corridor_rules=corridor_building_rule["corridor_rules"],
                     growth_method=corridor_building_rule["growth_method"]).apply_to(solution)
        writer.save_plan_as_json(solution.serialize(), solution_file_name)
        return solution
    else:
        logging.info("No solution for this plan")
        return None


__all__ = ('get_solution',)
