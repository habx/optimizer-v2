"""
Cache Module
Get a previously cached plan or compute it
"""
from typing import TYPE_CHECKING, Tuple, Optional

if TYPE_CHECKING:
    from libs.plan.plan import Plan
    from libs.specification.specification import Specification


def get_plan(plan_name: str = "001",
             spec_name: str = "0",
             solution_number: int = 0) -> Tuple['Specification', Optional['Plan']]:
    """
    Returns a specification and the corresponding solution plan
    :param plan_name: The name of the file of the plan blueprint source
    :param spec_name: The number of the setup of the corresponding plan
    :param solution_number: The solution number (note if the solution number is higher than the
    total number of solutions found, it returns the last solution)
    """
    import logging

    import libs.io.reader as reader
    import libs.io.writer as writer
    from libs.modelers.grid import GRIDS
    from libs.modelers.seed import SEEDERS
    from libs.space_planner.space_planner import SpacePlanner
    from libs.plan.plan import Plan
    from libs.io.reader import DEFAULT_PLANS_OUTPUT_FOLDER

    spec_file_name = plan_name + "_setup" + spec_name + ".json"
    plan_file_name = plan_name + "_solution_" + str(solution_number) + ".json"
    folder = DEFAULT_PLANS_OUTPUT_FOLDER

    try:
        new_serialized_data = reader.get_plan_from_json(plan_file_name)
        plan = Plan(plan_name).deserialize(new_serialized_data)
        spec_dict = reader.get_json_from_file(spec_file_name, DEFAULT_PLANS_OUTPUT_FOLDER)
        spec = reader.create_specification_from_data(spec_dict, "new")
        spec.plan = plan
        return spec, plan

    except FileNotFoundError:
        plan = reader.create_plan_from_file(plan_name + ".json")
        spec = reader.create_specification_from_file(spec_file_name)

        GRIDS['optimal_grid'].apply_to(plan)
        SEEDERS["simple_seeder"].apply_to(plan)
        spec.plan = plan
        space_planner = SpacePlanner("test", spec)
        best_solutions = space_planner.solution_research()
        new_spec = space_planner.spec

        if best_solutions:
            # make sure the solution number is correct
            num_solutions = len(best_solutions)
            if solution_number >= num_solutions:
                logging.info("Cache: Get Plan : no solution for the specified number, "
                             "retrieving solution n°%i", num_solutions)
                solution_number = num_solutions - 1
            solution = best_solutions[solution_number]
            plan = solution.plan
            new_spec.plan = plan
            writer.save_plan_as_json(plan.serialize(), plan_file_name)
            writer.save_as_json(new_spec.serialize(), folder, spec_file_name)
            return new_spec, plan
        else:
            logging.info("No solution for this plan")
            return spec, None
