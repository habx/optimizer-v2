"""
Finisher module :
Applies a genetic algorithm to improve the plan according to several constraints :
• rooms sizes
• rooms shapes
• circulation

The module is inspired by the deap library (global toolbox for genetic algorithms) and
implements a simple version of the NSGA-II algorithm:
    [Deb2002] Deb, Pratab, Agarwal, and Meyarivan, "A fast elitist
    non-dominated sorting genetic algorithm for multi-objective
    optimization: NSGA-II", 2002.
"""
from typing import TYPE_CHECKING, Optional


if TYPE_CHECKING:
    from libs.plan.plan import Plan



if __name__ == '__main__':
    """
    1. get a plan
    2. get some specifications
    3. define mutations
    4. define fitness functions
    5. run algorithm
    """

    def try_plan():
        """Tries to shuffle a plan"""
        import time
        import logging

        import libs.io.reader as reader
        import libs.io.writer as writer
        from libs.modelers.grid import GRIDS
        from libs.modelers.seed import SEEDERS
        from libs.modelers.shuffle import SHUFFLES
        from libs.space_planner.space_planner import SpacePlanner
        from libs.plan.plan import Plan
        logging.getLogger().setLevel(logging.DEBUG)

        plan_name = "001"
        specification_number = "0"
        solution_number = 1

        try:
            new_serialized_data = reader.get_plan_from_json(plan_name + "_solution_"
                                                            + str(solution_number))
            plan = Plan(plan_name).deserialize(new_serialized_data)
        except FileNotFoundError:
            plan = reader.create_plan_from_file(plan_name + ".json")
            GRIDS['optimal_grid'].apply_to(plan)
            SEEDERS["simple_seeder"].apply_to(plan)
            plan.plot()
            spec = reader.create_specification_from_file(plan_name + "_setup"
                                                         + specification_number + ".json")
            spec.plan = plan
            space_planner = SpacePlanner("test", spec)
            best_solutions = space_planner.solution_research()
            if best_solutions:
                solution = best_solutions[solution_number]
                plan = solution.plan
                plan.plot()
                writer.save_plan_as_json(plan.serialize(), plan_name + "_solution_"
                                         + str(solution_number))
            else:
                logging.info("No solution for this plan")
                return

        plan.plot()
        time.sleep(0.5)
        SHUFFLES["bedrooms_corner"].apply_to(plan, show=True)
        plan.plot()

    try_plan()
