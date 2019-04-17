# coding=utf-8
"""
Finisher module :
Applies a genetic algorithm to improve the plan according to several constraints :
• rooms sizes
• rooms shapes
• circulation [TO DO]

The module is inspired by the DEAP library (global toolbox for genetic algorithms :
https://github.com/deap/deap)

It implements a simple version of the NSGA-II algorithm:

    [Deb2002] Deb, Pratab, Agarwal, and Meyarivan, "A fast elitist
    non-dominated sorting genetic algorithm for multi-objective
    optimization: NSGA-II", 2002.

TODO LIST:
    • refine grid prior to genetic search
    • guide mutation to refine choices and speed-up search
    • forbid the swap of a face close to a needed component (eg. duct for a bathroom)

"""
import random
import logging

from typing import TYPE_CHECKING, Optional, Tuple, Callable, List, Union
from libs.plan.plan import Plan

from libs.refiner import core, crossover, evaluation, mutation, nsga, population, support

if TYPE_CHECKING:
    from libs.specification.specification import Specification

# The type of an algorithm function
algorithmFunc = Callable[['core.Toolbox', Plan, Optional['support.HallOfFame']],
                         List['core.Individual']]


class Refiner:
    """
    Refiner Class.
    A refiner will try to improve the plan using a genetic algorithm.
    The refiner is composed of a :
    • Toolbox factory that will create the toolbox
    containing the main types and operators needed for the algorithm
    • the algorithm function that will be applied to the plan
    """
    def __init__(self,
                 fc_toolbox: Callable[['Specification'], 'core.Toolbox'],
                 algorithm: algorithmFunc):
        self._toolbox_factory = fc_toolbox
        self._algorithm = algorithm

    def apply_to(self, plan: 'Plan', spec: 'Specification') -> 'Plan':
        """
        Applies the refiner to the plan and returns the result.
        :param plan:
        :param spec:
        :return:
        """
        results = self.run(plan, spec)
        return max(results, key=lambda i: i.fitness)

    def run(self,
            plan: 'Plan',
            spec: 'Specification',
            with_hof: bool = False) -> Union[List['core.Individual'], 'support.HallOfFame']:
        """
        Runs the algorithm and returns the results
        :param plan:
        :param spec:
        :param with_hof: whether to return the results or a hall of fame
        :return:
        """
        toolbox = self._toolbox_factory(spec)
        _hof = support.HallOfFame(3) if with_hof else None
        # 1. refine mesh of the plan
        # TODO : implement this

        # 2. create plan cache for performance reason
        for floor in plan.floors.values():
            floor.mesh.compute_cache()

        # 3. run the algorithm
        initial_ind = toolbox.individual(plan)
        results = self._algorithm(toolbox, initial_ind, _hof)
        return results if not with_hof else _hof


# Toolbox factories

def fc_nsga_toolbox(spec: 'Specification') -> 'core.Toolbox':
    """
    Returns a toolbox
    :param spec: The specification to follow
    :return:
    """
    toolbox = core.Toolbox()
    toolbox.configure("fitness", (-1.0, -10.0, -8.0))
    toolbox.configure("individual", toolbox.fitness)
    scores_fc = [evaluation.fc_score_area(spec),
                 evaluation.score_corner,
                 evaluation.score_bounding_box]
    toolbox.register("evaluate", evaluation.compose(scores_fc))
    toolbox.register("mutate", mutation.mutate_simple)
    toolbox.register("mate", crossover.connected_differences)
    toolbox.register("select", nsga.select_nsga)
    toolbox.register("populate", population.fc_mutate(toolbox.mutate))

    return toolbox


# Algorithm functions

def simple_ga(toolbox: 'core.Toolbox',
              initial_ind: 'core.Individual',
              hof: Optional['support.HallOfFame']) -> List['core.Individual']:
    """
    A simple implementation of a genetic algorithm.
    :param toolbox: a refiner toolbox
    :param initial_ind: an initial individual
    :param hof: an optional hall of fame to store best individuals
    :return: the best plan
    """
    # algorithm parameters
    ngen = 20
    mu = 4 * 20  # Must be a multiple of 4 for tournament selection of NSGA-II
    cxpb = 0.8

    pop = toolbox.populate(initial_ind, mu)
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # This is just to assign the crowding distance to the individuals
    # no actual selection is done
    pop = toolbox.select(pop, len(pop))

    # Begin the generational process
    for gen in range(1, ngen):
        logging.info("Refiner: generation %i : %f prct", gen, gen / ngen * 100.0)
        # Vary the population
        offspring = nsga.select_tournament_dcd(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]

        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            if random.random() <= cxpb:
                toolbox.mate(ind1, ind2)
            toolbox.mutate(ind1)
            toolbox.mutate(ind2)
            ind1.fitness.clear()
            ind2.fitness.clear()

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Select the next generation population
        pop = toolbox.select(pop + offspring, mu)

        # store best individuals in hof
        if hof is not None:
            hof.update(pop)

    return pop


REFINERS = {
    "simple": Refiner(fc_nsga_toolbox, simple_ga)
}

if __name__ == '__main__':
    """
    1. get a plan
    2. get some specifications
    3. run algorithm
    """
    from libs.modelers.shuffle import SHUFFLES

    def get_plan(plan_name: str = "001",
                 spec_name: str = "0",
                 solution_number: int = 0) -> Tuple['Specification', Optional['Plan']]:
        """
        Returns a solution plan
        """
        import logging

        import libs.io.reader as reader
        import libs.io.writer as writer
        from libs.modelers.grid import GRIDS
        from libs.modelers.seed import SEEDERS
        from libs.space_planner.space_planner import SpacePlanner
        from libs.plan.plan import Plan
        from libs.io.reader import DEFAULT_PLANS_OUTPUT_FOLDER
        # logging.getLogger().setLevel(logging.DEBUG)

        spec_file_name = plan_name + "_setup" + spec_name + ".json"
        plan_file_name = plan_name + "_solution_" + str(solution_number)
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
            GRIDS['optimal_grid'].apply_to(plan)
            SEEDERS["simple_seeder"].apply_to(plan)
            spec = reader.create_specification_from_file(plan_name + "_setup"
                                                         + spec_name + ".json")
            spec.plan = plan
            space_planner = SpacePlanner("test", spec)
            best_solutions = space_planner.solution_research()
            new_spec = space_planner.spec

            if best_solutions:
                solution = best_solutions[solution_number]
                plan = solution.plan
                new_spec.plan = plan
                writer.save_plan_as_json(plan.serialize(), plan_file_name)
                writer.save_as_json(new_spec.serialize(), folder, spec_file_name)
                return new_spec, plan
            else:
                logging.info("No solution for this plan")
                return spec, None

    def main():
        """ test function """
        logging.getLogger().setLevel(logging.INFO)

        spec, plan = get_plan("008", "0", 0)
        if plan:
            plan.plot()
            SHUFFLES["bedrooms_corner"].apply_to(plan)
            hof = REFINERS["simple"].run(plan, spec, True)
            for i in hof:
                i.plot()

    main()
