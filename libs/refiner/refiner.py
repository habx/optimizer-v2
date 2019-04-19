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
    • create efficient all aligned edges mutation
    • check edge selector to make sure we are not eliminating needed scenarios
    • add similar function to create diversity in the hof
    • enable multiprocessing by achieving to separate the mesh from the plan... (hard)

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
        _hof = support.HallOfFame(4, lambda a, b: a.is_similar(b)) if with_hof else None
        # 1. refine mesh of the plan
        # TODO : implement this

        # 2. create plan cache for performance reason
        for floor in plan.floors.values():
            floor.mesh.compute_cache()

        # 3. run the algorithm
        initial_ind = toolbox.individual(plan)
        results = self._algorithm(toolbox, initial_ind, _hof)

        output = results if not with_hof else _hof
        toolbox.evaluate_pop(output)  # evaluate fitnesses for analysis
        return output


# Toolbox factories

def fc_nsga_toolbox(spec: 'Specification') -> 'core.Toolbox':
    """
    Returns a toolbox
    :param spec: The specification to follow
    :return:
    """
    toolbox = core.Toolbox()
    toolbox.configure("fitness", (-3.0, -1.0, -5.0))
    toolbox.configure("individual", toolbox.fitness)
    # Note : order is very important as tuples are evaluated lexicographically in python
    scores_fc = [evaluation.score_corner,
                 evaluation.score_bounding_box,
                 evaluation.fc_score_area(spec)]
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
    ngen = 100
    mu = 4 * 25  # Must be a multiple of 4 for tournament selection of NSGA-II
    cxpb = 0.5

    pop = toolbox.populate(initial_ind, mu)
    toolbox.evaluate_pop(pop)

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
        toolbox.evaluate_pop(offspring)

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

    def main():
        """ test function """
        import time
        import tools.cache

        logging.getLogger().setLevel(logging.INFO)

        spec, plan = tools.cache.get_plan("029") #052
        if plan:
            plan.name = "original"
            plan.plot()
            start = time.time()
            hof = REFINERS["simple"].run(plan, spec, True)
            sols = sorted(hof, key=lambda i: i.fitness.value, reverse=True)
            end = time.time()
            for n, ind in enumerate(sols):
                ind.name = str(n)
                ind.plot()
                print("Fitness: {} - {}".format(ind.fitness.value, ind.fitness.values))
            print("Time elapsed: {}".format(end - start))
            best = sols[0]
            item_dict = evaluation.create_item_dict(spec)
            for space in best.mutable_spaces():
                print("• Area {} : {} -> [{}, {}]".format(space.category.name,
                                                          round(space.cached_area()),
                                                          item_dict[space.id].min_size.area,
                                                          item_dict[space.id].max_size.area))


    main()
