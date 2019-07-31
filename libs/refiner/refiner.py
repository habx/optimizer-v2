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

TODO :
• merge circulation with livingRoom or entrance when only adjacent to livingRoom and entrance
• recompute spec areas to take into account corridor space ?
• profile time and add cache for edge lengths and edge angles
• make a bunch of tests

"""
import random
import math
import logging
import multiprocessing
from typing import TYPE_CHECKING, Optional, Callable, List, Union, Tuple

from libs.plan.plan import Plan
from libs.refiner import (
    core,
    crossover,
    evaluation,
    mutation,
    nsga,
    space_nsga,
    population,
    support,
    selection
)


if TYPE_CHECKING:
    from libs.refiner.core import Individual
    from libs.plan.plan import Plan
    from libs.space_planner.solution import Solution

# The type of an algorithm function
algorithmFunc = Callable[['core.Toolbox', Plan, dict, Optional['support.HallOfFame']],
                         List['core.Individual']]

# setting a seed for debugging
random.seed(0)


def merge_adjacent_circulation(ind: 'Individual') -> None:
    """
    Merges two adjacent corridors
    :param ind:
    :return:
    """
    try_again = True
    adjacency_length = 20.0

    while try_again:
        try_again = False
        circulations = list(ind.get_spaces("circulation"))
        for circulation in circulations:
            for other_circulation in circulations:
                if circulation.adjacent_to(other_circulation, adjacency_length):
                    circulation.merge(other_circulation)
                    try_again = True
                    break
            if try_again:
                break


def merge_circulation_living(ind: 'Individual') -> None:
    """
    Merges a circulation space with the living or the livingKitchen if
    their adjacency length is superior to a maximum length
    :return:
    """
    max_length = 130.
    adjacency_ratio = 0.25

    circulations = list(ind.get_spaces("circulation"))
    livings = list(ind.get_spaces("living", "livingKitchen"))

    for circulation in circulations:
        merged = False
        circulation_perimeter = circulation.perimeter * adjacency_ratio
        for living in livings:
            if circulation.adjacency_to(living) >= min(max_length, circulation_perimeter):
                living.merge(circulation)
                merged = True
                break
        if merged:
            break


def merge_circulation_entrance(ind: 'Individual') -> None:
    """
    Merges a circulation space with the entrance if
    their adjacency length is superior a certain ratio of the circulation perimeter or
    superior to a minimum length
    :return:
    """
    max_length = 130.
    adjacency_ratio = 0.25

    circulations = list(ind.get_spaces("circulation"))
    entrances = list(ind.get_spaces("entrance"))

    for circulation in circulations:
        circulation_perimeter = circulation.perimeter * adjacency_ratio
        merged = False
        for entrance in entrances:
            if circulation.adjacency_to(entrance) >= min(max_length, circulation_perimeter):
                entrance.merge(circulation)
                merged = True
                break
        if merged:
            break


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
                 fc_toolbox: Callable[['Solution', dict], 'core.Toolbox'],
                 algorithm: algorithmFunc):
        self._toolbox_factory = fc_toolbox
        self._algorithm = algorithm

    def apply_to(self,
                 solution: 'Solution',
                 params: dict) -> 'Solution':
        """
        Applies the refiner to the plan and returns the result.
        :param solution:
        :param params: the parameters of the genetic algorithm (ex. cxpb, mupb etc.)
        :return:
        """
        results = self.run(solution, params)
        output = max(results, key=lambda i: i.fitness.wvalue)

        # clean unnecessary circulation
        output.plot()
        merge_circulation_living(output)
        merge_circulation_entrance(output)
        merge_adjacent_circulation(output)
        solution.spec.plan = output
        solution.space_item = {output.get_space_from_id(i): item
                               for i, item in output.fitness.cache["space_to_item"].items()
                               if output.get_space_from_id(i)}
        return solution

    def run(self,
            solution: 'Solution',
            params: dict) -> Union[List['core.Individual'], 'support.HallOfFame']:
        """
        Runs the algorithm and returns the results
        :param solution:
        :param params:
        :return:
        """

        processes = params.get("processes", 1)
        hof = params.get("hof", 0)
        _hof = support.HallOfFame(hof, lambda a, b: a.is_similar(b)) if hof > 0 else None
        chunk_size = math.ceil(params["mu"]/processes)

        # 1. create plan cache for performance reason
        for floor in solution.spec.plan.floors.values():
            floor.mesh.compute_cache()

        solution.spec.plan.store_meshes_globally()
        # needed for multiprocessing (must be donne after the caching)
        toolbox = self._toolbox_factory(solution, params)

        # NOTE : the pool must be created after the toolbox in order to
        # pass the global objects created when configuring the toolbox
        # to the forked processes
        pool = None
        if processes > 1:
            pool = multiprocessing.Pool(processes)
            map_func = pool.imap
        else:
            def map_func(f, it, _):
                """ simple map function"""
                return map(f, it)

        toolbox.register("map", map_func)

        # 2. run the algorithm
        initial_ind = toolbox.individual(solution.spec.plan)
        results = self._algorithm(toolbox, initial_ind, params, _hof)

        output = results if hof == 0 else _hof
        toolbox.evaluate_pop(toolbox.map, toolbox.evaluate, output, chunk_size)

        # close the pool
        if pool:
            pool.close()
            pool.join()

        return output


# Toolbox factories

# Algorithm functions
def mate_and_mutate(mate_func,
                    mutate_func,
                    params: dict,
                    couple: Tuple['Individual', 'Individual']) -> Tuple['Individual', 'Individual']:
    """
    Specific function for nsga algorithm
    :param mate_func:
    :param mutate_func:
    :param params: a dict containing the arguments of the function
    :param couple:
    :return:
    """
    cxpb = params["cxpb"]
    _ind1, _ind2 = couple
    new_ind_1, new_ind_2 = couple
    if random.random() <= cxpb:
        new_ind_1, new_ind_2 = mate_func(_ind1, _ind2)

    if new_ind_1 is _ind1:
        mutate_func(new_ind_1)

    if new_ind_2 is _ind2:
        mutate_func(new_ind_2)

    return new_ind_1, new_ind_2


def fc_nsga_toolbox(solution: 'Solution', params: dict) -> 'core.Toolbox':
    """
    Returns a toolbox
    :param solution:
    :param params: The params of the algorithm
    :return: a configured toolbox
    """
    weights = (-20.0, -1.0, -50.0, -1.0, -50000.0,)
    # a tuple containing the weights of the fitness
    cxpb = params["cxpb"]  # the probability to mate a given couple of individuals

    toolbox = core.Toolbox()
    toolbox.configure("fitness", "CustomFitness", weights)
    toolbox.fitness.cache["space_to_item"] = evaluation.create_item_dict(solution)
    toolbox.configure("individual", "customIndividual", toolbox.fitness)
    # Note : order is very important as tuples are evaluated lexicographically in python
    scores_fc = [
        evaluation.score_corner,
        evaluation.score_area,
        evaluation.score_perimeter_area_ratio,
        evaluation.score_bounding_box,
        evaluation.score_connectivity,
        # evaluation.score_circulation_width
    ]
    toolbox.register("evaluate", evaluation.compose, scores_fc, solution.spec)

    mutations = ((mutation.add_face, {mutation.Case.DEFAULT: 0.1,
                                      mutation.Case.SMALL: 0.3,
                                      mutation.Case.BIG: 0.1}),
                 (mutation.remove_face, {mutation.Case.DEFAULT: 0.1,
                                         mutation.Case.SMALL: 0.1,
                                         mutation.Case.BIG: 0.3}),
                 (mutation.add_aligned_faces, {mutation.Case.DEFAULT: 0.4,
                                               mutation.Case.SMALL: 0.5,
                                               mutation.Case.BIG: 0.1}),
                 (mutation.remove_aligned_faces, {mutation.Case.DEFAULT: 0.4,
                                                  mutation.Case.SMALL: 0.1,
                                                  mutation.Case.BIG: 0.5}))

    toolbox.register("mutate", mutation.composite, mutations)
    toolbox.register("mate", crossover.best_spaces)
    toolbox.register("mate_and_mutate", mate_and_mutate, toolbox.mate, toolbox.mutate,
                     {"cxpb": cxpb})
    toolbox.register("select", nsga.select_nsga)
    toolbox.register("populate", population.fc_mutate(toolbox.mutate))

    return toolbox


def fc_space_nsga_toolbox(solution: 'Solution', params: dict) -> 'core.Toolbox':
    """
    Returns a toolbox for the space nsga algorithm
    :param solution:
    :param params: The params of the algorithm
    :return: a configured toolbox
    """
    scores = [
        (evaluation.score_corner, -20.0),
        (evaluation.score_area, -8.0),
        # (evaluation.score_shape, -1.0),
        (evaluation.score_width_depth_ratio, -500.0),
        (evaluation.score_bounding_box, -1.0),
        (evaluation.score_connectivity, -50000.0),
        (evaluation.score_window_area_ratio, -10.)
    ]
    scores_fc, weights = list(zip(*scores))
    # a tuple containing the weights of the fitness
    cxpb = params["cxpb"]  # the probability to mate a given couple of individuals

    toolbox = core.Toolbox()
    toolbox.configure("fitness", "CustomFitness", weights)
    toolbox.fitness.cache["space_to_item"] = evaluation.create_item_dict(solution)
    toolbox.configure("individual", "customIndividual", toolbox.fitness)
    toolbox.register("evaluate", evaluation.compose, scores_fc, solution.spec)

    mutations = ((mutation.add_face, {mutation.Case.DEFAULT: 0.1,
                                      mutation.Case.SMALL: 0.3,
                                      mutation.Case.BIG: 0.1}),
                 (mutation.remove_face, {mutation.Case.DEFAULT: 0.1,
                                         mutation.Case.SMALL: 0.1,
                                         mutation.Case.BIG: 0.3}),
                 (mutation.add_aligned_faces, {mutation.Case.DEFAULT: 0.4,
                                               mutation.Case.SMALL: 0.5,
                                               mutation.Case.BIG: 0.1}),
                 (mutation.remove_aligned_faces, {mutation.Case.DEFAULT: 0.4,
                                                  mutation.Case.SMALL: 0.1,
                                                  mutation.Case.BIG: 0.5}))

    toolbox.register("mutate", mutation.composite, mutations)
    toolbox.register("mate", crossover.best_spaces)
    toolbox.register("mate_and_mutate", mate_and_mutate, toolbox.mate, toolbox.mutate,
                     {"cxpb": cxpb})
    toolbox.register("elite_select", selection.elite_select, toolbox.mutate, params["elite"])
    toolbox.register("select", space_nsga.select_nsga)
    toolbox.register("populate", population.fc_mutate(toolbox.mutate))

    return toolbox


def nsga_ga(toolbox: 'core.Toolbox',
            initial_ind: 'core.Individual',
            params: dict,
            hof: Optional['support.HallOfFame']) -> List['core.Individual']:
    """
    A simple implementation of a genetic algorithm.
    :param toolbox: a refiner toolbox
    :param initial_ind: an initial individual
    :param params: the parameters of the algorithm
    :param hof: an optional hall of fame to store best individuals
    :return: the best plan
    """
    # algorithm parameters
    ngen = params["ngen"]
    mu = params["mu"]  # Must be a multiple of 4 for tournament selection of NSGA-II
    chunk_size = math.ceil(mu / params["processes"])
    initial_ind.all_spaces_modified()
    initial_ind.fitness.sp_values = toolbox.evaluate(initial_ind)
    pop = toolbox.populate(initial_ind, mu)
    toolbox.evaluate_pop(toolbox.map, toolbox.evaluate, pop, chunk_size)

    # This is just to assign the crowding distance to the individuals
    # no actual selection is done
    pop = toolbox.select(pop, len(pop))

    # Begin the generational process
    for gen in range(1, ngen + 1):
        logging.info("Refiner: generation %i : %.2f prct", gen, gen / ngen * 100.0)
        # Vary the population
        offspring = nsga.select_tournament_dcd(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]

        # note : list is needed because map lazy evaluates
        modified = list(toolbox.map(toolbox.mate_and_mutate, zip(offspring[::2], offspring[1::2]),
                                    math.ceil(chunk_size/2)))
        offspring = [i for t in modified for i in t]

        # Evaluate the individuals with an invalid fitness
        toolbox.evaluate_pop(toolbox.map, toolbox.evaluate, offspring, chunk_size)

        # best score
        best_ind = max(offspring, key=lambda i: i.fitness.wvalue)
        logging.info("Best : {:.2f} - {}".format(best_ind.fitness.wvalue, best_ind.fitness.values))

        # Select the next generation population
        pop = toolbox.select(pop + offspring, mu)

        # store best individuals in hof
        if hof is not None:
            hof.update(pop, value=True)

    return pop


def space_nsga_ga(toolbox: 'core.Toolbox',
                  initial_ind: 'core.Individual',
                  params: dict,
                  hof: Optional['support.HallOfFame']) -> List['core.Individual']:
    """
    A simple implementation of a genetic algorithm. We try to select individuals according to the
    pareto fronts of the population for each space fitness value. The idea is to select the
    individuals with the best `non dominated` space and to mate them accordingly.
    This seems a better strategy than to select via the different objectives.
    :param toolbox: a refiner toolbox
    :param initial_ind: an initial individual
    :param params: the parameters of the algorithm
    :param hof: an optional hall of fame to store best individuals
    :return: the best plan
    """
    # algorithm parameters
    ngen = params["ngen"]
    mu = params["mu"]  # Must be a multiple of 4 for tournament selection of NSGA-II
    chunk_size = math.ceil(mu / params["processes"])
    initial_ind.all_spaces_modified()
    initial_ind.fitness.sp_values = toolbox.evaluate(initial_ind)
    pop = toolbox.populate(initial_ind, mu)
    toolbox.evaluate_pop(toolbox.map, toolbox.evaluate, pop, chunk_size)

    # This is just to assign the crowding distance to the individuals
    # no actual selection is done
    pop = toolbox.select(pop, len(pop))

    best_fitness = max(pop, key=lambda i: i.fitness.wvalue).fitness.wvalue
    no_improvement_count = 0

    # Begin the generational process
    for gen in range(1, ngen + 1):
        logging.info("Refiner: generation %i : %.2f prct", gen, gen / ngen * 100.0)
        # Vary the population
        offspring = space_nsga.select_tournament_dcd(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]

        # note : list is needed because map lazy evaluates
        modified = list(toolbox.map(toolbox.mate_and_mutate, zip(offspring[::2], offspring[1::2]),
                        math.ceil(chunk_size/2)))
        offspring = [i for t in modified for i in t]
        total_pop = pop + offspring

        # Evaluate the individuals with an invalid fitness
        toolbox.evaluate_pop(toolbox.map, toolbox.evaluate, total_pop, chunk_size)

        # Select the next generation population
        pop = toolbox.elite_select(total_pop, mu)

        # print best individual and check if we found a better solution
        best_ind = pop[0]
        best_ind_fitness = best_ind.fitness.wvalue
        if best_ind_fitness > best_fitness:
            best_fitness = best_ind_fitness
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        # store best individuals in hof
        if hof is not None:
            hof.update(pop, value=True)

        logging.info("Best x{}: {:.2f} - {}".format(no_improvement_count, best_ind.fitness.wvalue,
                                                    best_ind.fitness.values))

        # if we do not improve more than `max_tries times in a row we estimate we have reached t`
        # he global min and we can stop.
        if no_improvement_count > params.get("max_tries", 10):
            break

        # order individual on pareto front for tournament selection
        pop = toolbox.select(pop, mu)

    return pop


def naive_ga(toolbox: 'core.Toolbox',
             initial_ind: 'core.Individual',
             params: dict,
             hof: Optional['support.HallOfFame']) -> List['core.Individual']:
    """
    A simple implementation of a genetic algorithm.
    :param toolbox: a refiner toolbox
    :param initial_ind: an initial individual
    :param params: the parameters of the algorithm
    :param hof: an optional hall of fame to store best individuals
    :return: the best plan
    """
    # algorithm parameters
    ngen = params["ngen"]
    mu = params["mu"]  # Must be a multiple of 4 for tournament selection of NSGA-II
    chunk_size = math.ceil(mu / params["processes"])
    initial_ind.all_spaces_modified()  # set all spaces as modified for first evaluation
    initial_ind.fitness.sp_values = toolbox.evaluate(initial_ind)
    logging.info("Initial : {:.2f} - {}".format(initial_ind.fitness.wvalue,
                                                initial_ind.fitness.values))
    pop = toolbox.populate(initial_ind, mu)
    toolbox.evaluate_pop(toolbox.map, toolbox.evaluate, pop, chunk_size)

    # Begin the generational process
    for gen in range(1, ngen + 1):
        logging.info("Refiner: generation %i : %.2f prct", gen, gen / ngen * 100.0)
        # Vary the population
        offspring = [toolbox.clone(ind) for ind in pop]
        random.shuffle(offspring)

        # note : list is needed because map lazy evaluates
        modified = list(toolbox.map(toolbox.mate_and_mutate, zip(offspring[::2], offspring[1::2]),
                                    math.ceil(chunk_size/2)))
        offspring = [i for t in modified for i in t]

        # Evaluate the individuals with an invalid fitness
        toolbox.evaluate_pop(toolbox.map, toolbox.evaluate, offspring, chunk_size)

        # best score
        best_ind = max(offspring, key=lambda i: i.fitness.wvalue)
        logging.info("Best : {:.2f} - {}".format(best_ind.fitness.wvalue, best_ind.fitness.values))

        # Select the next generation population
        pop = sorted(pop + offspring, key=lambda i: i.fitness.wvalue, reverse=True)
        pop = pop[:mu]

        # store best individuals in hof
        if hof is not None:
            hof.update(pop, value=True)

    return pop


REFINERS = {
    "nsga": Refiner(fc_nsga_toolbox, nsga_ga),
    "naive": Refiner(fc_nsga_toolbox, naive_ga),
    "space_nsga": Refiner(fc_space_nsga_toolbox, space_nsga_ga)
}

if __name__ == '__main__':

    def apply():
        """
        Plan to check :
        • 049 / 050 / 027
        :return:
        """
        import tools.cache
        import time

        from libs.modelers.corridor import CORRIDOR_BUILDING_RULES, Corridor

        params = {"ngen": 120, "mu": 80, "cxpb": 0.5, "max_tries": 10, "elite": 0.2, "processes": 8}

        logging.getLogger().setLevel(logging.INFO)
        plan_number = "062"  # 062 006 020 061
        solution = tools.cache.get_solution(plan_number, grid="002", seeder="directional_seeder",
                                            solution_number=0)

        if solution:
            plan = solution.spec.plan
            plan.name = "original_" + plan_number
            plan.plot()

            Corridor(corridor_rules=CORRIDOR_BUILDING_RULES["no_cut"]["corridor_rules"],
                     growth_method=CORRIDOR_BUILDING_RULES["no_cut"]["growth_method"]
                     ).apply_to(solution)

            plan.name = "Corridor_" + plan_number
            plan.plot()
            # run genetic algorithm
            start = time.time()
            improved_plan = REFINERS["space_nsga"].apply_to(solution, params).spec.plan
            end = time.time()

            # display solution
            improved_plan.name = "Refined_" + plan_number
            improved_plan.plot()
            # analyse found solution
            logging.info("Time elapsed: {}".format(end - start))
            logging.info("Solution found : {} - {}".format(improved_plan.fitness.wvalue,
                                                           improved_plan.fitness.values))

            evaluation.check(improved_plan, solution)

    apply()
