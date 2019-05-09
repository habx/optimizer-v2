"""
Refiner Test Module
"""

import logging

from libs.refiner.refiner import REFINERS
from libs.refiner import evaluation
from libs.io import reader_test

INPUT_FILES = reader_test.BLUEPRINT_INPUT_FILES

PARAMS = {
            "weights": (-2.0, -1.0, -1.0),
            "ngen": 50,
            "mu": 20,
            "cxpb": 0.9
          }


def run():
    """ test function """
    import time
    import tools.cache

    logging.getLogger().setLevel(logging.INFO)

    spec, plan = tools.cache.get_plan("052", "1")  # 052

    if plan:
        plan.name = "original"
        plan.plot()

        # run genetic algorithm

        start = time.time()
        sols = REFINERS["simple"].run(plan, spec, PARAMS, processes=4, hof=1)
        end = time.time()

        # analyse found solutions
        for n, ind in enumerate(sols):
            ind.name = str(n)
            ind.plot()
            print("n°{} | Fitness: {} - {}".format(n, ind.fitness.value, ind.fitness.values))
        print("Time elapsed: {}".format(end - start))
        best = sols[0]
        item_dict = evaluation.create_item_dict(spec)
        for space in best.mutable_spaces():
            print("• Area {} : {} -> [{}, {}]".format(space.category.name,
                                                      round(space.cached_area()),
                                                      item_dict[space.id].min_size.area,
                                                      item_dict[space.id].max_size.area))


def apply():
    """ test function """
    import time
    import tools.cache

    logging.getLogger().setLevel(logging.INFO)

    spec, plan = tools.cache.get_plan("001")  # 052

    if plan:
        plan.name = "original"
        plan.plot()

        # run genetic algorithm
        start = time.time()
        improved_plan = REFINERS["simple"].apply_to(plan, spec, PARAMS, processes=4)
        end = time.time()
        improved_plan.name = "Refined"
        improved_plan.plot()
        # analyse found solutions
        logging.info("Time elapsed: {}".format(end - start))
        logging.info("Solution found : {} - {}".format(improved_plan.fitness.value,
                                                       improved_plan.fitness.values))

        evaluation.check_area(improved_plan, spec)


if __name__ == '__main__':
    apply()
