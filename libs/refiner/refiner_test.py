"""
Refiner Test Module
"""

import logging

from libs.refiner.refiner import REFINERS
from libs.refiner import evaluation
from libs.io import reader_test

INPUT_FILES = reader_test.BLUEPRINT_INPUT_FILES

PARAMS = {
            "weights": (-20.0, -1.0, -1.0, -1000.0, -20.0),
            "ngen": 120,
            "mu": 28,
            "cxpb": 0.2
          }

def apply():
    """ test function """
    import time
    import tools.cache

    logging.getLogger().setLevel(logging.INFO)

    spec, plan = tools.cache.get_plan("003", grid="optimal_grid")  # 052

    if plan:
        plan.name = "original"
        plan.remove_null_spaces()
        plan.plot()

        # run genetic algorithm
        start = time.time()
        improved_plan = REFINERS["simple"].apply_to(plan, spec, PARAMS, processes=4)
        end = time.time()
        improved_plan.name = "Refined"
        improved_plan.plot()
        # analyse found solutions
        logging.info("Time elapsed: {}".format(end - start))
        logging.info("Solution found : {} - {}".format(improved_plan.fitness.wvalue,
                                                       improved_plan.fitness.values))

        evaluation.check(improved_plan, spec)


if __name__ == '__main__':
    apply()
