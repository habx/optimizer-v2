"""
Refiner Test Module
"""

import logging
import pytest
import time

from libs.refiner.refiner import REFINERS
from libs.io import reader_test

import tools.cache

INPUT_FILES = reader_test.BLUEPRINT_INPUT_FILES

PARAMS = {"ngen": 50, "mu": 20, "cxpb": 0.2}


@pytest.mark.parametrize("input_file", INPUT_FILES)
def refiner_simple(input_file):
    """
    Test refiner on all plan files
    051 / 009 / 062 / 055
    :return:
    """
    logging.getLogger().setLevel(logging.INFO)
    plan_number = input_file[:len(input_file) - 5]

    spec, plan = tools.cache.get_plan(plan_number, grid="001", seeder="directional_seeder")

    if plan:
        plan.name = "original" + "_" + plan_number
        plan.remove_null_spaces()

        # run genetic algorithm
        start = time.time()
        improved_plan = REFINERS["nsga"].apply_to(plan, spec, PARAMS, processes=4)
        end = time.time()
        improved_plan.name = "Refined_" + plan_number

        # analyse found solutions
        logging.info("Time elapsed: {}".format(end - start))
        logging.info("Solution found : {} - {}".format(improved_plan.fitness.wvalue,
                                                       improved_plan.fitness.values))

        assert improved_plan.check()
