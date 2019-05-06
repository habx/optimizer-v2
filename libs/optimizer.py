# coding=utf-8
"""
module used to run optimizer
"""

import logging
from typing import List, Dict
import time
import json
import os

from libs.io import reader
from libs.io.writer import generate_output_dict
from libs.modelers.grid import GRIDS
from libs.modelers.seed import SEEDERS
from libs.modelers.shuffle import SHUFFLES
from libs.refiner.refiner import REFINERS
from libs.space_planner.space_planner import SpacePlanner
from libs.version import VERSION as OPTIMIZER_VERSION
import libs.io.plot


class Response:
    """
    Response of an optimizer run. Contains solutions and all run data.
    """

    def __init__(self,
                 solutions: List[dict],
                 elapsed_times: Dict[str, float]):
        self.solutions = solutions
        self.elapsed_times = elapsed_times

    @property
    def as_dict(self) -> dict:
        """
        return all response data in a dict
        """
        return self.__dict__

    def to_json_file(self, filepath) -> None:
        """
        save all response data in a json file
        """
        with open(filepath, "w") as output_file:
            json.dump(self.as_dict, output_file, sort_keys=True, indent=2)


class ExecParams:
    """
    Dict wrapper, mostly useful for auto-completion

         TODO : the params structure does not seem generic enough.
                The structure should be the same for each step of the pipe.
                For example, we could do something nicer such as :
                params = {
                           'grid': {'name': 'optimal_grid', 'params': {}},
                           'seeder': {'name': 'simple_seeder, 'params': {}},
                           'space_planner': {'name': 'default_planner', 'params': {}},
                           'shuffle': {'name': 'simple_shuffle', 'params': {}, 'run': False},
                           'refiner': {'name': 'simple', 'params': {'mu': 28, 'ngen': 100 ...},
                                       'run': True}
                          }
                To use as follow:
                if self.params.shuffle['run']:
                    (SHUFFLES[self.params.shuffle['name']]
                        .apply_to(plan, params=self.params.shuffle['params']))

    """

    def __init__(self, params):
        if params is None:
            params = {}

        refiner_params = {
            "weights": (-2.0, -1.0, -1.0),
            "ngen": 120,
            "mu": 28,
            "cxpb": 0.9
        }

        self.grid_type = params.get('grid_type', 'optimal_grid')
        self.seeder_type = params.get('seeder_type', 'simple_seeder')
        self.do_plot = params.get('do_plot', False)
        self.shuffle_type = params.get('shuffle_type', 'bedrooms_corner')
        self.do_shuffle = params.get('do_shuffle', False)
        self.do_refiner = params.get('do_refiner', False)
        self.refiner_type = params.get('refiner_type', 'simple')
        self.refiner_params = params.get('refiner_params', refiner_params)


class Optimizer:
    """
    Class used to run Optimizer with defined parameters.
    TODO: why are we using a Class here? We could just use functions and the module namespace.
    """

    VERSION = OPTIMIZER_VERSION
    """Current version"""

    def run_from_file_names(self,
                            lot_file_name: str = "011.json",
                            setup_file_name: str = "011_setup0.json",
                            params: dict = None,
                            local_params: dict = None) -> Response:
        """
        Run Optimizer from file names.
        :param lot_file_name: name of lot file, file has to be in resources/blueprints
        :param setup_file_name: name of setup file, file has to be in resources/specifications
        :param params: Execution parameters
        :return: optimizer response
        """
        lot = reader.get_json_from_file(lot_file_name)
        setup = reader.get_json_from_file(setup_file_name,
                                          reader.DEFAULT_SPECIFICATION_INPUT_FOLDER)

        return self.run(lot, setup, params, local_params)

    def run(self,
            lot: dict,
            setup: dict,
            params_dict: dict = None,
            local_params: dict = None) -> Response:
        """
        Run Optimizer
        :param lot: lot data
        :param setup: setup data
        :param params_dict: execution parameters
        :return: optimizer response
        """
        assert "v2" in lot.keys(), "lot must contain v2 data"

        params = ExecParams(params_dict)

        # output dir
        if local_params is not None and 'output_dir' in local_params:
            libs.io.plot.output_path = local_params['output_dir']
            if not os.path.exists(libs.io.plot.output_path):
                os.makedirs(libs.io.plot.output_path)

        # times
        elapsed_times = {}
        t0_total = time.process_time()
        t0_total_real = time.time()

        # reading lot
        logging.info("Read lot")
        t0_reader = time.process_time()
        plan = reader.create_plan_from_data(lot)
        elapsed_times["reader"] = time.process_time() - t0_reader
        logging.info("Lot read in %f", elapsed_times["reader"])

        # grid
        logging.info("Grid")
        t0_grid = time.process_time()
        GRIDS[params.grid_type].apply_to(plan)
        elapsed_times["grid"] = time.process_time() - t0_grid
        logging.info("Grid achieved in %f", elapsed_times["grid"])

        # seeder
        logging.info("Seeder")
        t0_seeder = time.process_time()
        SEEDERS[params.seeder_type].apply_to(plan)

        if params.do_plot:
            plan.plot()
        elapsed_times["seeder"] = time.process_time() - t0_seeder
        logging.info("Seeder achieved in %f", elapsed_times["seeder"])

        # reading setup
        logging.info("Read setup")
        t0_setup = time.process_time()
        spec = reader.create_specification_from_data(setup)
        logging.debug(spec)
        spec.plan = plan
        spec.plan.remove_null_spaces()
        elapsed_times["setup"] = time.process_time() - t0_setup
        logging.info("Setup read in %f", elapsed_times["setup"])

        # space planner
        t0_space_planner = time.process_time()
        logging.info("Space planner")
        space_planner = SpacePlanner("test", spec)
        best_solutions = space_planner.solution_research()
        logging.debug(best_solutions)
        elapsed_times["space planner"] = time.process_time() - t0_space_planner
        logging.info("Space planner achieved in %f", elapsed_times["space planner"])

        # shuffle
        t0_shuffle = time.process_time()
        if params.do_shuffle:
            logging.info("Shuffle")
            if best_solutions:
                for sol in best_solutions:
                    SHUFFLES[params.shuffle_type].apply_to(sol.plan)
                    if params.do_plot:
                        sol.plan.plot()
        elapsed_times["shuffle"] = time.process_time() - t0_shuffle
        logging.info("Shuffle achieved in %f", elapsed_times["shuffle"])

        # refiner
        t0_shuffle = time.process_time()
        if params.do_refiner:
            logging.info("Refiner")
            if best_solutions and space_planner:
                spec = space_planner.spec
                for sol in best_solutions:
                    spec.plan = sol.plan
                    sol.plan = REFINERS[params.refiner_type].apply_to(sol.plan, spec,
                                                                      params.refiner_params)
                    if params.do_plot:
                        sol.plan.plot()
        elapsed_times["refiner"] = time.process_time() - t0_shuffle
        logging.info("Refiner achieved in %f", elapsed_times["refiner"])

        # output
        t0_output = time.process_time()
        logging.info("Output")
        solutions = [generate_output_dict(lot, sol) for sol in best_solutions]
        elapsed_times["output"] = time.process_time() - t0_output
        logging.info("Output written in %f", elapsed_times["output"])

        elapsed_times["total"] = time.process_time() - t0_total
        elapsed_times["totalReal"] = time.time() - t0_total_real
        logging.info("Run complete in %f (process time), %f (real time)",
                     elapsed_times["total"],
                     elapsed_times["totalReal"])

        return Response(solutions, elapsed_times)


if __name__ == '__main__':
    def main():
        """
        Useful simple main
        """
        logging.getLogger().setLevel(logging.INFO)
        executor = Optimizer()
        response = executor.run_from_file_names(
            "012.json",
            "012_setup0.json",
            {
                "grid_type": "optimal_grid",
                "do_plot": True,
            }
        )
        logging.info("Time: %i", int(response.elapsed_times["total"]))
        logging.info("Nb solutions: %i", len(response.solutions))


    main()
