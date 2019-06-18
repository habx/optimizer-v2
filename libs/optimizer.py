#!/usr/bin/env python3
# coding=utf-8
"""
module used to run optimizer
"""

import logging
from typing import List, Dict, Optional
import time
import mimetypes
import os

from libs.io import reader
from libs.io.writer import generate_output_dict
from libs.modelers.grid import GRIDS
from libs.modelers.seed import SEEDERS
from libs.modelers.corridor import Corridor, CORRIDOR_BUILDING_RULES
from libs.refiner.refiner import REFINERS
from libs.space_planner.space_planner import SPACE_PLANNERS
from libs.version import VERSION as OPTIMIZER_VERSION
import libs.io.plot


class LocalContext:
    """Local execution context"""

    def __init__(self):
        self.files: Dict[str, Dict] = {}
        self.output_dir: str = None

    def add_file(
            self,
            name: str,
            ftype: Optional[str] = '?',
            title: Optional[str] = None,
            mime: str = None
    ):
        if not title:
            title = name
        self.files[name] = {
            'type': ftype,
            'title': title,
            'mime': mime,
        }


class Response:
    """
    Response of an optimizer run. Contains solutions and all run data.
    """

    def __init__(self,
                 solutions: List[dict],
                 elapsed_times: Dict[str, float]
                 ):
        self.solutions = solutions
        self.elapsed_times = elapsed_times


class ExecParams:
    """
    Dict wrapper, mostly useful for auto-completion

         TODO : the params structure does not seem generic enough.
                The structure should be the same for each step of the pipe.
                For example, we could do something nicer such as :
                params = {
                           'grid': {'name': 'optimal_grid', 'params': {}},
                           'seeder': {'name': 'simple_seeder, 'params': {}},
                           'space_planner': {'name': 'default_space_planner', 'params': {}},
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
            "ngen": 60,
            "mu": 40,
            "cxpb": 0.9
        }

        self.grid_type = params.get('grid_type', '002')
        self.seeder_type = params.get('seeder_type', 'directional_seeder')
        self.space_planner_type = params.get('space_planner_type', 'standard_space_planner')
        self.do_plot = params.get('do_plot', False)
        self.max_nb_solutions = params.get('max_nb_solutions', 3)
        self.do_corridor = params.get('do_corridor', False)
        self.corridor_type = params.get('corridor_params', 'no_cut')
        self.do_refiner = params.get('do_refiner', False)
        self.refiner_type = params.get('refiner_type', 'space_nsga')
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
                            local_context: dict = None) -> Response:
        """
        Run Optimizer from file names.
        :param lot_file_name: name of lot file, file has to be in resources/blueprints
        :param setup_file_name: name of setup file, file has to be in resources/specifications
        :param params: Execution parameters
        :param local_context: Local execution parameters
        :return: optimizer response
        """
        lot = reader.get_json_from_file(lot_file_name)
        setup = reader.get_json_from_file(setup_file_name,
                                          reader.DEFAULT_SPECIFICATION_INPUT_FOLDER)

        return self.run(lot, setup, params, local_context)

    @staticmethod
    def get_generated_files(output_dir) -> Dict[str, Dict]:
        mimetypes.init()
        files: Dict[str, Dict] = {}
        for file in os.listdir(output_dir):
            extension = os.path.splitext(file)[-1].lower()
            if extension in (".tif", ".tiff",
                             ".jpeg", ".jpg", ".jif", ".jfif",
                             ".jp2", ".jpx", ".j2k", ".j2c",
                             ".gif", ".svg", ".fpx", ".pcd", ".png", ".pdf"):
                files[file] = {
                    'type': os.path.splitext(file)[0],
                    'title': os.path.splitext(file)[0].capitalize(),
                    'mime': mimetypes.types_map[extension],
                }
        return files

    def run(self,
            lot: dict,
            setup: dict,
            params_dict: dict = None,
            local_context: LocalContext = None) -> Response:
        """
        Run Optimizer
        :param lot: lot data
        :param setup: setup data
        :param params_dict: execution parameters
        :param local_context: local context parameters
        :return: optimizer response
        """
        assert "v2" in lot.keys(), "lot must contain v2 data"

        # OPT-119: If we don't have a local_context, we create one
        if not local_context:
            local_context = LocalContext()

        params = ExecParams(params_dict)

        # output dir
        if local_context is not None and local_context.output_dir:
            libs.io.plot.output_path = local_context.output_dir
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
        if params.do_plot:
            plan.plot(name="grid")
        elapsed_times["grid"] = time.process_time() - t0_grid
        logging.info("Grid achieved in %f", elapsed_times["grid"])

        # seeder
        logging.info("Seeder")
        t0_seeder = time.process_time()
        SEEDERS[params.seeder_type].apply_to(plan)
        if params.do_plot:
            plan.plot(name="seeder")
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
        logging.info("Space planner")
        t0_space_planner = time.process_time()
        space_planner = SPACE_PLANNERS[params.space_planner_type]
        best_solutions = space_planner.apply_to(spec, params.max_nb_solutions)
        logging.debug(best_solutions)
        elapsed_times["space planner"] = time.process_time() - t0_space_planner
        logging.info("Space planner achieved in %f", elapsed_times["space planner"])

        # corridor
        t0_corridor = time.process_time()
        if params.do_corridor:
            logging.info("Corridor")
            if best_solutions and space_planner:
                spec = space_planner.spec
                for sol in best_solutions:
                    spec.plan = sol.plan
                    corridor_building_rule = CORRIDOR_BUILDING_RULES[params.corridor_type]
                    Corridor(corridor_rules=corridor_building_rule["corridor_rules"],
                             growth_method=corridor_building_rule["growth_method"]).apply_to(
                        sol.plan, spec)
                    if params.do_plot:
                        sol.plan.plot()
        elapsed_times["corridor"] = time.process_time() - t0_corridor
        logging.info("Corridor achieved in %f", elapsed_times["corridor"])

        # refiner
        t0_refiner = time.process_time()
        if params.do_refiner:
            logging.info("Refiner")
            if best_solutions and space_planner:
                spec = space_planner.spec
                for sol in best_solutions:
                    spec.plan = sol.plan
                    sol.plan = REFINERS[params.refiner_type].apply_to(sol.plan, spec,
                                                                      params.refiner_params,
                                                                      processes=2)
                    if params.do_plot:
                        sol.plan.plot()
        elapsed_times["refiner"] = time.process_time() - t0_refiner
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

        # OPT-114: This is how we will transmit the generated files
        local_context.files = Optimizer.get_generated_files(libs.io.plot.output_path)

        return Response(solutions, elapsed_times)


if __name__ == '__main__':
    def main():
        """
        Useful simple main
        """
        logging.getLogger().setLevel(logging.INFO)
        executor = Optimizer()
        response = executor.run_from_file_names(
            "016.json",
            "016_setup0.json",
            {
                "grid_type": "001",
                "seeder_type": "directional_seeder",
                "do_plot": True,
            }
        )
        logging.info("Time: %i", int(response.elapsed_times["total"]))
        logging.info("Nb solutions: %i", len(response.solutions))


    main()
