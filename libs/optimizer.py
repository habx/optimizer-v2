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
from libs.io.writer import generate_output_dict, save_plan_as_json
from libs.modelers.grid import GRIDS
from libs.modelers.seed import SEEDERS
from libs.modelers.corridor import Corridor, CORRIDOR_BUILDING_RULES
from libs.refiner.refiner import REFINERS
from libs.space_planner.space_planner import SPACE_PLANNERS
from libs.equipments.doors import place_doors, door_plot
from libs.version import VERSION as OPTIMIZER_VERSION
from libs.scoring.scoring import final_scoring
from libs.space_planner.solution import spec_adaptation, reference_plan_solution
import libs.io.plot
import matplotlib.pyplot as plt
import urllib, json


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
                 elapsed_times: Dict[str, float],
                 ref_plan_score:float,
                 ref_plan_score_components: Optional[dict] = None
                 ):
        self.solutions = solutions
        self.elapsed_times = elapsed_times
        self.ref_plan_score = ref_plan_score
        self.ref_plan_score_components = ref_plan_score_components


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

        refiner_params = {"ngen": 80, "mu": 80, "cxpb": 0.9, "max_tries": 10, "elite": 0.1,
                          "processes": 8}
        space_planner_params = {"space_planner_type": "standard_space_planner", "processes": 8}

        self.grid_type = params.get('grid_type', '002')
        self.seeder_type = params.get('seeder_type', 'directional_seeder')
        # self.space_planner_type = params.get('space_planner_type', 'standard_space_planner')
        self.space_planner_params = params.get('space_planner_params', space_planner_params)
        self.do_plot = params.get('do_plot', False)
        self.save_ll_bp = params.get('save_ll_bp', False)
        self.max_nb_solutions = params.get('max_nb_solutions', 3)
        self.do_corridor = params.get('do_corridor', False)
        self.corridor_type = params.get('corridor_params', 'no_cut')
        self.do_refiner = params.get('do_refiner', False)
        self.refiner_type = params.get('refiner_type', 'space_nsga')
        self.refiner_params = params.get('refiner_params', refiner_params)
        self.do_door = params.get('do_door', False)
        self.ref_plan_url = params.get('ref_plan_url', None)
        self.do_final_scoring = params.get('do_final_scoring', False)


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
            if (extension in (".tif", ".tiff",
                              ".jpeg", ".jpg", ".jif", ".jfif",
                              ".jp2", ".jpx", ".j2k", ".j2c",
                              ".gif", ".svg", ".fpx", ".pcd", ".png", ".pdf")
                    or extension == ".json"):
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

        # reading setup
        logging.info("Read setup")
        t0_setup = time.process_time()
        setup_spec = reader.create_specification_from_data(setup)
        logging.debug(setup_spec)
        setup_spec.plan = plan
        setup_spec.plan.remove_null_spaces()
        area_matching = setup_spec.area_checker()
        elapsed_times["setup"] = time.process_time() - t0_setup
        logging.info("Setup read in %f", elapsed_times["setup"])
        if not area_matching:
            return Response([], elapsed_times, None, None)

        # grid
        logging.info("Grid")
        t0_grid = time.process_time()
        GRIDS[params.grid_type].apply_to(plan)
        if params.do_plot:
            plan.plot(name="grid")
        if params.save_ll_bp:
            save_plan_as_json(plan.serialize(), "grid", libs.io.plot.output_path)
        elapsed_times["grid"] = time.process_time() - t0_grid
        logging.info("Grid achieved in %f", elapsed_times["grid"])

        # seeder
        logging.info("Seeder")
        t0_seeder = time.process_time()
        SEEDERS[params.seeder_type].apply_to(plan)
        if params.do_plot:
            plan.plot(name="seeder")
        if params.save_ll_bp:
            save_plan_as_json(plan.serialize(), "seeder", libs.io.plot.output_path)
        elapsed_times["seeder"] = time.process_time() - t0_seeder
        logging.info("Seeder achieved in %f", elapsed_times["seeder"])

        # space planner
        logging.info("Space planner")
        t0_space_planner = time.process_time()
        space_planner = SPACE_PLANNERS[params.space_planner_params['space_planner_type']]
        best_solutions = space_planner.apply_to(setup_spec, params.max_nb_solutions,
                                                params.space_planner_params['processes'])
        logging.debug(best_solutions)
        elapsed_times["space planner"] = time.process_time() - t0_space_planner
        logging.info("Space planner achieved in %f", elapsed_times["space planner"])

        # corridor
        t0_corridor = time.process_time()
        if params.do_corridor:
            logging.info("Corridor")
            if best_solutions:
                for i, sol in enumerate(best_solutions):
                    corridor_building_rule = CORRIDOR_BUILDING_RULES[params.corridor_type]
                    Corridor(corridor_rules=corridor_building_rule["corridor_rules"],
                             growth_method=corridor_building_rule["growth_method"]).apply_to(sol)
                    # specification update
                    spec_adaptation(sol, space_planner.solutions_collector)
                    if params.do_plot:
                        sol.spec.plan.plot(name=f"corridor sol {i + 1}")
                    if params.save_ll_bp:
                        save_plan_as_json(sol.spec.plan.serialize(), f"corridor sol {i + 1}",
                                          libs.io.plot.output_path)
        elapsed_times["corridor"] = time.process_time() - t0_corridor
        logging.info("Corridor achieved in %f", elapsed_times["corridor"])

        # refiner
        t0_refiner = time.process_time()
        if params.do_refiner:
            logging.info("Refiner")
            if best_solutions:
                for i, sol in enumerate(best_solutions):
                    REFINERS[params.refiner_type].apply_to(sol, params.refiner_params)
                    if params.do_plot:
                        sol.spec.plan.plot(name=f"refiner sol {i + 1}")
                    if params.save_ll_bp:
                        save_plan_as_json(sol.spec.plan.serialize(), f"refiner sol {i + 1}",
                                          libs.io.plot.output_path)
        elapsed_times["refiner"] = time.process_time() - t0_refiner
        logging.info("Refiner achieved in %f", elapsed_times["refiner"])

        # placing doors
        t0_door = time.process_time()
        if params.do_door:
            logging.info("Door")
            if best_solutions:
                for sol in best_solutions:
                    place_doors(sol.spec.plan)
                    if params.do_plot:
                        door_plot(sol.spec.plan)
        elapsed_times["door"] = time.process_time() - t0_door
        logging.info("Door placement achieved in %f", elapsed_times["door"])

        # scoring
        ref_final_score = None
        ref_final_score_components = None
        if params.do_final_scoring :
            if params.ref_plan_url is not None:
                with urllib.request.urlopen(params.ref_plan_url) as url:
                    data = json.loads(url.read().decode())
                    ref_plan = reader.create_plan_from_data(data)
                    if params.do_plot:
                        ref_plan.plot()
                    ref_solution = reference_plan_solution(ref_plan, setup_spec)
                    ref_final_score, ref_final_score_components = final_scoring(ref_solution)

            if best_solutions:
                for sol in best_solutions:
                    final_score, final_score_components = final_scoring(sol)
                    sol.final_score = final_score
                    sol.final_score_components = final_score_components
                    if params.do_plot:
                        sol.spec.plan.plot()
                plt.close()

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

        # TODO: once scoring has been added, add the ref_plan_score to the solution
        return Response(solutions, elapsed_times, ref_final_score, ref_final_score_components)


if __name__ == '__main__':
    def main():
        """
        Useful simple main
        """
        logging.getLogger().setLevel(logging.INFO)
        executor = Optimizer()
        response = executor.run_from_file_names(
            "france-confort_A1E0H01.json",
            "france-confort_A1E0H01_setup3.json",
            {
                "grid_type": "002",
                "seeder_type": "directional_seeder",
                "do_plot": False,
                "do_corridor": False,
                #"do_refiner":False,
                "max_nb_solutions": 5,
                #"do_door": False,
                #"do_final_scoring": True,
                #"ref_plan_url": "https://cdn.habx.fr/optimizer-lots/plans%20base/ARCH014_plan.json"
            }
        )
        logging.info("Time: %i", int(response.elapsed_times["total"]))
        logging.info("Nb solutions: %i", len(response.solutions))


    main()
