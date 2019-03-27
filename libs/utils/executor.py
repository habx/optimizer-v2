# coding=utf-8
"""
module used to run optimizer
"""

import logging
from typing import List, Optional
import time
import json

from libs.io import reader
from libs.io.writer import generate_output_dict
from libs.modelers.grid import GRIDS
from libs.modelers.seed import Seeder, GROWTH_METHODS
from libs.operators.selector import SELECTORS
from libs.modelers.shuffle import SHUFFLES
from libs.space_planner.space_planner import SpacePlanner, SQM
from libs.version import VERSION as OPTIMIZER_VERSION


class Response:
    """
    Response of an optimizer run. Contains solutions and all run data.
    """

    def __init__(self, solutions: List[dict], elapsed_time: float):
        self.solutions: List[dict] = solutions
        self.elapsed_time: float = elapsed_time

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
    """Dict wrapper, mostly useful for auto-completion"""
    def __init__(self, params):
        if not params:
            params = {}

        self.grid_type = params.get('grid_type', 'optimal_grid')
        self.do_plot = params.get('do_plot', False)
        self.shuffle_type = params.get('shuffle_type', 'square_shape_shuffle_rooms')


class Executor:
    """
    Class used to run Optimizer with defined parameters.
    """

    VERSION = OPTIMIZER_VERSION
    """Current version"""

    def run_from_file_names(self,
                            lot_file_name: str = "grenoble_101.json",
                            setup_file_name: str = "grenoble_101_setup0.json",
                            params: dict = None) -> Response:
        """
        Run Optimizer from file names.
        :param lot_file_name: name of lot file, file has to be in resources/blueprints
        :param setup_file_name: name of setup file, file has to be in resources/specifications
        :param params: Execution parameters
        :return: optimizer response
        """
        lot = reader.get_json_from_file(lot_file_name,
                                        reader.DEFAULT_BLUEPRINT_INPUT_FOLDER)
        setup = reader.get_json_from_file(setup_file_name,
                                          reader.DEFAULT_SPECIFICATION_INPUT_FOLDER)

        return self.run(lot, setup, params)

    def run(self, lot: dict, setup: dict, params_dict: dict) -> Response:
        """
        Run Optimizer
        :param lot: lot data
        :param setup: setup data
        :param params_dict: execution parameters
        :return: optimizer response
        """
        params = ExecParams(params_dict)

        # time
        t0 = time.time()

        # reading lot
        logging.info("Read lot")
        assert "v2" in lot.keys(), "lot must contain v2 data"
        plan = reader.create_plan_from_v2_data(lot["v2"])

        # grid
        logging.info("Grid")
        GRIDS[params.grid_type].apply_to(plan, show=params.do_plot)

        # seeder
        logging.info("Seeder")
        seeder = Seeder(plan, GROWTH_METHODS).add_condition(SELECTORS['seed_duct'], 'duct')
        if params.do_plot:
            plan.plot()

        (seeder.plant()
         .grow(show=params.do_plot)
         .divide_along_seed_borders(SELECTORS["not_aligned_edges"])
         .from_space_empty_to_seed()
         .merge_small_cells(min_cell_area=1 * SQM,
                            excluded_components=["loadBearingWall"],
                            show=params.do_plot))

        if params.do_plot:
            plan.plot()

        # reading setup
        logging.info("Read setup")
        spec = reader.create_specification_from_data(setup)
        logging.debug(spec)
        spec.plan = plan
        spec.plan.remove_null_spaces()

        # space planner
        logging.info("Space planner")
        space_planner = SpacePlanner("test", spec)
        best_solutions = space_planner.solution_research(show=False)
        logging.debug(best_solutions)

        # shuffle
        if best_solutions:
            for sol in best_solutions:
                SHUFFLES[params.shuffle_type].run(sol.plan, show=params.do_plot)
                if params.do_plot:
                    sol.plan.plot()

        # output
        logging.info("Output")
        solutions = [generate_output_dict(lot["v2"], sol) for sol in best_solutions]
        elapsed_time = time.time() - t0
        return Response(solutions, elapsed_time)


if __name__ == '__main__':
    def main():
        """
        Useful simple main
        """
        logging.getLogger().setLevel(logging.INFO)
        executor = Executor()
        response = executor.run_from_file_names(
            "grenoble_102.json",
            "grenoble_102_setup0.json",
            {
                "grid_type": "optimal_grid",
                "do_plot": False,
            }
        )
        logging.info("Time: %i", int(response.elapsed_time))
        logging.info("Nb solutions: %i", len(response.solutions))


    main()
