# coding=utf-8
"""
module used to run optimizer
"""

import logging
from typing import List, Optional, Dict
import time
import json

from libs.io import reader
from libs.io.writer import generate_output_dict
from libs.modelers.grid import GRIDS
from libs.modelers.seed import Seeder, GROWTH_METHODS
from libs.operators.selector import SELECTORS
from libs.modelers.shuffle import SHUFFLES
from libs.space_planner.space_planner import SpacePlanner, SQM


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


class Executor:
    """
    Class used to run Optimizer with defined parameters.
    """

    def __init__(self):
        self.grid_type: Optional[str] = None
        self.shuffle_type: Optional[str] = None
        # OPT-72: Setup name is not pat of the execution context, it should be handled on a
        # per-plan basis, and most probably directly in the input data.
        # self.setup_name: Optional[str] = None
        self.do_plot: Optional[bool] = None
        # NOTE: a seeder_name should be chosen too
        self.reset_to_default()

    def reset_to_default(self):
        self.grid_type = "optimal_grid"
        self.shuffle_type = "square_shape_shuffle_rooms"
        # self.setup_name = "unnamed"
        self.do_plot = False

    def set_execution_parameters(self,
                                 grid_type: Optional[str] = None,
                                 shuffle_type: Optional[str] = None,
                                 # setup_name: Optional[str] = None,
                                 do_plot: Optional[bool] = None) -> None:
        """
        Change parameters of executor. If one is not specified (None), default value is used.
        """
        if grid_type is not None:
            self.grid_type = grid_type
        if shuffle_type is not None:
            self.shuffle_type = shuffle_type
        # if setup_name is not None:
        #    self.setup_name = setup_name
        if do_plot is not None:
            self.do_plot = do_plot

    def run_from_file_names(self,
                            lot_file_name: str = "grenoble_101.json",
                            setup_file_name: str = "grenoble_101_setup0.json") -> Response:
        """
        Run Optimizer from file names.
        :param lot_file_name: name of lot file, file has to be in resources/blueprints
        :param setup_file_name: name of setup file, file has to be in resources/specifications
        :return: optimizer response
        """
        lot = reader.get_json_from_file(lot_file_name,
                                        reader.DEFAULT_BLUEPRINT_INPUT_FOLDER)
        setup = reader.get_json_from_file(setup_file_name,
                                          reader.DEFAULT_SPECIFICATION_INPUT_FOLDER)
        # self.setup_name = os.path.splitext(setup_file_name)[0]
        return self.run(lot, setup)

    def run(self, lot: dict, setup: dict) -> Response:
        """
        Run Optimizer
        :param lot: lot data
        :param setup: setup data
        :return: optimizer response
        """
        assert "v2" in lot.keys(), "lot must contain v2 data"

        # times
        elapsed_times = {}
        t0_total = time.process_time()
        t0_total_real = time.time()

        # reading lot
        logging.info("Read lot")
        t0_reader = time.process_time()
        plan = reader.create_plan_from_v2_data(lot["v2"])
        elapsed_times["reader"] = time.process_time() - t0_reader
        logging.info("Lot read in %f", elapsed_times["reader"])

        # grid
        logging.info("Grid")
        t0_grid = time.process_time()
        GRIDS[self.grid_type].apply_to(plan, show=self.do_plot)
        elapsed_times["grid"] = time.process_time() - t0_grid
        logging.info("Grid achieved in %f", elapsed_times["grid"])

        # seeder
        logging.info("Seeder")
        t0_seeder = time.process_time()
        seeder = Seeder(plan, GROWTH_METHODS).add_condition(SELECTORS['seed_duct'], 'duct')
        if self.do_plot:
            plan.plot()
        (seeder.plant()
         .grow(show=self.do_plot)
         .divide_along_seed_borders(SELECTORS["not_aligned_edges"])
         .from_space_empty_to_seed()
         .merge_small_cells(min_cell_area=1 * SQM,
                            excluded_components=["loadBearingWall"],
                            show=self.do_plot))
        if self.do_plot:
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
        best_solutions = space_planner.solution_research(show=False)
        logging.debug(best_solutions)
        elapsed_times["space planner"] = time.process_time() - t0_space_planner
        logging.info("Space planner achieved in %f", elapsed_times["space planner"])

        # shuffle
        t0_shuffle = time.process_time()
        logging.info("Shuffle")
        if best_solutions:
            for sol in best_solutions:
                SHUFFLES[self.shuffle_type].run(sol.plan, show=self.do_plot)
                if self.do_plot:
                    sol.plan.plot()
        elapsed_times["shuffle"] = time.process_time() - t0_shuffle
        logging.info("Shuffle achieved in %f", elapsed_times["shuffle"])

        # output
        t0_output = time.process_time()
        logging.info("Output")
        solutions = [generate_output_dict(lot["v2"], sol) for sol in best_solutions]
        elapsed_times["output"] = time.process_time() - t0_output
        logging.info("Output written in %f", elapsed_times["output"])

        elapsed_times["total"] = time.process_time() - t0_total
        elapsed_times["total_real"] = time.time() - t0_total_real
        logging.info("Run complete in %f (process time), %f (real time)",
                     elapsed_times["total"],
                     elapsed_times["total_real"])

        return Response(solutions, elapsed_times)


if __name__ == '__main__':
    def main():
        """
        Useful simple main
        """
        logging.getLogger().setLevel(logging.INFO)
        executor = Executor()
        executor.set_execution_parameters(grid_type="optimal_grid", do_plot=False)
        response = executor.run_from_file_names("grenoble_102.json", "grenoble_102_setup0.json")
        logging.info("Time: %i", int(response.elapsed_times["all"]))
        logging.info("Nb solutions: %i", len(response.solutions))


    main()
