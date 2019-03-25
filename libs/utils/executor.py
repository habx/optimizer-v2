# coding=utf-8
"""
module used to run optimizer
"""

import logging
from typing import List, Optional
import time
import os
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


class Executor:
    """
    Class used to run Optimizer with defined parameters.
    """

    def __init__(self):
        self.grid_type: Optional[str] = None
        self.shuffle_type: Optional[str] = None
        self.setup_name: Optional[str] = None
        self.do_plot: Optional[bool] = None
        # NOTE: a seeder_name should be chosen too
        self.reset_to_default()

    def reset_to_default(self):
        self.grid_type = "optimal_grid"
        self.shuffle_type = "square_shape_shuffle_rooms"
        self.setup_name = "unnamed"
        self.do_plot = False

    def set_execution_parameters(self,
                                 grid_type: Optional[str] = None,
                                 shuffle_type: Optional[str] = None,
                                 setup_name: Optional[str] = None,
                                 plot: Optional[bool] = None) -> None:
        """
        Change parameters of executor. If one is not specified (None), default value is used.
        """
        if grid_type is not None:
            self.grid_type = grid_type
        if shuffle_type is not None:
            self.shuffle_type = shuffle_type
        if setup_name is not None:
            self.setup_name = setup_name
        if plot is not None:
            self.do_plot = plot

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
        self.setup_name = os.path.splitext(setup_file_name)[0]
        return self.run(lot, setup)

    def run_from_file_paths(self, lot_file_path, setup_file_path) -> Response:
        """
        Run Optimizer from file paths.
        :return: optimizer response
        """
        lot = reader.get_json_from_file(lot_file_path, "")
        setup = reader.get_json_from_file(setup_file_path, "")
        self.setup_name = os.path.splitext(os.path.basename(setup_file_path))[0]
        return self.run(lot, setup)

    def run(self, lot: dict, setup: dict) -> Response:
        """
        Run Optimizer
        :param lot: lot data
        :param setup: setup data
        :return: optimizer response
        """
        # time
        t0 = time.time()

        # reading lot
        logging.info("Read lot")
        assert "v2" in lot.keys(), "lot must contain v2 data"
        plan = reader.create_plan_from_v2_data(lot["v2"])

        # grid
        logging.info("Grid")
        GRIDS[self.grid_type].apply_to(plan, show=self.do_plot)

        # seeder
        logging.info("Seeder")
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
                SHUFFLES[self.shuffle_type].run(sol.plan, show=self.do_plot)
                if self.do_plot:
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
        executor.set_execution_parameters(grid_type="optimal_grid")
        response = executor.run_from_file_names("grenoble_102.json", "grenoble_102_setup0.json")
        logging.info("Time: %i", int(response.elapsed_time))
        logging.info("Nb solutions: %i", len(response.solutions))


    main()
