# coding=utf-8
"""
Space Planner module

The space planner finds the best rooms layouts according to a plan with given seed spaces
and a customer input setup

"""
import logging
from typing import List, Optional, Dict
from libs.specification.specification import Specification, Item
from libs.space_planner.solution import SolutionsCollector, Solution
from libs.plan.plan import Plan, Space
from libs.space_planner.constraints_manager import ConstraintsManager
from libs.plan.category import SPACE_CATEGORIES
import libs.io.writer as writer

SQM = 10000


class SpacePlanner:
    """
    Space planner Class
    """

    def __init__(self, name: str):
        self.name = name
        self.spec = None
        self.manager = None
        self.solutions_collector = None

    def __repr__(self):
        output = "SpacePlanner" + self.name
        return output

    def _plan_cleaner(self, min_area: float = 100) -> None:
        """
        Plan cleaner for little spaces
        TODO: This means that we are breaking the assumption that every face of the mesh has an
              assigned space. This means that we must then always check for None results when we
              fetch the space of a face. Not sure this is optimal.
        :return: None
        """
        self.spec.plan.remove_null_spaces()
        for space in self.spec.plan.spaces:
            if space.cached_area() < min_area:
                self.spec.plan.remove(space)

    def _rooms_building(self, plan: 'Plan', matrix_solution) -> ('Plan', Dict['Item', 'Space']):
        """
        Builds the rooms requested in the specification from the matrix and seed spaces.
        :param: plan
        :param: matrix_solution
        :return: built plan
        """
        dict_space_item = {}
        for i_item, item in enumerate(self.solutions_collector.spec_with_circulation.items):
            item_space = []
            if item.category.name != "circulation":
                for j_space, space in enumerate(plan.mutable_spaces()):
                    if matrix_solution[i_item][j_space] == 1:
                        space.category = item.category
                        item_space.append(space)
                dict_space_item[item] = item_space

        # circulation case :
        for j_space, space in enumerate(plan.mutable_spaces()):
            if space.category.name == "seed":
                space.category = SPACE_CATEGORIES["circulation"]

        # To know the space associated with the item
        dict_items_space = {}
        for item in self.solutions_collector.spec_with_circulation.items:
            if item.category.name != "circulation":
                item_space = dict_space_item[item]
                if len(item_space) > 1:
                    space_ini = item_space[0]
                    item_space.remove(item_space[0])
                    i = 0
                    iter_max = len(item_space) ** 2
                    while (len(item_space) > 0) and i < iter_max:
                        i += 1
                        for space in item_space:
                            if space.adjacent_to(space_ini):
                                item_space.remove(space)
                                space_ini.merge(space)
                                plan.remove_null_spaces()
                                break
                    dict_items_space[space_ini] = item
                else:
                    if item_space:
                        dict_items_space[item_space[0]] = item

        # OPT-72: If we really want to enable it, it should be done through some execution context
        # parameters.
        # assert plan.check()

        return plan, dict_items_space

    def solution_research(self, show=False) -> Optional[List['Solution']]:
        """
        Looks for all possible solutions then find the three best solutions
        :return: None
        """

        self.manager.solver.solve()

        if len(self.manager.solver.solutions) == 0:
            logging.warning(
                "SpacePlanner : solution_research : Plan without space planning solution")
        else:
            logging.info("SpacePlanner : solution_research : Plan with {0} solutions".format(
                len(self.manager.solver.solutions)))
            if len(self.manager.solver.solutions) > 0:
                for i, sol in enumerate(self.manager.solver.solutions):
                    plan_solution = self.spec.plan.clone()
                    plan_solution, dict_space_item = self._rooms_building(plan_solution, sol)
                    solution_spec = Specification('Solution'+str(i)+'Specification', plan_solution)
                    for current_item in self.spec.items:
                        if current_item not in dict_space_item.values():
                            # entrance case
                            if current_item.category.name == "entrance":
                                for space, item in dict_space_item.items():
                                    if "frontDoor" in space.components_category_associated():
                                        item.min_size.area += current_item.min_size.area
                                        item.max_size.area += current_item.max_size.area

                    for item in self.spec.items:
                        if item in dict_space_item.values():
                            solution_spec.add_item(item)

                    solution_spec.plan.mesh.compute_cache()
                    self.solutions_collector.add_solution(solution_spec, dict_space_item)
                    logging.debug(plan_solution)

                    if show:
                        plan_solution.plot()

        return []

    def apply_to(self, spec: 'Specification', max_nb_solutions: int) -> List['Solution']:
        """
        Runs the space planner
        :param spec:
        :param max_nb_solutions
        :return: SolutionsCollector
        """
        self.solutions_collector = SolutionsCollector(spec, max_nb_solutions)
        self.spec = self.solutions_collector.spec_without_circulation

        self.spec.plan.mesh.compute_cache()
        self._plan_cleaner()
        logging.debug(self.spec)

        self.manager = ConstraintsManager(self)

        self.solution_research()

        self.solutions_collector.space_planner_best_results()
        print("SETUP SPACE PLANNER AREA", int(sum(item.required_area for item in self.spec.items)))
        print("PLAN AREA", self.spec.plan.indoor_area)

        return self.solutions_collector.best_solutions


standard_space_planner = SpacePlanner("standard")

SPACE_PLANNERS = {
    "standard_space_planner": standard_space_planner
}

if __name__ == '__main__':
    import libs.io.reader as reader
    from libs.modelers.grid import GRIDS
    from libs.modelers.seed import SEEDERS

    import argparse
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--plan_index", help="choose plan index",
                        default=0)
    #logging.getLogger().setLevel(logging.DEBUG)
    args = parser.parse_args()
    plan_index = int(args.plan_index)


    def space_planning():
        """
        Test
        :return:
        """
        #input_file = reader.get_list_from_folder(DEFAULT_BLUEPRINT_INPUT_FOLDER)[plan_index]
        input_file = "013.json"
        t00 = time.process_time()
        plan = reader.create_plan_from_file(input_file)
        logging.info("input_file %s", input_file)
        # print("input_file", input_file, " - area : ", plan.indoor_area)
        logging.debug(("P2/S ratio : %i", round(plan.indoor_perimeter ** 2 / plan.indoor_area)))

        GRIDS['002'].apply_to(plan)
        SEEDERS["directional_seeder"].apply_to(plan)

        plan.plot()
        # print(list(space.components_category_associated() for space in plan.mutable_spaces()))
        # print(list(space.cached_area() for space in plan.mutable_spaces()))

        input_file_setup = input_file[:-5] + "_setup0.json"
        spec = reader.create_specification_from_file(input_file_setup)
        logging.debug(spec)
        spec.plan = plan
        spec.plan.remove_null_spaces()

        logging.debug("number of mutables spaces, %i",
                      len([space for space in spec.plan.spaces if space.mutable]))

        t0 = time.process_time()
        space_planner = SPACE_PLANNERS["standard_space_planner"]
        best_solutions = space_planner.apply_to(spec, 3)
        logging.debug(space_planner.spec)
        logging.debug("space_planner time : %f", time.process_time() - t0)
        # surfaces control
        logging.debug("PLAN AREA : %i", int(space_planner.spec.plan.indoor_area))
        logging.debug("Setup AREA : %i",
                      int(sum(item.required_area for item in space_planner.spec.items)))
        logging.debug("Setup max AREA : %i", int(sum(item.max_size.area
                                                     for item in space_planner.spec.items)))
        logging.debug("Setup min AREA : %i", int(sum(item.min_size.area
                                                     for item in space_planner.spec.items)))
        plan_ratio = round(space_planner.spec.plan.indoor_perimeter
                           ** 2 / space_planner.spec.plan.indoor_area)
        logging.debug("PLAN Ratio : %i", plan_ratio)
        logging.debug("space_planner time : ", time.process_time() - t0)
        logging.debug("number of solutions : ", len(space_planner.solutions_collector.solutions))
        logging.debug("solution_research time: %f", time.process_time() - t0)
        logging.debug(best_solutions)

        # Output
        if best_solutions:
            for sol in best_solutions:
                sol.spec.plan.plot()
                logging.debug(sol, sol.space_planning_score)
                for space in sol.spec.plan.mutable_spaces():
                    logging.debug(space.category.name, " : ", space.cached_area())
                solution_dict = writer.generate_output_dict_from_file(input_file, sol)
                writer.save_json_solution(solution_dict, sol.id)

        # shuffle
        # if best_solutions:
        #     for sol in best_solutions:
        #         SHUFFLES['square_shape_shuffle_rooms'].run(sol.plan, show=True)
        #         sol.plan.plot()

        logging.debug("total time :", time.process_time() - t00)


    def space_planning_nico():
        """
        Test
        :return:
        """
        input_file = "001.json"
        t00 = time.process_time()
        plan = reader.create_plan_from_file(input_file)
        logging.info("input_file %s", input_file)
        # print("input_file", input_file, " - area : ", plan.indoor_area)
        logging.debug(("P2/S ratio : %i", round(plan.indoor_perimeter ** 2 / plan.indoor_area)))

        plan_name = None
        if plan_index < 10:
            plan_name = '00' + str(plan_index)
        elif 10 <= plan_index < 100:
            plan_name = '0' + str(plan_index)

        # plan_name = '007'

        try:
            new_serialized_data = reader.get_plan_from_json(plan_name + ".json")
            plan = Plan(plan_name).deserialize(new_serialized_data)
        except FileNotFoundError:
            plan = reader.create_plan_from_file(plan_name + ".json")
            GRIDS["optimal_finer_grid"].apply_to(plan)
            SEEDERS["directional_seeder"].apply_to(plan)
            writer.save_plan_as_json(plan.serialize(), plan_name + ".json")

        plan.remove_null_spaces()
        plan.plot()

        input_file_setup = plan_name + "_setup0.json"
        spec = reader.create_specification_from_file(input_file_setup)
        logging.debug(spec)
        spec.plan = plan
        spec.plan.remove_null_spaces()

        logging.debug("number of mutables spaces, %i",
                      len([space for space in spec.plan.spaces if space.mutable]))

        t0 = time.process_time()
        space_planner = SPACE_PLANNERS["standard_space_planner"]
        best_solutions = space_planner.apply_to(spec, 3)
        logging.debug(space_planner.spec)
        logging.debug("space_planner time : %f", time.process_time() - t0)
        # surfaces control
        logging.debug("PLAN AREA : %i", int(space_planner.spec.plan.indoor_area))
        logging.debug("Setup AREA : %i",
                      int(sum(item.required_area for item in space_planner.spec.items)))
        logging.debug("Setup max AREA : %i", int(sum(item.max_size.area
                                                     for item in space_planner.spec.items)))
        logging.debug("Setup min AREA : %i", int(sum(item.min_size.area
                                                     for item in space_planner.spec.items)))
        plan_ratio = round(space_planner.spec.plan.indoor_perimeter
                           ** 2 / space_planner.spec.plan.indoor_area)
        logging.debug("PLAN Ratio : %i", plan_ratio)
        logging.debug("solution_research time : ", time.process_time() - t0)
        logging.debug("number of solutions : ", len(space_planner.solutions_collector.solutions))
        logging.debug("solution_research time: %f", time.process_time() - t0)
        logging.debug(best_solutions)

        # Output
        if best_solutions:
            for sol in best_solutions:
                sol.spec.plan.plot()
                logging.debug(sol, sol.space_planning_score)
                for space in sol.spec.plan.mutable_spaces():
                    logging.debug(space.category.name, " : ", space.cached_area())
                solution_dict = writer.generate_output_dict_from_file(input_file, sol)
                writer.save_json_solution(solution_dict, sol.id)

        logging.debug("total time :", time.process_time() - t00)


    space_planning()
    # space_planning_nico()
