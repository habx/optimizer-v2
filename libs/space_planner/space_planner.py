# coding=utf-8
"""
Space Planner module

The space planner finds the best rooms layouts according to a plan with given seed spaces
and a customer input setup

"""
import logging
from typing import List, Optional, Dict
from libs.specification.specification import Specification, Item
from libs.specification.size import Size
from libs.space_planner.solution import SolutionsCollector, Solution
from libs.plan.plan import Plan, Space
from libs.space_planner.constraints_manager import ConstraintsManager
from libs.plan.category import SPACE_CATEGORIES
from resources import DEFAULT_BLUEPRINT_INPUT_FOLDER
import libs.io.writer as writer

SQM = 10000


class SpacePlanner:
    """
    Space planner Class
    """

    def __init__(self, name: str, spec: 'Specification'):
        self.name = name
        self._init_spec(spec)
        self._plan_cleaner()
        logging.debug(self.spec)

        self.manager = ConstraintsManager(self)

        self.solutions_collector = SolutionsCollector(self.spec)

    def __repr__(self):
        output = "SpacePlanner" + self.name
        return output

    def _init_spec(self, spec: 'Specification') -> None:
        """
        change reader specification :
        living + kitchen : opensOn --> livingKitchen
        area convergence
        :return: None
        """
        space_planner_spec = Specification('SpacePlannerSpecification', spec.plan)

        # entrance
        size_min = Size(area=2 * SQM)
        size_max = Size(area=5 * SQM)
        new_item = Item(SPACE_CATEGORIES["entrance"], "s", size_min, size_max)
        space_planner_spec.add_item(new_item)

        for item in spec.items:
            if item.category.name == "circulation":
                continue
            elif ((item.category.name != "living" or "kitchen" not in item.opens_on) and
                  (item.category.name != "kitchen" or len(item.opens_on) == 0)):
                space_planner_spec.add_item(item)
            elif item.category.name == "living" and "kitchen" in item.opens_on:
                kitchens = spec.category_items("kitchen")
                for kitchen_item in kitchens:
                    if "living" in kitchen_item.opens_on:
                        size_min = Size(area=(kitchen_item.min_size.area + item.min_size.area))
                        size_max = Size(area=(kitchen_item.max_size.area + item.max_size.area))
                        opens_on = item.opens_on.remove("kitchen")
                        new_item = Item(SPACE_CATEGORIES["livingKitchen"], item.variant, size_min,
                                        size_max, opens_on, item.linked_to)
                        space_planner_spec.add_item(new_item)

        category_name_list = ["entrance", "toilet", "bathroom", "laundry", "dressing", "kitchen",
                              "living", "livingKitchen", "dining", "bedroom", "study", "misc",
                              "circulation"]
        space_planner_spec.init_id(category_name_list)

        # area
        invariant_categories = ["entrance", "wc", "bathroom", "laundry", "dressing", "circulation",
                                "misc"]
        invariant_area = sum(item.required_area for item in space_planner_spec.items
                             if item.category.name in invariant_categories)
        coeff = (int(space_planner_spec.plan.indoor_area - invariant_area) / int(sum(
            item.required_area for item in space_planner_spec.items if
            item.category.name not in invariant_categories)))

        for item in space_planner_spec.items:
            if item.category.name not in invariant_categories:
                item.min_size.area = round(item.min_size.area * coeff)
                item.max_size.area = round(item.max_size.area * coeff)
        logging.debug("SP - PLAN AREA : %i", int(space_planner_spec.plan.indoor_area))
        logging.debug("SP - Setup AREA : %i", int(sum(item.required_area
                                                      for item in space_planner_spec.items)))

        self.spec = space_planner_spec

    def _plan_cleaner(self, min_area: float = 100) -> None:
        """
        Plan cleaner for little spaces
        :return: None
        """
        self.spec.plan.remove_null_spaces()
        for space in self.spec.plan.spaces:
            if space.area < min_area:
                self.spec.plan.remove(space)

    def _rooms_building(self, plan: 'Plan', matrix_solution) -> ('Plan', Dict['Item', 'Space']):
        """
        Builds the rooms requested in the specification from the matrix and seed spaces.
        :param: plan
        :param: matrix_solution
        :return: built plan
        """
        dict_items_spaces = {}
        for i_item, item in enumerate(self.spec.items):
            item_space = []
            for j_space, space in enumerate(plan.mutable_spaces()):
                if matrix_solution[i_item][j_space] == 1:
                    space.category = item.category
                    item_space.append(space)
            dict_items_spaces[item] = item_space

        # circulation case :
        for j_space, space in enumerate(plan.mutable_spaces()):
            if space.category.name == "seed":
                space.category = SPACE_CATEGORIES["circulation"]

        # To know the space associated with the item
        dict_items_space = {}
        for item in self.spec.items:
            item_space = dict_items_spaces[item]
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
                dict_items_space[item] = space_ini
            else:
                if item_space:
                    dict_items_space[item] = item_space[0]

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
                    plan_solution, dict_items_spaces = self._rooms_building(plan_solution, sol)
                    self.solutions_collector.add_solution(plan_solution, dict_items_spaces)
                    logging.debug(plan_solution)

                    if show:
                        plan_solution.plot()

                best_sol = self.solutions_collector.best()
                for sol in best_sol:
                    logging.debug(sol)
                    if show:
                        sol.plan.plot(save=True)
                return best_sol

        return []


if __name__ == '__main__':
    import libs.io.reader as reader
    from libs.modelers.grid import GRIDS
    from libs.modelers.seed import SEEDERS

    import argparse
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--plan_index", help="choose plan index",
                        default=0)
    logging.getLogger().setLevel(logging.DEBUG)
    args = parser.parse_args()
    plan_index = int(args.plan_index)


    def space_planning():
        """
        Test
        :return:
        """
        input_file = reader.get_list_from_folder(DEFAULT_BLUEPRINT_INPUT_FOLDER)[plan_index]
        # input_file = "001.json"
        t00 = time.process_time()
        plan = reader.create_plan_from_file(input_file)
        logging.info("input_file %s", input_file)
        # print("input_file", input_file, " - area : ", plan.indoor_area)
        logging.debug(("P2/S ratio : %i", round(plan.indoor_perimeter ** 2 / plan.indoor_area)))

        GRIDS['optimal_grid'].apply_to(plan)

        nbr_grid_cells = 0
        for space in plan.spaces:
            if space.category.name == "empty":
                nbr_grid_cells += len(list(space.faces))
        logging.debug("nbr_grid_cells : ", nbr_grid_cells)

        if nbr_grid_cells > 25:
            SEEDERS["simple_seeder"].apply_to(plan)
        else:
            new_space_list = []
            for space in plan.spaces:
                if space.category.name == "empty":
                    for face in space.faces:
                        new_space = Space(plan, space.floor, face.edge, SPACE_CATEGORIES["seed"])
                        new_space_list.append(new_space)
            has_empty_space = True
            while has_empty_space:
                has_empty_space = False
                for space in plan.spaces:
                    if space.category.name == "empty":
                        plan.remove(space)
                        has_empty_space = True
                plan.remove_null_spaces()

        plan.plot()
        # print(list(space.components_category_associated() for space in plan.mutable_spaces()))
        # print(list(space.area for space in plan.mutable_spaces()))

        input_file_setup = input_file[:-5] + "_setup0.json"
        spec = reader.create_specification_from_file(input_file_setup)
        logging.debug(spec)
        spec.plan = plan
        spec.plan.remove_null_spaces()

        logging.debug("number of mutables spaces, %i",
                      len([space for space in spec.plan.spaces if space.mutable]))

        t0 = time.process_time()
        space_planner = SpacePlanner("test", spec)
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
        t1 = time.process_time()
        best_solutions = space_planner.solution_research()
        logging.debug("solution_research time : ", time.process_time() - t1)
        logging.debug("number of solutions : ", len(space_planner.solutions_collector.solutions))
        logging.debug("solution_research time: %f", time.process_time() - t1)
        logging.debug(best_solutions)

        # Output
        for sol in best_solutions:
            sol.plan.plot()
            logging.debug(sol, sol.score)
            for space in sol.plan.mutable_spaces():
                logging.debug(space.category.name, " : ", space.area)
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
        input_file = reader.get_list_from_folder(DEFAULT_BLUEPRINT_INPUT_FOLDER)[plan_index]
        # input_file = "001.json"
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

        plan_name = '011'

        try:
            new_serialized_data = reader.get_plan_from_json(plan_name + ".json")
            plan = Plan(plan_name).deserialize(new_serialized_data)
        except FileNotFoundError:
            plan = reader.create_plan_from_file(plan_name + ".json")
            GRIDS["optimal_finer_grid"].apply_to(plan)
            SEEDERS["trames_seeder"].apply_to(plan)
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
        space_planner = SpacePlanner("test", spec)
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
        t1 = time.process_time()
        best_solutions = space_planner.solution_research()
        logging.debug("solution_research time : ", time.process_time() - t1)
        logging.debug("number of solutions : ", len(space_planner.solutions_collector.solutions))
        logging.debug("solution_research time: %f", time.process_time() - t1)
        logging.debug(best_solutions)

        # Output
        for sol in best_solutions:
            sol.plan.plot()
            logging.debug(sol, sol.score)
            for space in sol.plan.mutable_spaces():
                logging.debug(space.category.name, " : ", space.area)
            solution_dict = writer.generate_output_dict_from_file(input_file, sol)
            writer.save_json_solution(solution_dict, sol.id)

        logging.debug("total time :", time.process_time() - t00)

        for sp in plan.spaces:
            print(sp, "sp.arae", sp.area,"sp.peroimeter",sp.perimeter)


    # space_planning()
    space_planning_nico()
