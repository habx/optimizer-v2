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
from copy import deepcopy

from sklearn.cluster import DBSCAN
import numpy as np

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

    def _rooms_building(self, plan: 'Plan',i: int, matrix_solution) -> ('Plan', Dict['Item', 'Space']):
        """
        Builds the rooms requested in the specification from the matrix and seed spaces.
        :param: plan
        :param: matrix_solution
        :return: built plan
        """
        new_spec = deepcopy(self.solutions_collector.spec_with_circulation)
        space_item = {}
        seed_space_to_remove = []
        for i_item, item in enumerate(new_spec.items):
            current_space = None
            if item.category.name != "circulation":
                for j_space, space in enumerate(plan.mutable_spaces()):
                    if matrix_solution[i_item][j_space] == 1:
                        if not current_space:
                            space.category = item.category
                            current_space = space
                            space_item[current_space] = item
                        else:
                            seed_space_to_remove.append(space)
                            for face_id in space.faces_id:
                                current_space.add_face_id(face_id)
                if current_space:
                    current_space.set_edges()

        while seed_space_to_remove:
            for space in plan.mutable_spaces():
                if space in seed_space_to_remove:
                    plan.remove(space)
                    seed_space_to_remove.remove(space)

        # circulation case :
        for j_space, space in enumerate(plan.mutable_spaces()):
            if space.category.name == "seed":
                space.category = SPACE_CATEGORIES["circulation"]

        # OPT-72: If we really want to enable it, it should be done through some execution context
        # parameters.
        # assert plan.check()

        solution_spec = Specification('Solution' + str(i) + 'Specification', plan)
        for current_item in new_spec.items:
            if current_item not in space_item.values():
                # entrance case
                if current_item.category.name == "entrance":
                    for space, item in space_item.items():
                        if "frontDoor" in space.components_category_associated():
                            item.min_size.area += current_item.min_size.area
                            item.max_size.area += current_item.max_size.area

        for item in new_spec.items:
            if item in space_item.values():
                solution_spec.add_item(item)

        solution_spec.plan.mesh.compute_cache()

        return solution_spec, space_item

    def solution_distance(self, sol1, sol2) -> float:

        distance = 0
        for i, sol1_line in enumerate(sol1):
            sol2_line = sol2[i]
            for j, sol1_el in enumerate(sol1_line):
                sol2_el = sol2_line[j]
                distance += abs(sol1_el-sol2_el)

        return distance

    def clustering_distance_matrix(self, input_solutions: []):
        """
        distance matrix between solutions for the clustering
        :param: solutions
        :return: distance matrix
        """
        # seed space coeff
        seed_space_coeff = []
        for space in self.spec.plan.mutable_spaces():
            coeff = 1
            compo = space.components_category_associated()
            if "duct" in compo or "frontDoor" in compo:
                coeff += 1
            if "window" in compo or "doorWindow" in compo:
                coeff += 2
            coeff = coeff*space.cached_area()/sum(space.area for space in self.spec.plan.mutable_spaces())
            seed_space_coeff.append(coeff)

        solutions = deepcopy(input_solutions)
        for i, i_sol in enumerate(solutions):
            for i_line, line in enumerate(i_sol):
                for a, el in enumerate(line):
                    solutions[i][i_line][a] = el * seed_space_coeff[a]

        # distance matrix
        # stats
        min_dist = 1000
        max_dist = 0
        dist_moy = 0

        matrix = []
        for i_sol in solutions:
            matrix.append([0] * len(solutions))
        for i, i_sol in enumerate(solutions):
            for j, j_sol in enumerate(solutions):
                if i < j:
                    distance = self.solution_distance(i_sol, j_sol)
                    matrix[i][j] = distance
                    matrix[j][i] = distance
                    # stats
                    if distance < min_dist:
                        min_dist = distance
                    if distance > max_dist:
                        max_dist = distance
                    dist_moy += distance
        dist_moy = dist_moy/((len(solutions)**2)/2-len(solutions))

        print("min_dist", min_dist)
        print("max_dist", max_dist)
        print("dist_moy", dist_moy)
        return matrix

    def solution_research(self, show=False):
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

            matrix = self.clustering_distance_matrix(self.manager.solver.solutions)
            X_matrix = np.array(matrix)
            db = DBSCAN(eps=0.5, min_samples=10, metric="precomputed", n_jobs=None).fit(X_matrix)
            print("labels_", db.labels_)
            labels = db.labels_

            # Number of clusters in labels, ignoring noise if present.
            print(set(labels))
            for i in set(labels):
                print("number of elements", i, list(labels).count(i))

            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise_ = list(labels).count(-1)

            print('Estimated number of clusters: %d' % n_clusters_)
            print('Estimated number of noise points: %d' % n_noise_)
            list_labels = list(set(labels))
            clustering_solutions = []
            for i, sol in enumerate(self.manager.solver.solutions):
                if labels[i] in list_labels:
                    print(i)
                    clustering_solutions.append(sol)
                    list_labels.remove(labels[i])
                if len(list_labels) == 0:
                    break

            print("len clustering_solutions", len(clustering_solutions))
            for i, sol in enumerate(clustering_solutions):
                print(sol)
                plan_solution = self.spec.plan.clone()
                solution_spec, dict_space_item = self._rooms_building(plan_solution, i, sol)
                self.solutions_collector.add_solution(solution_spec, dict_space_item)
                logging.debug(solution_spec.plan)

                if show:
                    plan_solution.plot()

        return self.solutions_collector.solutions

            # for i, sol in enumerate(self.manager.solver.solutions):
            #     #print(sol)
            #     plan_solution = self.spec.plan.clone()
            #     solution_spec, dict_space_item = self._rooms_building(plan_solution, i, sol)
            #     self.solutions_collector.add_solution(solution_spec, dict_space_item)
            #     logging.debug(solution_spec.plan)
            #
            #     if show:
            #         plan_solution.plot()

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

        clustering_solutions = self.solution_research()

        self.solutions_collector.space_planner_best_results()
        #self.solutions_collector.best_solutions = clustering_solutions

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
        input_file = "037.json"
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

        print("number of mutables spaces, %i",
                      len([space for space in spec.plan.spaces if space.mutable]))
        print("mutables spaces area",
              sum([space.area for space in spec.plan.spaces if space.mutable]))
        t0 = time.process_time()
        space_planner = SPACE_PLANNERS["standard_space_planner"]
        best_solutions = space_planner.apply_to(spec, 3)
        logging.debug(space_planner.spec)
        logging.debug("space_planner time : %f", time.process_time() - t0)
        # surfaces control
        print("PLAN AREA : %i", int(space_planner.spec.plan.indoor_area))
        print("Setup AREA : %i",
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
        print(best_solutions)

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
        #         SHUFFLES['square_shape_shuffle_rooms'].run(sol.spec.plan, show=True)
        #         sol.spec.plan.plot()

        print("total time :", time.process_time() - t00)


    def space_planning_nico():
        """
        Test
        :return:
        """
        input_file = "020.json"
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
