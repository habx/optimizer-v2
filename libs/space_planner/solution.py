# coding=utf-8
"""
Solution collector module
Creates the following classes:
• SolutionsCollector: finds the best rooms layouts in given solution list
• Solution : rooms layout solution
TODO : fusion of the entrance for small apartment untreated

"""
import multiprocessing
from typing import List, Dict
from libs.specification.specification import Specification, Item
from libs.plan.plan import Space
from libs.scoring.scoring import space_planning_scoring
import logging
import functools
import operator

CORRIDOR_SIZE = 120
SQM = 10000


class SolutionsCollector:
    """
    Solutions Collector class
    """

    def __init__(self, spec: 'Specification', max_solutions: int = 3, processes: int = 8):
        self.solutions: List['Solution'] = []
        self.best_solutions: List['Solution'] = []
        self.architect_plans: List['Solution'] = []
        self.spec = spec
        self.max_results = max_solutions
        self.processes = processes

    def add_solution(self, spec: 'Specification', dict_space_item: Dict['Space', 'Item']) -> None:
        """
        creates and add plan solution to the list
        :param: plan
        :return: None
        """
        sol = Solution(spec, dict_space_item, len(self.solutions))
        self.solutions.append(sol)

    def add_plan(self, spec: 'Specification', dict_space_item: Dict['Space', 'Item']) -> None:
        """
        creates and add plan solution to the list
        :param: plan
        :return: None
        """
        archi_plan = Solution(self, spec, dict_space_item, len(self.architect_plans))
        self.architect_plans.append(archi_plan)

    @property
    def solutions_distance_matrix(self) -> [float]:
        """
        Distance between all solutions of the solution collector
        """
        # Distance matrix
        distance_matrix = []
        for i in range(len(self.solutions)):
            distance_matrix.append([])
            for j in range(len(self.solutions)):
                distance_matrix[i].append(0)

        for i, sol1 in enumerate(self.solutions):
            for j, sol2 in enumerate(self.solutions):
                if i < j:
                    distance = sol1.distance(sol2)
                    distance_matrix[i][j] = distance
                    distance_matrix[j][i] = distance

        logging.debug("SolutionsCollector : Distance_matrix : {0}".format(distance_matrix))
        return distance_matrix

    def distance_from_all_solutions(self, sol: 'Solution') -> [float]:
        """
        Distance between all solutions of the given solution
        """

        # Distance array
        distance = []
        for i, sol1 in enumerate(self.solutions):
            dist = sol.distance(sol1)
            distance.append(dist)

        return distance

    def compute_results(self, list_scores, index_best_sol) -> List['Solution']:

        best_sol_list = [self.solutions[index_best_sol]]
        best_sol = self.solutions[index_best_sol]
        dist_from_best_sol = self.distance_from_all_solutions(best_sol)
        distance_from_results = [dist_from_best_sol]

        for i in range(self.max_results - 1):
            current_score = None
            index_current_sol = None
            for i_sol in range(len(self.solutions)):
                current_distance_from_results = functools.reduce(operator.mul, [list_dist[i_sol]
                                                                                for list_dist in
                                                                                distance_from_results])
                if ((current_score is None and current_distance_from_results > 0)
                        or (current_score is not None
                            and list_scores[
                                i_sol] * current_distance_from_results > current_score)):
                    index_current_sol = i_sol
                    current_score = list_scores[i_sol] * current_distance_from_results
            if current_score:
                best_sol_list.append(self.solutions[index_current_sol])
                logging.debug("SolutionsCollector : Second solution : index : %i, score : %f",
                              index_current_sol, current_score)
                current_sol = self.solutions[index_current_sol]
                dist_from_current_sol = self.distance_from_all_solutions(current_sol)
                distance_from_results.append(dist_from_current_sol)
            else:
                break

        return best_sol_list

    def space_planner_best_results(self) -> List['Solution']:
        """
        Find best solutions of the list
        the best solution is the one with the highest score
        the second solution has the best score of the solutions distant from a minimum distance
        of the first solution...
        :param: plan
        :return: best solutions
        """

        if not self.solutions:
            logging.warning("Solution : 0 solutions")
            return []

        list_solutions = list([solution for solution in self.solutions])
        pool = multiprocessing.Pool(self.processes)
        list_scores = pool.map(space_planning_scoring, list_solutions)

        # Choose the best solution :
        best_score = max(list_scores)
        index_best_sol = list_scores.index(best_score)
        logging.debug("SolutionsCollector : Best solution : index : %i, score : %f", index_best_sol,
                      best_score)

        self.best_solutions = self.compute_results(list_scores, index_best_sol)


class Solution:
    """
    Solution Class
    item layout solution in a given plan
    """

    def __init__(self,
                 spec: 'Specification',
                 dict_space_item: Dict['Space', 'Item'],
                 _id: int):
        self._id = _id
        self.spec = spec
        self.spec.plan.name = self.spec.plan.name + "_Solution_Id" + str(self._id)
        self.space_item: Dict['Space', 'Item'] = dict_space_item
        self.space_planning_score:float = None
        self.final_score:float = None
        self.final_score_components:Dict[str, float] = None
        self.compute_cache()

    def __repr__(self):
        output = 'Solution Id' + str(self._id)
        return output

    def compute_cache(self):
        """
        Computes the cached values for area / length of the mesh elements
        :return:
        """
        for space in self.spec.plan.mutable_spaces():
            space._cached_immutable_components = space.immutable_components()

    @property
    def id(self):
        """
        Returns the id of the solution
        :return:
        """
        return self._id

    def init_space_item(self):
        """
        Dict item --> space initialization
        """
        for item in self.spec.items:
            for space in self.spec.plan.mutable_spaces():
                if item.category == space.category:
                    self.space_item[space] = item

    def get_rooms(self, category_name: str) -> ['Space']:
        """
        Retrieves all spaces corresponding to the category_name
        :param category_name: str
        :return: ['Spaces']
        """
        rooms_list = []
        for space in self.spec.plan.mutable_spaces():
            if space.category.name == category_name:
                rooms_list.append(space)

        return rooms_list

    def distance(self, other_solution: 'Solution') -> float:
        """
        Distance with an other solution
        the distance is calculated from difference of fixed items distribution
        the inversion of two rooms within the same group gives a zero distance
        :return: distance : float
        """
        window_list = ["livingKitchen", "living", "kitchen", "dining", "bedroom", "study", "misc"]
        duct_list = ["bathroom", "toilet", "laundry", "wardrobe"]

        distance = 0
        if len(self.space_item) != len(other_solution.space_item):
            distance += 1
        for space, item in self.space_item.items():
            if item not in other_solution.space_item.values():
                continue
            other_solution_space = \
                [o_space for o_space, o_item in other_solution.space_item.items() if
                 o_item == item][0]
            if not space or not other_solution_space:
                continue
            if item.category.name in window_list:
                for comp in space.cached_immutable_components:
                    if (comp.category.name in ["window", "doorWindow"]
                            and (comp not in other_solution_space.cached_immutable_components)
                            and [other_space for other_space in
                                 other_solution.spec.plan.get_spaces()
                                 if (comp in other_space.cached_immutable_components
                                     and other_space.category.name == space.category.name)] == []):
                        distance += 1
            elif item.category.name in duct_list:
                for comp in space.cached_immutable_components:
                    if (comp.category.name == "duct"
                            and comp not in other_solution_space.cached_immutable_components
                            and [other_space for other_space in
                                 other_solution.spec.plan.get_spaces()
                                 if (comp in other_space.cached_immutable_components
                                     and other_space.category.name == space.category.name)] == []):
                        distance += 1
        return distance
