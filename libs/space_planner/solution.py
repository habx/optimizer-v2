# coding=utf-8
"""
Solution collector module
Creates the following classes:
• SolutionsCollector: finds the best rooms layouts in given solution list
• Solution : rooms layout solution
TODO : fusion of the entrance for small apartment untreated

"""
from typing import List, Dict, Optional
from libs.specification.specification import Specification, Item
from libs.plan.plan import Space
from libs.plan.plan import Plan
from libs.scoring.scoring import space_planning_scoring, initial_spec_adaptation, create_item_dict
import logging
import functools
import operator
from copy import deepcopy

CORRIDOR_SIZE = 120
SQM = 10000


class SolutionsCollector:
    """
    Solutions Collector class
    """

    def __init__(self, spec: 'Specification', max_solutions: int = 3):
        self._init_specifications(spec)
        self.max_results = max_solutions
        self.solutions: List['Solution'] = []
        self.best_solutions: List['Solution'] = []

    def _init_specifications(self, spec: 'Specification') -> None:
        """
        change reader specification :
        living + kitchen : opensOn --> livingKitchen
        area convergence
        :return: None
        """
        self.spec_with_circulation = initial_spec_adaptation(spec, spec.plan,
                                                             'SpecificationWithCirculation', True)
        self.spec_with_circulation.plan.mesh.compute_cache()

        self.spec_without_circulation= initial_spec_adaptation(spec, spec.plan,
                                                               'SpecificationWithoutCirculation',
                                                               False)
        self.spec_without_circulation .plan.mesh.compute_cache()

    def add_solution(self, spec: 'Specification', dict_space_item: Dict['Space', 'Item']) -> None:
        """
        creates and add plan solution to the list
        :param: plan
        :return: None
        """
        sol = Solution(spec, dict_space_item, len(self.solutions))
        self.solutions.append(sol)

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
        """
        Create the list of the best solutions
        :param list_scores:
        :param index_best_sol:
        :return:
        """
        best_sol_list = [self.solutions[index_best_sol]]
        best_sol = self.solutions[index_best_sol]
        dist_from_best_sol = self.distance_from_all_solutions(best_sol)
        dist_from_results = [dist_from_best_sol]

        for i in range(self.max_results-1):
            current_score = None
            index_current_sol = None
            for i_sol in range(len(self.solutions)):
                current_distance_from_results = functools.reduce(operator.mul,
                                                                 [list_dist[i_sol]
                                                                  for list_dist
                                                                  in dist_from_results])
                if ((current_score is None and current_distance_from_results > 0)
                        or (current_score is not None
                            and list_scores[i_sol]*current_distance_from_results > current_score)):
                    index_current_sol = i_sol
                    current_score = list_scores[i_sol]*current_distance_from_results
            if current_score:
                best_sol_list.append(self.solutions[index_current_sol])
                logging.debug("SolutionsCollector : Second solution : index : %i, score : %f",
                              index_current_sol, current_score)
                current_sol = self.solutions[index_current_sol]
                dist_from_current_sol = self.distance_from_all_solutions(current_sol)
                dist_from_results.append(dist_from_current_sol)
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

        list_scores = []
        for solution in self.solutions:
            solution.space_planning_score = space_planning_scoring(solution)
            list_scores.append(solution.space_planning_score)

        # Choose the best solution :
        best_score = max(list_scores)
        index_best_sol = list_scores.index(best_score)
        logging.debug("SolutionsCollector : Best solution : index : %i, score : %f", index_best_sol,
                      best_score)

        self.best_solutions = self.compute_results(list_scores, index_best_sol)


def spec_adaptation(solution: 'Solution', collector: 'SolutionsCollector'):
    """
    Modify the specification instance to insert circulation items
    :param solution:
    :param collector:
    :return:
    """
    circulation_spaces = [space for space in solution.spec.plan.mutable_spaces()
                          if space.category.name == "circulation"]
    circulation_item = deepcopy([item for item in collector.spec_with_circulation.items
                             if item.category.name == "circulation"][0])

    if circulation_spaces:
        for space in solution.spec.plan.mutable_spaces():
            if space.category.name == "circulation":
                solution.spec.add_item(circulation_item)

    # area
    invariant_categories = ["entrance", "toilet", "bathroom", "laundry", "wardrobe", "circulation"
                            "misc"]
    invariant_area = sum(item.required_area for item in solution.spec.items
                         if item.category.name in invariant_categories)
    mutable_spaces_area = sum([space.cached_area() for space in solution.spec.plan.mutable_spaces()])
    coeff = (int(mutable_spaces_area - invariant_area) / int(sum(
        item.required_area for item in solution.spec.items if
        item.category.name not in invariant_categories)))

    for item in solution.spec.items:
        if item.category.name not in invariant_categories:
            item.min_size.area = round(item.min_size.area * coeff)
            item.max_size.area = round(item.max_size.area * coeff)


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
        self.space_planning_score: Optional[float] = None
        self.final_score: Optional[float] = None
        self.final_score_components: Optional[Dict[str, float]] = None
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

    def serialize(self) -> dict:
        """
        Returns a dict containing the solution information that can be saved as a json
        :return:
        """
        output = {
            "id": self.id,
            "spec": self.spec.serialize(),
            "plan": self.spec.plan.serialize(),
            "space_item": {s.id: i.id for s, i in self.space_item.items()},
            "space_planning_score": self.space_planning_score,
            "final_score": self.final_score,
            "final_score_components": self.final_score_components
        }

        return output

    @classmethod
    def deserialize(cls, data: dict) -> 'Solution':
        """
        Returns a solution instance from the data specified
        :param data:
        :return:
        """
        _id = data["id"]
        spec = Specification.deserialize(data["spec"])
        plan = Plan().deserialize(data["plan"])
        spec.plan = plan
        space_item = {plan.get_space_from_id(int(space_id)): spec.get_item_from_id(item_id)
                      for space_id, item_id in data["space_item"].items()}
        solution = cls(spec, space_item, _id)
        solution.final_score_components = data["final_score_components"]
        solution.final_score = data["final_score"]
        solution.space_planning_score = data["space_planning_score"]

        return solution

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
            other_solution_space = [o_space for o_space, o_item in other_solution.space_item.items()
                                    if o_item == item][0]
            if not space or not other_solution_space:
                continue
            if item.category.name in window_list:
                for comp in space.cached_immutable_components:
                    if (comp.category.name in ["window", "doorWindow"]
                            and (comp not in other_solution_space.cached_immutable_components)
                            and [other_space for other_space
                                 in other_solution.spec.plan.get_spaces()
                                 if (comp in other_space.cached_immutable_components
                                     and other_space.category.name == space.category.name)] == []):
                        distance += 1
            elif item.category.name in duct_list:
                for comp in space.cached_immutable_components:
                    if (comp.category.name == "duct"
                            and comp not in other_solution_space.cached_immutable_components
                            and [other_space for other_space
                                 in other_solution.spec.plan.get_spaces()
                                 if (comp in other_space.cached_immutable_components
                                     and other_space.category.name == space.category.name)] == []):
                        distance += 1
        return distance

def reference_plan_solution(reference_plan:'Plan', setup_spec: 'Specification') -> Solution:
    """
    reference plan solution building
    :param reference_plan: Solution
    :param setup_spec: Specification
    :return: Solution
    """
    reference_plan.remove_null_spaces()
    if [space for space in reference_plan.spaces if space.category.name == "circulation"]:
        ref_plan_spec = initial_spec_adaptation(setup_spec, reference_plan, "ReferencePlanSpec",
                                                True)
    else:
        ref_plan_spec = initial_spec_adaptation(setup_spec, reference_plan, "ReferencePlanSpec",
                                                False)
    ref_plan_spec.plan = reference_plan
    ref_space_item = create_item_dict(ref_plan_spec, reference_plan)
    reference_plan.plot()

    return Solution(ref_plan_spec, ref_space_item, 99999)
