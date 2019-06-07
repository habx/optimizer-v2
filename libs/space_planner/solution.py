# coding=utf-8
"""
Solution collector module
Creates the following classes:
• SolutionsCollector: finds the best rooms layouts in given solution list
• Solution : rooms layout solution
TODO : fusion of the entrance for small apartment untreated

"""
from typing import List, Dict
from libs.specification.specification import Specification, Item
from libs.plan.plan import Plan, Space
from libs.space_planner.circulation import Circulator, CostRules
from libs.space_planner.constraints_manager import WINDOW_ROOMS
import logging

CORRIDOR_SIZE = 120
SQM = 10000


class SolutionsCollector:
    """
    Solutions Collector class
    """

    def __init__(self, spec: 'Specification', max_solutions: int):
        self.solutions: List['Solution'] = []
        self.spec = spec
        self.max_results = max_solutions

    def add_solution(self, plan: 'Plan', dict_items_spaces: Dict['Item', 'Space']) -> None:
        """
        creates and add plan solution to the list
        :param: plan
        :return: None
        """
        sol = Solution(self, plan, dict_items_spaces, len(self.solutions))
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

        def product(dist_list):
            result = 1
            for x in dist_list:
                result *= x
            return result

        best_sol_list = list()
        best_sol_list.append(self.solutions[index_best_sol])
        best_sol = self.solutions[index_best_sol]
        dist_from_best_sol = self.distance_from_all_solutions(best_sol)
        distance_from_results = [dist_from_best_sol]

        for i in range(self.max_results-1):
            current_score = None
            index_current_sol = None
            for i_sol in range(len(self.solutions)):
                current_distance_from_results = product([list_dist[i_sol]
                                                         for list_dist in distance_from_results])
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
                distance_from_results.append(dist_from_current_sol)
            else:
                break

        return best_sol_list

    def results(self) -> List['Solution']:
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
            list_scores.append(solution.score)

        # Choose the best solution :
        best_score = max(list_scores)
        index_best_sol = list_scores.index(best_score)
        logging.debug("SolutionsCollector : Best solution : index : %i, score : %f", index_best_sol,
                      best_score)

        best_sol_list = self.compute_results(list_scores, index_best_sol)

        return best_sol_list


class Solution:
    """
    Solution Class
    item layout solution in a given plan
    """

    def __init__(self,
                 collector: 'SolutionsCollector',
                 plan: 'Plan',
                 dict_items_spaces: Dict['Item', 'Space'],
                 _id: int):
        self._id = _id
        self.collector = collector
        self.plan = plan
        self.plan.name = self.plan.name + "_Solution_Id" + str(self._id)
        self.items_spaces: Dict['Item', 'Space'] = dict_items_spaces
        self.score = None
        self._score()
        self.compute_cache()


    def __repr__(self):
        output = 'Solution Id' + str(self._id)
        return output

    def compute_cache(self):
        """
        Computes the cached values for area / length of the mesh elements
        :return:
        """
        for space in self.plan.mutable_spaces():
            space._cached_immutable_components = space.immutable_components()

    @property
    def id(self):
        """
        Returns the id of the solution
        :return:
        """
        return self._id

    def init_items_spaces(self):
        """
        Dict item --> space initialization
        """
        for item in self.collector.spec.items:
            for space in self.plan.mutable_spaces():
                if item.category == space.category:
                    self.items_spaces[item] = space

    def get_rooms(self, category_name: str) -> ['Space']:
        """
        Retrieves all spaces corresponding to the category_name
        :param category_name: str
        :return: ['Spaces']
        """
        rooms_list = []
        for space in self.plan.mutable_spaces():
            if space.category.name == category_name:
                rooms_list.append(space)

        return rooms_list

    def _area_score(self) -> float:
        """
        Area score
        :return: score : float
        """
        good_overflow_categories = ["living", "livingKitchen", "dining"]

        area_score = 0
        area_penalty = 0
        nbr_rooms = 0
        for item, space in self.items_spaces.items():
            nbr_rooms += 1
            # Min < SpaceArea < Max
            if item.min_size.area <= space.cached_area() <= item.max_size.area:
                item_area_score = 100
            # good overflow
            elif (item.max_size.area < space.cached_area() and
                  space.category.name in good_overflow_categories):
                item_area_score = 100
            # overflow
            else:
                item_area_score = (100 - abs(item.required_area - space.cached_area()) /
                                   item.required_area * 100)
                if space.category.name == "entrance":
                    if space.cached_area() < 15000:
                        area_penalty += 2
                    elif space.cached_area() > item.required_area:
                        area_penalty += 1
                elif space.category.name == "toilet":
                    if space.cached_area() < 10000:
                        area_penalty += 2
                    elif space.cached_area() > item.max_size.area:
                        area_penalty += 3
                elif space.category.name == "bathroom":
                    if space.cached_area() < 20000:
                        area_penalty += 2
                elif space.category.name == "bedroom":
                    if space.cached_area() < 90000:
                        area_penalty += 2
                elif space.category.name == "circulation":
                    if space.cached_area() > item.max_size.area:
                        area_penalty += 1

            # Area score
            area_score += item_area_score
            logging.debug("Solution %i: Area score : %f, room : %s", self._id, item_area_score,
                          item.id)

        area_score = round(area_score / nbr_rooms, 2) - area_penalty * 20
        logging.debug("Solution %i: Area score : %f", self._id, area_score)
        return area_score

    def _shape_score(self) -> float:
        """
        Shape score
        Related with difference between a room and its boundary box
        :return: score : float
        """
        shape_score = 0
        logging.debug("Solution %i: P2/A", self._id)
        nbr_spaces = 0
        for item, space in self.items_spaces.items():
            if item.category.name in ["toilet", "bathroom"]:
                logging.debug("room %s: P2/A : %i", item.id,
                              int((space.perimeter_without_duct *
                                   space.perimeter_without_duct) / space.cached_area()))
            else:
                logging.debug("room %s: P2/A : %i", item.id,
                              int((space.perimeter * space.perimeter) / space.cached_area()))
            area = space.cached_area()
            box = space.bounding_box()
            difference = (box[0] * box[1] - area)
            item_shape_score = min(100.0, 100.0 - (difference / (2 * area)) * 100)
            logging.debug("Solution %d: Shape score : %f, room : %s", self._id, item_shape_score,
                          item.category.name)

            shape_score += item_shape_score
            nbr_spaces += 1
        shape_score = shape_score / nbr_spaces
        logging.debug("Solution %d: Shape score : %f", self._id, shape_score)

        return shape_score

    def _good_size_bonus(self) -> float:
        """
        Good ordering items size bonus
        :return: score : float
        """
        for item1 in self.collector.spec.items:
            for item2 in self.collector.spec.items:
                if (item1 != item2 and item1.category.name not in ["entrance", "circulation"] and
                        item2.category.name not in ["entrance", "circulation"]):
                    if (item1.required_area < item2.required_area and
                            self.items_spaces[item1].cached_area() > self.items_spaces[item2].cached_area()):
                        logging.debug("Solution %i: Size bonus : %i", self._id, 0)
                        return 0
        logging.debug("Solution %i: Size bonus : %i", self._id, 10)
        return 10

    def _windows_good_distribution_bonus(self) -> float:
        """
        Good ordering windows area bonus
        :return: score : float
        """
        item_windows_area = {}
        for item in self.items_spaces:
            windows_area = 0
            for component in self.items_spaces[item].immutable_components():
                if component.category.name == "window":
                    windows_area += component.length * 100
                elif component.category.name == "doorWindow":
                    windows_area += component.length * 200
            item_windows_area[item.id] = windows_area

        for item1 in self.collector.spec.items:
            for item2 in self.collector.spec.items:
                if (item1.required_area < item2.required_area
                        and item1.category.name in WINDOW_ROOMS
                        and item2.category.name in WINDOW_ROOMS):
                    if item_windows_area[item1.id] > item_windows_area[item2.id]:
                        logging.debug("Solution %i: Windows bonus : %i", self._id, 0)
                        return 0
        logging.debug("Solution %i: Windows bonus : %i", self._id, 10)
        return 10

    def _entrance_bonus(self) -> float:
        """
        Entrance bonus
        :return: score : float
        """
        if (self.collector.spec.typology > 2
                and [item for item in self.items_spaces if item.category.name == "entrance"]):
            return 10
        elif (self.collector.spec.typology <= 2
              and [item for item in self.items_spaces if item.category.name == "entrance"]):
            return -10
        return 0

    def _externals_spaces_bonus(self) -> float:
        """
        Good ordering externals spaces size bonus
        :return: score : float
        """
        for item1 in self.collector.spec.items:
            for item2 in self.collector.spec.items:
                if (item1 != item2 and self.items_spaces[item1].connected_spaces()
                        and self.items_spaces[item2].connected_spaces()):
                    item1_ext_spaces_area = sum([ext_space.cached_area()
                                                 for ext_space in
                                                 self.items_spaces[item1].connected_spaces()
                                                 if ext_space.category.external])
                    item2_ext_spaces_area = sum([ext_space.cached_area()
                                                 for ext_space in
                                                 self.items_spaces[item1].connected_spaces()
                                                 if ext_space.category.external])

                    if (item1.required_area < item2.required_area and
                            item1_ext_spaces_area > item2_ext_spaces_area):
                        logging.debug("Solution %i: External spaces bonus : %i", self._id, 0)
                        return 0
        logging.debug("Solution %i: External spaces : %i", self._id, 10)
        return 10

    def _circulation_penalty(self) -> float:
        """
        Circulation penalty
        :return: score : float
        """
        circulator = Circulator(plan=self.plan, spec=self.collector.spec, cost_rules=CostRules)
        circulator.connect()
        cost = circulator.cost
        circulation_penalty = 0

        # NOTE : what a weird thing to do (can't we just get the cost right from the start ?)
        if cost > CostRules.water_room_less_than_two_ducts.value:
            circulation_penalty += 100
        elif cost > CostRules.window_room_default.value:
            circulation_penalty += 50
        elif cost > CostRules.water_room_default.value:
            circulation_penalty += 30
        elif cost - (self.collector.spec.typology - 1) * 300 > 0:
            circulation_penalty += 5
        logging.debug("Solution %i: circulation penalty : %i", self._id, circulation_penalty)

        return circulation_penalty

    def _night_and_day_score(self) -> float:
        """
        Night and day score
        day / night distribution of rooms
        :return: score : float
        """
        first_level = self.plan.first_level
        day_list = ["living", "kitchen", "livingKitchen", "dining"]

        night_list = ["bedroom", "bathroom", "laundry"]

        day_polygon_list = []
        night_polygon_list = []
        for i_floor in range(self.plan.floor_count):
            day_polygon_list.append(None)
            night_polygon_list.append(None)

        for item in self.items_spaces:
            associated_space = self.items_spaces[item]
            level = associated_space.floor.level
            # Day
            if (item.category.name in day_list
                    or (item.category.name == "toilet"
                        and associated_space == self.get_rooms("toilet")[0])):
                if day_polygon_list[level - first_level] is None:
                    day_polygon_list[level - first_level] = associated_space.as_sp
                else:
                    day_polygon_list[level - first_level] = day_polygon_list[
                        level - first_level].union(associated_space.as_sp)

            # Night
            elif (item.category.name in night_list or
                  (item.category.name == "toilet" and
                   associated_space != self.get_rooms("toilet")[0])):
                if night_polygon_list[level - first_level] is None:
                    night_polygon_list[level - first_level] = associated_space.as_sp
                else:
                    night_polygon_list[level - first_level] = night_polygon_list[
                        level - first_level].union(associated_space.as_sp)

        number_of_day_level = 0
        number_of_night_level = 0
        day_polygon = None
        night_polygon = None
        for i_floor in range(self.plan.floor_count):
            if day_polygon_list[i_floor] is not None:
                number_of_day_level += 1
                day_polygon = day_polygon_list[i_floor]
            if night_polygon_list[i_floor] is not None:
                number_of_night_level += 1
                night_polygon = night_polygon_list[i_floor]

        # groups of rooms
        groups_score = 100
        if number_of_day_level > 1:
            groups_score -= 50
        elif self.plan.floor_count < 2 and day_polygon and day_polygon.geom_type != "Polygon":
            if [item for item in self.items_spaces if item.category.name == "entrance"]:
                day_polygon = day_polygon.union(
                    self.get_rooms("entrance")[0].as_sp.buffer(1))
            if day_polygon.geom_type != "Polygon":
                groups_score -= 50

        if number_of_night_level > 1:
            if self.collector.spec.typology <= 2 or self.collector.spec.number_of_items < 6:
                groups_score -= 50
            else:
                groups_score -= 25
        if self.plan.floor_count < 2 and night_polygon and night_polygon.geom_type != "Polygon":
            if [item for item in self.items_spaces if item.category.name == "entrance"]:
                night_polygon_with_entrance = night_polygon.union(
                    self.get_rooms("entrance")[0].as_sp.buffer(CORRIDOR_SIZE))
            else:
                night_polygon_with_entrance = night_polygon
            if night_polygon_with_entrance.geom_type != "Polygon":
                if ((len(night_polygon) > 2 and len(night_polygon_with_entrance) > 2)
                        or (self.collector.spec.typology <= 2
                            or self.collector.spec.number_of_items < 6)):
                    groups_score -= 50
                else:
                    groups_score -= 25

        logging.debug("Solution %i: Night and day score : %i", self._id, groups_score)
        return groups_score

    def _position_score(self) -> float:
        """
        Position room score :
        - one of the toilets must be near the entrance
        - bedrooms must be near a bathroom
        - toilets and bathrooms must be accessible from corridors or the entrance
        - the living must be near the entrance
        :return: score : float
        """

        position_score = 0
        nbr_room_position_score = 0
        front_door = self.plan.front_door()
        corridor = None  # TODO
        for item in self.items_spaces:
            space = self.items_spaces[item]
            item_position_score = 0
            if item.category.name == "toilet" and space == self.get_rooms("toilet")[0]:
                nbr_room_position_score += 1
                # distance from the entrance
                if self.plan.front_door().floor == space.floor:
                    # distance from the entrance
                    plan_area = self.collector.spec.plan.area
                    criteria = plan_area ** 0.5
                    distance_toilet_fd = space.distance_to_linear(front_door, "min")
                    if distance_toilet_fd < criteria:
                        item_position_score = (criteria - distance_toilet_fd) * 100 / criteria
            elif item.category.name == "bathroom":
                nbr_room_position_score += 1
                # non adjacent bathroom / bathroom
                for item_test in self.items_spaces:
                    if (item_test.category.name == "bathroom" and
                            self.items_spaces[item_test].floor == space.floor):
                        if space.adjacent_to(self.items_spaces[item_test]):
                            item_position_score = 0
                            break
            elif item.category.name == "bedroom":
                nbr_room_position_score += 1
                # distance from a bedroom / bathroom
                for item_test in self.items_spaces:
                    if (item_test.category.name in ["bathroom", "circulation"] and
                            self.items_spaces[item_test].floor == space.floor):
                        if space.adjacent_to(self.items_spaces[item_test]):
                            item_position_score = 100
                            break
            if item.category.name == "toilet" or item.category.name == "bathroom":
                # could be private
                private = False
                for circulation_space in self.plan.circulation_spaces():
                    if circulation_space.adjacent_to(space):
                        private = True
                        item_position_score += 50
                        break
                if not private and corridor:
                    if not corridor.adjacent_to(space):
                        item_position_score -= 50
            elif item.category.name == "living" or item.category.name == "livingKitchen":
                nbr_room_position_score += 1
                if "frontDoor" in space.components_category_associated():
                    item_position_score = 100
                else:
                    # distance from the entrance
                    if space.distance_to_linear(front_door, "min") < CORRIDOR_SIZE * 2:
                        item_position_score = 100

            logging.debug("Solution %i: Position score : %i, room : %s, %f", self._id,
                          item_position_score, item.category.name, nbr_room_position_score)
            position_score = position_score + item_position_score

        position_score = position_score / nbr_room_position_score
        if len(self.get_rooms("toilet")) > 1 and self.plan.floor_count > 1:
            if self.get_rooms("toilet")[0].floor != self.get_rooms("toilet")[1]:
                position_score += 100 / nbr_room_position_score
        logging.debug("Solution %i: Position score : %f", self._id, position_score)

        return position_score

    def _something_inside_score(self) -> float:
        """
        Something inside score
        duct or bearing wall or pillar or isolated room must not be inside a room
        :return: score : float
        """
        something_inside_score = 100
        for item in self.items_spaces:
            space = self.items_spaces[item]
            #  duct or pillar or small bearing wall
            if space.has_holes:
                logging.debug("Solution %i: Something Inside score : %f, room : %s, has_holes",
                              self._id, 0, item.category.name)
                return 0
            #  isolated room
            list_of_non_concerned_room = ["entrance", "circulation", "wardrobe", "study", "laundry",
                                          "misc"]
            convex_hull = space.as_sp.convex_hull
            for i_item in self.items_spaces:
                if (i_item != item and
                        i_item.category.name not in list_of_non_concerned_room and space.floor ==
                        self.items_spaces[i_item].floor):
                    if (self.items_spaces[i_item].as_sp.is_valid and convex_hull.is_valid and
                            (round((convex_hull.intersection(self.items_spaces[i_item].as_sp)).area)
                             == round(self.items_spaces[i_item].as_sp.area))):
                        logging.debug(
                            "Solution %i: Something Inside score : %f, room : %s - isolated room",
                            self._id, 0, i_item.category.name)
                        return 0
                    elif (self.items_spaces[i_item].as_sp.is_valid and convex_hull.is_valid and
                          (convex_hull.intersection(self.items_spaces[i_item].as_sp)).area > (
                                  space.cached_area() / 8)):
                        # Check i_item adjacency
                        other_room_adj = False
                        for j_item in self.items_spaces:
                            if j_item != i_item and j_item != item:
                                if self.items_spaces[i_item].adjacent_to(self.items_spaces[j_item]):
                                    other_room_adj = True
                                    break
                        if not other_room_adj:
                            logging.debug("Solution %i: Something Inside score : %f, room : %s, "
                                          "isolated room", self._id, something_inside_score,
                                          item.category.name)
                            return 0

        logging.debug("Solution %i: Something Inside score : %f", self._id, something_inside_score)
        return something_inside_score

    def _score(self) -> None:
        """
        Scoring
        compilation of different scores
        :return: score : float
        """
        solution_score = (self._area_score() + self._shape_score() + self._night_and_day_score()) / 3
        solution_score = (solution_score + self._good_size_bonus() + self._entrance_bonus())
        logging.debug("Solution %i: Final score : %f", self._id, solution_score)

        self.score = solution_score

    def distance(self, other_solution: 'Solution') -> float:
        """
        Distance with an other solution
        the distance is calculated from the groups of rooms day and night
        the inversion of two rooms within the same group gives a zero distance
        :return: distance : float
        """
        window_list = ["livingKitchen", "living", "kitchen", "dining", "bedroom", "study", "misc"]
        duct_list = ["bathroom", "toilet", "laundry", "wardrobe"]

        distance = 0
        for item in self.items_spaces:
            if len(self.items_spaces) != len(other_solution.items_spaces):
                distance += 1
            if item not in other_solution.items_spaces:
                continue
            space = self.items_spaces[item]
            other_solution_space = other_solution.items_spaces[item]
            if not space or not other_solution_space:
                continue
            if item.category.name in window_list:
                for comp in space.cached_immutable_components:
                    if (comp.category.name in ["window", "doorWindow"]
                            and (comp not in other_solution_space.cached_immutable_components)
                            and [other_space for other_space in other_solution.plan.get_spaces()
                                 if (comp in other_space.cached_immutable_components
                                     and other_space.category.name == space.category.name )] == []):
                        distance += 1
            elif item.category.name in duct_list:
                for comp in space.cached_immutable_components:
                    if (comp.category.name == "duct"
                            and comp not in other_solution_space.cached_immutable_components
                            and [other_space for other_space in other_solution.plan.get_spaces()
                                 if (comp in other_space.cached_immutable_components
                                     and other_space.category.name == space.category.name)] == []):
                        distance += 1
        return distance