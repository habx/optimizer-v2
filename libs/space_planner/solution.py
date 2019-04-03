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
from libs.plan.plan import Plan, Space
from libs.space_planner.circulation import Circulator, COST_RULES
import logging

CORRIDOR_SIZE = 120


class SolutionsCollector:
    """
    Solutions Collector class
    """

    def __init__(self, spec: 'Specification'):
        self.solutions: List['Solution'] = []
        self.spec = spec

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

    def best(self) -> Optional[List['Solution']]:
        """
        Find best solutions of the list
        the best solution is the one with the highest score
        the second solution has the best score of the solutions distant from a minimum distance
        of the first solution
        the third solution has the best score of the solutions distant from a minimum distance
        of the first and second solution
        Just the best solution is given for small apartment
        :param: plan
        :return: best solutions
        """

        if not self.solutions:
            logging.warning("Solution : 0 solutions")
            return []

        best_sol_list = []

        list_scores = []
        for solution in self.solutions:
            list_scores.append(solution.score())

        # Choose the tree best distributions :
        best_score = max(list_scores)
        index_best_sol = list_scores.index(best_score)
        logging.debug("SolutionsCollector : Best solution : index : %i, score : %f", index_best_sol,
                      best_score)

        best_sol_list.append(self.solutions[index_best_sol])

        if self.spec.number_of_items <= 4:
            return best_sol_list

        best_sol = self.solutions[index_best_sol]
        dist_from_best_sol = self.distance_from_all_solutions(best_sol)

        second_score = None
        index_second_sol = None
        for i in range(len(self.solutions)):
            if dist_from_best_sol[i] > 25 and ((second_score is None) or
                                               list_scores[i] > second_score):
                index_second_sol = i
                second_score = list_scores[i]

        if second_score:
            best_sol_list.append(self.solutions[index_second_sol])
            logging.debug("SolutionsCollector : Second solution : index : %i, score : %f",
                          index_second_sol, second_score)

            index_third_sol = None
            third_score = None
            second_sol = self.solutions[index_second_sol]
            dist_from_second_sol = self.distance_from_all_solutions(second_sol)
            for i in range(len(self.solutions)):
                if (dist_from_best_sol[i] > 25 and dist_from_second_sol[i] > 25 and
                        (third_score is None or list_scores[i] > third_score)):
                    index_third_sol = i
                    third_score = list_scores[i]
            if third_score:
                best_sol_list.append(self.solutions[index_third_sol])
                logging.debug(" SolutionsCollector : Third solution : index %i, score : %f",
                              index_third_sol, third_score)

            best_distribution_list = [index_best_sol, index_second_sol, index_third_sol]
            for i in best_distribution_list:
                for j in best_distribution_list:
                    if i and j:
                        logging.debug(
                            "SolutionsCollector : Distance solutions : %i and %i : %f", i, j,
                            (self.solutions[i]).distance(self.solutions[j]))

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
        self.plan.name = self.plan.name[:-5] + "_Solution_Id" + str(self._id)
        self.items_spaces: Dict['Item', 'Space'] = dict_items_spaces

    def __repr__(self):
        output = 'Solution Id' + str(self._id)
        return output

    @property
    def get_id(self):
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
        good_overflow_categories = ["living", "dining"]

        area_score = 0
        area_penalty = 0
        nbr_rooms = 0
        for item in self.collector.spec.items:
            if item in self.items_spaces.keys():
                space = self.items_spaces[item]
                nbr_rooms += 1
                # Min < SpaceArea < Max
                if item.min_size.area <= space.area <= item.max_size.area:
                    item_area_score = 100
                # good overflow
                elif (item.max_size.area < space.area and
                      space.category.name in good_overflow_categories):
                    item_area_score = 100
                # overflow
                else:
                    item_area_score = (100 - abs(item.required_area - space.area) /
                                       item.required_area * 100)
                    if space.category.name == "entrance":
                        if space.area < 20000:
                            area_penalty += 1
                    elif space.category.name == "toilet":
                        if space.area < 10000:
                            area_penalty += 1
                        elif space.area > item.max_size.area:
                            area_penalty += 3
                    elif space.category.name == "bathroom" or space.category.name == "toiletBathroom":
                        if space.area < 20000:
                            area_penalty += 1
                    elif space.category.name == "bedroom":
                        if space.area < 75000:
                            area_penalty += 1

                # Area score
                area_score += item_area_score
                logging.debug("Solution %i: Area score : %f, room : %s", self._id, item_area_score,
                              item.id)

        for space in self.plan.spaces:
            if space.category.name == "circulationSpace":
                area_penalty += 1

        area_score = round(area_score / nbr_rooms, 2) - area_penalty * 20
        logging.debug("Solution %i: Area score : %f", self._id, area_score)
        return area_score

    def _shape_score(self) -> float:
        """
        Shape score
        Related with difference between a room and its convex hull
        :return: score : float
        """
        shape_score = 100
        logging.debug("Solution %i: P2/A", self._id)
        for item in self.items_spaces.keys():
            space = self.items_spaces[item]
            if item.category.name in ["toilet", "bathroom"]:
                logging.debug("room %s: P2/A : %i", item.id,
                              int((space.perimeter_without_duct *
                                   space.perimeter_without_duct)/space.area))
            else:
                logging.debug("room %s: P2/A : %i", item.id,
                              int((space.perimeter*space.perimeter)/space.area))
            sp_space = space.as_sp
            convex_hull = sp_space.convex_hull
            if convex_hull.is_valid and sp_space.is_valid:
                outside = convex_hull.difference(sp_space)
                item_shape_score = min(100, 100 - (
                        (outside.area - sp_space.area / 7) / (sp_space.area / 4) * 100))
                logging.debug(
                    "Solution %i: Shape score : %f, room : %s", self._id, item_shape_score,
                    item.category.name)
            else:
                logging.warning("Solution %i: Invalid shapely space")
                item_shape_score = 100

            shape_score = min(item_shape_score, shape_score)
        logging.debug("Solution %i: Shape score : %f", self._id, shape_score)

        return shape_score

    def _good_size_bonus(self) -> float:
        """
        Good ordering items size bonus
        :return: score : float
        """
        for item1 in self.collector.spec.items:
            for item2 in self.collector.spec.items:
                if (item1 != item2 and item1.category.name != "entrance" and
                        item2.category.name != "entrance"):
                    if (item1.required_area < item2.required_area and
                            self.items_spaces[item1].area > self.items_spaces[item2].area):
                        logging.debug("Solution %i: Size bonus : %i", self._id, 0)
                        return 0
        logging.debug("Solution %i: Size bonus : %i", self._id, 10)
        return 10

    def _externals_spaces_bonus(self) -> float:
        """
        Good ordering externals spaces size bonus
        :return: score : float
        """
        for item1 in self.collector.spec.items:
            for item2 in self.collector.spec.items:
                if (item1 != item2 and self.items_spaces[item1].connected_spaces()
                        and self.items_spaces[item2].connected_spaces()):
                    item1_ext_spaces_area = sum([ext_space.area
                                                 for ext_space in
                                                 self.items_spaces[item1].connected_spaces()
                                                 if ext_space.category.external])
                    item2_ext_spaces_area = sum([ext_space.area
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
        circulator = Circulator(plan=self.plan, cost_rules=COST_RULES)
        circulator.connect()
        cost = circulator.circulation_cost
        circulation_penalty = 0

        if cost > COST_RULES["water_room_less_than_two_ducts"]:
            circulation_penalty += 100
        elif cost > COST_RULES["window_room_default"]:
            circulation_penalty += 50
        elif cost > COST_RULES["water_room_default"]:
            circulation_penalty += 30
        elif cost - (self.collector.spec.typology - 1) * 200 > 0:
            circulation_penalty += 15
        logging.debug("Solution %i: circulation penalty : %i", self._id, circulation_penalty)
        return circulation_penalty

    def _night_and_day_score(self) -> float:
        """
        Night and day score
        day / night distribution of rooms
        :return: score : float
        """
        first_level = self.plan.first_level
        day_list = ["living", "kitchen", "cellar", "dining"]

        night_list = ["bedroom", "bathroom", "laundry"]

        day_polygon_list = []
        night_polygon_list = []
        for i_floor in range(self.plan.floor_count):
            day_polygon_list.append(None)
            night_polygon_list.append(None)

        for item in self.items_spaces.keys():
            associated_space = self.items_spaces[item]
            level = associated_space.floor.level
            # Day
            if (item.category.name in day_list or (item.category.name == "toilet" and
                                                   associated_space == self.get_rooms("toilet")[0])):
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
            night_polygon_with_entrance = night_polygon.union(
                self.get_rooms("entrance")[0].as_sp.buffer(CORRIDOR_SIZE))
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
        - toilets and bathrooms must be accessible from the corridors or the entrance
        - the living must be near the entrance
        :return: score : float
        """

        position_score = 0
        nbr_room_position_score = 0
        front_door = self.plan.front_door().as_sp
        corridor_poly = None  # TODO
        for item in self.items_spaces.keys():
            space = self.items_spaces[item]
            item_position_score = 0
            if item.category.name == "toilet" and space == self.get_rooms("toilet")[0]:
                nbr_room_position_score += 1
                # distance from the entrance
                if self.plan.front_door().floor == space.floor:
                    # distance from the entrance
                    plan_area = self.collector.spec.plan.area
                    criteria = plan_area ** 0.5
                    distance_toilet_fd = space.as_sp.distance(front_door)
                    if distance_toilet_fd < criteria:
                        item_position_score = (criteria - distance_toilet_fd) * 100 / criteria
            elif item.category.name == "bedroom":
                nbr_room_position_score += 1
                # distance from a bedroom / bathroom
                for item_test in self.collector.spec.items:
                    if (item_test.category.name == "bathroom" and
                            self.items_spaces[item_test].floor == space.floor):
                        polygon_bathroom = self.items_spaces[item_test].as_sp
                        connected = False
                        for circulation_space in self.plan.circulation_spaces():
                            if polygon_bathroom.intersection(
                                    circulation_space.as_sp.union(space.as_sp)):
                                connected = True
                                item_position_score = 100
                                break
                        if not connected and polygon_bathroom.distance(space.as_sp) < CORRIDOR_SIZE:
                            item_position_score = 100
                            break
            if item.category.name == "toilet" or item.category.name == "bathroom":
                # could be private
                private = False
                for circulation_space in self.plan.circulation_spaces():
                    if circulation_space.as_sp.intersection(space.as_sp):
                        private = True
                        item_position_score = 100
                        break
                if not private and corridor_poly:
                    if not corridor_poly.intersection(space.as_sp):
                        item_position_score -= 50
                if item.category.name == "bathroom":
                    nbr_room_position_score += 1
            elif item.category.name == "living":
                nbr_room_position_score += 1
                if "frontDoor" in space.components_category_associated():
                    item_position_score = 100
                elif self.get_rooms("entrance")[0].floor == space.floor:
                    # distance from the entrance
                    if self.get_rooms("entrance")[0].as_sp.distance(space.as_sp) < CORRIDOR_SIZE:
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
        for item in self.items_spaces.keys():
            space = self.items_spaces[item]
            #  duct or pillar or small bearing wall
            if space.has_holes:
                item_something_inside_score = 0
                something_inside_score = min(something_inside_score,
                                             item_something_inside_score)
                logging.debug("Solution %i: Something Inside score : %f, room : %s, has_holes",
                              self._id, something_inside_score, item.category.name)
                continue
            #  isolated room
            list_of_non_concerned_room = ["entrance", "circulationSpace", "dressing", "cellar",
                                          "study", "laundry"]
            sp_space = space.as_sp
            convex_hull = sp_space.convex_hull
            for i_item in self.collector.spec.items:
                if (i_item != item and not (
                        i_item.category.name in list_of_non_concerned_room) and space.floor ==
                        self.items_spaces[i_item]):
                    if (convex_hull.intersection(self.items_spaces[i_item].as_sp)).area > (
                            space.area / 8):
                        # Check jroom adjacency
                        other_room_adj = False
                        for j_item in self.collector.spec.items:
                            if j_item != i_item and j_item != item:
                                if space.adjacent_to(self.items_spaces[j_item]):
                                    other_room_adj = True
                        if not other_room_adj:
                            item_something_inside_score = 0
                            something_inside_score = min(something_inside_score,
                                                         item_something_inside_score)
                            logging.debug("Solution %i: Something Inside score : %f, room : %s, "
                                          "isolated room", self._id, something_inside_score,
                                          item.category.name)
                            break

        logging.debug("Solution %i: Something Inside score : %f", self._id, something_inside_score)
        return something_inside_score

    def score(self) -> float:
        """
        Scoring
        compilation of different scores
        :return: score : float
        """
        solution_score = (self._area_score() + self._shape_score() + self._night_and_day_score()
                          + self._position_score() + self._something_inside_score()) / 5
        solution_score = solution_score + self._good_size_bonus()  # - self._circulation_penalty()
        logging.debug("Solution %i: Final score : %f", self._id, solution_score)
        return solution_score

    def distance(self, other_solution: 'Solution') -> float:
        """
        Distance with an other solution
        the distance is calculated from the groups of rooms day and night
        the inversion of two rooms within the same group gives a zero distance
        :return: distance : float
        """
        # Day group
        day_list = ["livingKitchen", "living", "kitchen", "dining", "cellar"]
        # Night group
        night_list = ["bedroom", "bathroom", "toilet", "laundry", "dressing", "study", "misc"]

        difference_area = 0
        mesh_area = 0
        for floor in self.plan.floors.values():
            for face in floor.mesh.faces:
                space = self.plan.get_space_of_face(face)
                if space.category.mutable:
                    mesh_area += face.area
                    other_space = other_solution.plan.get_space_of_face(face)
                    if ((space.category.name in day_list and
                         other_space.category.name not in day_list) or
                            (space.category.name in night_list and
                             other_space.category.name not in night_list)):
                        difference_area += face.area

        distance = difference_area * 100 / mesh_area
        return distance
