# coding=utf-8
"""
Solution collector

"""
from typing import List
from libs.specification import Specification
from libs.plan import Plan, Space
import logging

CORRIDOR_SIZE = 120


class SolutionsCollector:
    """
    Solutions Collector
    """

    def __init__(self, spec: 'Specification'):
        self.solutions: List['Solution'] = []
        self.spec = spec

    def add_plan(self, plan: 'Plan') -> None:
        """
        add solutions to the list
        :return: None
        """
        sol = Solution(self, plan, len(self.solutions))
        self.solutions.append(sol)


class Solution:
    """
    Space planner Class
    """

    def __init__(self, collector: 'SolutionsCollector', plan: 'Plan', id: float):
        self.id = id
        self.collector = collector
        self.plan = plan
        self.items_spaces = {}
        self.init_items_spaces()

    def __repr__(self):
        output = 'Solution Id' + str(self.id)
        return output

    def init_items_spaces(self):
        for item in self.collector.spec.items:
            for space in self.plan.mutable_spaces:
                if item.category.name == space.category.name:
                    self.items_spaces[item] = space

    def get_rooms(self, category_name: str) -> list('Space'):
        """
        Retrieves all spaces corresponding to the category_name
        :param category_name: str
        :return: List('Spaces')
        """
        rooms_list = []
        for item in self.collector.spec.items:
            if item.category.name == category_name:
                rooms_list.append(self.items_spaces[item])

        return rooms_list

    def _area_score(self) -> float:
        """
        Area score
        :return: score : float
        """
        good_overflow_categories = ['living', 'dining']

        area_score = 0
        area_penalty = 0
        for item in self.collector.spec.items:
            space = self.items_spaces[item]
            # Min < SpaceArea < Max
            if item.min_size.area <= space.area <= item.max_size.area:
                item_area_score = 100
            # good overflow
            elif item.max_size.area < space.area and \
                    space.category.name in good_overflow_categories:
                item_area_score = 100
            # overflow
            else:
                item_area_score = (100 - abs(item.required_area - space.area) /
                                   item.required_area * 100)
                if space.category.name == 'entrance':
                    if space.area < 20000:
                        area_penalty += 1
                elif space.category.name == 'wc':
                    if space.area < 10000:
                        area_penalty += 1
                    elif space.area > item.max_size.area:
                        area_penalty += 3
                elif space.category.name == 'bathroom' or space.category.name == 'wcBathroom':
                    if space.area < 20000:
                        area_penalty += 1
                elif space.category.name == 'bedroom':
                    if space.area < 75000:
                        area_penalty += 1

            # Area score
            area_score += item_area_score
            logging.debug('Area score : {0}, room : {1}'.format(item_area_score, item.id))

        area_score = round(area_score / self.collector.spec.number_of_items, 2) - area_penalty * 20
        logging.debug('Area score : {0}'.format(area_score))
        return area_score

    def _shape_score(self) -> float:
        """
        Shape score
        :return: score : float
        """
        shape_score = 100
        for item in self.collector.spec.items:
            space = self.items_spaces[item]
            sp_space = space.as_sp
            convex_hull = sp_space.convex_hull
            if convex_hull.is_valid and sp_space.is_valid:
                outside = convex_hull.difference(sp_space)
                item_shape_score = min(100, 100 - (
                            (outside.area - sp_space.area / 7) / (sp_space.area / 4) * 100))
                logging.debug('Shape score : {0}, room : {1}'.format(item_shape_score, item.id))
            else:
                logging.warning('Invalid shapely space')
                item_shape_score = 100

            shape_score = min(item_shape_score, shape_score)

        return shape_score

    def _good_size_bonus(self) -> float:
        """
        Good ordering items size bonus
        :return: score : float
        """
        for item1 in self.collector.spec.items:
            for item2 in self.collector.spec.items:
                if item1 != item2 and item1.category.name != 'entrance' and \
                        item2.category.name != 'entrance':
                    if item1.required_area < item2.required_area and \
                            self.items_spaces[item1].area > self.items_spaces[item2].area:
                        print(item1.category.name, item2.category.name)
                        logging.debug('Size bonus : {0}'.format(0))
                        return 0
        logging.debug('Size bonus : {0}'.format(10))
        return 10

    def _night_and_day_score(self) -> float:
        """
        Night and day score
        :return: score : float
        """

        day_list = ['living', 'kitchen', 'cellar', 'dining']

        night_list = ['bedroom', 'bathroom', 'wcBathroom', 'laundry']

        day_polygon = None
        night_polygon = None
        for item in self.collector.spec.items:
            # Day
            if item.category.name in day_list or \
                    (item.category.name == 'wc' and
                     self.items_spaces[item] == self.get_rooms('wc')[0]):
                if not day_polygon:
                    day_polygon = self.items_spaces[item].as_sp
                else:
                    day_polygon = day_polygon.union(self.items_spaces[item].as_sp)

            # Night
            elif item.category.name in night_list or \
                    (item.category.name == 'wc' and
                     self.items_spaces[item] != self.get_rooms('wc')[0]):
                if not night_polygon:
                    night_polygon = self.items_spaces[item].as_sp
                else:
                    night_polygon = night_polygon.union(self.items_spaces[item].as_sp)

        print('day_polygon', day_polygon)
        print('night_polygon', night_polygon)
        # groups of rooms
        groups_score = 100
        if day_polygon.geom_type != 'Polygon':
            day_polygon = day_polygon.union(
                self.get_rooms('entrance')[0].as_sp.buffer(1))
            if day_polygon.geom_type != 'Polygon':
                groups_score -= 50

        if night_polygon.geom_type != 'Polygon':
            night_polygon_with_entrance = night_polygon.union(
                self.get_rooms('entrance')[0].as_sp.buffer(CORRIDOR_SIZE))
            if night_polygon_with_entrance.geom_type != 'Polygon':
                if (len(night_polygon) > 2 and len(night_polygon_with_entrance) > 2) \
                        or (self.collector.spec.typology <= 2
                            or self.collector.spec.number_of_items < 6):
                    groups_score -= 50
                else:
                    groups_score -= 25

        logging.debug('Night and day score : {0}'.format(groups_score))
        return groups_score

    def _position_score(self) -> float:
        """
        Night and day score
        :return: score : float
        """

        position_score = 100
        nbr_room_position_score = 0
        for item in self.collector.spec.items:
            item_position_score = 100
            if item.category.name == 'wc' and self.items_spaces[item] == self.get_rooms('wc')[0]:
                nbr_room_position_score += 1
                # distance from the entrance
                if False:  # front door not in this level:  self.items_spaces[item].level:
                    item_position_score = 0
                else:
                    # distance from the entrance
                    entrance_poly = self.get_rooms('entrance')[0].as_sp.buffer(CORRIDOR_SIZE)
                    union_poly = entrance_poly.union(self.items_spaces[item].as_sp)
                    if union_poly.geom_type != 'Polygon':
                        # distance from the entrance / front door
                        plan_area = self.collector.spec.plan.area
                        criteria = plan_area ** 0.5
                        distance_wc_fd = self.items_spaces[item].distance(front_door_poly)
                        if distance_wc_fd < criteria:
                            item_position_score = (criteria-distance_wc_fd) * 100 / criteria
                        else:
                            item_position_score = 0

            elif room_type == 'bedroom':
                nbr_room_position_score += 1
                # distance from a bedroom / bathroom
                first_pass_bath = True
                for j, jroom in enumerate(list(PolygonRoomsDf['Type'])):
                    if (jroom == 'bathroom' or jroom == 'wcBathroom') and first_pass_bath:
                        first_pass_bath = False
                        polygon_bathroom = PolygonRoomsDf.ix[j, 'Room']
                    elif jroom == 'bathroom' or jroom == 'wcBathroom':
                        polygon_bathroom = polygon_bathroom.union(PolygonRoomsDf.ix[j, 'Room'])
                if not first_pass_bath:
                    if polygon_bathroom.distance(
                            PolygonRoomsDf.at[room, 'Room']) > CORRIDOR_SIZE and groups_score != 100:
                        item_position_score = 0

            if PolygonRoomsDf.at[room, 'Type'] == 'wc' or PolygonRoomsDf.at[
                room, 'Type'] == 'bathroom' or PolygonRoomsDf.at[room, 'Type'] == 'wcBathroom':
                entrance_inter = []
                # distance from the entrance
                if PolygonRoomsDf.at['entrance', 'Room'] and PolygonRoomsDf.at[
                    'entrance', 'Level'] == room_level:
                    entrance_poly = PolygonRoomsDf.at['entrance', 'Room'].buffer(settings.epsilon)
                    entrance_inter = entrance_poly.intersection(PolygonRoomsDf.at[room, 'Room'])
                if not entrance_inter and corridor_poly:
                    corridor_inter = corridor_poly.intersection(PolygonRoomsDf.at[room, 'Room'])
                    if not corridor_inter:
                        item_position_score -= 50
                if PolygonRoomsDf.at[room, 'Type'] == 'bathroom' or PolygonRoomsDf.at[
                    room, 'Type'] == 'wcBathroom':
                    nbr_room_position_score += 1
            elif PolygonRoomsDf.at[room, 'Type'] == 'living' or PolygonRoomsDf.at[
                room, 'Type'] == 'livingKitchen':
                nbr_room_position_score += 1
                # distance from the entrance
                if PolygonRoomsDf.at['entrance', 'Room']:
                    entrance_poly = PolygonRoomsDf.at['entrance', 'Room'].buffer(CORRIDOR_SIZE)
                    union_poly = entrance_poly.union(PolygonRoomsDf.at[room, 'Room'])
                    # diff = union_poly.difference(lbw.buffer(settings.epsilon))
                    if union_poly.geom_type != 'Polygon':
                        item_position_score = 0

            position_score = position_score + item_position_score

    @property
    def score(self) -> float:
        """
        Scoring
        :return: score : float
        """
        solution_score = (self._area_score() + self._shape_score() + self._night_and_day_score()) / 3
        solution_score = solution_score - self._good_size_bonus()

        return solution_score

    def distance(self, other_solution: 'Solution') -> float:
        """
        Constraints list initialization
        :return: distance : float
        """
