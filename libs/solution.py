# coding=utf-8
"""
Solution collector

"""
from typing import List, Dict, Callable, Optional
from libs.specification import Specification, Item
from libs.plan import Plan
import logging


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
        sol = Solution(self, plan)
        self.solutions.append(sol)


class Solution:
    """
    Space planner Class
    """

    def __init__(self, collector: 'SolutionCollector', plan: 'Plan'):
        self.collector = collector
        self.plan = plan
        self.items_spaces = {}
        self.init_items_spaces()

    def init_items_spaces(self):
        for item in self.collector.spec.items:
            for space in self.plan.mutable_spaces:
                if item.category.name == space.category.name:
                    self.items_spaces[item] = space

    def area_score(self) -> float:
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

        area_score = round(area_score / self.collector.spec.number_of_items, 2) - area_penalty * 20

        return area_score

    def shape_score(self) -> float:

        shape_score = 100
        for item in self.collector.spec.items:
            space = self.items_spaces[item]
            sp_space = space.as_sp
            convex_hull = sp_space.convex_hull
            if convex_hull.is_valid and sp_space.is_valid:
                outside = convex_hull.difference(sp_space)
                item_shape_score = min(100, 100 - (
                            (outside.area - sp_space.area / 7) / (sp_space.area / 4) * 100))
                logging.debug('Shape score : ', item_shape_score, 'room : ', item.id)
            else:
                logging.warning('Invalid shapely space')
                item_shape_score = 100

            shape_score = min(item_shape_score, shape_score)

        return shape_score

    def good_size_bonus_score(self) -> float:

        for item1 in self.collector.spec.items:
            for item2 in self.collector.spec.items:
                if item1 != item2 and item1.category.name != 'entrance' and \
                        item2.category.name != 'entrance':
                    if item1.required_area < item2.required_area and \
                            self.items_spaces[item1].area > self.items_spaces[item2].area:
                        return 0

        return 10

    def score(self) -> float:
        """
        Scoring
        :return: score : float
        """

    def distance(self, other_solution: 'Solution') -> float:
        """
        Constraints list initialization
        :return: distance : float
        """
