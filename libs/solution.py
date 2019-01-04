# coding=utf-8
"""
Solution collector

"""
from typing import List, Dict, Callable, Optional
from libs.specification import Specification, Item
from libs.plan import Plan


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
        sol_id = len(self.solutions)
        sol = Solution(sol_id, plan)
        self.solutions.append(sol)


class Solution:
    """
    Space planner Class
    """

    def __init__(self, sol_id: int, plan: 'Plan'):
        self.id = sol_id
        self.plan = plan

    def area_score(self) -> float:

        for i_space, space in enumerate(self.plan.mutable_spaces):
            if room_type != 'circulationSpace':
                # Min < VoronoiArea < Max
                if floor_plan.RoomsDf.at[room, 'RequiredMinArea'] <= PolygonRoomsDf.at[room, 'Area'] and \
                        floor_plan.RoomsDf.at[room, 'RequiredMaxArea'] >= PolygonRoomsDf.at[
                    room, 'Area']:
                    PolygonRoomsDf.at[room, 'AreaScore'] = 100
                # good overflow
                elif floor_plan.RoomsDf.at[room, 'RequiredMaxArea'] < PolygonRoomsDf.at[
                    room, 'Area'] and \
                        (room_type == 'living' or room_type == 'livingKitchen'
                         or room_type == 'livingKitchenDining' or room_type == 'dining'
                         or room_type == 'kitchenDining'):
                    PolygonRoomsDf.at[room, 'AreaScore'] = 100
                # overflow
                else:
                    PolygonRoomsDf.at[room, 'AreaScore'] = (100 - abs(
                        floor_plan.RoomsDf.at[room, 'RequiredArea'] - PolygonRoomsDf.at[room, 'Area']) /
                                                            floor_plan.RoomsDf.at[
                                                                room, 'RequiredArea'] * 100)
                    if room_type == 'entrance':
                        if PolygonRoomsDf.at[room, 'Area'] < 20000:
                            area_penalty += 1
                    elif room_type == 'wc':
                        if PolygonRoomsDf.at[room, 'Area'] < 10000:
                            area_penalty += 1
                        elif PolygonRoomsDf.at[room, 'Area'] > floor_plan.RoomsDf.at[
                            room, 'RequiredArea']:
                            area_penalty += 3
                    elif room_type == 'bathroom' or room_type == 'wcBathroom':
                        if PolygonRoomsDf.at[room, 'Area'] < 20000:
                            area_penalty += 1
                    elif room_type == 'bedroom':
                        if PolygonRoomsDf.at[room, 'Area'] < 75000:
                            area_penalty += 1
            else:
                if not PolygonRoomsDf.at[room, 'Area'] and PolygonRoomsDf.at[room, 'Area'] == 0:
                    PolygonRoomsDf.at[room, 'AreaScore'] = 100
                else:
                    PolygonRoomsDf.at[room, 'AreaScore'] = 50

            # Area score
            area_score += PolygonRoomsDf.at[room, 'AreaScore']

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

