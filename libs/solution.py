# coding=utf-8
"""
Solution collector

"""
from typing import List, Dict, Callable, Optional
import logging

import matplotlib.pyplot as plt
from libs.specification import Specification, Item


class SolutionsCollector:
    """
    Solutions Collector
    """

    def __init__(self):
        self.solutions: List['Solution'] = []

    def add_plan(self, plan: 'Plan') -> None:
        """
        add solutions to the list
        :return: None
        """
        id = len(self.solutions)
        sol = Solution(id, plan)
        self.solutions.append(sol)


class Solution:
    """
    Space planner Class
    """

    def __init__(self, id: int, plan: 'Plan'):
        self.id = id
        self.plan = plan

    def score(self) -> float:
        """
        scoring
        :return: score : float
        """

    def distance(self, other_solution: 'Solution') -> float:
        """
        Constraints list initialization
        :return: distance : float
        """




if __name__ == '__main__':
    import libs.reader as reader
    import libs.seed
    from libs.selector import SELECTORS
    from libs.grid import GRIDS
    from libs.shuffle import SHUFFLES

    logging.getLogger().setLevel(logging.DEBUG)


    def space_planning():
        """
        Test
        :return:
        """

        input_file = 'Bussy_Regis.json'  # 5 Levallois_Letourneur / Antony_A22
        plan = reader.create_plan_from_file(input_file)

        seeder = libs.seed.Seeder(plan, libs.seed.GROWTH_METHODS)
        seeder.add_condition(SELECTORS['seed_duct'], 'duct')
        GRIDS['ortho_grid'].apply_to(plan)

        seeder.plant()
        seeder.grow(show=True)
        SHUFFLES['square_shape'].run(plan, show=True)

        logging.debug(plan)
        logging.debug(seeder)

        plan.plot(show=True)
        # seeder.plot_seeds(ax)
        plt.show()
        seed_empty_furthest_couple_middle = SELECTORS[
            'seed_empty_furthest_couple_middle_space_area_min_100000']
        seed_empty_area_max_100000 = SELECTORS['area_max=100000']
        seed_methods = [
            (
                seed_empty_furthest_couple_middle,
                libs.seed.GROWTH_METHODS_FILL,
                "empty"
            ),
            (
                seed_empty_area_max_100000,
                libs.seed.GROWTH_METHODS_SMALL_SPACE_FILL,
                "empty"
            )
        ]

        filler = libs.seed.Filler(plan, seed_methods)
        filler.apply_to(plan)
        plan.remove_null_spaces()
        fuse_selector = SELECTORS['fuse_small_cell']

        logging.debug("num_mutable_spaces before merge: {0}".format(plan.count_mutable_spaces()))

        filler.fusion(fuse_selector)

        logging.debug("num_mutable_spaces after merge: {0}".format(plan.count_mutable_spaces()))

        SHUFFLES['square_shape'].run(plan, show=True)

        input_file = 'Bussy_Regis_setup.json'
        spec = reader.create_specification_from_file(input_file)
        spec.plan = plan

        space_planner = SpacePlanner('test', spec)
        space_planner.add_spaces_constraints()
        space_planner.add_item_constraints()
        space_planner.rooms_building()

        plan.plot(show=True)
        # seeder.plot_seeds(ax)
        plt.show()
        assert spec.plan.check()


    space_planning()
