# coding=utf-8
"""
Space grower module
A grower creates spaces in a plan from seeds point and according to a specification
"""
from typing import List, Optional, Dict, Callable
import copy
import logging

import matplotlib.pyplot as plt

from libs.mesh import Edge, Face
from libs.plan import Space, Plan, PlanComponent
from libs.specification import Specification
import libs.reader as reader
from libs.grid import sequence_grid, edge_length
from libs.category import space_categories

from libs.utils.geometry import barycenter, move_point


class Grower:
    """
    Grower class
    """
    def __init__(self, specification: Specification, seeds: List['Seed']):
        self.specification = specification
        self.seeds = seeds

    def __repr__(self):
        output = 'Grower: \n'
        output += self.specification.__repr__()
        for seed in self.seeds:
            output += seed.__repr__()

        return output

    def grow(self) -> Plan:
        """
        Creates a plan with new spaces
        :return:
        """
        if self.specification.plan is None:
            raise ValueError('Cannot grow from a specification with no plan:' +
                             '{0}'.format(self.specification))

        empty_plan = self.specification.plan
        if not empty_plan.is_empty:
            raise ValueError('Cannot grow inside a non empty plan: {0}'.format(empty_plan))

        new_plan = copy.deepcopy(empty_plan)

        # add a new space for each seed
        # DO SOMETHING

        return new_plan


class Seeder:
    """
    Seeder Class
    """
    def __init__(self, plan):
        self.plan = plan
        self.seeds: List['Seed'] = []
        self.conditions: Dict[str] = {}

    def __repr__(self):
        output = 'Seeder:\n'
        for seed in self.seeds:
            output += '• ' + seed.__repr__() + '\n'
        return output

    def plant(self):
        """
        Creates the seeds
        :return:
        """
        for component in self.plan.get_component():
            if not component.category.mutable:
                # create seed for each edge touching an empty space
                for edge in component.edges:
                    seed_edge = edge.pair if isinstance(component, Space) else edge
                    if not self.check_condition(component.category.name, edge):
                        continue
                    if seed_edge.space and seed_edge.space.category.name == 'empty':
                        # check if a seed already exist with the same face
                        for seed in self.seeds:
                            if seed.edge.face is seed_edge.face:
                                seed.add_component(component)
                                break
                        else:
                            new_seed = Seed(self, seed_edge, component)
                            self.add_seed(new_seed)

    def grow(self):
        """
        Creates the space for each seed
        :return:
        """
        # create the seeds
        self.plant()

        # grow the seeds
        while True:
            faces_added = []
            for seed in self.seeds:
                face_added = seed.grow()
                if face_added:
                    faces_added.append(face_added)
            if not faces_added:
                break

    def add_seed(self, seed: 'Seed'):
        """
        Adds a seed to the seeder
        :param seed:
        :return:
        """
        self.seeds.append(seed)

    def add_condition(self, condition: Callable, category_name: Optional[str] = None):
        """
        Adds a condition to create a seed.
        A condition is a predicate that takes an edge as input and returns a boolean
        :param condition:
        :param category_name:
        :return:
        """
        key = category_name if category_name else 'general'
        if key in self.conditions:
            self.conditions[key].append(condition)
        else:
            self.conditions[key] = [condition]

    def check_condition(self, category_name: str, edge: Edge) -> bool:
        """
        Verify the condition
        :param category_name:
        :param edge:
        :return:
        """
        if category_name != 'general' and not self.check_condition('general', edge):
            return False
        if category_name not in self.conditions:
            return True
        for condition in self.conditions[category_name]:
            if not condition(edge):
                return False
        return True

    def plot(self, ax):
        """
        Plots the seeds point
        :param ax:
        :return:
        """
        for seed in self.seeds:
            ax = seed.plot(ax)

        return ax


class Seed:
    """
    Seed class
    An edge from which to grow a space
    """
    def __init__(self,
                 seeder: Seeder,
                 edge: Edge,
                 plan_component: Optional[PlanComponent] = None):
        self.seeder = seeder
        self.edge = edge  # the reference edge of the seed
        self.components = [plan_component] if PlanComponent else []
        self.space: Optional[Space] = None

    def __repr__(self):
        return 'Seed: {0}, {1}, {2}'.format(self.edge, self.components, self.space)

    def grow(self) -> Optional[Face]:
        """
        Tries to grow the seed space by one face
        Returns the face added
        :param self:
        :return:
        """
        if self.space is None:
            self.edge.face.space.remove_face(self.edge.face)
            self.space = Space(self.seeder.plan, self.edge, space_categories['seed'])
            self.seeder.plan.add_space(self.space)
            return self.edge.face

        # This growth mecanism does crazing things (spiraling and such). We should create
        # more appropriate rules to select the best face to add to the space
        for edge in self.space.edges:
            face = edge.pair.face
            if face is None:
                continue
            if face.space and face.space.category.name != 'empty':
                continue
            face.space.remove_face(face)
            self.space.add_face(face)
            return face

        return None

    def max_size(self):
        """
        returns the max size of the space
        :param self:
        :return:
        """
        pass

    def add_component(self, component: PlanComponent):
        """
        Adds a plan component to the seed
        :param component:
        :return:
        """
        self.components.append(component)

    def plot(self, ax):
        """
        Plots the seed
        :param ax:
        :return:
        """
        seed_distance_to_edge = 15
        point = barycenter(self.edge.start.coords, self.edge.end.coords, 0.5)
        point = move_point(point, self.edge.normal, seed_distance_to_edge)
        ax.plot([point[0]], [point[1]], 'ro', color='r')
        return ax


if __name__ == '__main__':

    logging.getLogger().setLevel(logging.INFO)

    def grow_a_plan():
        """
        Test
        :return:
        """
        plan = reader.create_plan_from_file('Noisy_A145.json')

        new_plan = sequence_grid.apply_to(plan)

        seeder = Seeder(new_plan)
        seeder.add_condition(edge_length(50.0), 'duct')

        seeder.grow()
        print(seeder)

        ax = new_plan.plot(save=False, options=('fill', 'border'))
        seeder.plot(ax)
        plt.show()

    grow_a_plan()
