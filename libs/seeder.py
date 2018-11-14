# coding=utf-8
"""
Space grower module
A grower creates spaces in a plan from seeds point and according to a specification
"""
from typing import List, Optional, Dict, Callable, Generator, Tuple
import copy
import logging

import matplotlib.pyplot as plt

from libs.mesh import Edge, Face
from libs.plan import Space, Plan, PlanComponent
from libs.specification import Specification
import libs.reader as reader
from libs.grid import sequence_grid, edge_length
from libs.category import space_categories
from libs.utils.custom_types import Vector2d

from libs.utils.geometry import barycenter, move_point, same_half_plane, ccw_angle, pseudo_equal, opposite_vector


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
            if component.category.seedable:
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
            # stop to grow once we cannot grow anymore
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
        :param condition: predicate
        :param category_name: the str name of a category
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
        self.components = [plan_component] if PlanComponent else []  # the components of the seed
        self.space: Optional[Space] = None  # the seed space
        # edge pointers used for growth
        self.horizontal_growth = True

    def __repr__(self):
        return 'Seed: {0}, {1}, {2}'.format(self.edge, self.components, self.space)

    @property
    def size(self) -> Tuple[float, float]:
        """
        Returns the width of the seed space
        :return: width, depth
        """
        return self.space.bounding_box(self.edge.unit_vector)

    def select(self, vector: Vector2d, epsilon: float = 10.0) -> Generator[Edge, None, None]:
        """
        Returns the edges of the seed space which normals are quasi parallel to the specified vector
        :param vector:
        :param epsilon:
        :return:
        """
        for edge in self.space.edges:
            if pseudo_equal(ccw_angle(edge.normal, vector), 180.0, epsilon):
                yield edge

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
        max_width = 300
        max_depth = 400

        self_size = self.size
        added_face = None

        if self.horizontal_growth:
            directions = self.edge.unit_vector, opposite_vector(self.edge.unit_vector)
        else:
            directions = (self.edge.normal,)

        for direction in directions:
            for edge in list(self.select(direction)):
                face = edge.pair.face
                if face and (not face.space or face.space.category.name == 'empty'):
                    face_size = face.bounding_box(direction)
                    correct_size = (self_size[0] + face_size[0] <= max_width
                                    if self.horizontal_growth
                                    else self_size[1] + face_size[1] <= max_depth)
                    if correct_size:
                        face.space.remove_face(face)
                        self.space.add_face(face)
                        added_face = face
                    else:
                        break
            if added_face is None and self.horizontal_growth:
                self.horizontal_growth = False

        return added_face

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
        plan = reader.create_plan_from_file('Massy_C204.json')

        new_plan = sequence_grid.apply_to(plan)

        seeder = Seeder(new_plan)
        seeder.add_condition(edge_length(50.0), 'duct')

        seeder.grow()
        print(seeder)

        ax = new_plan.plot(save=False, options=('fill', 'border'))
        seeder.plot(ax)
        plt.show()

    grow_a_plan()
