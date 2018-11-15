# coding=utf-8
"""
Space grower module
A grower creates spaces in a plan from seeds point and according to a specification
"""
from typing import List, Optional, Dict, Callable, Generator, Tuple, TYPE_CHECKING
import logging

import matplotlib.pyplot as plt

from libs.category import space_categories
from libs.plan import Space, PlanComponent

from libs.utils.geometry import barycenter, move_point

if TYPE_CHECKING:
    from libs.mesh import Edge, Face


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

    def check_condition(self, category_name: str, edge: 'Edge') -> bool:
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
                 edge: 'Edge',
                 plan_component: Optional[PlanComponent] = None):
        self.seeder = seeder
        self.edge = edge  # the reference edge of the seed
        self.components = [plan_component] if PlanComponent else []  # the components of the seed
        self.space: Optional[Space] = None  # the seed space
        # growth methods / should be stored in the space category
        self.growth_methods = plan_component.category.seed_category.methods
        self.growth_method_index = 0

    def __repr__(self):
        return 'Seed: {0}, {1}, {2}'.format(self.edge, self.components, self.space)

    @property
    def size(self) -> Tuple[float, float]:
        """
        Returns the width of the seed space
        TODO : we should implement a size method for the space that will return a Size object
        :return: width, depth
        """
        return self.space.bounding_box(self.edge.unit_vector)

    @property
    def check_size(self) -> bool:
        """
        Returns True if the space size is within the provided limits
        :return:
        """
        max_size = self.max_size
        return self.size[0] <= max_size[0] and self.size[1] <= max_size[1]

    @property
    def max_size(self) -> Tuple[float, float]:
        """
        Returns the max size for the seed space according to its component
        :return:
        """
        max_width = 0
        max_depth = 0
        for component in self.components:
            max_width = max(max_width, component.category.seed_category.size.max_width)
            max_depth = max(max_depth, component.category.seed_category.size.max_depth)

        return max_width, max_depth

    @property
    def neighbors(self) -> Generator['Face', None, None]:
        """
        Returns adjacent faces of the corresponding growth method
        :return:
        """
        yield from self.growth_method(self)

    @property
    def growth_method(self):
        """
        Returns the current growth method
        :return:
        """
        return self.growth_methods[self.growth_method_index]

    def add_face(self, face: 'Face') -> 'Face':
        """
        Adds a face to the seed space
        :param face:
        :return:
        """
        added_face = None

        if face.space.category.name == 'empty':
            initial_space = face.space
            initial_space.remove_face(face)
            self.space.add_face(face)
            # check size
            if self.check_size:
                added_face = face
            else:
                self.space.remove_face(face)
                initial_space.add_face(face)
        return added_face

    def grow(self) -> Optional['Face']:
        """
        Tries to grow the seed space by one face
        Returns the face added
        :param self:
        :return:
        """
        if self.growth_method.name == 'done':
            return None

        # initialize first face
        if self.space is None:
            self.edge.face.space.remove_face(self.edge.face)
            self.space = Space(self.seeder.plan, self.edge, space_categories['seed'])
            self.seeder.plan.add_space(self.space)
            return self.edge.face

        added_face = None

        for face in self.neighbors:
            added_face = self.add_face(face)

        # if we couldn't add face : switch to the next growth strategy
        if added_face is None:
            self.growth_method_index += 1

        return added_face

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

    import libs.reader as reader
    from libs.grid import sequence_grid, edge_length

    logging.getLogger().setLevel(logging.DEBUG)

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

        ax = new_plan.plot(save=False, options=('fill', 'border', 'half-edge', 'face'))
        seeder.plot(ax)
        plt.show()

        assert new_plan.check()

    grow_a_plan()
