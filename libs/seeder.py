# coding=utf-8
"""
Space grower module
A grower creates spaces in a plan from seeds point and according to a specification
"""
from typing import TYPE_CHECKING, List, Optional, Dict, Callable, Sequence
import logging
import copy

import matplotlib.pyplot as plt

from libs.category import space_catalog
from libs.plan import Space, PlanComponent, Plan
from libs.plot import plot_point
from libs.size import Size
from libs.constraint import CONSTRAINTS

from libs.utils.geometry import barycenter, move_point

if TYPE_CHECKING:
    from libs.mesh import Edge
    from libs.action import Action


# TODO : to be deleted
PLOT_STEPS = False
if PLOT_STEPS:
    plt.ion()
    FIG, AX = plt.subplots()
    AX.set_aspect('equal')
######################

EPSILON_MAX_SIZE = 10.0


class Seeder:
    """
    Seeder Class
    """
    def __init__(self, plan: Plan):
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
            spaces_modified = []
            for seed in self.seeds:
                spaces_modified += seed.grow()
            # stop to grow once we cannot grow anymore
            if not spaces_modified:
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
        Verify the condition to create the seed
        :param category_name:
        :param edge:
        :return: True if the condition is satisfied
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
        self.growth_methods = plan_component.category.seed_category.operators
        self.growth_method_index = 0
        self.max_size = self.get_components_max_size()
        self.max_size_constraint = self.create_max_size_constraint()

    def __repr__(self):
        return ('Seed: {0}, area: {1}, width: {2}, depth: {3} - {4}, ' +
                '{5}').format(self.components, str(self.space.area), str(self.size.width),
                              str(self.size.depth), self.space, self.edge)

    @property
    def size(self) -> Size:
        """
        Returns the size of the space of the seed
        :return: size
        """
        return self.space.size

    def check_size(self,
                   size: Size) -> bool:
        """
        Returns True if the space size is within the provided limits
        :return:
        """
        return self.size <= size

    def update_max_size(self):
        """
        Updates the seed max sizes if one of its dimension is superior to the maximum values
        This is needed because the first face might have a large width or depth than the maximum
        size value
        :return:
        """
        self.max_size.width = max(self.size.width + EPSILON_MAX_SIZE, self.max_size.width)
        self.max_size.depth = max(self.size.depth + EPSILON_MAX_SIZE, self.max_size.depth)

    def get_components_max_size(self) -> Size:
        """
        Returns the max size for the seed space according to its component
        :return:
        """
        max_width = 0
        max_depth = 0
        max_area = 0
        for component in self.components:
            max_area = max(max_area, component.category.seed_category.param('max_size').area)
            max_width = max(max_width, component.category.seed_category.param('max_size').width)
            max_depth = max(max_depth, component.category.seed_category.param('max_size').depth)

        return Size(max_area, max_width, max_depth)

    def create_max_size_constraint(self):
        """
        Creates a max_size constraint
        :return:
        """
        return copy.deepcopy(CONSTRAINTS['max_size']).set(max_size=self.max_size)

    def update_max_size_constraint(self):
        """
        Updates the max_size constraint
        :return:
        """
        self.update_max_size()
        self.max_size_constraint.set(max_size=self.max_size)

    @property
    def growth_method(self) -> Optional['Action']:
        """
        Returns the current growth method
        :return:
        """
        if self.growth_method_index >= len(self.growth_methods):
            return None
        return self.growth_methods[self.growth_method_index]

    def grow(self) -> Sequence['Space']:
        """
        Tries to grow the seed space by one face
        Returns the list of the faces added
        :param self:
        :return:
        """
        if self.growth_method is None:
            return []

        # initialize first face
        if self.space is None:
            self.edge.face.space.remove_face(self.edge.face)
            self.space = Space(self.seeder.plan, self.edge, space_catalog['seed'])
            self.seeder.plan.add_space(self.space)
            self.update_max_size_constraint()
            return [self.space]

        modified_spaces = self.growth_method.apply_to(self.space, (self,),
                                                      (self.max_size_constraint,))


        # TODO: TO BE DELETED,
        # we should find a much faster and cleaner way to do plot animation
        if PLOT_STEPS:
            global AX
            AX = self.seeder.plan.plot(AX, save=False,
                                       options=('fill', 'border', 'half-edge', 'face'))
            self.seeder.plot(AX)
            plt.pause(0.000001)
            AX.clear()
        #################

        if not modified_spaces:
            self.growth_method_index += 1

        return modified_spaces

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
        seed_distance_to_edge = 15  # per convention
        point = barycenter(self.edge.start.coords, self.edge.end.coords, 0.5)
        point = move_point(point, self.edge.normal, seed_distance_to_edge)
        return plot_point([point[0]], [point[1]], ax, save=False)


if __name__ == '__main__':

    import libs.reader as reader
    from libs.grid import GRIDS
    from libs.selector import edge_length

    from libs.shuffle import simple_shuffle

    logging.getLogger().setLevel(logging.DEBUG)

    def grow_a_plan():
        """
        Test
        :return:
        """
        plan = reader.create_plan_from_file('Paris18_A302.json')

        new_plan = GRIDS['sequence_grid'].apply_to(plan)

        seeder = Seeder(new_plan)
        seeder.add_condition(edge_length(50.0), 'duct')
        seeder.grow()

        print(seeder)

        ax = new_plan.plot(save=False, options=('fill', 'border', 'face'))
        seeder.plot(ax)
        plt.show()
        plt.pause(60)

        assert new_plan.check()

        # simple_shuffle.apply_to(seeder)

    grow_a_plan()
