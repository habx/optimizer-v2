# coding=utf-8
"""
Seed module

A seeder can be applied to plant seeds in a plan.
The seeds are planted along the non empty components (space or linear) of the plan according
to specified rules.

After being planted the seeds can be grown according to provided actions.
These actions are stored in the seed category of the space or linear

"""
from typing import TYPE_CHECKING, List, Optional, Dict, Generator, Sequence
import logging
import copy

import matplotlib.pyplot as plt

from libs.plan import Space, PlanComponent, Plan, Linear
from libs.plot import plot_point, Plot
from libs.size import Size
from libs.utils.catalog import Catalog
from libs.action import Action

from libs.category import SPACE_CATEGORIES
from libs.constraint import CONSTRAINTS
from libs.selector import SELECTORS
from libs.mutation import MUTATIONS

from libs.utils.geometry import barycenter, move_point

if TYPE_CHECKING:
    from libs.mesh import Edge
    from libs.selector import Selector
    from libs.constraint import Constraint

EPSILON_MAX_SIZE = 10.0


class Seeder:
    """
    Seeder Class
    """
    def __init__(self, plan: Plan, growth_methods: Catalog):
        self.plan = plan
        self.seeds: List['Seed'] = []
        self.selectors: Dict[str, 'Selector'] = {}
        self.growth_methods = growth_methods
        self.plot = None

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

                if isinstance(component, Space):
                    for edge in self.space_seed_edges(component):
                        seed_edge = edge
                        self.add_seed(seed_edge, component)

                if isinstance(component, Linear):
                    seed_edge = component.edge
                    self.add_seed(seed_edge, component)

    def merge(self, edge: 'Edge', component: 'PlanComponent') -> bool:
        """
        Checks if a potential seed edge shares a face with a seed. If another seed is found,
        return True else False
        :param edge:
        :param component:
        :return:
        """
        for seed in self.seeds:
            if seed.edge.face is edge.face:
                seed.add_component(component)
                return True
        return False

    def grow(self, show: bool = False):
        """
        Creates the space for each seed
        :return:
        """
        if show:
            self.plot = Plot()
            plt.ion()
            self.plot.draw(self.plan)
            self.plot_seeds(self.plot.ax)
            plt.show()
            plt.pause(0.0001)

        # grow the seeds
        while True:
            all_spaces_modified = []
            for seed in self.seeds:
                spaces_modified = seed.grow()
                all_spaces_modified += spaces_modified

                if spaces_modified and show:
                    self.plot.update(spaces_modified)
            # stop to grow once we cannot grow anymore
            if not all_spaces_modified:
                break

    def add_seed(self, seed_edge: 'Edge', component: PlanComponent):
        """
        Adds a seed to the seeder
        :param seed_edge:
        :param component
        :return:
        """
        # check for none space
        if seed_edge.face is None:
            return
        # only add a seed if the seed edge points to an empty space
        if seed_edge.space and seed_edge.space.category.name != 'empty':
            return

        if not self.merge(seed_edge, component):
            new_seed = Seed(self, seed_edge, component)
            self.seeds.append(new_seed)

    def space_seed_edges(self, space: 'Space') -> Generator['Edge', bool, 'None']:
        """
        returns the space edges
        :param space:
        :return:
        """
        category_name = space.category.name
        if category_name not in self.selectors:
            for edge in space.edges:
                if edge.pair.face is not None:
                    yield edge.pair
        else:
            yield from self.selectors[category_name].yield_from(space)

    def add_condition(self, selector: 'Selector', category_name: str):
        """
        Adds a selector to create a seed from a space component.
        The selector returns the edges that will receive a seed.
        :param selector:
        :param category_name: the str name of a category
        """
        self.selectors[category_name] = selector

    def plot_seeds(self, ax):
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
        # per convention we apply the growth method corresponding
        # to the first component category name
        self.growth_methods = self.get_growth_methods()
        self.growth_action_index = 0
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

    def get_growth_methods(self) -> Sequence['GrowthMethod']:
        """
        Retrieves the growth_method of each component
        :return:
        """
        growth_methods = []
        for component in self.components:
            component_name = component.category.name
            growth_methods.append(self.seeder.growth_methods(component_name, 'default'))

        return growth_methods

    def get_components_max_size(self) -> Size:
        """
        Returns the max size for the seed space according to its component
        :return:
        """
        max_width = 0
        max_depth = 0
        max_area = 0
        for growth_method in self.growth_methods:
            max_area = max(max_area, growth_method.param('max_size').area)
            max_width = max(max_width, growth_method.param('max_size').width)
            max_depth = max(max_depth, growth_method.param('max_size').depth)

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
    def growth_action(self) -> Optional['Action']:
        """
        Returns the current growth action.
        Per convention we only use the actions of the first component
        :return:
        """
        if self.growth_action_index >= len(self.growth_methods[0].actions):
            return None
        return self.growth_methods[0].actions[self.growth_action_index]

    def grow(self) -> Sequence['Space']:
        """
        Tries to grow the seed space by one face
        Returns the list of the faces added
        :param self:
        :return:
        """
        if self.growth_action is None:
            return []

        # initialize first face
        if self.space is None:
            empty_space = self.edge.face.space
            if empty_space.category.name != 'empty':
                raise ValueError('The seed should point towards an empty space')
            empty_space.remove_face(self.edge.face)
            self.space = Space(self.seeder.plan, self.edge, SPACE_CATEGORIES['seed'])
            self.seeder.plan.add_space(self.space)
            self.update_max_size_constraint()
            return [self.space, empty_space]

        modified_spaces = self.growth_action.apply_to(self.space, (self,),
                                                      (self.max_size_constraint,))

        if not modified_spaces:
            self.growth_action_index += 1
        else:
            pass

        return modified_spaces

    def add_component(self, component: PlanComponent):
        """
        Adds a plan component to the seed
        :param component:
        :return:
        """
        self.components.append(component)
        self.growth_methods = self.get_growth_methods()
        self.max_size = self.get_components_max_size()

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


class GrowthMethod:
    """
    A category of a seed
    """
    def __init__(self, name: str, constraints: Optional[Sequence['Constraint']] = None,
                 actions: Optional[Sequence['Action']] = None):
        constraints = constraints or []
        self.name = name
        self.constraints = {constraint.name: constraint for constraint in constraints}
        self.actions = actions or None

    def __repr__(self):
        return 'Seed: {0}'.format(self.name)

    def param(self, param_name: str, constraint_name: Optional[str] = None):
        """
        Returns the constraints parameters of the given name
        :param param_name:
        :param constraint_name: optional, the name of the desired constraint. If not provided, will
        search for all constraint parameters and return the first parameter with the provided name.
        :return: the value of the parameter
        """
        if constraint_name is not None:
            constraint = self.constraint(constraint_name)
            return constraint.params[param_name]

        for constraint in self.constraints:
            if param_name in self.constraints[constraint].params:
                return self.constraints[constraint].params[param_name]
        else:
            raise ValueError('Parameter {0} not present in any of the seed constraints {1}'
                             .format(param_name, self))

    def constraint(self, constraint_name: str):
        """
        retrieve a constraint by name
        :param constraint_name:
        :return:
        """
        if constraint_name not in self.constraints:
            raise ValueError('Constraint {0} not present in seed category {1}'
                             .format(constraint_name, self))

        return self.constraints[constraint_name]


classic_seed_category = GrowthMethod(
    'default',
    (CONSTRAINTS['max_size_s'],),
    (
        Action(SELECTORS.factory['oriented_edges'](('horizontal',)), MUTATIONS['add_face']),
        Action(SELECTORS.factory['oriented_edges'](('vertical',)), MUTATIONS['add_face'], True),
        Action(SELECTORS['boundary_other_empty_space'], MUTATIONS['add_face'])
    )
)

duct_seed_category = GrowthMethod(
    'duct',
    (CONSTRAINTS['max_size_xs'],),
    (
        Action(SELECTORS.factory['oriented_edges'](('horizontal',)), MUTATIONS['add_face']),
        Action(SELECTORS['surround_seed_component'], MUTATIONS['add_face']),
        Action(SELECTORS.factory['oriented_edges'](('vertical',)), MUTATIONS['add_face'], True),
        Action(SELECTORS['boundary_other_empty_space'], MUTATIONS['add_face'])
    )
)

front_door_seed_category = GrowthMethod(
    'frontDoor',
    (CONSTRAINTS['max_size_xs'],),
    (
        Action(SELECTORS.factory['oriented_edges'](('horizontal',)), MUTATIONS['add_face']),
        Action(SELECTORS.factory['oriented_edges'](('vertical',)), MUTATIONS['add_face'], True),
        Action(SELECTORS['boundary_other_empty_space'], MUTATIONS['add_face'])
    )
)


GROWTH_METHODS = Catalog('seeds').add(
    classic_seed_category,
    duct_seed_category,
    front_door_seed_category)


if __name__ == '__main__':

    import libs.reader as reader
    from libs.grid import GRIDS
    from libs.selector import SELECTORS

    logging.getLogger().setLevel(logging.DEBUG)

    def grow_a_plan():
        """
        Test
        :return:
        """
        input_file = reader.BLUEPRINT_INPUT_FILES[9]  # 9 Antony B22, 13 Bussy 002
        plan = reader.create_plan_from_file(input_file)

        seeder = Seeder(plan, GROWTH_METHODS)
        seeder.add_condition(SELECTORS['seed_duct'], 'duct')
        GRIDS['ortho_grid'].apply_to(plan)

        seeder.plant()
        seeder.grow(show=False)

        ax = plan.plot(save=False)
        seeder.plot_seeds(ax)
        plt.show()

        print(seeder)

        assert plan.check()

    grow_a_plan()
