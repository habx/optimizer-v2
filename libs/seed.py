# coding=utf-8
"""
Seed module

A seeder can be applied to plant seeds in a plan.
The seeds are planted along the non empty components (space or linear) of the plan according
to specified rules.

After being planted the seeds can be grown according to provided actions.
These actions are stored in the seed category of the space or linear

Remaining empty spaces of the plan are seeded as well through a filler. Operation is performed
until the space is totally filled

"""

from typing import Tuple, TYPE_CHECKING, List, Optional, Dict, Generator, Sequence
import logging
import copy

import matplotlib.pyplot as plt

from libs.plan import Space, PlanComponent, Plan, Linear
from libs.plot import plot_point, Plot
from libs.size import Size
from libs.action import Action

from libs.constraint import CONSTRAINTS
from libs.selector import SELECTORS, SELECTOR_FACTORIES
from libs.mutation import MUTATIONS
from libs.category import SPACE_CATEGORIES

from libs.utils.geometry import barycenter, move_point

if TYPE_CHECKING:
    from libs.mesh import Edge, Face
    from libs.selector import Selector
    from libs.constraint import Constraint
    from libs.shuffle import Shuffle

EPSILON_MAX_SIZE = 10.0


class Seeder:
    """
    Seeder Class
    """

    def __init__(self,
                 plan: Plan,
                 growth_methods: Dict[str, 'GrowthMethod']):
        self.plan = plan
        self.seeds: List['Seed'] = []
        self.selectors: Dict[str, 'Selector'] = {}
        self.growth_methods = growth_methods
        self.plot: Optional['Plot'] = None

    def __repr__(self):
        output = 'Seeder:\n'
        for seed in self.seeds:
            output += '• ' + seed.__repr__() + '\n'
        return output

    def plant(self) -> 'Seeder':
        """
        Creates the seeds
        :return:
        """
        for component in self.plan.get_components():
            if ((component.category.seedable and self.growth_methods["default"])
                    or component.category.name in self.selectors):

                if isinstance(component, Space):
                    for edge in self.space_seed_edges(component):
                        seed_edge = edge
                        self.add_seed(seed_edge, component)

                if isinstance(component, Linear):
                    seed_edge = component.edge
                    self.add_seed(seed_edge, component)

        return self

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
                seed.add_component(edge, component)
                return True
        return False

    def grow(self, show: bool = False, plot: Optional['Plot'] = None) -> 'Seeder':
        """
        Creates the space for each seed
        :param show
        :param plot
        :return: the seeder
        """
        logging.debug("Seeder: Starting to grow")

        # Real time plot updates
        if show:
            self._initialize_plot(plot)

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

        self.plan.remove_null_spaces()

        return self

    def fill(self,
             growth_methods: ['GrowthMethod'],
             selector_and_category: Tuple['Selector', str],
             recursive: bool = False,
             show: bool = False) -> 'Seeder':
        """
        Fills the empty space
        :param growth_methods:
        :param selector_and_category:
        :param recursive: whether to repeat the fill until there are no empty spaces
        :param show: whether to display the plot in real time
        :return:
        """
        if show:
            self._initialize_plot()
        max_recursion = 10  # to prevent infinite loops
        while True:
            (Seeder(self.plan, growth_methods).add_condition(*selector_and_category)
             .plant()
             .grow(show=show, plot=self.plot))
            max_recursion -= 1
            if not recursive or self.plan.count_category_spaces("empty") == 0:
                break
            if max_recursion == 0:
                raise Exception("Seed: Fill max recursion reach")

        return self

    def simplify(self, selector: 'Selector', show: bool = False) -> 'Seeder':
        """
        Merges the seed spaces according to the selector
        :param selector:
        :param show:
        :return:
        """
        if show:
            self._initialize_plot()

        continue_merge = True
        while continue_merge:
            continue_merge = False
            merged_done = False
            for space in self.plan.get_spaces("seed"):
                for edge in selector.yield_from(space):
                    other_space = self.plan.get_space_of_edge(edge.pair)
                    space.merge(other_space)
                    merged_done = True
                    continue_merge = True
                    # update plot
                    if show:
                        self.plot.update([other_space, space])
                    break
                if merged_done:
                    logging.debug("Seed: Space has been merged: %s", space)
                    break

        self.plan.remove_null_spaces()

        return self

    def shuffle(self, shuffle: 'Shuffle', show: bool = False) -> 'Seeder':
        """
        Runs a shuffle on the plan
        :param shuffle:
        :param show: whether to show the plot
        :return:
        """
        shuffle.run(self.plan, [self], show=show, plot=self.plot)
        return self

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
        space = self.plan.get_space_of_edge(seed_edge)
        if space and space.category.name != 'empty':
            logging.debug("Seed: Cannot add a seed to an edge "
                          "that does not belong to an empty space")
            return

        if not self.merge(seed_edge, component):
            new_seed = Seed(self, seed_edge, component)
            self.seeds.append(new_seed)

    def space_seed_edges(self, space: 'Space') -> Generator['Edge', bool, 'None']:
        """
        returns the edges of the space that should produce a seed on their pair edge
        (for example : a duct)
        If the space category has a specific seed condition we apply it. Otherwise we seed
        every pair edge.
        :param space:
        :return:
        """
        category_name = space.category.name
        if category_name in self.selectors:  # note the space does not need to be seedable here
            yield from self.selectors[category_name].yield_from(space)
        elif space.category.seedable:
            for edge in space.edges:
                # only seed empty space
                seed_space = space.plan.get_space_of_edge(edge.pair)
                if seed_space and seed_space.category.name == "empty":
                    yield edge.pair

    def add_condition(self, selector: 'Selector', category_name: str) -> 'Seeder':
        """
        Adds a selector to create a seed from a space component.
        The selector returns the edges that will receive a seed.
        :param selector:
        :param category_name: the name of a category
        """
        self.selectors[category_name] = selector
        return self

    def _initialize_plot(self, plot: Optional['Plot'] = None):
        """
        Creates a plot
        :return:
        """
        # if the seeder has already a plot : do nothing
        if self.plot:
            return

        if not plot:
            self.plot = Plot()
            plt.ion()
            self.plot.draw(self.plan)
            self.plot_seeds(self.plot.ax)
            plt.show()
            plt.pause(0.0001)
        else:
            self.plot = plot

    def plot_seeds(self, ax):
        """
        Plots the seeds point
        :param ax:
        :return:
        """
        for seed in self.seeds:
            ax = seed.plot(ax)
        return ax

    def get_seed_from_space(self, space: 'Space') -> Optional['Seed']:
        """
        Return the seed corresponding to the space.
        Returns None if the space has no corresponding seed.
        :param space:
        :return:
        """
        for seed in self.seeds:
            if space is seed.space:
                return seed
        return None


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
        self.edges = [edge]  # the reference edge of the seed
        self.components = [plan_component] if PlanComponent else []  # the components of the seed
        self.space: Optional[Space] = None  # the seed space
        # per convention we apply the growth method corresponding
        # to the first component category name
        self.growth_methods = self.get_growth_methods()
        self.growth_action_index = 0
        self.max_size = self.get_components_max_size()
        self.max_size_constraint = self.create_max_size_constraint()

    def __repr__(self):
        if self.space is not None:
            return ('Seed: {0}, area: {1}, width: {2}, depth: {3} - {4}, ' +
                    '{5}').format(self.components, str(self.space.area), str(self.size.width),
                                  str(self.size.depth), self.space, self.edge)
        else:
            return 'Seed: {0} - {1})'.format(self.components, self.edge)

    @property
    def size(self) -> Size:
        """
        Returns the size of the space of the seed
        :return: size
        """
        return self.space.size

    @property
    def edge(self):
        """
        Returns the first edge of the seed
        :return:
        """
        return self.edges[0]

    def face_has_component(self, face: 'Face') -> bool:
        """
        Returns True if the face is linked to a component of the Space
        :param face:
        :return:
        """
        for edge in face.edges:
            if edge in self.edges:
                return True
        return False

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
            method = self.seeder.growth_methods.get(component_name,
                                                    self.seeder.growth_methods['default'])
            if method is not None:
                growth_methods.append(method)

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
        return copy.deepcopy(CONSTRAINTS["max_size_seed"]).set(max_size=self.max_size)

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
        if not self.growth_methods:
            return None
        if self.growth_action_index >= len(self.growth_methods[0].actions):
            return None
        return self.growth_methods[0].actions[self.growth_action_index]

    def _create_seed_space(self) -> ['Space']:
        """
        Creates the initial seed space
        Returns the modified spaces
        (the created space and the space from which the face was removed)
        :return:
        """
        logging.debug("Seed: Creating the seed space")

        face = self.edge.face
        empty_space = self.seeder.plan.get_space_of_face(face)

        if not empty_space or empty_space.category.name != 'empty':
            raise ValueError('The seed should point towards an empty space')

        modified_spaces = empty_space.remove_face(face)
        self.space = Space(self.seeder.plan, empty_space.floor, self.edge, SPACE_CATEGORIES["seed"])
        self.update_max_size_constraint()
        return [self.space] + modified_spaces

    def grow(self) -> Sequence['Space']:
        """
        Tries to grow the seed space by one face
        Returns the list of the faces added
        :param self:
        :return:
        """
        logging.debug("Seed: Growing %s", self)

        if self.growth_action is None:
            return []

        for growth_method in self.growth_methods:
            for action in growth_method.actions:
                action.flush()

        # initialize first face
        if self.space is None:
            return self._create_seed_space()

        modified_spaces = self.growth_action.apply_to(self.space, [self.seeder],
                                                      [self.max_size_constraint])

        if not modified_spaces:
            self.growth_action_index += 1
            logging.debug("Seed: Switching to next growth action : %i - %s",
                          self.growth_action_index, self)
        else:
            pass

        return modified_spaces

    def add_component(self, edge: 'Edge', component: PlanComponent):
        """
        Adds a plan component to the seed
        :param edge
        :param component:
        :return:
        """
        self.edges.append(edge)
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
    GrowthMethod class
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


# Growth Methods

fill_seed_category = GrowthMethod(
    'default',
    (CONSTRAINTS['max_size_s_seed'],),
    (
        Action(SELECTORS['homogeneous'], MUTATIONS['swap_face']),
    )
)

fill_small_seed_category = GrowthMethod(
    'empty',
    (CONSTRAINTS["max_size_seed"],),
    (
        Action(SELECTOR_FACTORIES['oriented_edges'](('horizontal',)), MUTATIONS['swap_face']),
        Action(SELECTOR_FACTORIES['oriented_edges'](('vertical',)), MUTATIONS['swap_face'], True),
        Action(SELECTORS['boundary_other_empty_space'], MUTATIONS['swap_face'])
    )
)

classic_seed_category = GrowthMethod(
    'default',
    (CONSTRAINTS["max_size_s_seed"],),
    (
        Action(SELECTOR_FACTORIES['oriented_edges'](('horizontal',)), MUTATIONS['swap_face']),
        Action(SELECTOR_FACTORIES['oriented_edges'](('vertical',)), MUTATIONS['swap_face'], True),
        Action(SELECTORS['boundary_other_empty_space'], MUTATIONS['swap_face'])
    )
)

duct_seed_category = GrowthMethod(
    'duct',
    (CONSTRAINTS["max_size_xs_seed"],),
    (
        Action(SELECTOR_FACTORIES['oriented_edges'](('horizontal',)), MUTATIONS['swap_face']),
        Action(SELECTORS['seed_component_boundary'], MUTATIONS['swap_face']),
        Action(SELECTOR_FACTORIES['oriented_edges'](('vertical',)), MUTATIONS['swap_face'], True),
        Action(SELECTORS['boundary_other_empty_space'], MUTATIONS['swap_face'])
    )
)

front_door_seed_category = GrowthMethod(
    'frontDoor',
    (CONSTRAINTS["max_size_xs_seed"],),
    (
        Action(SELECTOR_FACTORIES['oriented_edges'](('horizontal',)), MUTATIONS['swap_face']),
        Action(SELECTOR_FACTORIES['oriented_edges'](('vertical',)), MUTATIONS['swap_face'], True),
        Action(SELECTORS['boundary_other_empty_space'], MUTATIONS['swap_face'])
    )
)

GROWTH_METHODS = {
    "default": classic_seed_category,
    "duct": duct_seed_category,
    "frontDoor": front_door_seed_category,
}

FILL_METHODS_HOMOGENEOUS = {
    "empty": fill_seed_category,
    "default": None
}

FILL_METHODS = {
    "empty": classic_seed_category,
    "default": None
}

if __name__ == '__main__':
    import libs.reader as reader
    from libs.grid import GRIDS
    from libs.selector import SELECTORS
    from libs.shuffle import SHUFFLES
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--plan_index", help="choose plan index",
                        default=0)

    args = parser.parse_args()
    plan_index = int(args.plan_index)

    logging.getLogger().setLevel(logging.DEBUG)


    def grow_a_plan():
        """
        Test
        :return:
        """
        logging.debug("Start test")
        input_file = reader.get_list_from_folder()[
            plan_index]  # 9 Antony B22, 13 Bussy 002

        plan = reader.create_plan_from_file(input_file)

        GRIDS['finer_ortho_grid'].apply_to(plan)

        seeder = Seeder(plan, GROWTH_METHODS).add_condition(SELECTORS['seed_duct'], 'duct')
        plan.plot()
        (seeder.plant()
         .grow(show=True)
         .shuffle(SHUFFLES['seed_square_shape'], show=True)
         .fill(FILL_METHODS_HOMOGENEOUS, (SELECTORS["farthest_couple_middle_space_area_min_100000"],
                                          "empty"), show=True)
         .fill(FILL_METHODS_HOMOGENEOUS, (SELECTORS["single_edge"], "empty"), recursive=True,
               show=True)
         .simplify(SELECTORS["fuse_small_cell_without_components"], show=True)
         .shuffle(SHUFFLES['seed_square_shape'], show=True))

        plan.plot(show=True)
        plt.show()


    grow_a_plan()
