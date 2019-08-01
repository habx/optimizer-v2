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

from typing import (TYPE_CHECKING, List, Optional, Dict,
                    Generator, Sequence, Set, Tuple, Callable, Union)
import logging
import copy

import matplotlib.pyplot as plt

from libs.plan.plan import Space, PlanComponent, Plan, Linear
from libs.io.plot import Plot
from libs.specification.size import Size
from libs.operators.action import Action

from libs.operators.constraint import CONSTRAINTS
from libs.operators.selector import SELECTORS, SELECTOR_FACTORIES
from libs.operators.mutation import MUTATIONS
from libs.plan.category import SPACE_CATEGORIES
from libs.utils.geometry import ccw_angle, truncate

if TYPE_CHECKING:
    from libs.mesh.mesh import Edge, Face
    from libs.operators.selector import Selector
    from libs.operators.constraint import Constraint
    import uuid

FillMethod = MergeMethod = Callable[['Seeder', bool], List['Space']]

SQM = 10000

# TODO: these globals should really be members of the Seeder instance
EPSILON_MAX_SIZE = 10.0
SEEDER_ACTIVATION_NBR_CELLS = 18
MIN_SEEDER_SPACE_AREA = SQM


class Seeder:
    """
    Seeder Class

    A seeder needs to be given three lists of methods:
    • a seed method list : how the seeder must plant seed
    • a growth method list : how the seeder must grow the seeds
    • a fill method list : how the seeder must fill the remaining empty spaces with seed spaces

    Note : a space can be of the category "seed" but is not necessarily linked to a Seed instance.
           Seed instances are only used for the initial seed that correspond to the seedable
           components of the plan (eg: window, duct etc.)

    The seeder is expected to transform the plan by replacing its empty spaces with seed spaces.
    No empty space should remain in the plan once the seeder has been applied.

    If a plot is given to the seeder, it will use it to display in real time the changes occurring
    on the plan. It is useful to chain visualization of different stages in the optimizer pipeline
    on the same plot. For example : the grid, then the seeder, then the shuffle.
    """

    def __init__(self,
                 seed_methods: Dict[str, 'Selector'],
                 growth_methods: Dict[str, 'GrowthMethod'],
                 fill_methods: List[FillMethod],
                 merge_methods: Optional[List[MergeMethod]] = None,
                 plot: Optional['Plot'] = None):
        self.seed_methods = seed_methods
        self.growth_methods = growth_methods
        self.fill_methods = fill_methods
        self.merge_methods = merge_methods or []

        self.seeds: List['Seed'] = []
        self.plan: Plan = None

        self.plot = plot

    def __repr__(self):
        output = 'Seeder:\n'
        for seed in self.seeds:
            output += '• ' + seed.__repr__() + '\n'
        return output

    def _init(self):
        self.plan = None
        self.seeds = []

    def apply_to(self, plan: 'Plan', show: bool = False) -> 'Plan':
        """
        Runs the seeder
        :param plan:
        :param show: whether to display a real-time vizualisation of the seeder
        :return:
        """
        self._init()
        self.plan = plan
        # Real time plot updates
        if show:
            self._initialize_plot()

        # temporary dirty implementation
        nbr_grid_cells = len(list(f for s in plan.empty_spaces for f in s.faces))

        # If there are already less faces in the grid than the objective of the seeder
        # we can just transform each face in a new seed space directly
        # we still execute the merge methods in order to remove the small spaces
        if nbr_grid_cells <= SEEDER_ACTIVATION_NBR_CELLS:
            for space in plan.empty_spaces:
                for face in space.faces:
                    Space(plan, space.floor, face.edge, SPACE_CATEGORIES["seed"])
                plan.remove(space)
            return self.merge(show).plan

        return self.plant(show).grow(show).fill(show).merge(show).plan

    def plant(self, show: bool = False) -> 'Seeder':
        """
        Creates the seeds
        :return:
        """
        logging.debug("Seeder: Starting to plant")

        # Real time plot updates
        if show:
            self._initialize_plot()

        for component in self.plan.get_components():
            if ((component.category.seedable and self.growth_methods["default"])
                    or component.category.name in self.seed_methods):

                if isinstance(component, Space):
                    for edge in self._space_seed_edges(component):
                        seed_edge = edge
                        self._add_seed(seed_edge, component, show)

                if isinstance(component, Linear):
                    seed_edge = component.edge
                    self._add_seed(seed_edge, component, show)

        return self

    def grow(self, show: bool = False) -> 'Seeder':
        """
        Creates the space for each seed
        :param show
        :return: the seeder
        """
        logging.debug("Seeder: Starting to grow")

        # Real time plot updates
        if show:
            self._initialize_plot()

        # grow the seeds
        while True:
            all_done = True
            for seed in self.seeds:
                done = seed.grow(show=show)
                all_done = all_done and done
            # stop to grow when all growth method are done
            if all_done:
                break

        self.plan.remove_null_spaces()

        return self

    def fill(self, show: bool = False) -> 'Seeder':
        """
        Runs the fill methods
        :param show:
        :return:
        """
        logging.debug("Seeder: Starting to fill")
        # Real time plot updates
        if show:
            self._initialize_plot()

        for method in self.fill_methods:
            self._execute_fill_or_merge_method(method, show)

        return self

    def merge(self, show: bool = False) -> 'Seeder':
        """
        Runs the fill methods
        :param show:
        :return:
        """
        logging.debug("Seeder: Starting to merge")
        # Real time plot updates
        if show:
            self._initialize_plot()

        for method in self.merge_methods:
            self._execute_fill_or_merge_method(method, show)

        return self

    def _execute_fill_or_merge_method(self, method: Union[FillMethod, MergeMethod], show: bool):
        """
        Executes a fill method
        :param method:
        :return:
        """
        new_spaces = method(self, show)

        if new_spaces:
            self.plan.remove_null_spaces()  # TODO: is this really useful ?
            self._execute_fill_or_merge_method(method, show)

    def _merge_seeds(self, edge: 'Edge', component: 'PlanComponent') -> bool:
        """
        Checks if a potential seed edge already belongs to a seed space.
        If this is the case the component edge is added to the seed space.
        :param edge:
        :param component:
        :return: True if the edge already belongs to a seed space
        """
        for seed in self.seeds:
            if seed.space and seed.space.has_edge(edge):
                seed.add_component(edge, component)
                return True
        return False

    def _add_seed(self, seed_edge: 'Edge', component: PlanComponent, show: bool):
        """
        Adds a seed to the seeder
        :param seed_edge:
        :param component
        :param show
        :return:
        """
        # check for none space
        if seed_edge.face is None:
            return

        # only add a seed if the seed edge points to an empty space
        space = self.plan.get_space_of_edge(seed_edge)
        if not space or (space and space.category.name != 'empty'):
            logging.debug("Seed: Cannot add a seed to an edge "
                          "that does not belong to an empty space")
            return

        if not self._merge_seeds(seed_edge, component):
            new_seed = Seed(self, seed_edge, component)
            self.seeds.append(new_seed)
            # initialize first face
            modified_spaces = new_seed.create_seed_space()
            if show:
                self.plot.draw_seeds_points(self)
                self.plot.update(modified_spaces)

    def _space_seed_edges(self, space: 'Space') -> Generator['Edge', bool, 'None']:
        """
        returns the edges of the space that should produce a seed on their pair edge
        (for example : a duct)
        If the space category has a specific seed condition we apply it. Otherwise we seed
        every pair edge.
        :param space:
        :return:
        """
        category_name = space.category.name
        if category_name in self.seed_methods:  # note the space does not need to be seedable here
            yield from self.seed_methods[category_name].yield_from(space)
        elif space.category.seedable:
            for edge in space.edges:
                # only seed empty space
                seed_space = space.plan.get_space_of_edge(edge.pair)
                if seed_space and seed_space.category.name == "empty":
                    yield edge.pair

    def _initialize_plot(self, plot: Optional['Plot'] = None):
        """
        Creates a plot
        :return:
        """
        # if the seeder has already a plot : do nothing
        if self.plot:
            return

        if not plot:
            self.plot = Plot(self.plan)
            plt.ion()
            self.plot.draw(self.plan)
            self.plot.draw_seeds_points(self)
            plt.show()
            plt.pause(0.0001)
        else:
            self.plot = plot

    def get_seed_from_space(self, space: 'Space') -> Optional['Seed']:
        """
        Return the seed corresponding to the space.
        Returns None if the space has no corresponding seed.
        Used by selector if it needs to retrieve seed information
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
                 plan_component: Optional[PlanComponent] = None,
                 space: Optional['Space'] = None):
        self.seeder = seeder
        self.edges = [edge]  # the reference edge of the seed
        self.components = [plan_component] if plan_component else []  # the components of the seed
        self.space = space  # the seed space
        # per convention we apply the growth method corresponding
        # to the first component category name
        self.growth_methods = self.get_growth_methods()
        self._growth_action_index = 0
        self._number_of_pass = 0
        self.max_size = self.get_components_max_size()
        self.max_size_constraint = self.create_max_size_constraint()

    def __repr__(self):
        if self.space is not None:
            return ('Seed: {0} - {1}, area: {2}, width: {3}, depth: {4} - {5}, ' +
                    '{6}').format(id(self), self.components, str(self.space.area),
                                  str(self.size.width), str(self.size.depth), self.space, self.edge)
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
        self.max_size.width = max(self.size.width + EPSILON_MAX_SIZE, self.max_size.width or 0)
        self.max_size.depth = max(self.size.depth + EPSILON_MAX_SIZE, self.max_size.depth or 0)
        self.max_size.area = max(self.max_size.width * self.max_size.depth,
                                 self.size.area + EPSILON_MAX_SIZE ** 2,
                                 self.max_size.area or 0)

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
        # we make sure the growth methods are sorted per priority
        # this is needed in the case of space with several seedable items that each
        # have a different growth method and we want to control which growth method is
        # executed in priority
        growth_methods.sort(key=lambda g: g.priority, reverse=True)
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
        if self._growth_action_index >= len(self.growth_methods[0].actions):
            return None
        return self.growth_methods[0].actions[self._growth_action_index]

    def create_seed_space(self) -> ['Space']:
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
        self.space = Space(self.seeder.plan, empty_space.floor, self.edge,
                           SPACE_CATEGORIES["seed"])
        self.update_max_size_constraint()
        return [self.space] + modified_spaces

    def grow(self, show: bool = False) -> bool:
        """
        Tries to grow the seed space by one face
        Returns a boolean to indicate whether the growth is done
        :param show: whether to plot the growth
        :param self:
        :return:
        """

        if self.growth_action is None:
            logging.debug("Seed: No more growth action %s", self)
            return True

        logging.debug("Seed: Growing %s", self)

        for growth_method in self.growth_methods:
            for action in growth_method.actions:
                action.flush()

        modified_spaces = self.growth_action.apply_to(self.space, [self.seeder],
                                                      [self.max_size_constraint])
        if modified_spaces and show:
            self.seeder.plot.update(modified_spaces)
            # input("s")

        if not modified_spaces:
            if self._number_of_pass >= self.growth_action.number_of_pass - 1:
                self._number_of_pass = 0
                self._growth_action_index += 1
                logging.debug("Seed: Switching to next growth action : %i - %s",
                              self._growth_action_index, self)
            else:
                self._number_of_pass += 1

        return False

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


class GrowthMethod:
    """
    GrowthMethod class
    """

    def __init__(self,
                 name: str,
                 constraints: Optional[Sequence['Constraint']] = None,
                 actions: Optional[Sequence['Action']] = None,
                 priority: int = 0):
        constraints = constraints or []
        self.name = name
        self.constraints = {constraint.name: constraint for constraint in constraints}
        self.actions = actions or None
        self.priority = priority

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


# Seed Methods


SEED_METHODS = {
    "duct": SELECTORS['seed_duct']
}

# Growth Methods

GROWTH_METHODS = {
    "default": GrowthMethod(
        'default',
        (CONSTRAINTS["max_size_default_constraint_seed"],),
        (
            Action(SELECTOR_FACTORIES['oriented_edges'](('horizontal',)), MUTATIONS['swap_face'],
                   number_of_pass=2),
            Action(SELECTOR_FACTORIES['oriented_edges'](('vertical',)), MUTATIONS['swap_face'],
                   True),
            Action(SELECTORS['improved_aspect_ratio'], MUTATIONS['swap_face'])
        )
    ),
    "duct": GrowthMethod(
        'duct',
        (CONSTRAINTS["max_size_duct_constraint_seed"],),
        (
            Action(SELECTORS['along_duct_side'], MUTATIONS['swap_face']),
            Action(SELECTORS['best_aspect_ratio'], MUTATIONS['swap_face'])
        )
    ),
    "frontDoor": GrowthMethod(
        'frontDoor',
        (CONSTRAINTS["max_size_frontdoor_constraint_seed"],),
        (
            Action(SELECTORS['improved_aspect_ratio'], MUTATIONS['swap_face']),
        )
    ),
    "window": GrowthMethod(
        'window',
        (CONSTRAINTS["max_size_window_constraint_seed"],),
        (
            Action(SELECTORS['corner_fill'], MUTATIONS['swap_face'], True),
            Action(SELECTOR_FACTORIES['oriented_edges'](('horizontal',)), MUTATIONS['swap_face'],
                   number_of_pass=2),
            Action(SELECTOR_FACTORIES['oriented_edges'](('vertical',)), MUTATIONS['swap_face'],
                   True),
            Action(SELECTORS['improved_aspect_ratio'], MUTATIONS['swap_face'],
                   name="improved_aspect")
        ),
        priority=1
    ),
    "doorWindow": GrowthMethod(
        'doorWindow',
        (CONSTRAINTS["max_size_doorWindow_constraint_seed"],),
        (
            Action(SELECTORS['corner_fill'], MUTATIONS['swap_face'],
                   True),
            Action(SELECTOR_FACTORIES['oriented_edges'](('horizontal',)), MUTATIONS['swap_face'],
                   number_of_pass=2),
            Action(SELECTOR_FACTORIES['oriented_edges'](('vertical',)), MUTATIONS['swap_face'],
                   True),
            Action(SELECTORS['improved_aspect_ratio'], MUTATIONS['swap_face'],
                   name="improved_aspect")
        ),
        priority=1
    ),
    "startingStep": GrowthMethod(
        'startingStep',
        (CONSTRAINTS["max_size_frontdoor_constraint_seed"],),
        (
            Action(SELECTORS['improved_aspect_ratio'], MUTATIONS['swap_face']),
        )
    )
}


# FILL METHODS


def empty_to_seed(seeder: 'Seeder', show: bool) -> List['Space']:
    """
    Fill method
    Transforms each empty space into a seed space
    :param seeder:
    :param show:
    :return:
    """
    output = []
    modified_spaces = []
    for space in seeder.plan.get_spaces("empty"):
        modified_spaces.append(space)
        space.category = SPACE_CATEGORIES["seed"]
        output.append(space)
    if show:
        seeder.plot.update(modified_spaces)
    return output


def adjacent_faces(seeder: 'Seeder', show: bool) -> List['Space']:
    """
    Fill method
    Transforms each face adjacent to a seed space into a new seed space.
    Tries to merge newly created spaces together if adjacent to the same seed space.
    :param seeder:
    :param show:
    :return:
    """

    epsilon_aspect = 0.5

    def _get_adjacencies(_space: 'Space',
                         targeted_spaces: ['Space']) -> Set[Tuple['uuid.UUID', float]]:
        """
        Returns a set containing all the id of the spaces adjacent and the angle of the
        shared edge
        :param _space:
        :param targeted_spaces:
        :return:
        """
        _adjacencies = set()
        for _edge in _space.edges:
            _other = seeder.plan.get_space_of_edge(_edge.pair)
            if not _other or _other.category.name != "seed" or _other not in targeted_spaces:
                continue
            # store all adjacencies of each space
            _angle = truncate(ccw_angle(_edge.unit_vector) % 180.0)
            _adjacencies.add((_other.id, _angle))
        return _adjacencies

    seed_spaces = [seed.space for seed in seeder.seeds]

    # create new seed spaces
    new_spaces = []
    initial_spaces = []
    for seed_space in list(seeder.plan.get_spaces("seed")):
        initial_spaces.append(seed_space)
        for edge in seed_space.edges:
            other_space = seeder.plan.get_space_of_edge(edge.pair)
            if (not other_space
                    or other_space.category.name != "empty"
                    or other_space in new_spaces):
                continue

            modified_spaces = other_space.remove_face(edge.pair.face)
            # create a new seed space
            new_space = Space(seeder.plan, seed_space.floor, edge.pair, SPACE_CATEGORIES["seed"])
            new_spaces.append(new_space)
            if show:
                seeder.plot.update(modified_spaces + [new_space])

    # merge new seed spaces
    for seed_space in new_spaces:
        # compute adjacencies
        adjacencies = _get_adjacencies(seed_space, initial_spaces)
        for other in seed_space.adjacent_spaces():
            if other in seed_spaces or other.category.name != "seed":
                continue
            other_adjacencies = _get_adjacencies(other, initial_spaces)
            # extensive merge
            for angle in [a for _, a in adjacencies]:
                if (set(filter(lambda t: t[1] == angle, adjacencies)) ==
                        set(filter(lambda t: t[1] == angle, other_adjacencies)) != set()):
                    # check aspect ratio before merge
                    if (seed_space.aspect_ratio(other.faces)
                            <= seed_space.aspect_ratio() + epsilon_aspect):
                        seed_space.merge(other)
                        if other in new_spaces:
                            new_spaces.remove(other)
                        if show:
                            seeder.plot.update([seed_space, other])
                        break

    return new_spaces


def merge_small_cells(seeder: 'Seeder', show: bool) -> List['Space']:
    """
    Merges small spaces with neighbor space that has highest contact length
    If several neighbor spaces have same contact length, the smallest one is chosen
    Do not merge two spaces containing non mutable components
    Stop merge when the number of spaces is under the target number of spaces
    :param seeder:
    :param show:
    :return: the list of modified spaces
    """

    epsilon_length = 20
    if seeder.plan.indoor_area / SQM < 100:
        min_cell_area = MIN_SEEDER_SPACE_AREA
    else:
        min_cell_area = 1.5 * MIN_SEEDER_SPACE_AREA
    target_number_of_spaces = SEEDER_ACTIVATION_NBR_CELLS
    modified_spaces = []

    if len([s for s in seeder.plan.spaces if s.mutable]) < target_number_of_spaces:
        return modified_spaces

    for small_space in sorted((s for s in seeder.plan.get_spaces("seed") if s.area < min_cell_area),
                              key=lambda x: x.area):
        # adjacent mutable spaces of small_space
        adjacent_spaces = [s for s in small_space.adjacent_spaces() if s.mutable]
        if not adjacent_spaces:
            continue
        max_contact_length = max(map(lambda s: s.contact_length(small_space),
                                     adjacent_spaces))
        # select adjacent mutable spaces with highest contact length
        candidates = [adj for adj in adjacent_spaces if
                      adj.contact_length(small_space) > max_contact_length - epsilon_length]
        if not candidates:
            continue
        # in case there are several spaces with equal contact length,
        # merge with the smallest one
        selected = min(candidates, key=lambda s: s.area)

        # do not merge if the selected space contains a seed as well as the small space
        if seeder.get_seed_from_space(selected) and seeder.get_seed_from_space(small_space):
            continue

        selected.merge(small_space)
        modified_spaces = [selected, small_space]
        break

    if show:
        seeder.plot.update(modified_spaces)

    return modified_spaces


def merge_enclosed_faces(seeder: 'Seeder', show: bool) -> List['Space']:
    """
    Merge the enclosed face with the enclosing face
    :param seeder:
    :param show:
    :return:
    """
    ratio = 0.45
    modified_spaces = []
    for space in seeder.plan.get_spaces("seed"):
        adjacent_dict = {}
        perimeter = space.perimeter
        for edge in space.exterior_edges:
            other = space.plan.get_space_of_edge(edge.pair)
            if not other or other.category is not SPACE_CATEGORIES["seed"]:
                continue
            adjacent_dict[other.id] = adjacent_dict.get(other.id, 0) + edge.length
            if adjacent_dict[other.id] > perimeter * ratio:
                other.merge(space)
                modified_spaces += [space, other]
                break

    if show:
        seeder.plot.update(modified_spaces)

    return modified_spaces


def divide_along_line(space: 'Space', line_edges: List['Edge']) -> List['Space']:
    """
    Divides the space into sub-spaces, cut performed along the line formed by line_edges
    :param space:
    :param line_edges:
    :return:
    """

    def _face_on_side(_line_edges: List['Edge'], _space: 'Space') -> List['Face']:
        """
        List the faces of the space that are on one of both sides
        defined by line_edges
        """
        if _line_edges:
            face_ini = _line_edges[0].face
            list_side_face = [face_ini]
            add = [face_ini]
            added = True
            while added:
                added = False
                for face_ini in add:
                    for _face in _space.plan.get_space_of_face(face_ini).adjacent_faces(face_ini):
                        # adds faces adjacent to those already added
                        # do not add faces on the other side of the line
                        if (not [edge for edge in _line_edges if edge.pair in _face.edges]
                                and _face not in list_side_face):
                            list_side_face.append(_face)
                            add.append(_face)
                            added = True
            return list_side_face

    def _groups_of_adjacent_faces(_faces: List['Face']) -> List[List['Face']]:
        """
        get lists of faces in _faces forming adjacent groups
        :param _faces:
        :return:
        """
        if not _faces:
            return []
        ref_face = _faces[0]
        list_remaining = _faces[1:]
        groups = [[ref_face]]
        count = 0
        while list_remaining:
            adj = True
            while adj:
                adj = False
                for f in list_remaining[:]:
                    adjacent_f = [adj_f for adj_f in groups[count] if f.is_adjacent(adj_f)]
                    if adjacent_f:
                        groups[count].append(f)
                        list_remaining.remove(f)
                        adj = True
                        break
            if len(list_remaining) == 1:
                groups.append([list_remaining[0]])
                return groups
            if list_remaining:
                count += 1
                ref_face = list_remaining[0]
                list_remaining = list_remaining[1:]
                groups.append([ref_face])

        return groups

    if not line_edges:
        return []

    list_side_faces = _face_on_side(line_edges, space)
    if not list_side_faces:
        return []

    groups_of_adjacent_faces = _groups_of_adjacent_faces(list_side_faces)
    list_other_side_faces = [f for f in space.faces if not f in list_side_faces]
    groups_of_adjacent_faces_other_side = _groups_of_adjacent_faces(list_other_side_faces)

    groups_of_adjacent_faces += groups_of_adjacent_faces_other_side[1:]

    other_spaces = []
    for group in groups_of_adjacent_faces:
        if not group:
            continue
        other_space = Space(space.plan, space.floor,
                            group[0].edge,
                            SPACE_CATEGORIES[space.category.name])
        for face in group:
            if face in space.faces:
                space.remove_face_id(face.id)
                other_space.add_face_id(face.id)
        other_space.set_edges()
        other_spaces.append(other_space)
    space.set_edges()
    return [space] + other_spaces


def line_from_edge(plan: 'Plan', edge_origin: 'Edge') -> List['Edge']:
    """
    Returns list of edges forming contiguous lines from edge_origin
    and cutting empty spaces
    :param plan:
    :param edge_origin:
    :return:
    """

    contiguous_edges = []

    def _get_contiguous_edges(list_contiguous_edges: List['Edge'], current_edge: 'Edge'):
        while current_edge:
            current_edge = current_edge.aligned_edge or current_edge.continuous_edge
            if current_edge:
                space_of_current = plan.get_space_of_edge(current_edge)
                if (space_of_current and space_of_current.category
                        and space_of_current.category.name == "empty"
                        and not space_of_current.is_boundary(current_edge)):
                    list_contiguous_edges.append(current_edge)
                elif not current_edge.pair.face:
                    # case line is along the plan border
                    continue
                else:
                    break

    _get_contiguous_edges(contiguous_edges, edge_origin)
    _get_contiguous_edges(contiguous_edges, edge_origin.pair)
    return contiguous_edges


def divide_along_borders(seeder: 'Seeder', show: bool):
    """
    divide empty spaces along all lines drawn from selected edges
    Iterates though seed spaces and load bearing wall spaces, at each iteration :
    1 - a corner edge of the space is selected
    2 - the list of its contiguous edges is built
    3 - each empty space cut by a set of those contiguous edges is cut into two parts
    :param seeder:
    :param show:
    :return:
    """

    def border_division(space_category: 'str', selector: 'Selector' = None):
        continue_division = True
        while continue_division:
            spaces_before_division = [sp for sp in seeder.plan.spaces if
                                      sp.category.name in space_category]
            for division_space in spaces_before_division:
                # selector = selectors[division_space.category.name]
                if not division_space.edges:
                    continue
                selected_edges = list(e for e in selector.yield_from(division_space))
                for edge_selected in selected_edges:
                    # for edge_selected in selector.yield_from(division_space):
                    # lists of edges along which empty spaces division will be performed
                    contiguous_edges = line_from_edge(seeder.plan, edge_selected)

                    divided_spaces = []
                    for edge in contiguous_edges:
                        space = seeder.plan.get_space_of_edge(edge)
                        if space not in divided_spaces:
                            divided_spaces.append(space)
                        edges_in_space = list(
                            edge for edge in contiguous_edges if space.has_edge(edge))
                        modified_spaces = divide_along_line(space, edges_in_space)
                        if show:
                            seeder.plot.update(modified_spaces)
            seeder.plan.remove_null_spaces()
            spaces_after_division = [sp for sp in seeder.plan.spaces if
                                     sp.category.name is space_category]
            if len(spaces_after_division) == len(spaces_before_division):
                continue_division = False

    border_division("seed", SELECTORS["not_aligned_edges"])
    border_division("loadBearingWall", SELECTORS["not_aligned_edges"])
    border_division("hole", SELECTORS["not_aligned_edges"])
    border_division("empty", SELECTORS["not_aligned_edges_border"])

    return []


def merge_corners(seeder: 'Seeder', show: bool) -> List['Space']:
    """
    Merges small spaces with neighbor space that will induce a merged space with as few corners
    as possible. If several candidates are available, will select the space with the longest
    contact.
    :param seeder:
    :param show:
    :return: the list of modified spaces
    """
    if seeder.plan.indoor_area / SQM < 100:
        min_cell_area = MIN_SEEDER_SPACE_AREA
    else:
        min_cell_area = 1.5 * MIN_SEEDER_SPACE_AREA
    modified_spaces = []

    for small_space in (s for s in seeder.plan.get_spaces("seed") if s.area < min_cell_area):
        # adjacent mutable spaces of small_space
        adjacent_spaces = [s for s in small_space.adjacent_spaces() if s.mutable]
        if not adjacent_spaces:
            continue
        spaces_with_corners = list(map(lambda s: (s, small_space.number_of_corners(s)),
                                       adjacent_spaces))
        min_corner = min(spaces_with_corners, key=lambda t: t[1])[1]
        candidates = list(map(lambda t: t[0],
                              filter(lambda t: t[1] == min_corner, spaces_with_corners)))

        # in case there are several spaces with equal contact length,
        # merge with the smallest one
        selected = max(candidates, key=lambda s: s.contact_length(small_space))

        # do not merge if the selected space contains a seed as well as the small space
        if seeder.get_seed_from_space(selected) and seeder.get_seed_from_space(small_space):
            continue

        selected.merge(small_space)
        modified_spaces = [selected, small_space]
        break

    if show:
        seeder.plot.update(modified_spaces)

    return modified_spaces


# CREATE SEEDERS

SEEDERS = {
    "initial_seeder": Seeder(SEED_METHODS, GROWTH_METHODS,
                             [adjacent_faces, empty_to_seed], [merge_small_cells]),
    "simple_seeder": Seeder(SEED_METHODS, GROWTH_METHODS,
                            [adjacent_faces, empty_to_seed], [merge_corners]),
    "directional_seeder": Seeder(SEED_METHODS, GROWTH_METHODS,
                                 [divide_along_borders, empty_to_seed], [merge_small_cells,
                                                                         merge_enclosed_faces]),
}

if __name__ == '__main__':

    logging.getLogger().setLevel(logging.DEBUG)


    def try_plan():
        """
        Test
        :return:
        """
        from libs.modelers.grid import GRIDS
        from libs.operators.selector import SELECTORS
        import libs.io.writer as writer
        import libs.io.reader as reader

        import matplotlib
        matplotlib.use("TkAgg")

        logging.getLogger().setLevel(logging.INFO)

        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("-p", "--plan_index", help="choose plan index",
                            default=1)

        args = parser.parse_args()
        plan_index = int(args.plan_index)

        plan_name = None
        if plan_index < 10:
            plan_name = '00' + str(plan_index)
        elif 10 <= plan_index < 100:
            plan_name = '0' + str(plan_index)

        #plan_name = "001"

        # to not run each time the grid generation
        try:
            new_serialized_data = reader.get_plan_from_json(plan_name + ".json")
            plan = Plan(plan_name).deserialize(new_serialized_data)
        except FileNotFoundError:
            plan = reader.create_plan_from_file(plan_name + ".json")
            GRIDS["002"].apply_to(plan)
            writer.save_plan_as_json(plan.serialize(), plan_name + ".json")

        # SEEDERS["simple_seeder"].apply_to(plan, show=False)
        SEEDERS["directional_seeder"].apply_to(plan, show=False)
        plan.plot()
        plan.check()


    try_plan()
