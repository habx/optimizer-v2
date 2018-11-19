# coding=utf-8
"""
Category module : describes the type of space or linear that can be used in a program or a plan.
"""
from typing import TYPE_CHECKING, Sequence, Optional

from libs.action import Action
from libs.constraint import CONSTRAINTS
from libs.selector import SELECTORS
from libs.mutation import MUTATIONS

from libs.utils.catalog import Catalog

if TYPE_CHECKING:
    from libs.constraint import Constraint


CATEGORIES_COLORS = {
    'duct': 'k',
    'loadBearingWall': 'k',
    'window': '#FFCB19',
    'doorWindow': '#FFCB19',
    'entrance': 'r',
    'frontDoor': 'r',
    'empty': 'b',
    'space': 'b',
    'seed': '#6a006a'
}


class SeedCategory:
    """
    A category of seedable space or linear
    """
    def __init__(self, name: str, constraints: Optional[Sequence['Constraint']] = None,
                 operators: Optional[Sequence[Action]] = None):
        constraints = constraints or []
        self.name = name
        self.constraints = {constraint.name: constraint for constraint in constraints}
        self.operators = operators or None

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


class Category:
    """
    A category of a space or a linear
    """
    def __init__(self,
                 name: str,
                 mutable: bool = True,
                 seed_category: Optional[SeedCategory] = None,
                 color: str = 'b'):
        self.name = name
        self.mutable = mutable
        self.seed_category = seed_category
        self.color = CATEGORIES_COLORS.get(self.name, color)

    def __repr__(self) -> str:
        return self.name

    @property
    def seedable(self) -> bool:
        """
        Returns True if the space category is seedable (has a seed method)
        :return:
        """
        return getattr(self, 'seed_category', None)


class SpaceCategory(Category):
    """
    A category of a space
    Examples: duct, chamber, kitchen, wc, bathroom, entrance
    """
    pass


class LinearCategory(Category):
    """
    A category of a linear
    Examples : window, doorWindow, door, wall, entrance
    """
    def __init__(self,
                 name,
                 mutable: bool = True,
                 seed_category: Optional[SeedCategory] = None,
                 aperture: bool = False,
                 width: float = 5.0):
        super().__init__(name, mutable, seed_category)
        self.aperture = aperture
        self.width = width
        self.mutable = mutable


classic_seed_category = SeedCategory(
    'classic',
    (CONSTRAINTS['max_size_s'],),
    (
        Action(SELECTORS.factory['oriented_edges'](('horizontal',)), MUTATIONS['add_face']),
        Action(SELECTORS.factory['oriented_edges'](('vertical',)), MUTATIONS['add_face'], True),
        Action(SELECTORS['boundary_other_empty_space'], MUTATIONS['add_face'])
    )
)

duct_seed_category = SeedCategory(
    'duct',
    (CONSTRAINTS['max_size_xs'],),
    (
        Action(SELECTORS.factory['oriented_edges'](('horizontal',)), MUTATIONS['add_face']),
        Action(SELECTORS['surround_seed_component'], MUTATIONS['add_face']),
        Action(SELECTORS.factory['oriented_edges'](('vertical',)), MUTATIONS['add_face'], True),
        Action(SELECTORS['boundary_other_empty_space'], MUTATIONS['add_face'])
    )
)

front_door_seed_category = SeedCategory(
    'front_door',
    (CONSTRAINTS['max_size_xs'],),
    (
        Action(SELECTORS.factory['oriented_edges'](('horizontal',)), MUTATIONS['add_face']),
        Action(SELECTORS.factory['oriented_edges'](('vertical',)), MUTATIONS['add_face'], True),
        Action(SELECTORS['boundary_other_empty_space'], MUTATIONS['add_face'])
    )
)


seed_catalog = Catalog('seeds').add(
    classic_seed_category,
    duct_seed_category,
    front_door_seed_category)

linear_catalog = Catalog('linears').add(
    LinearCategory('window', False, seed_catalog['classic'], True, True),
    LinearCategory('door', aperture=True),
    LinearCategory('doorWindow', False, seed_catalog['classic'], True),
    LinearCategory('frontDoor', False, seed_catalog['front_door'], True),
    LinearCategory('wall'),
    LinearCategory('externalWall', False, width=2.0))

space_catalog = Catalog('spaces').add(
    SpaceCategory('empty'),
    SpaceCategory('seed'),
    SpaceCategory('duct', False, seed_catalog['duct']),
    SpaceCategory('loadBearingWall', False),
    SpaceCategory('chamber'),
    SpaceCategory('bedroom'),
    SpaceCategory('living'),
    SpaceCategory('entrance'),
    SpaceCategory('kitchen'),
    SpaceCategory('bathroom'),
    SpaceCategory('wcBathroom'),
    SpaceCategory('livingKitchen'),
    SpaceCategory('wc'))
