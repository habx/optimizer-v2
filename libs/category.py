# coding=utf-8
"""
Category module : describes the type of space or linear that can be used in a program or a plan.
"""
from typing import Sequence, Optional

from libs.size import Size
from libs.growth import GrowthMethod, GROWTH_METHODS

from libs.utils.catalog import Catalog


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
    def __init__(self, name: str, size: Size, methods: Sequence[GrowthMethod]):
        self.name = name
        self.size = size
        self.methods = methods


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


classic_seed_category = SeedCategory('classic',
                                     Size(min_area=20000, max_area=150000,
                                          max_width=350, max_depth=400),
                                     (GROWTH_METHODS['horizontal_growth'],
                                      GROWTH_METHODS['vertical_growth'],
                                      GROWTH_METHODS['free_growth'],
                                      GROWTH_METHODS['done']))

duct_seed_category = SeedCategory('duct',
                                  Size(min_area=20000, max_area=80000,
                                       max_width=200, max_depth=300),
                                  (GROWTH_METHODS['horizontal_growth'],
                                   GROWTH_METHODS['surround_growth'],
                                   GROWTH_METHODS['vertical_growth'],
                                   GROWTH_METHODS['horizontal_growth'],
                                   GROWTH_METHODS['free_growth'],
                                   GROWTH_METHODS['done']))


seed_catalog = Catalog('seeds').add(classic_seed_category, duct_seed_category)

linear_catalog = Catalog('linears').add(
    LinearCategory('window', False, seed_catalog['classic'], True, True),
    LinearCategory('door', aperture=True),
    LinearCategory('doorWindow', False, seed_catalog['classic'], True),
    LinearCategory('frontDoor', False, seed_catalog['classic'], True),
    LinearCategory('wall'),
    LinearCategory('externalWall', False, width=2.0)
)

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
    SpaceCategory('wc')
)
