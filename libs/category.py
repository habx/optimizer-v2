# coding=utf-8
"""
Category module : describes the type of space or linear that can be used in a program or a plan.
"""

from typing import List

CATEGORIES_COLORS = {
    'duct': 'k',
    'loadBearingWall': 'k',
    'window': '#FFCB19',
    'doorWindow': '#FFCB19',
    'frontDoor': 'r',
    'empty': 'b',
    'space': 'b',
    'seed': '#6a006a',
    'living': 'aquamarine',
    'dining': 'turquoise',
    'kitchen': 'paleturquoise',
    'bedroom': 'mistyrose',
    'wc': 'cornflowerblue',
    'bathroom': 'lightskyblue',
    'circulationSpace': 'lightgray',
    'entrance': 'grey',
    'dressing': 'pink',
    'laundry': 'lightsteelblue',
    'office': 'darkseagreen',
    'misc': 'lightsteelblue',
    'balcony': 'silver',
    'terrace': 'silver',
    'garden': 'green',
    'loggia': 'silver'
}


class Category:
    """
    A category of a space or a linear
    """

    def __init__(self,
                 name: str,
                 mutable: bool = True,
                 seedable: bool = False,
                 external: bool = False,
                 color: str = 'b'
                 ):
        self.name = name
        self.mutable = mutable
        self.seedable = seedable
        self.external = external
        self.color = CATEGORIES_COLORS.get(self.name, color)

    def __repr__(self) -> str:
        return self.name


class SpaceCategory(Category):
    """
    A category of a space
    Examples: duct, chamber, kitchen, wc, bathroom, entrance
    """

    def __init__(self,
                 name,
                 mutable: bool = True,
                 seedable: bool = False,
                 external: bool = False,
                 circulation: bool = False,
                 needed_linears: List['LinearCategory'] = [],
                 needed_spaces: List['SpaceCategory'] = []):
        super().__init__(name, mutable, seedable, external)
        self.circulation = circulation
        self.needed_linears = needed_linears
        self.needed_spaces = needed_spaces


class LinearCategory(Category):
    """
    A category of a linear
    Examples : window, doorWindow, door, wall, entrance
    """

    def __init__(self,
                 name,
                 mutable: bool = True,
                 seedable: bool = False,
                 aperture: bool = False,
                 window_type: bool = False,
                 width: float = 5.0):
        super().__init__(name, mutable, seedable)
        self.aperture = aperture
        self.width = width
        self.window_type = window_type


LINEAR_CATEGORIES = {
    "window": LinearCategory('window', mutable=False, seedable=True, aperture=True,
                             window_type=True),
    "door": LinearCategory('door', aperture=True),
    "doorWindow": LinearCategory('doorWindow', mutable=False, seedable=True, aperture=True,
                                 window_type=True),
    "frontDoor": LinearCategory('frontDoor', mutable=False, seedable=True, aperture=True),
    "wall": LinearCategory('wall'),
    "externalWall": LinearCategory('externalWall', False, width=2.0)
}

duct_space = SpaceCategory('duct', mutable=False, seedable=True)
window_linears = [LINEAR_CATEGORIES[name] for name in LINEAR_CATEGORIES.keys() if
                  LINEAR_CATEGORIES[name].window_type]

SPACE_CATEGORIES = {
    "empty": SpaceCategory('empty'),
    "seed": SpaceCategory('seed'),
    "duct": SpaceCategory('duct', mutable=False, seedable=True),
    "loadBearingWall": SpaceCategory('loadBearingWall', mutable=False),
    "chamber": SpaceCategory('chamber', needed_linears=window_linears),
    "bedroom": SpaceCategory('bedroom', needed_linears=window_linears),
    "living": SpaceCategory('living', circulation=True, needed_linears=window_linears),
    "entrance": SpaceCategory('entrance', circulation=True),
    "kitchen": SpaceCategory('kitchen', needed_spaces=[duct_space],
                             needed_linears=window_linears),
    "bathroom": SpaceCategory('bathroom', needed_spaces=[duct_space]),
    "dining": SpaceCategory('dining', circulation=True,
                            needed_linears=window_linears),
    "office": SpaceCategory('office', needed_linears=window_linears),
    "dressing": SpaceCategory('dressing', needed_spaces=[duct_space]),
    "laundry": SpaceCategory('laundry', needed_spaces=[duct_space]),
    "wc": SpaceCategory('wc', needed_spaces=[duct_space]),
    "corridor": SpaceCategory('corridor', circulation=True),
    "balcony": SpaceCategory('balcony', mutable=False, external=True),
    "garden": SpaceCategory('garden', mutable=False, external=True),
    "terrace": SpaceCategory('terrace', mutable=False, external=True),
    "loggia": SpaceCategory('loggia', mutable=False, external=True)
}
