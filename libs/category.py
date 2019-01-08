# coding=utf-8
"""
Category module : describes the type of space or linear that can be used in a program or a plan.
"""

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
    'bedroom': 'mistyrose',  ''
    'wc': 'cornflowerblue',
    'bathroom': 'lightskyblue',
    'circulationSpace': 'lightgray',
    'entrance': 'r',
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
                 needs_window: bool = False,
                 needs_duct: bool = False):
        super().__init__(name, mutable, seedable, external)
        self.circulation = circulation
        self.needs_window = needs_window
        self.needs_duct = needs_duct


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
    "window": LinearCategory('window', mutable=False, seedable=True, aperture=True, window_type=True),
    "door": LinearCategory('door', aperture=True),
    "doorWindow": LinearCategory('doorWindow', mutable=False, seedable=True, aperture=True, window_type=True),
    "frontDoor": LinearCategory('frontDoor', mutable=False, seedable=True, aperture=True),
    "wall": LinearCategory('wall'),
    "externalWall": LinearCategory('externalWall', False, width=2.0)
}

SPACE_CATEGORIES = {
    "empty": SpaceCategory('empty'),
    "seed": SpaceCategory('seed'),
    "duct": SpaceCategory('duct', mutable=False, seedable=True),
    "loadBearingWall": SpaceCategory('loadBearingWall', mutable=False),
    "chamber": SpaceCategory('chamber', needs_window=True),
    "bedroom": SpaceCategory('bedroom', needs_window=True),
    "living": SpaceCategory('living', circulation=True, needs_window=True),
    "entrance": SpaceCategory('entrance', circulation=True),
    "kitchen": SpaceCategory('kitchen', needs_window=True, needs_duct=True),
    "bathroom": SpaceCategory('bathroom', needs_duct=True),
    "dining": SpaceCategory('dining', circulation=True, needs_window=True),
    "office": SpaceCategory('office', needs_window=True),
    "dressing": SpaceCategory('dressing', needs_duct=True),
    "wc": SpaceCategory('wc', needs_duct=True),
    "corridor": SpaceCategory('corridor', circulation=True),
    "balcony": SpaceCategory('balcony', mutable=False, external=True),
    "garden": SpaceCategory('garden', mutable=False, external=True),
    "terrace": SpaceCategory('terrace', mutable=False, external=True),
    "loggia": SpaceCategory('loggia', mutable=False, external=True)
}
