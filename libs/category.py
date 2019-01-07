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
                 circulation: bool = False):
        super().__init__(name, mutable, seedable, external)
        self.circulation = circulation


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
                 width: float = 5.0):
        super().__init__(name, mutable, seedable)
        self.aperture = aperture
        self.width = width


LINEAR_CATEGORIES = {
    "window": LinearCategory('window', mutable=False, seedable=True, aperture=True),
    "door": LinearCategory('door', aperture=True),
    "doorWindow": LinearCategory('doorWindow', mutable=False, seedable=True, aperture=True),
    "frontDoor": LinearCategory('frontDoor', mutable=False, seedable=True, aperture=True),
    "wall": LinearCategory('wall'),
    "externalWall": LinearCategory('externalWall', False, width=2.0)
}

SPACE_CATEGORIES = {
    "empty": SpaceCategory('empty'),
    "seed": SpaceCategory('seed'),
    "duct": SpaceCategory('duct', mutable=False, seedable=True),
    "loadBearingWall": SpaceCategory('loadBearingWall', mutable=False),
    "chamber": SpaceCategory('chamber'),
    "bedroom": SpaceCategory('bedroom'),
    "living": SpaceCategory('living', circulation=True),
    "entrance": SpaceCategory('entrance', circulation=True),
    "kitchen": SpaceCategory('kitchen'),
    "bathroom": SpaceCategory('bathroom'),
    "dining": SpaceCategory('dining', circulation=True),
    "office": SpaceCategory('office'),
    "dressing": SpaceCategory('dressing'),
    "wc": SpaceCategory('wc'),
    "corridor": SpaceCategory('corridor', circulation=True),
    "balcony": SpaceCategory('balcony', mutable=False, external=True),
    "garden": SpaceCategory('garden', mutable=False, external=True),
    "terrace": SpaceCategory('terrace', mutable=False, external=True),
    "loggia": SpaceCategory('loggia', mutable=False, external=True)
}
