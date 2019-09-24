# coding=utf-8
"""
Category module : describes the type of space or linear that can be used in a program or a plan.
"""

from typing import List, Optional

CATEGORIES_COLORS = {
    'duct': 'dimgrey',
    'loadBearingWall': 'dimgrey',
    'window': 'white',
    'doorWindow': 'white',
    'door': 'white',
    'frontDoor': 'lightgray',
    'empty': 'b',
    'space': 'b',
    'seed': '#6a006a',
    'living': 'limegreen',
    'livingKitchen': 'limegreen',
    'dining': 'turquoise',
    'kitchen': 'orangered',
    'bedroom': 'orange',
    'toilet': 'blue',
    'bathroom': 'dodgerblue',
    'circulation': 'lightgray',
    'entrance': 'lightgrey',
    'wardrobe': 'pink',
    'laundry': 'plum',
    'study': 'peru',
    'misc': 'darkgrey',
    'balcony': 'silver',
    'terrace': 'silver',
    'garden': 'green',
    'loggia': 'silver',
    'wintergarden': 'gainsboro',
    'startingStep': 'r',
    'hole': 'lightblue',
    'stairsObstacle': 'brown'
}


class Category:
    """
    A category of a space or a linear
    """

    __slots__ = 'name', 'mutable', 'seedable', 'external', 'color'

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
    Examples: duct, bedroom, kitchen, toilet, bathroom, entrance
    """

    __slots__ = 'circulation', 'needed_linears', 'needed_spaces'

    def __init__(self,
                 name,
                 mutable: bool = True,
                 seedable: bool = False,
                 external: bool = False,
                 circulation: bool = False,
                 needed_linears: Optional[List['LinearCategory']] = None,
                 needed_spaces: Optional[List['SpaceCategory']] = None):
        super().__init__(name, mutable, seedable, external)
        self.circulation = circulation
        self.needed_linears = needed_linears or []
        self.needed_spaces = needed_spaces or []


class LinearCategory(Category):
    """
    A category of a linear
    Examples : window, doorWindow, door, wall, frontDoor
    """

    __slots__ = 'aperture', 'width', 'window_type'

    def __init__(self,
                 name,
                 mutable: bool = True,
                 seedable: bool = False,
                 aperture: bool = False,
                 window_type: bool = False,
                 width: float = 2.5):
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
    "startingStep": LinearCategory('startingStep', mutable=False, seedable=True, aperture=True),
    "wall": LinearCategory('wall'),
    "externalWall": LinearCategory('externalWall', False, width=2.0)
}

duct_space = SpaceCategory('duct', mutable=False, seedable=True)
window_linears = [LINEAR_CATEGORIES[name] for name in LINEAR_CATEGORIES.keys() if
                  LINEAR_CATEGORIES[name].window_type]

SPACE_CATEGORIES = {
    "empty": SpaceCategory('empty'),
    "seed": SpaceCategory('seed'),
    "duct": duct_space,
    "loadBearingWall": SpaceCategory('loadBearingWall', mutable=False),
    "hole": SpaceCategory('hole', mutable=False),
    "stairsObstacle": SpaceCategory('stairsObstacle', mutable=False),
    "bedroom": SpaceCategory('bedroom', needed_linears=window_linears),
    "living": SpaceCategory('living', circulation=True, needed_linears=window_linears),
    "livingKitchen": SpaceCategory('livingKitchen', circulation=True, needed_linears=window_linears,
                                   needed_spaces=[duct_space]),
    "entrance": SpaceCategory('entrance', circulation=True,
                              needed_linears=[LINEAR_CATEGORIES["frontDoor"]]),
    "kitchen": SpaceCategory('kitchen', needed_spaces=[duct_space],
                             needed_linears=window_linears),
    "bathroom": SpaceCategory('bathroom', needed_spaces=[duct_space]),
    "dining": SpaceCategory('dining', circulation=True,
                            needed_linears=window_linears),
    "study": SpaceCategory('study', needed_linears=window_linears),
    "wardrobe": SpaceCategory('wardrobe'),
    "misc": SpaceCategory('misc'),
    "laundry": SpaceCategory('laundry', needed_spaces=[duct_space]),
    "toilet": SpaceCategory('toilet', needed_spaces=[duct_space]),
    "circulation": SpaceCategory("circulation", circulation=True),
    "balcony": SpaceCategory('balcony', mutable=False, external=True),
    "garden": SpaceCategory('garden', mutable=False, external=True),
    "terrace": SpaceCategory('terrace', mutable=False, external=True),
    "loggia": SpaceCategory('loggia', mutable=False, external=True),
    "wintergarden": SpaceCategory('wintergarden', mutable=False, external=True)
}


if __name__ == '__main__':
    import matplotlib as mpl

    def colors():
        for cat, color in CATEGORIES_COLORS.items():
            rgb_color = mpl.colors.to_hex(color)
            print(cat, color, rgb_color)

    colors()

