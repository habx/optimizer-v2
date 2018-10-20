# coding=utf-8
"""
Category module : describes the type of space or linear that can be used in a program or a plan.
"""

CATEGORIES_COLORS = {
    'duct': 'k',
    'window': '#A3A3A3',
    'doorWindow': '#A3A3A3',
    'entrance': 'r',
    'frontDoor': 'r',
    'empty': 'b',
    'space': 'b'
}


class Category:
    """
    A category of a space or a linear
    """
    def __init__(self, name: str, mutable: bool = True, color: str = 'b'):
        self.name = name
        self.mutable = mutable
        self.color = CATEGORIES_COLORS.get(self.name, color)


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
    def __init__(self, name, mutable=True, aperture=False):
        super().__init__(name, mutable)
        self.aperture = aperture


linear_categories = {
    'window': LinearCategory('window', False, True),
    'door': LinearCategory('door', aperture=True),
    'doorWindow': LinearCategory('doorWindow', False, True),
    'frontDoor': LinearCategory('entranceDoor', False, True),
    'wall': LinearCategory('wall')
}

space_categories = {
    'empty': SpaceCategory('empty'),
    'doorWindow': SpaceCategory('doorWindow'),  # TODO : temporary fix : should be a linear
    'frontDoor': SpaceCategory('frontDoor'),  # TODO : temporary fix : should be a linear
    'window': SpaceCategory('window'),  # TODO : temporary fix : should be a linear
    'duct': SpaceCategory('duct', False),
    'chamber': SpaceCategory('chamber'),
    'living': SpaceCategory('living'),
    'entrance': SpaceCategory('entrance'),
    'kitchen': SpaceCategory('kitchen'),
    'bathroom': SpaceCategory('bathroom')
}
