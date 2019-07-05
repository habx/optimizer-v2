from typing import (
    TYPE_CHECKING,
    Sequence,
    Tuple,
    Optional,
    Dict,
    List
)
from libs.utils.geometry import rectangle, unit
from libs.plan.category import SpaceCategory, SPACE_CATEGORIES
from libs.io.plot import plot_polygon

if TYPE_CHECKING:
    from libs.utils.custom_types import FourCoords2d, Vector2d, Coords2d, ListCoords2d
    from libs.plan.plan import Plan, Space

# custom type: order
# ex: (SPACE_CATEGORIES["bathroom"], "m", ("bathtub", "sink"), True) -> last bool is for PRM
Order = Tuple[SpaceCategory, str, Tuple[str, ...], bool]


class Garnisher:
    """
    Instanciates and fits furniture in plan with rooms.
    """

    def __init__(self, name: str, orders: Sequence[Order]):
        self.name = name
        self.orders = orders

    def apply_to(self, plan: 'Plan') -> Plan:
        """
        Modify the plan by applying the successive orders.
        :param plan:
        :return: the plan
        """
        if plan.furniture_list is None:
            plan.furniture_list = FurnitureList()
        for order in self.orders:
            self._apply_order(plan, order)
        return plan

    def _apply_order(self, plan: 'Plan', order: Order) -> None:
        """
        Apply an oder by updating plan list of furniture and fitting them in space
        :param plan:
        :param
        :return:
        """
        for space in plan.spaces:
            if space.category == order[0] and space.variant :
                furniture = Furniture()
                plan.furniture_list.add(space, furniture)
                self._fit(space, furniture)


class FurnitureType:
    def __init__(self,
                 name: str,
                 polygon: 'ListCoords2d',
                 is_prm: bool):
        self.name = name
        self.polygon = polygon
        self.is_prm = is_prm


class FurnitureList:

    def __init__(self):
        self.furniture: Dict['Space', List['Furniture']] = {}

    def add(self, space: 'Space', furniture: 'Furniture'):
        self.furniture.setdefault(space, []).append(furniture)

    def get(self, space: 'Space') -> List['Furniture']:
        return self.furniture.get(space, [])


class Furniture:
    def __init__(self,
                 category: FurnitureType,
                 ref_point: Optional['Coords2d'] = None,
                 ref_vect: Optional['Vector2d'] = None):
        self.category = category
        self.ref_point = ref_point if ref_point is not None else (0, 0)
        self.ref_vect = unit(ref_vect) if ref_vect is not None else (1, 0)

    def bounding_box(self) -> 'FourCoords2d':
        """
        :return: rectangle shape of the furniture
        """
        size = Furniture.prm_sizes[self.category] if self.prm else Furniture.sizes[self.category]
        return rectangle(self.ref_point, self.ref_vect, *size)

    def plot(self, ax=None,
             options=('fill', 'border'),
             save: Optional[bool] = None):
        """
        Plots the face
        :return:
        """
        color = "black"
        bounding_box = self.bounding_box()
        x = [p[0] for p in bounding_box]
        y = [p[1] for p in bounding_box]
        return plot_polygon(ax, x, y, options, color, save)


FURNITURE_CATEGORIES = {
    "bed": (FurnitureType("bed", ((0, 0), (140, 0), (140, 190), (0, 190)), False),
            FurnitureType("bed_prm_1", ((0, 0), (320, 0), (320, 310), (0, 310)), True),
            FurnitureType("bed_prm_2", ((0, 0), (380, 0), (380, 280), (0, 280)), True)),
    "bathtub": (FurnitureType("bathtub", ((0, 0), (140, 0), (140, 190), (0, 190)), True))
}

GARNISHERS = {
    "default": Garnisher("default", [(SPACE_CATEGORIES["bedroom"], "bed", True)])
}
