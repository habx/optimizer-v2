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

if TYPE_CHECKING:
    from libs.utils.custom_types import FourCoords2d, Vector2d, Coords2d
from libs.plan.plan import Plan, Space


class Garnisher:

    def __init__(self, name: str, orders: Sequence[Tuple[SpaceCategory, str, bool]]):
        """

        :param name:
        :param to_do_list:
        """
        self.name = name
        self.orders = orders

    def apply_to(self, plan: 'Plan'):
        if plan.furniture_list is None:
            plan.furniture_list = FurnitureList()
        for order in self.orders:
            self._apply_order(plan, order)

    def _apply_order(self, plan: Plan, order: Tuple[SpaceCategory, str, bool]):
        for space in plan.spaces:
            if space.category == order[0]:
                furniture = Furniture(order[1], order[2])
                plan.furniture_list.add(space, furniture)
                self._fit(space, furniture)

    def _fit(self, space: Space, furniture: 'Furniture'):
        furniture.ref_point = space.centroid


class FurnitureList:

    def __init__(self):
        self.furniture: Dict['Space', List['Furniture']] = {}

    def add(self, space: Space, furniture: 'Furniture'):
        self.furniture.setdefault(space, []).append(furniture)


class Furniture:
    sizes = {
        "bed": (140, 190)
    }
    prm_sizes = {
        "bed": (320, 310)
    }

    def __init__(self,
                 category: str,
                 prm: bool,
                 ref_point: Optional[Coords2d] = None,
                 ref_vect: Optional[Vector2d] = None):
        self.category = category
        self.prm = prm
        self.ref_point = ref_point if ref_point is not None else (0, 0)
        self.ref_vect = unit(ref_vect) if ref_vect is not None else (1, 0)

    def bounding_box(self) -> FourCoords2d:
        """
        :return: rectangle shape of the furniture
        """
        size = Furniture.prm_sizes[self.category] if self.prm else Furniture.sizes[self.category]
        return rectangle(self.ref_point, self.ref_vect, *size)


prm_bed_garnisher = Garnisher("prm_bed", [(SPACE_CATEGORIES["bedroom"], "bed", True)])

GARNISHERS = {
    "prm_bed": prm_bed_garnisher
}
