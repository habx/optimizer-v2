from typing import (
    TYPE_CHECKING,
)
from libs.utils.geometry import rectangle, unit

if TYPE_CHECKING:
    from libs.utils.custom_types import FourCoords2d, Vector2d, Coords2d
    from libs.plan.plan import Plan

class Garnisher:

    def __init__(self, name : str, prm: bool):
        self.name = name
        self.prm = prm

    def apply_to(self, plan : 'Plan'):
        for space in plan.spaces:


class FurnituresList:

    def __init__(self):
        self.furnitures = []

class Furniture:
    sizes = {
        "bed": (140, 190)
    }
    prm_sizes = {
        "bed": ((380, 280),(320, 310))
    }

    def __init__(self, category: str, prm: bool, ref_point: Coords2d, ref_vect: Vector2d):
        self.category = category
        self.prm = prm
        self.ref_point = ref_point
        self.ref_vect = unit(ref_vect)

    def bounding_box(self) -> FourCoords2d:
        """
        :return: rectangle shape of the furniture
        """
        size = Furniture.prm_sizes[self.category] if self.prm else Furniture.sizes[self.category]
        return rectangle(self.ref_point, self.ref_vect, *size)

prm_bed_garnisher = Garnisher("prm_bed", True)

GARNISHERS = {
    "prm_bed": prm_bed_garnisher
}