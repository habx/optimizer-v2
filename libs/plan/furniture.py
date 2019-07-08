from typing import (
    TYPE_CHECKING,
    Sequence,
    Tuple,
    Optional
)
from libs.utils.geometry import (ccw_angle,
                                 unit,
                                 barycenter,
                                 rotate,
                                 minimum_rotated_rectangle,
                                 move_point)
from libs.plan.category import SpaceCategory, SPACE_CATEGORIES
from libs.io.plot import plot_polygon

if TYPE_CHECKING:
    from libs.utils.custom_types import FourCoords2d, Vector2d, Coords2d, ListCoords2d
    from libs.plan.plan import Space
    from libs.space_planner.solution import Solution
    from libs.specification.specification import Item

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

    def apply_to(self, solution: 'Solution') -> None:
        """
        Modify the solution plan by applying the successive orders.
        """
        for order in self.orders:
            self._apply_order(solution, order)

    def _apply_order(self, solution: 'Solution', order: Order) -> None:
        """
        Apply an order by updating plan list of furniture and fitting them in space
        :param plan:
        :param
        :return:
        """
        category, variant, furniture_names, is_prm = order
        plan = solution.spec.plan
        for space in plan.mutable_spaces():
            item = solution.space_item[space]
            if item.category == category and item.variant == variant:
                furnitures = [Furniture(next(furniture
                                             for furniture in FURNITURE_CATALOG[name]
                                             if furniture.is_prm == is_prm))
                              # today only 1st one is picked
                              for name in furniture_names]
                self._fit(space, furnitures)
                for furniture in furnitures:
                    plan.furnitures.setdefault(space, []).append(furniture)

    def _fit(self, space: 'Space', furnitures: Sequence['Furniture']):
        for furniture in furnitures:
            fit_bed_in_bedroom(furniture, space)


class FurnitureType:
    def __init__(self,
                 name: str,
                 polygon: 'ListCoords2d',
                 is_prm: bool):
        self.name = name
        self.polygon = polygon
        self.is_prm = is_prm


class Furniture:
    def __init__(self,
                 model: FurnitureType,
                 ref_point: Optional['Coords2d'] = None,
                 ref_vect: Optional['Vector2d'] = None):
        self.model = model
        self.ref_point = ref_point if ref_point is not None else (0, 0)
        self.ref_vect = unit(ref_vect) if ref_vect is not None else (0, 1)

    @property
    def middle_point(self) -> 'Coords2d':
        footprint = self.footprint()
        return barycenter(footprint[0], footprint[1], 0.5)

    @middle_point.setter
    def middle_point(self, value: 'Coords2d'):
        vect = self.ref_point[0] - self.middle_point[0], self.ref_point[1] - self.middle_point[1]
        self.ref_point = move_point(value, vect)

    def bounding_box(self) -> 'FourCoords2d':
        """
        Rectangle shape of the furniture
        """
        return minimum_rotated_rectangle(self.footprint())

    def footprint(self) -> 'ListCoords2d':
        """
        Real shape of the furniture, well oriented and located
        """
        angle = ccw_angle((0, 1), self.ref_vect)
        trans_x, trans_y = self.ref_point
        rotated = rotate(self.model.polygon, self.model.polygon[0], angle)
        translated = tuple([(x + trans_x, y + trans_y)
                            for x, y in rotated])
        return translated

    def plot(self, ax=None,
             options=('fill', 'border'),
             save: Optional[bool] = None):
        """
        Plots the face
        :return:
        """
        color = "black"
        footprint = self.footprint()
        x = [p[0] for p in footprint]
        y = [p[1] for p in footprint]
        return plot_polygon(ax, x, y, options, color, save)


FURNITURE_CATALOG = {
    "bed": (FurnitureType("bed", ((0, 0), (140, 0), (140, 190), (0, 190)), False),
            FurnitureType("bed_prm_1", ((0, 0), (320, 0), (320, 310), (0, 310)), True),
            FurnitureType("bed_prm_2", ((0, 0), (380, 0), (380, 280), (0, 280)), True)),
    "bathtub": (FurnitureType("bathtub", ((0, 0), (140, 0), (140, 190), (0, 190)), True))
}

GARNISHERS = {
    "default": Garnisher("default", [
        (SPACE_CATEGORIES["bedroom"], "xs", ("bed",), False),
        (SPACE_CATEGORIES["bedroom"], "s", ("bed",), False),
        (SPACE_CATEGORIES["bedroom"], "m", ("bed",), False),
        (SPACE_CATEGORIES["bedroom"], "l", ("bed",), True),
        (SPACE_CATEGORIES["bedroom"], "xl", ("bed",), True),
    ])
}


def fit_bed_in_bedroom(bed: Furniture, bedroom: 'Space'):
    """
    Move furniture to fit in space.
    :return:
    """
    longest_length = 0

    aligned_edges = bedroom.aligned_siblings(bedroom.edge)
    initial_edge = aligned_edges[0]
    start_edge = None

    while start_edge is not initial_edge:
        start_edge = aligned_edges[0]
        end_edge = aligned_edges[-1]

        length = start_edge.start.distance_to(end_edge.end)
        if length > longest_length:
            longest_length = length
            bed.ref_vect = start_edge.normal
            bed.middle_point = barycenter(start_edge.start.coords, end_edge.end.coords, 0.5)

        aligned_edges = bedroom.aligned_siblings(bedroom.next_edge(end_edge))
        # start_edge = aligned_edges[0]
