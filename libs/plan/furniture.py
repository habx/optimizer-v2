from typing import (
    TYPE_CHECKING,
    Sequence,
    Tuple,
    Optional,
    Dict
)
from libs.plan.category import SpaceCategory, SPACE_CATEGORIES, LinearCategory, LINEAR_CATEGORIES
from libs.io.plot import plot_polygon
from libs.utils.geometry import (ccw_angle,
                                 unit,
                                 barycenter,
                                 rotate,
                                 minimum_rotated_rectangle,
                                 polygons_collision,
                                 polygon_border_collision,
                                 polygon_linestring_collision,
                                 move_point,
                                 rectangle)

if TYPE_CHECKING:
    from libs.utils.custom_types import FourCoords2d, Vector2d, Coords2d, ListCoords2d
    from libs.plan.plan import Space
    from libs.space_planner.solution import Solution

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
            category, variant, furniture_names, is_prm = order
            plan = solution.spec.plan
            for space in plan.mutable_spaces():
                item = solution.space_item[space]
                if item.category == category and item.variant == variant:
                    furnitures = [Furniture(next(furniture
                                                 for furniture in FURNITURE_CATALOG[name]
                                                 if furniture.is_prm or not is_prm))
                                  # today only 1st one is picked
                                  for name in furniture_names]
                    for furniture in furnitures:
                        # TODO: map functions to furnitures
                        if fit_bed(furniture, space):
                            plan.furnitures.setdefault(space, []).append(furniture)


class FurnitureCategory:
    BY_NAME: Dict[str, 'FurnitureCategory'] = {}

    def __init__(self,
                 name: str,
                 polygon: 'ListCoords2d',
                 is_prm: bool,
                 required_space: 'ListCoords2d',
                 color: str = 'black'):
        """
        :param name:
        :param polygon: polygon of the furniture
        :param is_prm: ok for a prm room
        :param required_space: poltgon of the furniture + space around
        """
        self.name = name
        self.polygon = polygon
        self.is_prm = is_prm
        self.required_space = required_space
        self.color = color
        FurnitureCategory.BY_NAME[name] = self


class Furniture:
    def __init__(self,
                 category: Optional[FurnitureCategory] = None,
                 ref_point: Optional['Coords2d'] = None,
                 ref_vect: Optional['Vector2d'] = None):
        self.category = category
        self.ref_point = ref_point if ref_point is not None else (0, 0)
        self.ref_vect = unit(ref_vect) if ref_vect is not None else (0, 1)

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
        translation_vec = self.ref_point
        rotated = rotate(self.category.polygon, (0, 0), angle)
        translated = tuple([move_point(coord, translation_vec)
                            for coord in rotated])
        return translated

    def required_space(self) -> 'ListCoords2d':
        """
        Shape of the furniture + space required around it to
        :return:
        """
        angle = ccw_angle((0, 1), self.ref_vect)
        translation_vec = self.ref_point
        rotated = rotate(self.category.required_space, (0, 0), angle)
        translated = tuple([move_point(coord, translation_vec)
                            for coord in rotated])
        return translated

    def check_validity(self, space: 'Space') -> bool:
        """
        Return True if current furniture position is valid with space
        :param space:
        :return:
        """
        footprint = self.footprint()
        required_space = self.required_space()
        # furniture collision
        for furniture in space.furnitures():
            if furniture is not self and polygons_collision(furniture.footprint(), required_space):
                return False
        # border collision
        border = space.boundary_polygon()
        if polygon_border_collision(required_space, border, 1):
            return False
        # window collision
        for component in space.immutable_components():
            if isinstance(component.category, LinearCategory) and component.category.window_type:
                window_line = [component.edge.start.coords]
                window_line += [edge.end.coords for edge in component.edges]
                if polygon_linestring_collision(footprint, window_line, -1):
                    return False
        # door collision
        for linear in space.plan.linears:
            if linear.category == LINEAR_CATEGORIES["door"]:
                if linear.edge.space == space:
                    # door opens in the space
                    door_box = rectangle(linear.edge.start.coords, linear.edge.vector,
                                         linear.length, linear.length)
                    if polygons_collision(door_box, footprint, -1):
                        return False
                elif linear.edge.pair.space == space:
                    # door touches the space
                    door_line = [linear.edge.start.coords]
                    door_line += [edge.end.coords for edge in linear.edges]
                    if polygon_linestring_collision(footprint, door_line, -1):
                        return False

        return True

    def plot(self, ax=None,
             options=('fill', 'border'),
             save: Optional[bool] = None):
        """
        Plots the face
        :return:
        """
        color = self.category.color
        footprint = self.footprint()
        x = [p[0] for p in footprint]
        y = [p[1] for p in footprint]
        return plot_polygon(ax, x, y, options, color, save)

    def serialize(self) -> Dict:
        """
        Returns a serialized version of the furniture
        :return:
        """
        output = {
            "ref_point": list(self.ref_point),
            "ref_vect": list(self.ref_vect),
            "category": self.category.name
        }

        return output

    def deserialize(self, value: Dict) -> 'Furniture':
        """
        Fills the feature with serialized data.
        :return:
        """
        self.ref_point = tuple(value["ref_point"])
        self.ref_vect = tuple(value["ref_vect"])
        self.category = FurnitureCategory.BY_NAME[value["category"]]
        return self


FURNITURE_CATALOG = {
    "single_bed": (FurnitureCategory("single_bed",
                                     ((-45, 0), (45, 0), (45, 190), (-45, 190)),
                                     False,
                                     ((-45, 0), (45, 0), (45, 250), (-45, 250))),),
    "double_bed": (FurnitureCategory("double_bed",
                                     ((-70, 0), (70, 0), (70, 190), (-70, 190)),
                                     False,
                                     ((-130, 0), (130, 0), (130, 250), (-130, 250))),
                   FurnitureCategory("double_bed_pmr_1",
                                     ((-70, 0), (70, 0), (70, 190), (-70, 190)),
                                     True,
                                     ((-160, 0), (160, 0), (160, 310), (-160, 310))),
                   FurnitureCategory("double_bed_pmr_2",
                                     ((-70, 0), (70, 0), (70, 190), (-70, 190)),
                                     True,
                                     ((-130, 0), (130, 0), (130, 250), (-130, 250)))),
    "bathtub": (FurnitureCategory("bathtub",
                                  ((-90, 0), (90, 0), (90, 80), (-90, 80)),
                                  True,
                                  ((-90, 0), (90, 0), (90, 80), (-90, 80)),
                                  color='blue'),)
}

GARNISHERS = {
    "default": Garnisher("default", [
        (SPACE_CATEGORIES["bedroom"], "xs", ("single_bed",), False),
        (SPACE_CATEGORIES["bedroom"], "s", ("double_bed",), False),
        (SPACE_CATEGORIES["bedroom"], "m", ("double_bed",), False),
        (SPACE_CATEGORIES["bedroom"], "l", ("double_bed",), True),
        (SPACE_CATEGORIES["bedroom"], "xl", ("double_bed",), True),
        # (SPACE_CATEGORIES["bathroom"], "xs", ("bathtub",), False),
        # (SPACE_CATEGORIES["bathroom"], "s", ("bathtub",), False),
        # (SPACE_CATEGORIES["bathroom"], "m", ("bathtub",), False),
        # (SPACE_CATEGORIES["bathroom"], "l", ("bathtub",), False),
        # (SPACE_CATEGORIES["bathroom"], "xl", ("bathtub",), False),
    ])
}


def fit_bed(bed: Furniture, space: 'Space') -> bool:
    """
    Move furniture to fit in space, designed especially for bed.
    :return:
    """
    window_edges = [component.edge
                    for component in space.immutable_components()
                    if isinstance(component.category, LinearCategory)
                    and component.category.window_type]
    space_perimeter = space.perimeter

    # init loop
    aligned_edges = space.aligned_siblings(space.edge)
    initial_edge = aligned_edges[0]
    start_edge = None
    possibilities = []
    # find all possibilities
    while start_edge is not initial_edge:
        start_edge = aligned_edges[0]
        end_edge = aligned_edges[-1]

        # compute perimeter percentage
        length = start_edge.start.distance_to(end_edge.end)
        perimeter_proportion = length / space_perimeter

        # score each line
        no_window = True
        for edge in window_edges:
            if edge in aligned_edges:
                # at least one window on the line
                no_window = False
                break
        possibilities.append({
            "ref_point": barycenter(start_edge.start.coords, end_edge.end.coords, 0.5),
            "ref_vect": start_edge.normal,
            "score": perimeter_proportion * 100 + no_window * 100
        })
        for i in range(1, 5):
            possibilities.append({
                "ref_point": barycenter(start_edge.start.coords, end_edge.end.coords,
                                        0.5 - (0.1 * i)),
                "ref_vect": start_edge.normal,
                "score": perimeter_proportion * 100 - i + no_window * 100
            })
            possibilities.append({
                "ref_point": barycenter(start_edge.start.coords, end_edge.end.coords,
                                        0.5 + (0.1 * i)),
                "ref_vect": start_edge.normal,
                "score": perimeter_proportion * 100 - i + no_window * 100
            })

        # prepare next loop
        aligned_edges = space.aligned_siblings(space.next_edge(end_edge))
        start_edge = aligned_edges[0]

    # sort possibilites
    possibilities.sort(key=lambda p: p["score"], reverse=True)

    # try all possibilites
    for possibility in possibilities:
        bed.ref_vect = possibility["ref_vect"]
        bed.ref_point = possibility["ref_point"]
        if bed.check_validity(space):
            return True

    return False
