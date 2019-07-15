from typing import (
    TYPE_CHECKING,
    Sequence,
    Tuple,
    Dict,
    Optional,
    Callable)
from libs.plan.fitting import (fit_along_walls,
                               fit_in_corners,
                               fit_in_center)
from libs.plan.category import SpaceCategory, SPACE_CATEGORIES, LinearCategory, LINEAR_CATEGORIES
from libs.io.plot import plot_polygon
from libs.utils.geometry import (ccw_angle,
                                 unit,
                                 rotate,
                                 minimum_rotated_rectangle,
                                 polygons_collision,
                                 polygon_border_collision,
                                 polygon_linestring_collision,
                                 move_point,
                                 rectangle,
                                 centroid)

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
                    # order is applied in space
                    for name in furniture_names:
                        placed = False
                        possible_furnitures = [Furniture(category)
                                               for category in FURNITURE_CATALOG[name]
                                               if not is_prm or category.is_prm]
                        for furniture in possible_furnitures:
                            for func in furniture.category.fitting:
                                if func(furniture, space):
                                    placed = True
                                    plan.furnitures.setdefault(space, []).append(furniture)
                                    break
                            if placed:
                                break


class FurnitureCategory:
    BY_NAME: Dict[str, 'FurnitureCategory'] = {}

    def __init__(self,
                 name: str,
                 polygon: 'ListCoords2d',
                 is_prm: bool,
                 required_space: 'ListCoords2d',
                 fitting: [Callable],
                 color: str = 'black'):
        """
        :param name:
        :param polygon: polygon of the furniture: ref point at 0,0: all ys should be > 0, shape
        should be symmetric along y axis
        :param is_prm: is ok for a prm room
        :param required_space: polygon of the furniture + space around
        :param fitting: functions used to fit it, starting with the most appropriate ones
        :param color: color to show it
        """
        self.name = name
        self.polygon = polygon
        self.is_prm = is_prm
        self.required_space = required_space
        self.fitting = fitting
        self.color = color
        FurnitureCategory.BY_NAME[name] = self
        self.vect_to_centroid = centroid(polygon)


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
        space_edges = [e for e in space.edges]
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
                if linear.edge in space_edges:
                    # door opens in the space
                    door_box = rectangle(linear.edge.start.coords, linear.edge.vector,
                                         linear.length, linear.length)
                    if polygons_collision(door_box, footprint, -1):
                        return False
                elif linear.edge.pair in space_edges:
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
    "single_bed": (FurnitureCategory("single_bed_1",
                                     ((-45, 0), (45, 0), (45, 190), (-45, 190)),
                                     False,
                                     ((-105, 0), (45, 0), (45, 250), (-105, 250)),
                                     [fit_along_walls, fit_in_corners, fit_in_center]),
                   FurnitureCategory("single_bed_1",
                                     ((-45, 0), (45, 0), (45, 190), (-45, 190)),
                                     False,
                                     ((-45, 0), (105, 0), (105, 250), (-45, 250)),
                                     [fit_along_walls, fit_in_corners, fit_in_center]),),
    "double_bed": (FurnitureCategory("double_bed",
                                     ((-70, 0), (70, 0), (70, 190), (-70, 190)),
                                     False,
                                     ((-130, 0), (130, 0), (130, 250), (-130, 250)),
                                     [fit_along_walls, fit_in_corners, fit_in_center]),
                   FurnitureCategory("double_bed_pmr_1",
                                     ((-70, 0), (70, 0), (70, 190), (-70, 190)),
                                     True,
                                     ((-160, 0), (160, 0), (160, 310), (-160, 310)),
                                     [fit_along_walls, fit_in_corners, fit_in_center]),
                   FurnitureCategory("double_bed_pmr_2",
                                     ((-70, 0), (70, 0), (70, 190), (-70, 190)),
                                     True,
                                     ((-130, 0), (130, 0), (130, 250), (-130, 250)),
                                     [fit_along_walls, fit_in_corners, fit_in_center])),
    "bathtub": (FurnitureCategory("bathtub",
                                  ((-90, 0), (90, 0), (90, 80), (-90, 80)),
                                  True,
                                  ((-90, 0), (90, 0), (90, 80), (-90, 80)),
                                  [fit_in_corners, fit_along_walls, fit_in_center],
                                  color='blue'),),
    "toilet_seat": (FurnitureCategory("toilet_seat",
                                      ((-28, 0), (28, 0), (28, 32), (27.46, 37.46), (24.7, 45.2),
                                       (19.8, 51.8), (13.2, 56.7), (5.46, 59.46), (-2.7, 59.9),
                                       (-10.7, 57.9), (-17.8, 53.7), (-23.3, 47.5), (-26.8, 40.1),
                                       (-28, 32)),
                                      True,
                                      ((-45, 0), (45, 0), (45, 130), (-45, 130)),
                                      [fit_in_corners, fit_along_walls, fit_in_center],
                                      color='blue'),),
    "table": (FurnitureCategory("table",
                                ((100.0, 100.0),
                                 (86.60254037844389, 50.00000000000004),
                                 (50.000000000000085, 13.39745962155618),
                                 (1.3934999695075554e-13, 0.0),
                                 (-49.99999999999983, 13.397459621556024),
                                 (-86.60254037844373, 49.99999999999978),
                                 (-100.0, 99.99999999999967),
                                 (-86.60254037844406, 149.99999999999966),
                                 (-50.00000000000034, 186.60254037844368),
                                 (-4.624589118372728e-13, 200.0),
                                 (49.999999999999545, 186.60254037844413),
                                 (86.60254037844356, 150.0000000000005),
                                 (100.0, 100.0)),
                                True,
                                ((100.0, 100.0),
                                 (86.60254037844389, 50.00000000000004),
                                 (50.000000000000085, 13.39745962155618),
                                 (1.3934999695075554e-13, 0.0),
                                 (-49.99999999999983, 13.397459621556024),
                                 (-86.60254037844373, 49.99999999999978),
                                 (-100.0, 99.99999999999967),
                                 (-86.60254037844406, 149.99999999999966),
                                 (-50.00000000000034, 186.60254037844368),
                                 (-4.624589118372728e-13, 200.0),
                                 (49.999999999999545, 186.60254037844413),
                                 (86.60254037844356, 150.0000000000005),
                                 (100.0, 100.0)),
                                [fit_in_center, fit_along_walls, fit_in_corners],
                                color='brown'),)
}

GARNISHERS = {
    "default": Garnisher("default", [
        (SPACE_CATEGORIES["bedroom"], "xs", ("single_bed",), False),
        (SPACE_CATEGORIES["bedroom"], "s", ("double_bed",), False),
        (SPACE_CATEGORIES["bedroom"], "m", ("double_bed",), False),
        (SPACE_CATEGORIES["bedroom"], "l", ("double_bed",), True),
        (SPACE_CATEGORIES["bedroom"], "xl", ("double_bed",), True),

        (SPACE_CATEGORIES["bathroom"], "xs", ("bathtub",), False),
        (SPACE_CATEGORIES["bathroom"], "s", ("bathtub",), False),
        (SPACE_CATEGORIES["bathroom"], "m", ("bathtub",), False),
        (SPACE_CATEGORIES["bathroom"], "l", ("bathtub",), False),
        (SPACE_CATEGORIES["bathroom"], "xl", ("bathtub",), False),

        (SPACE_CATEGORIES["toilet"], "xs", ("toilet_seat",), False),
        (SPACE_CATEGORIES["toilet"], "s", ("toilet_seat",), False),
        (SPACE_CATEGORIES["toilet"], "m", ("toilet_seat",), False),
        (SPACE_CATEGORIES["toilet"], "l", ("toilet_seat",), False),
        (SPACE_CATEGORIES["toilet"], "xl", ("toilet_seat",), False),

        (SPACE_CATEGORIES["livingKitchen"], "xs", ("table",), False),
        (SPACE_CATEGORIES["livingKitchen"], "s", ("table",), False),
        (SPACE_CATEGORIES["livingKitchen"], "m", ("table",), False),
        (SPACE_CATEGORIES["livingKitchen"], "l", ("table",), False),
        (SPACE_CATEGORIES["livingKitchen"], "xl", ("table",), False),

        (SPACE_CATEGORIES["living"], "xs", ("table",), False),
        (SPACE_CATEGORIES["living"], "s", ("table",), False),
        (SPACE_CATEGORIES["living"], "m", ("table",), False),
        (SPACE_CATEGORIES["living"], "l", ("table",), False),
        (SPACE_CATEGORIES["living"], "xl", ("table",), False),

    ])
}
