# coding=utf-8
"""
Constraint module : describes the type of constraint that can be applied to a space or a linear
These constraints could be used by a genetic algorithm to compute a multi-objective cost function

A constraint computes a score

"""
import math
from typing import TYPE_CHECKING, Callable, Union, Dict, Optional, Any

from libs.utils.geometry import ccw_angle, pseudo_equal
from libs.specification.size import Size

if TYPE_CHECKING:
    from libs.plan.plan import Space

scoreFunction = Callable[[Union['Space', 'Linear']], float]
scoreFunctionFactory = Callable[..., scoreFunction]


class Constraint:
    """
    A type of constraint
    An imperative constraint must be verified. A mutation cannot be applied if it will
    break an imperative constraint.
    If a constraint is not imperative, it will be considered an objective constraint.
    In this case, a mutation will be applied if it will increase the score of the modified spaces
    according to the constraint score function.
    """

    def __init__(self,
                 score_factory: scoreFunctionFactory,
                 params: Dict[str, Any], name: str = "",
                 imperative: bool = True):
        self.score_factory = score_factory  # the min the better
        self.params = params
        self.name = name
        self.imperative = imperative

    def __repr__(self):
        return 'Constraint: {0}'.format(self.name)

    def check(self,
              *spaces: 'Space',
              other_constraint: Optional['Constraint'] = None) -> bool:
        """
        Returns True if the constraint is satisfied. A constraint is deemed satisfied if
        its score is inferior or equal to zero
        :param spaces:
        :param other_constraint: an other constraint
        :return:
        """
        for space in spaces:
            # per convention if the params specify a category name, we only apply the constraint
            # to spaces of the specified category
            if ("category_name" in self.params
                    and space.category.name != self.params["category_name"]):
                continue
            if other_constraint is not None:
                if self.score(space) > 0 and other_constraint.score(space) > 0:
                    return False
            if self.score(space) > 0:
                return False
        return True

    def set(self, **params) -> 'Constraint':
        """
        Sets the params of the constraint
        For example a maximum width
        :param params:
        :return:
        """
        for key, value in params.items():
            if key in self.params:
                self.params[key] = value
            else:
                raise ValueError('A parameter can only be modified not added: {0}'.format(key))

        return self

    def score(self, space: 'Space') -> float:
        """
        Computes a score of how the space respects the constraint
        :param space:
        :return: a score
        """
        return self.score_factory(self.params)(space)


class SpaceConstraint(Constraint):
    """
    A space constraint
    A rule that can be applied to a Space instance
    Examples : 'square'
    """
    pass


class LinearConstraint(Constraint):
    """
    A type of linear constraint
    """
    pass


# score functions factories

def square_shape(_: Dict) -> scoreFunction:
    """
    Scores the area / perimeter ratio of a space.
    :return:
    """

    def _score(space: 'Space') -> float:
        if (space is None or space.size is None or
                space.size.depth is None or space.size.width is None):
            return 0.0
        rectangle_area = space.size.depth * space.size.width
        return max((rectangle_area - space.area) / rectangle_area, 0.0)

    return _score


def component_surface_objective(params: Dict) -> scoreFunction:
    """
    Scores the respect of surface margins for cells containing components defined in params
    :param params:
    :return:
    """

    def shift_from_interval(val_min: float, val_max: float, val: float) -> float:
        return max((val - val_max) / val_max, (val_min - val) / val, 0.0)

    def _score(space: 'Space') -> float:
        list_components = list(
            p for p in params.keys() if p in space.components_category_associated())

        if space.category.name is "empty":
            return 0

        score = 0
        if not list_components:
            list_components = ["default"]
        if list_components:
            min_area = max(params[component]["min_area"] for component in list_components)
            max_area = max(params[component]["max_area"] for component in list_components)
            score = shift_from_interval(min_area,
                                        max_area, space.area)
        score = params["weight"] * score
        return score

    return _score


def component_aspect_constraint(params: Dict) -> scoreFunction:
    """
    Scores the respect of shape size and aspect ratio for cells containing components defined
    in params
    :param params:
    :return:
    """

    def shift_from_interval(val_min: float, val_max: float, val: float) -> float:
        return max((val - val_max) / val_max, (val_min - val) / val, 0.0)

    def _score(space: 'Space') -> float:
        list_components = list(
            p for p in params.keys() if p in space.components_category_associated())

        if space.category.name is "empty":
            return 0

        score = 0
        if not list_components:
            list_components = ["default"]
        if list_components:
            size_ratio = space.size.depth / space.size.width
            aspect_ratio = min(size_ratio, 1 / size_ratio)
            score += shift_from_interval(params["min_aspect_ratio"],
                                         1 / params["min_aspect_ratio"], aspect_ratio)

            score = params["weight"] * score

        return score

    return _score


def number_of_components(params: Dict):
    """
    penalizes spaces containing high number of components, for components considered in params
    :return:
    """

    def _score(space: 'Space') -> float:
        if space is None:
            return 0.0
        list_components = space.components_category_associated()
        number_components = sum(comp in params.keys() for comp in list_components)
        for adjacent_space in space.adjacent_spaces():
            list_components = adjacent_space.components_category_associated()
            number_components += sum(comp in params.keys() for comp in list_components)
        return number_components ** 2

    return _score


def few_corners(params: Dict) -> scoreFunction:
    """
    Scores a space by counting the number of corners.
    Note : we only count the exterior corners of the space
    We do not count the corners of internal holes.
    A corner is defined as an angle between two boundaries edge superior to 20.0
    :param params:
    :return:
    """
    min_corners = params['min_corners']

    def _score(space: 'Space') -> float:
        number_of_corners = space.number_of_corners()
        if number_of_corners == 0:
            return 0

        return math.fabs((number_of_corners - min_corners) / number_of_corners)

    return _score


def alignments(_: Dict):
    """
    Scores a space by counting the number of corners and t-edges
    A corner is defined as an angle between two boundary edges superior to 20.0
    :param:
    :return:
    """

    corner_min_angle = 20.0

    def _score(space: 'Space') -> float:
        number_of_corners = 0
        for edge in space.exterior_edges:
            if not edge.is_mesh_boundary and ccw_angle(edge.vector, space.next_edge(
                    edge).vector) >= corner_min_angle:
                number_of_corners += 1
        return space.count_t_edges() + number_of_corners

    return _score


def max_size(params: Dict) -> scoreFunction:
    """
    Checks if a space has a size inferior to the specified size
    :param params
    :return:
    """
    _max_size: Size = params['max_size']

    def _score(space: 'Space') -> float:
        if space.size <= _max_size:
            return 0
        else:
            return _max_size.distance(space.size)

    return _score


def min_size(params: Dict) -> scoreFunction:
    """
    Checks if a space has a size inferior to the specified size
    :param params
    :return:
    """
    _min_size: Size = params['min_size']

    def _score(space: 'Space') -> float:
        if space.size >= _min_size:
            return 0
        else:
            return _min_size.distance(space.size)

    return _score


# Imperative constraints
min_size_constraint = SpaceConstraint(min_size,
                                      {"min_size": Size(3900, 60, 60)},
                                      "min-size")

max_size_constraint = SpaceConstraint(max_size,
                                      {"max_size": Size(100000, 1000, 1000)},
                                      "max_size")

max_size_constraint_seed = SpaceConstraint(max_size,
                                           {"max_size": Size(100000, 1000, 1000),
                                            "category_name": "seed"},
                                           "max_size")

max_size_s_constraint_seed = SpaceConstraint(max_size,
                                             {"max_size": Size(180000, 500, 450),
                                              "category_name": "seed"},
                                             "max_size_s")

max_size_xs_constraint_seed = SpaceConstraint(max_size,
                                              {"max_size": Size(90000, 300, 300),
                                               "category_name": "seed"},
                                              "max_size_xs")

ELEMENT_CONSTRAINT_SEED = {
    'duct': SpaceConstraint(max_size,
                            {"max_size": Size(30000, 150, 200),
                             "category_name": "seed"},
                            "max_size_duct"),
    'frontDoor': SpaceConstraint(max_size,
                                 {"max_size": Size(40000, 200, 200),
                                  "category_name": "seed"},
                                 "max_size_frontDoor"),
    'window': SpaceConstraint(max_size,
                              {"max_size": Size(12000, 300, 400),
                               "category_name": "seed"},
                              "max_size_window"),
    'doorWindow': SpaceConstraint(max_size,
                                  {"max_size": Size(16000, 400, 400),
                                   "category_name": "seed"},
                                  "max_size_doorWindow")
}

max_size_default_constraint_seed = SpaceConstraint(max_size,
                                                   {"max_size": Size(90000, 300, 300),
                                                    "category_name": "seed"},
                                                   "max_size_xs")

# objective constraints
component_surface_objective = SpaceConstraint(component_surface_objective,
                                              {
                                                  "weight": 10,
                                                  "default": {"min_area": 10000,
                                                              "max_area": 70000},
                                                  "frontDoor": {"min_area": 20000,
                                                                "max_area": 50000},
                                                  "duct": {"min_area": 15000,
                                                           "max_area": 30000},
                                                  "window": {"min_area": 50000,
                                                             "max_area": 120000},
                                                  "doorWindow": {"min_area": 50000,
                                                                 "max_area": 120000},
                                                  "startingStep": {"min_area": 10000,
                                                                   "max_area": 30000},
                                              },
                                              "component_surface_objective", imperative=False)

min_width = 1000
min_depth = 1000
max_width = 4000
max_depth = 4000
component_aspect_constraint = SpaceConstraint(component_aspect_constraint,
                                              {
                                                  "weight": 10,
                                                  "min_aspect_ratio": 0.3,
                                                  "default": {"min_width": min_width,
                                                              "min_depth": min_depth,
                                                              "max_width": max_width,
                                                              "max_depth": max_depth},
                                                  "frontDoor": {"min_width": min_width,
                                                                "min_depth": min_depth,
                                                                "max_width": max_width,
                                                                "max_depth": max_depth},
                                                  "duct": {"min_width": min_width,
                                                           "min_depth": min_depth,
                                                           "max_width": max_width,
                                                           "max_depth": max_depth},
                                                  "window": {"min_width": min_width,
                                                             "min_depth": min_depth,
                                                             "max_width": max_width,
                                                             "max_depth": max_depth},
                                                  "doorWindow": {"min_width": min_width,
                                                                 "min_depth": min_depth,
                                                                 "max_width": max_width,
                                                                 "max_depth": max_depth},
                                                  "startingStep": {"min_width": min_width,
                                                                   "min_depth": min_depth,
                                                                   "max_width": max_width,
                                                                   "max_depth": max_depth},
                                              },
                                              "component_aspect_constraint", imperative=False)

number_of_components = SpaceConstraint(number_of_components,
                                       {
                                           "duct": [],
                                           "window": [],
                                           "doorWindow": [],
                                           "frontDoor": []
                                       },
                                       "number_of_components", imperative=False)

square_shape = SpaceConstraint(square_shape,
                               {"max_ratio": 100.0},
                               "square_shape", imperative=False)

few_corners_constraint = SpaceConstraint(few_corners,
                                         {"min_corners": 4},
                                         "few_corners",
                                         imperative=False)

alignments_constraint = SpaceConstraint(alignments,
                                        {},
                                        imperative=False)

CONSTRAINTS = {
    "few_corners": few_corners_constraint,
    "min_size": min_size_constraint,
    "max_size": max_size_constraint,
    "max_size_seed": max_size_constraint_seed,
    "max_size_s_seed": max_size_s_constraint_seed,
    "max_size_xs_seed": max_size_xs_constraint_seed,
    "max_size_duct_constraint_seed": ELEMENT_CONSTRAINT_SEED['duct'],
    "max_size_frontdoor_constraint_seed": ELEMENT_CONSTRAINT_SEED['frontDoor'],
    "max_size_window_constraint_seed": ELEMENT_CONSTRAINT_SEED['window'],
    "max_size_doorWindow_constraint_seed": ELEMENT_CONSTRAINT_SEED['doorWindow'],
    "max_size_default_constraint_seed": max_size_default_constraint_seed,
    "square_shape": square_shape,
    "component_surface_objective": component_surface_objective,
    "component_aspect_constraint": component_aspect_constraint,
    "number_of_components": number_of_components,
    "alignments": alignments_constraint
}
