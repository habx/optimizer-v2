# coding=utf-8
"""
Constraint module : describes the type of constraint that can be applied to a space or a linear
These constraints could be used by a genetic algorithm to compute a multi-objective cost function

A constraint computes a score

"""
import math
from typing import TYPE_CHECKING, Callable, Union, Dict, Optional, Any

from libs.utils.geometry import ccw_angle
from libs.size import Size

if TYPE_CHECKING:
    from libs.plan import Space

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
    Scores checks surface margins for cells containing components defined in params
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
        score = 10 * score
        return score

    return _score


def number_of_components(params: Dict):
    """
    penalizes spaces containing high number of components, for components considered in params
    :return:
    """

    def _score(space: 'Space') -> float:
        if (space is None):
            return 0.0
        list_components = space.components_category_associated()
        number_components = sum(comp in params.keys() for comp in list_components)
        for adjacent_space in space.adjacent_spaces():
            list_components = adjacent_space.components_category_associated()
            number_components += sum(comp in params.keys() for comp in list_components)
        return number_components ** 2

    return _score


def aligned_spaces(params: Dict):
    """
    better scores when corner of spaces are corner of adjacent spaces as well
    :return:
    """

    def _score(space: 'Space') -> float:
        if (space is None or space.category.name in params[
            "excluded"]):
            return 0.0
        non_proper_arrangement = 0
        corner_edges = [edge for edge in space.edges if edge.pair and
                        (space.previous_is_orho(edge) or space.next_is_orho(
                            edge))]

        for edge in corner_edges:
            space_pair = space.plan.get_space_of_edge(edge.pair)

            if space_pair and space_pair.category.name not in params[
                "excluded"] and not (
                    space_pair.next_is_orho(edge.pair) or space_pair.previous_is_orho(edge.pair)):
                non_proper_arrangement += 1

        print("space", space, "non_proper_arrangement", non_proper_arrangement)

        return non_proper_arrangement

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
    corner_min_angle = 20.0

    def _score(space: 'Space') -> float:
        number_of_corners = 0
        for edge in space.exterior_edges:
            if ccw_angle(edge.vector, space.next_edge(edge).vector) >= corner_min_angle:
                number_of_corners += 1

        if number_of_corners == 0:
            return 0

        return math.fabs((number_of_corners - min_corners) / number_of_corners)

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
                            {"max_size": Size(35000, 200, 200),
                             "category_name": "seed"},
                            "max_size_duct"),
    'frontDoor': SpaceConstraint(max_size,
                                 {"max_size": Size(50000, 200, 200),
                                  "category_name": "seed"},
                                 "max_size_frontDoor"),
    'window': SpaceConstraint(max_size,
                              {"max_size": Size(80000, 300, 300),
                               "category_name": "seed"},
                              "max_size_window"),
    'doorWindow': SpaceConstraint(max_size,
                                  {"max_size": Size(90000, 400, 300),
                                   "category_name": "seed"},
                                  "max_size_doorWindow")
}

max_size_default_constraint_seed = SpaceConstraint(max_size,
                                                   {"max_size": Size(60000, 300, 300),
                                                    "category_name": "seed"},
                                                   "max_size_xs")

# objective constraints
component_surface_objective = SpaceConstraint(component_surface_objective,
                                              {
                                                  "default": {"min_area": 15000,
                                                              "max_area": 70000},
                                                  "frontDoor": {"min_area": 20000,
                                                                "max_area": 50000},
                                                  "duct": {"min_area": 10000,
                                                           "max_area": 30000},
                                                  "window": {"min_area": 50000,
                                                             "max_area": 80000},
                                                  "doorWindow": {"min_area": 50000,
                                                                 "max_area": 80000},
                                                  "startingStep": {"min_area": 10000,
                                                                   "max_area": 30000},
                                              },
                                              "component_surface_objective", imperative=False)

number_of_components = SpaceConstraint(number_of_components,
                                       {
                                           "duct": [],
                                           "window": [],
                                           "doorWindow": [],
                                           "frontDoor": []
                                       },
                                       "number_of_components", imperative=False)

aligned_spaces = SpaceConstraint(aligned_spaces, {"excluded": ['empty']},
                                 "aligned_spaces", imperative=False)

square_shape = SpaceConstraint(square_shape,
                               {"max_ratio": 100.0},
                               "square_shape", imperative=False)

few_corners_constraint = SpaceConstraint(few_corners,
                                         {"min_corners": 4},
                                         "few_corners",
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
    "number_of_components": number_of_components,
    "aligned_spaces": aligned_spaces
}
