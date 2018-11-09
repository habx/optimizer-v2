# coding=utf-8
"""
Grid module
"""
from typing import Tuple, Sequence, Callable, Generator, Optional
import copy
import logging
import matplotlib.pyplot as plt
from functools import reduce


import libs.reader as reader
from libs.mesh import Edge, Face, MIN_ANGLE, ANGLE_EPSILON
from libs.plan import Plan
import libs.transformation as transformation

from libs.utils.geometry import ccw_angle


class Grid:
    """
    Creates a grid inside a plan.
    """
    def __init__(self, name: str, operators: Sequence[Tuple['Selector', 'Slicer']]):
        self.name = name
        self.operators = operators or []

    def apply_to(self, plan: Plan) -> Plan:
        """
        Returns the modified plan with the created grid
        :param plan:
        :return: a copy of the plan with the created grid
        """
        _plan = copy.deepcopy(plan)
        for operator in self.operators:
            self.iterate(_plan, operator)
            # we simplify the mesh between each operator
            plan.mesh.simplify()

        return _plan

    def iterate(self, plan: Plan, operator: Tuple['Selector', 'Slicer']):
        """
        Apply operation to each face of the empty spaces of the plan
        TODO: this is brutal as we try each face each time a cut happens
        a better way could be found !!
        :param plan:
        :param operator:
        :return:
        """
        for empty_space in plan.empty_spaces:
            for face in empty_space.faces:
                mesh_has_changed = self.select_and_slice(face, operator)
                if mesh_has_changed:
                    return self.iterate(plan, operator)
        return

    @staticmethod
    def select_and_slice(face: Face, operator: Tuple['Selector', 'Slicer']) -> bool:
        """
        Selects the correct edges and applies the slice transformation to them
        :param face:
        :param operator:
        :return:
        """
        selector, slicer = operator
        for edge in selector.yield_from(face):
            mesh_has_changed = slicer.apply_to(edge)
            if mesh_has_changed:
                return True
        return False


class Selector:
    """
    Returns an iterator on a given mesh face
    """
    def __init__(self, *predicates: Callable, name: str = ''):
        self.name = name or reduce(lambda x, y: x + '__' + y.__name__, predicates, '')
        self.predicates = predicates

    def yield_from(self, face: Face) -> Generator[Edge, None, None]:
        """
        Runs the selector
        :param face:
        :return:
        """
        for edge in face.edges:
            filtered = True
            for predicate in self.predicates:
                filtered = filtered and predicate(edge)
                if not filtered:
                    break
            else:
                yield edge


class Slicer:
    """
    Will cut a face in half
    """
    def __init__(self, action: Callable, name: Optional[str] = None):
        self.name = name or action.__name__
        self.action = action

    def apply_to(self, edge: Edge) -> bool:
        """
        Executes the action
        :param edge:
        :return: the newly created face
        """
        return self.action(edge)


# examples

# selectors

# predicates


def _is_not_space_boundary(edge: Edge):
    """
    Predicate
    Returns True if the edge is not a space boundary
    :param edge:
    :return:
    """
    return not edge.is_space_boundary


# predicates factories

def is_not(predicate: Callable) -> Callable:
    """
    Returns the opposite predicate
    :return:
    """
    def _predicate(edge: Edge) -> bool:
        return not predicate(edge)

    return _predicate


def edge_angle(min_angle: Optional[float] = None,
               max_angle: Optional[float] = None,
               previous: bool = False) -> Callable:
    """
    Predicate factory
    Returns a predicate indicating if an edge angle with its next edge
    is comprised between the two provided values. Only one of the value can be provided
    :param min_angle:
    :param max_angle:
    :param previous : whether to check for the angle to the next edge or to the previous edge
    :return: boolean
    """
    if min_angle is None and max_angle is None:
        raise ValueError('A min or a max angle must be provided to the angle predicate factory')

    def _predicate(edge: Edge) -> bool:
        _angle = edge.next_angle if not previous else edge.previous_angle
        if min_angle is not None and max_angle is not None:
            return min_angle <= _angle <= max_angle
        if min_angle is not None:
            return _angle >= min_angle
        if max_angle is not None:
            return _angle <= max_angle
        return True

    return _predicate


def edge_length(min_length: float = None, max_length: float = None) -> Callable:
    """
    Predicate factory
    Returns a predicate indicating if an edge has a length inferior or equal to the provided max
    length, or a superior length to the provided min_length or both
    :param min_length:
    :param max_length
    :return:
    """
    def _predicate(edge: Edge) -> bool:
        if min_length is not None and max_length is not None:
            return min_length <= edge.length <= max_length
        if min_length is not None:
            return edge.length >= min_length
        if max_length is not None:
            return edge.length <= max_length

    return _predicate


def is_linear(*category_names: str) -> Callable:
    """
    Predicate Factory
    :param category_names:
    :return: a predicate
    """

    def _predicate(edge: Edge) -> bool:
        return edge.linear and edge.linear.category.name in category_names

    return _predicate


def touches_linear(*category_names: str, position: str = 'before') -> Callable:
    """
    Predicate factory
    Returns a predicate indicating if an edge is between two linears of the provided category
    :param category_names: tuple of linear category names
    :param position : where should the edge be : before, after, between
    :return:
    """

    position_valid_values = 'before', 'after', 'between'

    if position not in position_valid_values:
        raise ValueError('Wrong position value in predicate factory touches_linear:' +
                         ' {0}'.format(position))

    def _predicate(edge: Edge) -> bool:
        if position == 'before':
            return edge.next.linear and edge.next.linear.category.name in category_names
        if position == 'after':
            return edge.next.linear and edge.next.linear.category.name in category_names
        if position == 'between':
            if edge.previous.linear and edge.previous.linear.category.name in category_names:
                if edge.next.linear and edge.next.linear.category.name in category_names:
                    return True
            return False

    return _predicate


def close_to_linear(*category_names: str, min_distance: float = 50.0) -> Callable:
    """
    Predicate factory
    Returns a predicate indicating if the edge is closer than
    the provided minimum distance to a linear of the provided category
    :param min_distance:
    :param category_names
    :return: function
    """
    def _predicate(edge: Edge):
        linear_edges = []
        for sibling in edge.siblings:
            if sibling.linear and sibling.linear.category.name in category_names:
                linear_edges.append(sibling)

        if not linear_edges:
            return False

        for linear_edge in linear_edges:
            start_point_dist, end_point_dist = None, None
            if ccw_angle(edge.vector, linear_edge.opposite_vector) >= 90.0 - MIN_ANGLE:
                continue
            start_point_line = linear_edge.start.sp_line(linear_edge.normal)
            end_point_line = linear_edge.end.sp_line(linear_edge.normal)
            start_point_intersection = start_point_line.intersection(edge.as_sp)
            end_point_intersection = end_point_line.intersection(edge.as_sp)
            if (not start_point_intersection.is_empty
                    and start_point_intersection.geom_type == 'Point'):
                start_point_dist = start_point_intersection.distance(linear_edge.start.as_sp)
            if (not end_point_intersection.is_empty
                    and end_point_intersection.geom_type == 'Point'):
                end_point_dist = end_point_intersection.distance(linear_edge.end.as_sp)
            if not start_point_dist and not end_point_dist:
                continue
            dist_to_edge = min(d for d in (start_point_dist, end_point_dist) if d is not None)
            if dist_to_edge <= min_distance:
                return True

        return False

    return _predicate

# slicers


def _cut_has_changed_the_mesh(cut_data):
    """
    Used to check if the cut has changed the mesh (meaning a new face has been created)
    :param cut_data:
    :return:
    """
    return cut_data and cut_data[2] is not None


def _remove_edge(edge):
    return edge.remove()


remove_edge = Slicer(_remove_edge, 'remove_edge')


def _ortho_cut(edge: Edge):
    cut_data = edge.ortho_cut()
    return _cut_has_changed_the_mesh(cut_data)


ortho_projection_cut = Slicer(_ortho_cut, 'ortho_projection_cut')


def barycenter_cut(coeff: float, angle: float = 90.0, traverse: str = 'relative') -> Callable:
    """
    Action Factory
    :param coeff:
    :param angle:
    :param traverse:
    :return: an action function
    """
    def _action(edge: Edge) -> bool:
        cut_data = edge.recursive_barycenter_cut(coeff, angle, traverse)
        return _cut_has_changed_the_mesh(cut_data)

    return _action


def translation_cut(dist: float, angle: float = 90.0, traverse: str = 'relative') -> Callable:
    """
    Action Factory
    :param dist:
    :param angle:
    :param traverse:
    :return: an action function
    """
    def _action(edge: Edge) -> bool:
        # check the length of the edge. It cannot be inferior to the translation distance
        if edge.length < dist:
            return False
        # create a translated vertex
        vertex = (transformation.get['translation']
                  .config(vector=edge.unit_vector, coeff=dist)
                  .apply_to(edge.start))
        cut_data = edge.recursive_cut(vertex, angle=angle, traverse=traverse)
        return _cut_has_changed_the_mesh(cut_data)

    return _action

# grids


sequence_grid = Grid('sequence_grid', [
    (
        Selector(edge_angle(180.0 + ANGLE_EPSILON, 270.0 - ANGLE_EPSILON, previous=True)),
        ortho_projection_cut
    ),
    (
        Selector(edge_angle(180.0 + ANGLE_EPSILON, 270.0 - ANGLE_EPSILON)),
        ortho_projection_cut
    ),
    (
        Selector(edge_angle(90.0 + ANGLE_EPSILON, 180.0 - ANGLE_EPSILON, previous=True)),
        ortho_projection_cut
    ),
    (
        Selector(edge_angle(90.0 + ANGLE_EPSILON, 180.0 - ANGLE_EPSILON)),
        ortho_projection_cut
    ),
    (
        Selector(edge_angle(270.0 - ANGLE_EPSILON, 270.0 + ANGLE_EPSILON, previous=True)),
        ortho_projection_cut
    ),
    (
        Selector(close_to_linear('window', 'doorWindow', min_distance=90.0),
                 _is_not_space_boundary),
        remove_edge
    ),
    (
        Selector(touches_linear('window', 'doorWindow', position='between')),
        Slicer(barycenter_cut(0.5))
    ),
    (
        Selector(edge_angle(180.0 - ANGLE_EPSILON, 180.0 + ANGLE_EPSILON),
                 is_not(touches_linear('window', 'doorWindow')),
                 is_not(is_linear('window', 'doorWindow'))),
        Slicer(barycenter_cut(1.0))
    ),
    (
        Selector(edge_length(min_length=150.0)),
        Slicer(barycenter_cut(0.5))
    ),
    (
        Selector(close_to_linear('window', 'doorWindow', min_distance=90.0),
                 _is_not_space_boundary),
        remove_edge
    )
])

if __name__ == '__main__':

    logging.getLogger().setLevel(logging.INFO)

    def create_a_grid():
        """
        Test
        :return:
        """
        input_file = reader.BLUEPRINT_INPUT_FILES[17]  # 16: Edison_10
        plan = reader.create_plan_from_file(input_file)

        new_plan = sequence_grid.apply_to(plan)
        new_plan.check()

        new_plan.plot(save=False)
        plt.show()

    create_a_grid()
