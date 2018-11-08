# coding=utf-8
"""
Grid module
"""
from typing import Tuple, Sequence, Callable, Generator, Optional
import copy
import logging
import matplotlib.pyplot as plt


import libs.reader as reader
from libs.mesh import Edge, Face
from libs.plan import Plan
import libs.transformation as transformation


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
    def __init__(self, name: str, predicates: Sequence[Callable]):
        self.name = name
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
    def __init__(self, name: str, action: Callable):
        self.name = name
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

def _non_ortho_angle(edge):
    if edge.face.area < 50.0:
        return False
    if edge.previous_angle <= 95.0:
        return False
    return not edge.previous_is_ortho


def _far_from_corner(edge):
    min_dist = 30.0
    close_to_next_corner = edge.next_is_ortho and edge.length < min_dist
    close_to_previous_corner = edge.previous.previous_is_ortho and edge.previous.length < min_dist
    return not close_to_next_corner and not close_to_previous_corner


def _far_from_boundary(edge):
    min_dist = 100.0
    close_to_next_corner = (edge.next.is_mesh_boundary
                            and edge.next_is_ortho
                            and edge.length < min_dist)

    close_to_previous_corner = (edge.previous.previous.is_mesh_boundary
                                and edge.previous.previous_is_ortho
                                and edge.previous.length < min_dist)

    return not close_to_next_corner and not close_to_previous_corner


def _non_aligned_angle(edge):
    return not (edge.previous_is_aligned and edge.pair.face is None)


def _is_aligned(edge):
    return not _non_aligned_angle(edge)


def _non_ortho_not_aligned(edge):
    return _non_ortho_angle(edge) and _non_aligned_angle(edge)


def _is_window(edge):
    return edge.linear and edge.linear.category.name in ("window", "doorWindow")


def _next_is_window(edge):
    return _is_window(edge.next)


def _previous_is_window(edge):
    return _is_window(edge.previous)


def _close_edge(edge):
    select = edge.next_is_ortho and edge.next.next_is_ortho
    if not select:
        return False
    close = edge.next.length < 10.0  # for example
    return close


def _is_not_space_boundary(edge: Edge):
    return not edge.is_space_boundary


def _min_length(min_length: float) -> Callable:
    def _predicate(edge: Edge):
        return edge.length >= min_length

    return _predicate


close_edge_selector = Selector('close_edge', [_is_not_space_boundary, _close_edge])

is_aligned_selector = Selector('is_aligned', [_is_aligned, _far_from_corner])

is_window_selector = Selector('window', [_is_window, _far_from_boundary, _far_from_corner])

next_is_window_selector = Selector('before_window', [_next_is_window, _min_length(25)])

previous_is_window = Selector('previous_window',
                              [_previous_is_window, _far_from_boundary, _far_from_corner])

non_ortho_selector = Selector('non_ortho', [_non_ortho_angle, _non_aligned_angle])

non_ortho_or_aligned = Selector('non_ortho_or_aligned', [_far_from_corner, _non_ortho_not_aligned])

# slicers


def _cut_has_changed_the_mesh(cut_data):
    return cut_data and cut_data[2] is not None


def _remove_edge(edge):
    return edge.remove()


remove_edge_action = Slicer('remove_edge', _remove_edge)


def _ortho_cut(edge: Edge):
    cut_data = edge.ortho_cut()
    return _cut_has_changed_the_mesh(cut_data)


ortho_cut_slicer = Slicer('ortho_cut', _ortho_cut)


def _cut_ortho_to_edge(edge):
    cut_data = edge.cut_at_barycenter(0.0)
    return _cut_has_changed_the_mesh(cut_data)


start_slicer = Slicer('boundary', _cut_ortho_to_edge)


def _cut_ortho_to_previous(edge):
    cut_data = edge.previous.cut_at_barycenter(1)
    return _cut_has_changed_the_mesh(cut_data)


end_slicer = Slicer('boundary', _cut_ortho_to_previous)


def _ortho_laser_cut(edge: Edge) -> Optional[Face]:
    distance_to_window = 15
    vertex = (transformation.get['translation']
                            .config(vector=edge.unit_vector, coeff=-1*distance_to_window)
                            .apply_to(edge.end))
    cut_data = edge.space.cut(edge, vertex, traverse='relative')
    return _cut_has_changed_the_mesh(cut_data)


ortho_laser_cut = Slicer('ortho_laser_cut', _ortho_laser_cut)

# grids

ortho_grid = Grid('ortho_grid', [(non_ortho_selector, ortho_cut_slicer)])

rectilinear_grid = Grid('rectilinear', [
                                        (non_ortho_or_aligned, ortho_cut_slicer),
                                        (non_ortho_or_aligned, start_slicer),
                                        (non_ortho_or_aligned, end_slicer),
                                        (next_is_window_selector, ortho_laser_cut)
                                       ])

if __name__ == '__main__':

    logging.getLogger().setLevel(logging.DEBUG)

    def create_a_grid():
        """
        Test
        :return:
        """
        input_file = reader.INPUT_FILES[7]
        plan = reader.create_plan_from_file(input_file)

        new_plan = rectilinear_grid.apply_to(plan)
        new_plan.mesh.check()

        new_plan.plot(save=False)
        plt.show()

    create_a_grid()
