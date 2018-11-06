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


class Grid:
    """
    Creates a grid inside a plan.
    1. We select an edge according to the selectors,
    2. We apply the slicers
    3. if new edges are created we recursively apply the grid
    4. Once, we're finished we apply the reducers
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

        return _plan

    def iterate(self, plan: Plan, operator: Tuple['Selector', 'Slicer']):
        """
        Apply operation to each face of the empty spaces of the plan
        :param plan:
        :param operator:
        :return:
        """
        for empty_space in plan.empty_spaces:
            for face in empty_space.faces:
                new_face = self.select_and_slice(face, operator)
                if new_face is not None:
                    return self.iterate(plan, operator)
        return

    @staticmethod
    def select_and_slice(face: Face, operator: Tuple['Selector', 'Slicer']) -> Optional[Face]:
        """
        Selects the correct edges and applies the slice transformation to them
        :param face:
        :param operator:
        :return:
        """
        selector, slicer = operator
        for edge in selector.yield_from(face):
            new_face = slicer.apply_to(edge)
            if new_face is not None:
                return new_face
        return None


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

    def apply_to(self, edge: Edge) -> Optional[Face]:
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
    if edge.previous_angle <= 90.0:
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


def _previous_is_window(edge):
    return edge.previous.linear and edge.previous.linear.category.name in ("window", "doorWindow")


def _close_edge(edge):
    select = edge.next_is_ortho and edge.next.next_is_ortho
    if not select:
        return False
    close = edge.next.length < 10.0  # for example
    return close


def _is_not_space_boundary(edge):
    return not edge.is_space_boundary


close_edge_selector = Selector('close_edge', [_is_not_space_boundary, _close_edge])

is_aligned_selector = Selector('is_aligned', [_is_aligned, _far_from_corner])

is_window_selector = Selector('window', [_is_window, _far_from_boundary, _far_from_corner])

previous_is_window = Selector('previous_window',
                              [_previous_is_window, _far_from_boundary, _far_from_corner])

non_ortho_selector = Selector('non_ortho', [_non_ortho_angle, _non_aligned_angle])

non_ortho_or_aligned = Selector('non_ortho_or_aligned', [_far_from_corner, _non_ortho_not_aligned])

# slicers

remove_edge_action = Slicer('remove_edge', lambda edge: edge.remove())

ortho_cut_slicer = Slicer('ortho_cut', lambda edge: edge.ortho_cut())

boundary_slicer = Slicer('boundary', lambda edge: edge.cut_at_barycenter(0.0))

# grids

ortho_grid = Grid('ortho_grid', [(non_ortho_selector, ortho_cut_slicer)])

rectilinear_grid = Grid('rectilinear', [
                                        (non_ortho_or_aligned, ortho_cut_slicer),
                                        (is_window_selector, boundary_slicer),
                                        (previous_is_window, boundary_slicer),
                                        (is_aligned_selector, ortho_cut_slicer),
                                        (close_edge_selector, remove_edge_action)
                                       ])

if __name__ == '__main__':

    logging.getLogger().setLevel(logging.INFO)


    def create_a_grid():
        """
        Test
        :return:
        """
        # input_file = reader.INPUT_FILES[12]
        plan = reader.create_plan_from_file("Edison_10.json")

        new_plan = rectilinear_grid.apply_to(plan)
        new_plan.mesh.check()

        new_plan.plot(save=False)
        plt.show()

    create_a_grid()
