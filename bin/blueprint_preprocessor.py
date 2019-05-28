# -*- coding: utf-8 -*-
"""
Blueprint preprocessor
"""
from __future__ import print_function
import os
from libs.plan.plan import Plan, Edge, Space
import libs.io.reader as reader
from libs.modelers.grid import GRIDS
from libs.modelers.seed import SEEDERS
from libs.optimizer import ExecParams
import logging

MAXIMUM_DUCT_SPACES = 3 # maximum duct spaces for one duct

def count_duct_space(current_plan: 'Plan') -> int:
    """
    counts the number of ducts spaces
    for duct along wall, only spaces on both sides are considered
    :return: float
    """

    def _wall_edge(_edge: 'Edge'):
        # returns true if the edge pair is outside the apartment
        if not _edge.pair.face:
            return True
        space_pair = current_plan.get_space_of_edge(_edge.pair)
        if space_pair and space_pair.category.external:
            return True
        return False

    def _against_wall(_space: 'Space'):
        # returns true if the space is along the apartment border
        for space_edge in _space.edges:
            if _wall_edge(space_edge):
                return True
        return False

    ducts = [sp for sp in current_plan.spaces if sp.category.name == "duct"]
    adjacent_duct_spaces = []
    overflow_count = 0
    for duct in ducts:
        count = 0
        if _against_wall(duct):
            for edge in duct.edges:
                if (_wall_edge(duct.previous_edge(edge))
                        or _wall_edge(duct.next_edge(edge))):
                    sp = current_plan.get_space_of_edge(edge.pair)
                    if sp and sp.mutable and not sp in adjacent_duct_spaces:
                        number_linears = sum(sp.has_linear(component) for component in sp.plan.linears)
                        if number_linears > 0:
                            continue
                        adjacent_duct_spaces.append(sp)
                        count += 1
                        if count > MAXIMUM_DUCT_SPACES:
                            overflow_count += 1

        else:
            for sp in duct.adjacent_spaces():
                if sp.mutable and not sp in adjacent_duct_spaces:
                    number_linears = sum(sp.has_linear(component) for component in sp.plan.linears)
                    if number_linears > 0:
                        continue
                    adjacent_duct_spaces.append(sp)
                    count += 1
                    if count > MAXIMUM_DUCT_SPACES:
                        overflow_count += 1

    return len(adjacent_duct_spaces) - overflow_count


def count_window_space(plan: 'Plan') -> int:
    """
    counts the number of spaces that contain a window
    """
    window_space = 0
    mutable_spaces = [sp for sp in plan.spaces if sp.mutable]
    for sp in mutable_spaces:
        number_windows = sum(
            sp.has_linear(component) and component.category.window_type for component in
            sp.plan.linears)
        if number_windows > 0:
            window_space += 1
    return window_space


def run(input_file: str, directory_path: str, do_plot: bool = False):
    params = ExecParams(None)
    current_plan = reader.create_plan_from_file(input_file, directory_path)
    GRIDS[params.grid_type].apply_to(current_plan)
    SEEDERS[params.seeder_type].apply_to(current_plan)
    if do_plot:
        current_plan.plot()
    return current_plan


if __name__ == '__main__':

    def main():
        """
        Useful simple main
        """
        directory_path = "/Users/graziella/Desktop/test" # your path
        do_plot = True
        count_dict = {}
        for file_name in os.listdir(directory_path):
            if ".json" in file_name:
                file_plan = run(file_name, directory_path, do_plot)
                nb_window_space = count_window_space(file_plan)
                nb_duct_space = count_duct_space(file_plan)
                count_dict[file_name] = {"nb_window_space": nb_window_space,
                                         "nb_duct_space": nb_duct_space}
                if nb_window_space < 1:
                    logging.warning("blueprint preprocessor : nb_window_space %i", nb_window_space)
                if nb_duct_space < 1:
                    logging.warning("blueprint preprocessor : nb_duct_space %i", nb_duct_space)

                print(file_name, count_dict[file_name])

    main()