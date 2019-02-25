# coding=utf-8
"""
Testing module for the solver module
"""
from libs.cpsolver.solver import Solver
from typing import List
import math

# the perimeter must be < sqrt(Area) * PERIMETER_RATIO
PERIMETER_RATIO = 4.8
# the max distance between to cells must be < sqrt(space.area) * MAX_SIZE_RATIO
MAX_SIZE_RATIO = 1.6


def adjacency_matrix(size: int) -> List[List[int]]:
    """
    Returns a square (size ** 2 x size ** 2) grid adjacency matrix
    :param size:
    :return:
    """
    output = [[0 for _ in range(size**2)] for _ in range(size**2)]
    for i in range(size ** 2):
        if i >= size:
            output[i][i - size] = 1
            output[i - size][i] = 1
        if i < size * (size - 1):
            output[i][i + size] = 1
            output[i + size][i] = 1
        if i % size != 0:
            output[i][i - 1] = 1
            output[i - 1][i] = 1
        if (i + 1) % size != 0:
            output[i][i + 1] = 1
            output[i + 1][i] = 1

    return output


def distance_matrix(size: int):
    """
    Returns a square (size x size) grid distance matrix
    :param size:
    :return:
    """
    output = [[0 for _ in range(size**2)] for _ in range(size**2)]
    for i in range(size ** 2):
        col_i = i % size
        row_i = i // size
        for j in range(i + 1, size ** 2):
            col_j = j % size
            row_j = j // size
            output[i][j] = math.sqrt((abs(col_j - col_i) + 1)**2 + (abs(row_j - row_i) + 1)**2)
            output[j][i] = output[i][j]

    return output


def simple_problem():
    """
    Testing of a simple solver
    :return:
    """
    num_col = 3
    spaces = [
        {"area": 3, "components": {"window": 1}},
        {"area": 3, "components": {"window": 1}},
        {"area": 3, "components": {"window": -1}}
    ]

    num_spaces = len(spaces)
    num_cells = num_col ** 2

    matrix = adjacency_matrix(num_col)

    my_solver = Solver(matrix)
    domain = set(range(num_spaces))
    my_solver.add_values(domain)

    # create cells
    for ix in range(num_cells):
        if ix in (3, 7):
            props = {"area": 1, "components": ("window",)}
        else:
            props = {"area": 1, "components": {}}

        my_solver.add_cell(domain, props, ix)

    # add  constraints
    for ix, space in enumerate(spaces):

        # component constraint
        if space["components"]:
            my_solver.add_component_constraint(ix, space["components"])

        # connectivity constraint
        my_solver.add_connectivity_constraint(ix, matrix)

        # area constraint
        area = space["area"]
        min_area = area * 0.99
        max_area = area * 1.01
        my_solver.add_area_constraint(ix, min_area, max_area)

    # symmetry constraint
    my_solver.add_symmetry_breaker_constraint(0, 1)

    assert(len(my_solver.solve()) == 8)


def solve_complex():
    """
    Testing of a harder case :
    An apartment of 108 sqm
    -> 108 m2
    -> 36 squared cells of 3m2 each
    -> 9 spaces
    :return:
    """
    num_col = 6
    spaces = [
        {"area": 24, "components": {"window": 2}},  # 0. living
        {"area": 12, "components": {"window": 1, "duct": 1}},  # 1. kitchen
        {"area": 18, "components": {"window": 1}},  # 2. bedroom 1
        {"area": 18, "components": {"window": 1}},  # 3. bedroom 2
        {"area": 12, "components": {"window": 1}},  # 4. bedroom 3
        {"area": 9, "components": {"duct": 1}},  # 5. bathroom 1
        {"area": 6, "components": {"duct": 1}},  # 6. bathroom 2
        {"area": 3, "components": {"duct": 1, "window": -1}},  # 7. wc
        {"area": 6, "components": {"frontDoor": 1, "window": -1}},  # 8. entrance
    ]

    num_spaces = len(spaces)
    num_cells = num_col ** 2

    matrix = adjacency_matrix(num_col)
    distances = distance_matrix(num_col)
    logging.debug(distances)

    params = {
        "num_solutions": 100,
        "num_fails": 300000,
        "num_restarts": 0
    }

    my_solver = Solver(matrix, distances, params=params)
    domain = set(range(num_spaces))
    my_solver.add_values(domain)

    # create cells
    for ix in range(num_cells):
        if ix in (14, 20, 24, 29, 30):
            props = {"area": 3, "perimeter": 4, "components": ("duct",)}
        elif ix in (0, 1, 2, 3, 4, 5, 33, 35):
            props = {"area": 3, "perimeter": 4, "components": ("window",)}
        elif ix == 31:
            props = {"area": 3, "perimeter": 4, "components": ("frontDoor",)}
        else:
            props = {"area": 3, "perimeter": 4, "components": {}}

        my_solver.add_cell(domain, props, ix)

    my_solver.add_symmetry_breaker_constraint(2, 3)

    # add  constraints
    for ix, space in enumerate(spaces):
        # component constraint
        my_solver.add_component_constraint(ix, space["components"])

        # max size constraint
        my_solver.add_max_size_constraint(ix, math.sqrt(space["area"] / 3.0) * 1.65)

        # connectivity constraint
        my_solver.add_connectivity_constraint(ix, matrix)

        # area constraint
        min_area = space["area"] * 0.99
        max_area = space["area"] * 1.01
        my_solver.add_area_constraint(ix, min_area, max_area)
        my_solver.add_max_perimeter_constraint(ix, math.sqrt(space["area"]/3)*4.8)

    assert(len(my_solver.solve()) == 2)


if __name__ == '__main__':
    import logging
    logging.getLogger().setLevel(logging.DEBUG)
    solve_complex()
