# coding=utf-8
"""
Testing module for the solver module
"""
from libs.cpsolver.solver import Solver


def adjacency_matrix(size: int):
    """
    Returns a square (size x size) grid adjacency matrix
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


def test_simple_problem():
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
        {"area": 36, "components": {"window": 3}},  # living
        {"area": 12, "components": {"window": 1, "duct": 1}},  # kitchen
        {"area": 18, "components": {"window": 2}},  # bedroom 1
        {"area": 12, "components": {"window": 1}},  # bedroom 2
        {"area": 9, "components": {"window": 1}},  # bedroom 3
        {"area": 6, "components": {"duct": 1}},  # bathroom 1
        {"area": 6, "components": {"duct": 1}},  # bathroom 2
        {"area": 3, "components": {"duct": 1, "window": -1}},  # wc
        {"area": 6, "components": {"frontDoor": 1, "window": -1}},  # entrance
    ]

    num_spaces = len(spaces)
    num_cells = num_col ** 2

    matrix = adjacency_matrix(num_col)
    params = {
        "num_solutions": 10,
        "num_fails": 1500,
        "num_restarts": 36
    }

    my_solver = Solver(matrix, params)
    domain = set(range(num_spaces))
    my_solver.add_values(domain)

    # create cells
    for ix in range(num_cells):
        if ix in (14, 20, 24, 29, 30):
            props = {"area": 3, "components": ("duct",)}
        elif ix in (0, 1, 2, 3, 4, 5, 33, 35):
            props = {"area": 3, "components": ("window",)}
        elif ix == 31:
            props = {"area": 3, "components": ("frontDoor",)}
        else:
            props = {"area": 3, "components": {}}

        my_solver.add_cell(domain, props, ix)

    # add  constraints
    for ix, space in enumerate(spaces):
        my_solver.add_component_constraint(ix, space["components"])

    for ix, space in enumerate(spaces):

        # connectivity constraint
        my_solver.add_connectivity_constraint(ix, matrix)

        # area constraint
        min_area = space["area"] * 0.99
        max_area = space["area"] * 1.01
        my_solver.add_area_constraint(ix, min_area, max_area)

    assert(len(my_solver.solve()) == 10)
