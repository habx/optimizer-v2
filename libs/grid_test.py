"""for testing purposes"""
import json
import math
import logging
from libs.utils.decorator_cache import DecoratorCache
from libs.mesh import plt, Mesh
from libs.grid import get_perimeter, get_fixed_items_perimeters
import libs.optimizer_input as Input
import libs.logsetup


import libs.logsetup as ls
ls.init()


def _plot_perimeters(perimeters):
    """
    Test
    :param perimeters:
    :return:
    """
    if len(perimeters) == 1:
        plot_and_cut_perimeter(perimeters[0][0], perimeters[0][1])
    else:
        n_col = 2
        n = int(math.ceil(len(perimeters) / n_col))
        fig, axs = plt.subplots(n, n_col)

        for i, perimeter_fixed_items in enumerate(perimeters):
            perimeter, fixed_items = perimeter_fixed_items
            row, col = math.floor(i / n_col), i % n_col
            if n == 0 or n == 1:
                plot_and_cut_perimeter(perimeter, fixed_items, ax=axs[col])
            else:
                plot_and_cut_perimeter(perimeter, fixed_items, ax=axs[col, row])

        fig.tight_layout()
        # plt.show()
        plt.savefig('output/grid_test_apartments.svg')


def plot_and_cut_perimeter(perimeter=None, fixed_items=None, ax=None):
    """
    Test
    :param perimeter:
    :param fixed_items:
    :param ax:
    :return:
    """

    perimeter = perimeter or [(0, 0), (2, 0), (2, 2), (1, 2), (1, 1), (0, 1)]
    mesh = Mesh()
    mesh.from_boundary(perimeter)
    main_face = mesh.faces[0]

    for fixed_item in fixed_items:
        fixed_item_mesh = Mesh().from_boundary(fixed_item)
        main_face = main_face.insert_face(fixed_item_mesh.faces[0])

    for edge in mesh.boundary_edges():
        if edge.length > 50:
            edge.pair.cut_at_barycenter(0.5)  # 513 867 - 541 797
            if ax is None:
                mesh.plot(options=('fill', 'edges', 'half-edges', 'boundary-edges', 'vertices'))
                mesh.check()
                plt.show()

    mesh.plot(ax=ax, options=('fill', 'edges'))
    mesh.check()
    return mesh


def _get_infos():
    """
    Test
    :return:
    """
    input_files = [
        ("Antony_A22.json", "Antony_A22_setup.json"),
        ("Bussy_A001.json", "Bussy_A001_setup.json"),
        ("Bussy_B104.json", "Bussy_B104_setup.json"),
        ("Levallois_Parisot.json", "Levallois_Parisot_setup.json")
    ]
    input_folder = "resources/blueprints"
    output_folder = "output"
    cache_folder = "output/cache/grid_test"

    input_files_path = [tuple(input_folder + "/" + file for file in files) for files in input_files]

    logging.debug("input_files_path = %s", input_files_path)

    # retrieve data from json files
    infos_array = []
    for input_file in input_files_path:
        with open(input_file[0], "r") as floor_plan_file:
            input_floor_plan_dict = json.load(floor_plan_file)

        with open(input_file[1], "r") as setup_file:
            input_setup_dict = json.load(setup_file)

        # modify the data into the infos and settings class
        @DecoratorCache(['InputOptimizer.py'], cache_folder)
        def retrieve_input(_input_floor_plan_dict, _input_setup_dict,
                           _output_folder, save_output, save_cache, save_log):
            _settings = Input.AlgoSettings()
            _infos = Input.Infos(
                _input_floor_plan_dict, _input_setup_dict,
                output_folder, save_output, save_cache,
                save_log, _settings
            )
            return _settings, _infos

        settings, infos = retrieve_input(
            input_floor_plan_dict, input_setup_dict, output_folder,
            True, True, True
        )
        infos_array.append((infos, settings))

    return infos_array


def test_apartments():
    """
    Test
    :return:
    """
    infos = _get_infos()
    perimeters = [(get_perimeter(info[0]), get_fixed_items_perimeters(info[0])) for info in infos]
    _plot_perimeters(perimeters)

