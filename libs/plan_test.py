# coding=utf-8

"""
Test module for plan module
"""
import json
import logging

from libs.plan import Plan
from libs.utils.decorator_cache import DecoratorCache
import libs.optimizer_input as json_input
import libs.logsetup as ls

ls.init()


def get_perimeter(infos):
    """
    Returns a vertices list of the perimeter points of an apartment
    :param infos:
    :return:
    """
    apartment = infos.input_floor_plan_dict['apartment']
    perimeter_walls = apartment['externalWalls']
    vertices = apartment['vertices']
    return [(vertices[i]['x'], vertices[i]['y']) for i in perimeter_walls]


def get_fixed_items_perimeters(infos):
    """
    Returns a list with the perimeter of each fixed items.
    NOTE: we are using the pandas dataframe because we do not want to recalculate
    the absolute geometry of each fixed items. As a general rule,
    it would be much better to read and store the geometry
    of each fixed items as list of vertices instead of the way it's done by using barycentric and
    width data. It would be faster and enable us any fixed item shape.
    :param infos:
    :return: list
    """
    fixed_items_polygons = infos.floor_plan.fixed_items['Polygon']
    fixed_items_category = infos.floor_plan.fixed_items['Type']
    output = []
    for i, polygon in enumerate(fixed_items_polygons):
        output.append((polygon.exterior.coords[1:], fixed_items_category[i]))

    return output


def get_infos():
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
    input_folder = "../resources/blueprints"
    output_folder = "../output"
    cache_folder = "../output/cache/grid_test"

    input_files_path = [tuple(input_folder + "/" + file for file in files)
                        for files in input_files]

    logging.debug("input_files_path = %s", input_files_path)

    # retrieve data from json files
    infos_array = []
    for input_file in input_files_path:
        with open(input_file[0]) as floor_plan_file:
            input_floor_plan_dict = json.load(floor_plan_file)

        with open(input_file[1]) as setup_file:
            input_setup_dict = json.load(setup_file)

        # modify the data into the infos and settings class
        @DecoratorCache(['InputOptimizer.py'], cache_folder)
        def retrieve_input(_input_floor_plan_dict, _input_setup_dict,
                           _output_folder, save_output=True, save_cache=True, save_log=True):
            """
            Retrieves data from json json_input
            :param _input_floor_plan_dict:
            :param _input_setup_dict:
            :param _output_folder:
            :param save_output:
            :param save_cache:
            :param save_log:
            :return:
            """
            _settings = json_input.AlgoSettings()
            _infos = json_input.Infos(_input_floor_plan_dict, _input_setup_dict, output_folder,
                                      save_output, save_cache, save_log, _settings)
            return _settings, _infos

        settings, infos = retrieve_input(input_floor_plan_dict, input_setup_dict, output_folder)
        infos_array.append((infos, settings))

    return infos_array


def test():
    """
    Test
    :return:
    """
    infos = get_infos()[3][0]  # first floor plan
    perimeter = get_perimeter(infos)

    plan = Plan().from_boundary(perimeter)
    empty_space = plan.empty_space

    fixed_items = get_fixed_items_perimeters(infos)

    for fixed_item in fixed_items:
        empty_space.add_fixed_space(*fixed_item)

    for edge in list(empty_space.edges):
        empty_space.cut_at_barycenter(edge)

    assert plan.check()


def test_add_duct_to_space():
    """
    Test
    :return:
    """

    perimeter = [(0, 0), (1000, 0), (1000, 1000), (0, 1000)]
    duct = [(200, 0), (400, 0), (400, 400), (200, 400)]

    # add border duct
    plan = Plan().from_boundary(perimeter)
    plan.empty_space.add_fixed_space(duct, 'duct')

    # add inside duct
    inside_duct = [(600, 200), (800, 200), (800, 400), (600, 400)]
    plan.empty_space.add_fixed_space(inside_duct, 'duct')

    # add touching duct
    touching_duct = [(0, 800), (200, 800), (200, 1000), (0, 1000)]
    plan.empty_space.add_fixed_space(touching_duct, 'duct')

    # add separating duct
    separating_duct = [(700, 800), (1000, 700), (1000, 800), (800, 1000), (700, 1000)]
    plan.empty_space.add_fixed_space(separating_duct, 'duct')

    # add single touching point
    point_duct = [(0, 600), (200, 500), (200, 700)]
    plan.empty_space.add_fixed_space(point_duct, 'duct')

    # add complex duct
    complex_duct = [(300, 1000), (300, 600), (600, 600), (600, 800), (500, 1000),
                    (450, 800), (400, 1000), (350, 1000)]
    plan.empty_space.add_fixed_space(complex_duct, 'duct')

    # add a face from another space
    duct_spaces = list(plan.get_spaces('duct'))
    print(duct_spaces)

    assert plan.check()


def test_add_face():
    """
    Test
    :return:
    """
    perimeter = [(0, 0), (1000, 0), (1000, 1000), (0, 1000)]

    plan = Plan().from_boundary(perimeter)

    # add complex face
    complex_face = [(700, 800), (1000, 700), (1000, 800), (800, 1000), (700, 1000)]
    plan.empty_space.face.insert_face_from_boundary(complex_face)

    face_to_remove = list(plan.empty_space.faces)[1]
    plan.empty_space.remove_face(face_to_remove)

    plan.plot()

    plan.empty_space.add_face(face_to_remove)
    plan.plot()

    assert plan.check()
