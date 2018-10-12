# coding=utf-8
"""
Grid Module
We define a grid as a mesh with specific functionalities

Ideas :
- a face should have a type (space, object, reserved_space) and a nature from type:
"""
import math
from mesh import Mesh
import matplotlib.pyplot as plt


class Plan:
    """
    Main class containing the floor plan of the appartement
    • mesh
    • spaces
    • walls
    • objects
    • input
    """


class Transformation:
    """
    Transformation class
    Describes a transformation of a mesh
    A transformation queries the edges of the mesh and modify them as prescribed
    • a query
    • a transformation
    """

    pass


class Factory:
    """
    Grid Factory Class
    Creates a mesh according to transformations, boundaries and fixed-items input
    and specific rules
    """
    pass


class Query:
    """
    Queries a mesh and returns a generator with the required edges
    """


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
    NOTE: we are using the dataframe because we do not want to recalculate the absolute geometry
    of each fixed items. As a general rule, it would be much better to read and store the geometry
    of each fixed items as list of vertices instead of the way it's done by using barycentric and
    width data. It would be faster and enable us any fixed item shape.
    :param infos:
    :return: list
    """
    fixed_items_polygons = infos.floor_plan.fixed_items['Polygon']
    output = [polygon.exterior.coords[1:] for polygon in fixed_items_polygons]
    return output


if __name__ == "__main__":
    # for testing purposes
    # TODO : replace this by cleaner unit tests
    # (@florent idea on the best library and files structure for this ?)
    from DecoratorCache import DecoratorCache
    import InputOptimizer as Input
    import json


    def plot_perimeters(perimeters):
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
            plt.show()

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
        input_folder = "Plans/Input"
        output_folder = "Plans/Output"
        cache_folder = "Cache"

        input_files_path = [tuple(input_folder + "/" + file for file in files) for files in input_files]

        print(input_files_path)

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
                _infos = Input.Infos(_input_floor_plan_dict, _input_setup_dict, output_folder, save_output, save_cache,
                                     save_log, _settings)
                return _settings, _infos

            settings, infos = retrieve_input(input_floor_plan_dict, input_setup_dict, output_folder, True, True, True)
            infos_array.append((infos, settings))

        return infos_array

    def test_apartments():
        """
        Test
        :return:
        """
        infos = get_infos()
        perimeters = [(get_perimeter(info[0]), get_fixed_items_perimeters(info[0])) for info in infos]
        plot_perimeters(perimeters)
        return

    test_apartments()