from typing import (
    TYPE_CHECKING,
    Dict
)
from libs.plan.category import LinearCategory
from libs.utils.geometry import (barycenter,
                                 distance_point_border,
                                 is_inside,
                                 move_point)

if TYPE_CHECKING:
    from libs.plan.plan import Space
    from libs.plan.furniture import Furniture


def fit_along_walls(furniture: 'Furniture', space: 'Space') -> bool:
    """
    Move furniture to fit in space, designed especially for bed.
    :return:
    """
    window_edges = [component.edge
                    for component in space.immutable_components()
                    if isinstance(component.category, LinearCategory)
                    and component.category.window_type]
    space_perimeter = space.perimeter

    # init loop
    aligned_edges = space.aligned_siblings(space.edge)
    initial_edge = aligned_edges[0]
    start_edge = None
    possibilities = []
    # find all possibilities
    while start_edge is not initial_edge:
        start_edge = aligned_edges[0]
        end_edge = aligned_edges[-1]

        # compute perimeter percentage
        length = start_edge.start.distance_to(end_edge.end)
        perimeter_proportion = length / space_perimeter

        # score each line
        no_window = True
        for edge in window_edges:
            if edge in aligned_edges:
                # at least one window on the line
                no_window = False
                break
        possibilities.append({
            "ref_point": barycenter(start_edge.start.coords, end_edge.end.coords, 0.5),
            "ref_vect": start_edge.normal,
            "score": perimeter_proportion * 100 + no_window * 100
        })
        for i in range(1, 5):
            possibilities.append({
                "ref_point": barycenter(start_edge.start.coords, end_edge.end.coords,
                                        0.5 - (0.1 * i)),
                "ref_vect": start_edge.normal,
                "score": perimeter_proportion * 100 - i + no_window * 100
            })
            possibilities.append({
                "ref_point": barycenter(start_edge.start.coords, end_edge.end.coords,
                                        0.5 + (0.1 * i)),
                "ref_vect": start_edge.normal,
                "score": perimeter_proportion * 100 - i + no_window * 100
            })

        # prepare next loop
        aligned_edges = space.aligned_siblings(space.next_edge(end_edge))
        start_edge = aligned_edges[0]
    return _try_possibilities(furniture, space, possibilities)


def fit_in_corners(furniture: 'Furniture', space: 'Space') -> bool:
    possibilities = []
    return _try_possibilities(furniture, space, possibilities)


def fit_in_center(furniture: 'Furniture', space: 'Space') -> bool:
    possibilities = []
    directions = space.directions
    coords = space.boundary_polygon()
    xs, ys = zip(*coords)
    min_x, max_x, min_y, max_y = int(min(xs)), int(max(xs)), int(min(ys)), int(max(ys))
    for x in range(min_x, max_x, 10):
        for y in range(min_y, max_y, 10):
            if is_inside((x,y), coords):
                for direction in directions:
                    possibilities.append({
                        "ref_point": move_point((x, y),furniture.category.vect_to_centroid, -1),
                        "ref_vect": direction,
                        "score": distance_point_border((x, y), coords)
                    })
    return _try_possibilities(furniture, space, possibilities)


def _try_possibilities(furniture: 'Furniture', space: 'Space', possibilities: [Dict]) -> bool:
    """
    Try each placement possibility, starting with the best scored, until it fits.
    :param furniture:
    :param space:
    :param possibilities: dicts with keys: "ref_vect", "ref_point" and "score"
    :return: has been fitted or not
    """
    # sort possibilites
    possibilities.sort(key=lambda p: p["score"], reverse=True)

    # try all possibilites
    for possibility in possibilities:
        furniture.ref_vect = possibility["ref_vect"]
        furniture.ref_point = possibility["ref_point"]
        if furniture.check_validity(space):
            return True

    return False

if __name__ == '__main__':
    import logging
    from typing import Dict, Tuple

    from libs.io import reader
    from libs.modelers.grid import GRIDS
    from libs.modelers.seed import SEEDERS
    from libs.specification.specification import Specification
    from libs.plan.plan import Space, Plan
    from libs.modelers.corridor import Corridor, CORRIDOR_BUILDING_RULES
    from libs.refiner.refiner import REFINERS
    from libs.space_planner.space_planner import SPACE_PLANNERS
    from libs.equipments.doors import place_doors
    from libs.space_planner.solution import spec_adaptation

    BLACKLIST = ()

    def compute_or_load_plan(input_file: str = "001.json") -> Tuple['Plan', 'Specification']:
        """
        Try to load plan from cache
        :param input_file:
        :return:
        """

        import libs.io.reader as reader
        import libs.io.writer as writer
        from libs.space_planner.space_planner import SPACE_PLANNERS
        from libs.io.reader import DEFAULT_PLANS_OUTPUT_FOLDER

        folder = DEFAULT_PLANS_OUTPUT_FOLDER

        spec_file_name = input_file[:-5] + "_setup0"
        plan_file_name = input_file

        try:
            new_serialized_data = reader.get_plan_from_json(input_file)
        except FileNotFoundError:
            # reading lot
            plan = reader.create_plan_from_file(input_file)
            # grid
            GRIDS["002"].apply_to(plan)
            # seeder
            SEEDERS["directional_seeder"].apply_to(plan)
            # reading setup
            spec = reader.create_specification_from_file(input_file[:-5] + "_setup0" + ".json")
            spec.plan = plan
            spec.plan.remove_null_spaces()
            # space planner
            space_planner = SPACE_PLANNERS["standard_space_planner"]
            best_solutions = space_planner.apply_to(spec, 3)
            ###
            if best_solutions:
                solution = best_solutions[0]
                # corridor
                corridor_building_rule = CORRIDOR_BUILDING_RULES["no_cut"]
                Corridor(corridor_rules=corridor_building_rule["corridor_rules"],
                         growth_method=corridor_building_rule["growth_method"])\
                    .apply_to(solution, space_planner.solutions_collector.spec_with_circulation)
                spec_adaptation(solution, space_planner.solutions_collector)
                # refiner
                REFINERS['space_nsga'].apply_to(solution, {"ngen": 80, "mu": 80, "cxpb": 0.9, "max_tries": 10, "elite": 0.1, "processes": 1})
                spec_adaptation(solution, space_planner.solutions_collector)
                # doors
                place_doors(solution.spec.plan)
                # save
                writer.save_plan_as_json(solution.spec.plan.serialize(), plan_file_name)
                writer.save_as_json(solution.spec.serialize(), folder, spec_file_name + ".json")
                return plan, solution.spec
            else:
                logging.info("No solution for this plan")
        else:
            plan = Plan(input_file[:-5]).deserialize(new_serialized_data)
            spec_dict = reader.get_json_from_file(spec_file_name + ".json",
                                                  folder)
            spec = reader.create_specification_from_data(spec_dict, "new")
            spec.plan = plan
            return plan, spec

    def main():
        logging.getLogger().setLevel(logging.INFO)
        for i in range(1,65):
            name = str(i).zfill(3) + ".json"
            if name not in BLACKLIST:
                logging.info(name)
                try:
                    plan, spec = compute_or_load_plan(name)
                except Exception as e:
                    print(name, e)
                else:
                    if plan is None or spec is None:
                        logging.info("No solution")
                        continue

    main()