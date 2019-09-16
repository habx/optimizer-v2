from typing import (
    TYPE_CHECKING,
    Dict
)

from libs.utils.geometry import (barycenter,
                                 distance_point_border,
                                 is_inside,
                                 move_point,
                                 ANGLE_EPSILON)

from libs.plan.category import LinearCategory, LINEAR_CATEGORIES

if TYPE_CHECKING:
    from libs.plan.plan import Space
    from libs.equipments.furniture import Furniture


def fit_along_walls(furniture: 'Furniture', space: 'Space', **kwargs) -> bool:
    """
    Move furniture to fit in space along walls. Avoid walls with windows, and prefer centers.
    :return: did success
    """

    avoid_windows = kwargs.get("avoid_windows", True)
    avoid_doors = kwargs.get("avoid_doors", False)

    space_perimeter = space.perimeter  # to compute it only once
    window_edges = [component.edge
                    for component in space.immutable_components()
                    if avoid_windows
                    and isinstance(component.category, LinearCategory)
                    and component.category.window_type]
    space_edges = [e for e in space.edges]
    doors_positions = [linear.edge.start.coords
                       for linear in space.plan.linears
                       if avoid_doors
                       and linear.category == LINEAR_CATEGORIES["door"]
                       and (linear.edge in space_edges
                            or linear.edge.pair in space_edges)]
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

        # bonus/malus
        bonus = 0
        if avoid_windows:
            for edge in window_edges:
                if edge in aligned_edges:
                    # at least one window on the line
                    bonus -= 100
                    break
        for i in range(-4, 5):
            x, y = barycenter(start_edge.start.coords, end_edge.end.coords, 0.5 + (0.1 * i))
            if avoid_doors:
                for position in doors_positions:
                    bonus += ((position[0] - x) ** 2 + (position[1] - y) ** 2) ** 0.5
            vect = start_edge.normal
            possibilities.append({
                "ref_point": (x, y),
                "ref_vect": vect,
                "score": perimeter_proportion * 100 - abs(i) + bonus
            })
        # prepare next loop
        aligned_edges = space.aligned_siblings(space.next_edge(end_edge))
        start_edge = aligned_edges[0]
    return _try_possibilities(furniture, space, possibilities)


def fit_in_corners(furniture: 'Furniture', space: 'Space') -> bool:
    """
    Move furniture to fit in space in one of convex corners.
    :return: did success
    """
    # init loop
    aligned_edges = space.aligned_siblings(space.edge)
    initial_edge = aligned_edges[0]
    start_edge = None
    possibilities = []
    # find all possibilities
    while start_edge is not initial_edge:
        start_edge = aligned_edges[0]
        end_edge = aligned_edges[-1]

        if 90 - ANGLE_EPSILON <= space.previous_angle(start_edge) <= 180 + ANGLE_EPSILON:
            possibilities.append({
                "ref_point": move_point(start_edge.start.coords, start_edge.unit_vector,
                                        furniture.category.width / 2),
                "ref_vect": start_edge.normal,
                "score": 0
            })
        if 90 - ANGLE_EPSILON <= space.next_angle(end_edge) <= 180 + ANGLE_EPSILON:
            possibilities.append({
                "ref_point": move_point(end_edge.end.coords, end_edge.unit_vector,
                                        - furniture.category.width / 2),
                "ref_vect": end_edge.normal,
                "score": 0
            })

        # prepare next loop
        aligned_edges = space.aligned_siblings(space.next_edge(end_edge))
        start_edge = aligned_edges[0]
    return _try_possibilities(furniture, space, possibilities)


def fit_in_center(furniture: 'Furniture', space: 'Space') -> bool:
    """
    Move furniture to fit in space, as far as possible from the walls
    :return: did success
    """
    possibilities = []
    directions = space.directions
    coords = space.boundary_polygon()
    xs, ys = zip(*coords)
    min_x, max_x, min_y, max_y = int(min(xs)), int(max(xs)), int(min(ys)), int(max(ys))
    for x in range(min_x, max_x, 30):
        for y in range(min_y, max_y, 30):
            if is_inside((x, y), coords):
                for direction in directions:
                    possibilities.append({
                        "ref_point": move_point((x, y), direction,
                                                -furniture.category.height / 2),
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

    # ONLY FOR TESTING PURPOSES

    import logging
    from typing import Dict, Tuple

    from libs.io import reader
    from libs.specification.specification import Specification
    from libs.plan.plan import Space, Plan
    from libs.space_planner.solution import reference_plan_solution
    from libs.equipments.furniture import GARNISHERS
    import time


    def load_plan(input_file: str = "001.json") -> Tuple['Plan', 'Specification']:
        """
        Load plan from cache
        """
        from libs.io.reader import DEFAULT_PLANS_OUTPUT_FOLDER

        folder = DEFAULT_PLANS_OUTPUT_FOLDER
        spec_file_name = input_file[:-5] + "_setup0"

        new_serialized_data = reader.get_plan_from_json(input_file)
        plan = Plan(input_file[:-5]).deserialize(new_serialized_data)
        spec_dict = reader.get_json_from_file(spec_file_name + ".json", folder)
        spec = reader.create_specification_from_data(spec_dict, "new")
        spec.plan = plan
        return plan, spec


    def main():
        logging.getLogger().setLevel(logging.INFO)
        for i in range(1, 65):
            name = str(i).zfill(3) + ".json"
            logging.info(name)
            try:
                plan, spec = load_plan(name)
                solution = reference_plan_solution(plan, spec)
            except Exception as e:
                logging.warning(f"{name} ignored: {str(e)}")
                continue
            start = time.process_time()
            GARNISHERS['default'].apply_to(solution)
            stop = time.process_time()
            logging.info(f"Garnisher time : {stop-start}")
            solution.spec.plan.plot(name=name)


    main()
