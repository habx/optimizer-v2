import logging
import math
import os
from typing import Dict

import matplotlib.pyplot as plt
import shapely as sp
from shapely.geometry import Point, LineString

from libs.io.plot import plot_save
from libs.plan.category import SPACE_CATEGORIES
from libs.plan.plan import Plan, Face, Linear
from libs.space_planner.circulation import Circulator, CostRules
from libs.space_planner.constraints_manager import WINDOW_ROOMS
from libs.specification.size import Size
from libs.specification.specification import Specification, Item
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from libs.space_planner.solution import Solution

SQM = 10000
CORRIDOR_SIZE = 120

"""
Scoring functions
"""
def corner_scoring(solution: 'Solution') -> float:
    """
    :param solution
    :return: score : float
    """
    nbr_corner = {
        "bedroom": 6,
        "living": 8,
        "livingKitchen": 8,
        "dining": 8,
        "toilet": 4,
        "bathroom": 6,
        "wardrobe": 6,
        "study": 6,
        "kitchen": 6,
        "laundry": 6,
        "entrance": 6,
        "circulation": 6,
    }
    corner_score = 0
    has_holes = 0
    for space, item in solution.space_item.items():
        space_score = 100 - max(space.number_of_corners() - nbr_corner[item.category.name], 0) * 25
        corner_score += space_score
        if space.has_holes:
            has_holes += 1
    corner_score = round(corner_score / len(solution.space_item), 2) - has_holes * 25
    logging.debug("Corner score : %f", corner_score)
    return corner_score


def bounding_box_scoring(solution: 'Solution') -> float:
    """
    Computes a score as the difference between the area of a space and its bounding box
    :param solution
    :return: score : float
    """
    ratios = {
        "bedroom": 2.0,
        "wc": 2.0,
        "default": 1.0
    }
    score = {}
    bounding_box_score = 0
    for space, item in solution.space_item.items():

        if space.category.name == "circulation":
            score[space] = 100
            bounding_box_score += 100
            continue
        else:
            area = space.cached_area()
            vector = space.directions[0]
            box = space.bounding_box(vector)
            difference = (box[0] * box[1] - area)
            space_score = 100.0 - (difference * 100 / area) * \
                          ratios.get(space.category.name, ratios["default"])
            score[space] = space_score
            bounding_box_score += space_score

    bounding_box_score = bounding_box_score / len(solution.space_item)
    logging.debug("bounding_box_score %f", bounding_box_score)
    return bounding_box_score


def area_scoring(solution: 'Solution') -> float:
    """
    Area score
    :param solution
    :return: score : float
    """
    good_overflow_categories = ["living", "livingKitchen", "dining"]

    area_score = 0
    area_penalty = 0
    nbr_rooms = 0
    circulation_area = 0
    circulation_max_area = sum(item.max_size.area for item in solution.spec.items
                               if item.category.name in ["entrance", "circulation"])
    for space, item in solution.space_item.items():
        if space.category.name not in ["entrance", "circulation"]:
            nbr_rooms += 1
            # Min < SpaceArea < Max
            if item.min_size.area <= space.cached_area() <= item.max_size.area:
                item_area_score = 100
            # good overflow
            elif (item.max_size.area < space.cached_area() and
                  space.category.name in good_overflow_categories):
                item_area_score = 100
            # overflow
            else:
                item_area_score = max(100 - (abs(item.required_area - space.cached_area()) * 200 /
                                             item.required_area), 0)
                if space.category.name == "toilet":
                    if space.cached_area() < 12000:
                        area_penalty += 5
                    elif space.cached_area() > item.max_size.area:
                        area_penalty += 3
                elif space.category.name == "bathroom":
                    if space.cached_area() < 23000:
                        area_penalty += 3
                elif space.category.name == "bedroom":
                    if space.cached_area() < 90000:
                        area_penalty += 5
                elif space.category.name == "laundry":
                    if space.cached_area() < 20000:
                        area_penalty += 5
            # Area score
            area_score += item_area_score

        else:
            if space.category.name == "entrance":
                circulation_area += space.cached_area()
                if space.cached_area() < 15000:
                    area_penalty += 2
            elif space.category.name == "circulation":
                circulation_area += space.cached_area()

    if circulation_area > circulation_max_area:
        area_penalty += 2

    area_score = round(area_score / nbr_rooms, 2) - area_penalty * 10
    logging.debug("Area score : %f", area_score)

    return area_score


def position_scoring(solution: 'Solution') -> float:
    """
    Position room score :
    - one of the toilets must be near the entrance
    - bedrooms must be near a bathroom
    - toilets and bathrooms must be accessible from corridors or the entrance
    - the living must be near the entrance
    :param solution
    :return: score : float
    """
    position_score = 0
    front_door = solution.spec.plan.front_door()
    toilet_score = 0
    nbr_room_position_score = 0
    entrance = [space for space, item in solution.space_item.items() if
                item.category.name == "entrance"]
    circulation_spaces = [space for space, item in solution.space_item.items() if
                          item.category.circulation]
    for space, item in solution.space_item.items():
        memo = 0
        item_position_score = None
        if item.category.name == "toilet":
            item_position_score = 0
            # distance from the entrance
            if solution.spec.plan.front_door().floor.level == space.floor.level:
                if entrance != [] and entrance[0].adjacent_to(space):
                    item_position_score = 100
                    toilet_score = 100
                else:
                    # distance from the entrance
                    plan_area = solution.spec.plan.area
                    criteria = plan_area ** 0.5
                    distance_toilet_fd = space.distance_to_linear(front_door, "min")
                    if distance_toilet_fd < criteria:
                        score = (criteria - distance_toilet_fd) * 100 / criteria
                        if score > toilet_score:
                            memo = toilet_score
                            toilet_score = score
                            item_position_score = score
        elif item.category.name == "bathroom":
            item_position_score = 100
            # non adjacent bathroom / bathroom
            for space_test, item_test in solution.space_item.items():
                if (item_test.category.name == "bathroom" and
                        space_test.floor == space.floor):
                    if space.adjacent_to(space_test):
                        item_position_score = 0
                        break
        elif item.category.name == "bedroom":
            nbr_room_position_score += 1
            item_position_score = 0
            # distance from a bedroom / bathroom
            for space_test, item_test in solution.space_item.items():
                if (item_test.category.name in ["bathroom", "circulation", "entrance"] and
                        space_test.floor == space.floor):
                    if space.adjacent_to(space_test):
                        item_position_score = 100
                        break
        if item.category.name == "toilet" or item.category.name == "bathroom":
            nbr_room_position_score += 1
            # could be private
            private = False
            for circulation_space in circulation_spaces:
                if circulation_space.adjacent_to(space):
                    private = True
                    break
            if not private:
                item_position_score = max(item_position_score - 50, 0)
        elif item.category.name == "living" or item.category.name == "livingKitchen":
            nbr_room_position_score += 1
            item_position_score = 0
            if "frontDoor" in space.components_category_associated():
                item_position_score = 100
            else:
                # distance from the entrance
                if ((entrance != [] and entrance[0].adjacent_to(space))
                        or space.distance_to_linear(front_door, "min") < CORRIDOR_SIZE * 2):
                    item_position_score = 100

        if item_position_score is not None:
            position_score += item_position_score - memo

    position_score = position_score / nbr_room_position_score

    logging.debug("position_score %f", position_score)

    return position_score


def windows_scoring(solution: 'Solution') -> float:
    """
    Windows area ratio constraint : NF HABITAT HQE Good ordering windows area bonus
    :param solution
    :return: score : float
    """
    nbr_window_room = 0
    windows_score = 100
    for space, item in solution.space_item.items():
        if item.category.name in WINDOW_ROOMS and [comp for comp in space.immutable_components() if
                                                   comp.category.name in ["window", "doorWindow"]]:
            nbr_window_room += 1
            if item.category.name in ["living", "livingKitchen", "dining"]:
                ratio = 18
            elif item.category.name in ["bedroom"] and len(item.opens_on) == 0:
                ratio = 15
            elif item.category.name in ["kitchen", "study", "bathroom"] and len(item.opens_on) == 0:
                ratio = 10
            else:
                ratio = 0

            windows_area = 0
            for component in space.immutable_components():
                if component.category.name == "window":
                    windows_area += component.length * 100
                elif component.category.name == "doorWindow":
                    windows_area += component.length * 200
            if (space.area * ratio) <= (windows_area * 100):
                space_windows_score = 100
            else:
                space_windows_score = 0
            windows_score = min(space_windows_score, windows_score)

    logging.debug("Windows score : %f", windows_score)
    return windows_score


def minimal_dimensions_scoring(solution: 'Solution') -> float:
    """
    minimal dimension for each item category
    :param solution
    :return: score : float
    """
    min_length = {
        "bedroom": 250,  # non PMR
        "living": 300,
        "livingKitchen": 340,
        "dining": 280,
        "toilet": 95,  # non PMR
        "bathroom": 150,  # non PMR
        "dressing": 155,
        "study": 210,
        "kitchen": 180,  # non PMR
        "laundry": 95,
        "entrance": 120,  # PRM
        "circulation": 100,  # non PRM
    }
    max_length = {  # racine carrée de la surface min
        "bedroom": 300,
        "living": 380,
        "livingKitchen": 440,
        "dining": 340,
        "toilet": 130,
        "bathroom": 180,
        "dressing": 150,
        "study": 300,
        "kitchen": 220,
        "laundry": 150,
        "entrance": 220,
        "circulation": None,
    }

    minimal_dimensions_score = 100
    for space, item in solution.space_item.items():
        space_minimal_dimensions_score = 100
        vector = space.directions[0]
        box = space.bounding_box(vector)
        if ((max_length.get(item.category.name) and max(box) < max_length.get(
                item.category.name)) or
                (min_length.get(item.category.name) and min(box) < min_length.get(
                    item.category.name))):
            space_minimal_dimensions_score = 0
        if space_minimal_dimensions_score == 0:
            minimal_dimensions_score = max(minimal_dimensions_score - 25, 0)

    logging.debug("minimal_dimensions_score : %f", minimal_dimensions_score)
    return minimal_dimensions_score


def sun_ray(plan: 'Plan') -> Dict['Face', Dict['Linear', 'LineString']]:
    """
    sun ray from window to each face
    :param plan
    :return: sun_ray dict
    """
    sun_ray_dict = {}
    for floor in plan.floors.values():
        windows_list = [lin for lin in plan.linears
                        if lin.category.window_type and lin.floor == floor]
        for face in floor.mesh.faces:
            face_dict = {}
            for window in windows_list:
                ray = sp.geometry.LineString([[window.as_sp.centroid.xy[0][0],
                                               window.as_sp.centroid.xy[1][0]],
                                              [face.as_sp.centroid.xy[0][0],
                                               face.as_sp.centroid.xy[1][0]]])
                face_dict[window] = ray
            sun_ray_dict[face] = face_dict
    return sun_ray_dict


def luminosity_scoring(solution: 'Solution') -> float:
    """
    Natural luminosity surface ratio
    :param solution
    :return: score : float
    """
    distance_max = 600
    face_luminosity = {}
    sun_ray_dict = sun_ray(solution.spec.plan)
    rooms_faces = 0
    for space, item in solution.space_item.items():
        windows_list = [lin for lin in solution.spec.plan.linears
                        if (lin.category.window_type and lin in space.immutable_components()
                            and lin.floor == space.floor)]
        if windows_list:
            for face in space.faces:
                face_luminosity[face] = 0
                rooms_faces += 1
                for lin in windows_list:
                    try:
                        ray = sun_ray_dict[face][lin]
                    except ValueError:
                        continue

                    if ray.length <= distance_max:
                        inside_intersection = ray.intersection(space.as_sp)
                        if round(inside_intersection.length) == round(ray.length):
                            for test_room in solution.spec.plan.mutable_spaces():
                                if test_room != space:
                                    environnement_inside_intersection = ray.intersection(
                                        test_room.as_sp)
                                    if environnement_inside_intersection:
                                        break
                            face_luminosity[face] = 100
                            break

        else:
            for face in space.faces:
                rooms_faces += 1
                face_luminosity[face] = 0

    luminosity_score = 0
    area_sum = 0
    for space, item in solution.space_item.items():
        if item.category.name in WINDOW_ROOMS and [comp for comp in space.immutable_components() if
                                                   comp.category.name in ["window", "doorWindow"]]:
            for face in space.faces:
                if face_luminosity[face] == 0:
                    area_sum += 5 * face.area
                else:
                    luminosity_score += face.area * face_luminosity[face]
                    area_sum += face.area
        else:
            for face in space.faces:
                luminosity_score += face.area * face_luminosity[face]
                area_sum += face.area
    luminosity_score = luminosity_score / area_sum
    luminosity_plot(solution, face_luminosity)

    return luminosity_score


def night_and_day_scoring(solution: 'Solution') -> float:
    """
    Night and day score
    day / night distribution of rooms
    :param solution
    :return: score : float
    """
    first_level = solution.spec.plan.first_level
    day_list = ["living", "kitchen", "livingKitchen", "dining"]

    night_list = ["bedroom", "bathroom"]

    day_polygon_list = []
    night_polygon_list = []
    for i_floor in range(solution.spec.plan.floor_count):
        day_polygon_list.append(None)
        night_polygon_list.append(None)

    for space, item in solution.space_item.items():
        level = space.floor.level
        # Day
        if (item.category.name in day_list
                or (item.category.name == "toilet"
                    and space == solution.get_rooms("toilet")[0])):
            if day_polygon_list[level - first_level] is None:
                day_polygon_list[level - first_level] = space.as_sp
            else:
                day_polygon_list[level - first_level] = day_polygon_list[
                    level - first_level].union(space.as_sp)

        # Night
        elif (item.category.name in night_list or
              (item.category.name == "toilet" and
               space != solution.get_rooms("toilet")[0])):
            if night_polygon_list[level - first_level] is None:
                night_polygon_list[level - first_level] = space.as_sp
            else:
                night_polygon_list[level - first_level] = night_polygon_list[
                    level - first_level].union(space.as_sp)

    number_of_day_level = 0
    number_of_night_level = 0
    day_polygon = None
    night_polygon = None
    for i_floor in range(solution.spec.plan.floor_count):
        if day_polygon_list[i_floor] is not None:
            number_of_day_level += 1
            day_polygon = day_polygon_list[i_floor]
        if night_polygon_list[i_floor] is not None:
            number_of_night_level += 1
            night_polygon = night_polygon_list[i_floor]

    # groups of rooms
    groups_score = 100
    if number_of_day_level > 1:
        groups_score -= 50
    elif solution.spec.plan.floor_count < 2 and day_polygon and day_polygon.geom_type != "Polygon":
        if [item for item in solution.space_item.values() if item.category.name == "entrance"]:
            day_polygon = day_polygon.union(
                solution.get_rooms("entrance")[0].as_sp.buffer(1))
        if day_polygon.geom_type != "Polygon":
            groups_score -= 50

    if number_of_night_level > 1:
        if solution.spec.typology <= 2 or solution.spec.number_of_items < 6:
            groups_score -= 50
        else:
            groups_score -= 25
    if solution.spec.plan.floor_count < 2 and night_polygon and night_polygon.geom_type != "Polygon":
        if [item for item in solution.space_item.values() if item.category.name == "entrance"]:
            night_polygon_with_entrance = night_polygon.union(
                solution.get_rooms("entrance")[0].as_sp.buffer(CORRIDOR_SIZE))
        else:
            night_polygon_with_entrance = night_polygon
        if night_polygon_with_entrance.geom_type != "Polygon":
            if ((len(night_polygon) > 2 and len(night_polygon_with_entrance) > 2)
                    or (solution.spec.typology <= 2
                        or solution.spec.number_of_items < 6)):
                groups_score -= 50
            else:
                groups_score -= 25

    logging.debug("Solution %i: Night and day score : %i", solution.id, groups_score)
    return groups_score


def something_inside_scoring(solution: 'Solution') -> float:
    """
    Something inside score
    duct or bearing wall or pillar or isolated room must not be inside a room
    :param solution
    :return: score : float
    """
    something_inside_score = 100
    for space, item in solution.space_item.items():
        #  duct or pillar or small bearing wall
        if space.has_holes:
            logging.debug("Solution %i: Something Inside score : %f, room : %s, has_holes",
                          solution.id, 0, item.category.name)
            return 0
        #  isolated room
        list_of_non_concerned_room = ["entrance", "circulation", "wardrobe", "study", "laundry",
                                      "misc"]
        convex_hull = space.as_sp.convex_hull
        for i_space, i_item in solution.space_item.items():
            if (i_item != item and
                    i_item.category.name not in list_of_non_concerned_room and space.floor ==
                    i_space.floor):
                if (i_space.as_sp.is_valid and convex_hull.is_valid and
                        (round((convex_hull.intersection(i_space.as_sp)).area)
                         == round(i_space.as_sp.area))):
                    logging.debug(
                        "Solution %i: Something Inside score : %f, room : %s - isolated room",
                        solution.id, 0, i_item.category.name)
                    return 0
                elif (i_space.as_sp.is_valid and convex_hull.is_valid and
                      (convex_hull.intersection(i_space.as_sp)).area > (
                              space.cached_area() / 8)):
                    # Check i_item adjacency
                    other_room_adj = False
                    for j_space, j_item in solution.space_item.items():
                        if j_item != i_item and j_item != item:
                            if i_space.adjacent_to(j_space):
                                other_room_adj = True
                                break
                    if not other_room_adj:
                        logging.debug("Solution %i: Something Inside score : %f, room : %s, "
                                      "isolated room", solution.id, something_inside_score,
                                      item.category.name)
                        return 0

    logging.debug("Solution %i: Something Inside score : %f", solution.id, something_inside_score)
    return something_inside_score


def good_size_bonus(solution: 'Solution') -> float:
    """
    Good ordering items size bonus
    :param solution
    :return: score : float
    """
    for space1, item1 in solution.space_item.items():
        for space2, item2 in solution.space_item.items():
            if (item1 != item2 and item1.category.name not in ["entrance", "circulation"] and
                    item2.category.name not in ["entrance", "circulation"]):
                if (item1.required_area < item2.required_area and
                        space1.cached_area() > space2.cached_area()):
                    logging.debug("Solution %i: Size bonus : %i", solution.id, 0)
                    return 0
    logging.debug("Solution %i: Size bonus : %i", solution.id, 10)
    return 10


def windows_good_distribution_bonus(solution: 'Solution') -> float:
    """
    Good ordering windows area bonus
    :param solution
    :return: score : float
    """
    item_windows_area = {}
    for space, item in solution.space_item.items():
        windows_area = 0
        for component in space.immutable_components():
            if component.category.name == "window":
                windows_area += component.length * 100
            elif component.category.name == "doorWindow":
                windows_area += component.length * 200
        item_windows_area[item.id] = windows_area

    for item1 in solution.spec.items:
        for item2 in solution.spec.items:
            if (item1.required_area < item2.required_area
                    and item1.category.name in WINDOW_ROOMS
                    and item2.category.name in WINDOW_ROOMS):
                if item_windows_area[item1.id] > item_windows_area[item2.id]:
                    logging.debug("Solution %i: Windows bonus : %i", solution.id, 0)
                    return 0
    logging.debug("Solution %i: Windows bonus : %i", solution.id, 10)
    return 10


def entrance_bonus(solution: 'Solution') -> float:
    """
    Entrance bonus
    :param solution
    :return: score : float
    """
    if (solution.spec.typology > 2
            and [item for item in solution.space_item.values() if
                 item.category.name == "entrance"]):
        return 10
    elif (solution.spec.typology <= 2
          and [item for item in solution.space_item.values() if item.category.name == "entrance"]):
        return -10
    return 0


def externals_spaces_bonus(solution: 'Solution') -> float:
    """
    Good ordering externals spaces size bonus
    :param solution
    :return: score : float
    """
    for space1, item1 in solution.space_item.items():
        for space2, item2 in solution.space_item.items():
            if (item1 != item2 and space1.connected_spaces()
                    and space2.connected_spaces()):
                item1_ext_spaces_area = sum([ext_space.cached_area()
                                             for ext_space in
                                             space1.connected_spaces()
                                             if ext_space.category.external])
                item2_ext_spaces_area = sum([ext_space.cached_area()
                                             for ext_space in
                                             space2.connected_spaces()
                                             if ext_space.category.external])

                if (item1.required_area < item2.required_area and
                        item1_ext_spaces_area > item2_ext_spaces_area):
                    logging.debug("Solution %i: External spaces bonus : %i", solution.id, 0)
                    return 0
    logging.debug("Solution %i: External spaces : %i", solution.id, 10)
    return 10


def circulation_penalty(solution: 'Solution') -> float:
    """
    Circulation penalty
    :param solution
    :return: score : float
    """
    circulator = Circulator(plan=solution.spec.plan, spec=solution.spec, cost_rules=CostRules)
    circulator.connect()
    cost = circulator.cost
    penalty = 0

    # NOTE : what a weird thing to do (can't we just get the cost right from the start ?)
    if cost > CostRules.water_room_less_than_two_ducts.value:
        penalty += 100
    elif cost > CostRules.window_room_default.value:
        penalty += 50
    elif cost > CostRules.water_room_default.value:
        penalty += 30
    elif cost - (solution.spec.typology - 1) * 300 > 0:
        penalty += 5
    logging.debug("Solution %i: circulation penalty : %i", solution.id, penalty)

    return penalty


def space_planning_scoring(solution: 'Solution') -> float:
    """
    Space planning scoring
    compilation of different scores
    :param solution
    :return: score : float
    """
    solution_score = (area_scoring(solution)
                      + position_scoring(solution)
                      + corner_scoring(solution)
                      + night_and_day_scoring(solution)) / 4
    solution_score = (solution_score + entrance_bonus(solution))
    logging.debug("Solution %i: Final score : %f", solution.id, solution_score)

    return solution_score

def final_scoring(solution: 'Solution') -> [float]:
    """
    Final scoring
    compilation of different scores
    :param solution
    :return: score : float
    :return: score_components = [float]
    """
    score_components = dict()
    score_components["area"] = area_scoring(solution)
    score_components["corner"] = corner_scoring(solution)
    score_components["bounding_box"] = bounding_box_scoring(solution)
    score_components["position"] = position_scoring(solution)
    score_components["luminosity"] = luminosity_scoring(solution)
    score_components["minimal_dimensions"] = minimal_dimensions_scoring(solution)
    plan_score = sum(score_components.values()) / len(score_components)

    return plan_score, score_components

"""
Scoring plot tools
"""

def luminosity_plot(solution: 'Solution', face_luminosity: Dict['Face', int]):
    """
    luminosity plan plot
    :param solution
    :param face_luminosity
    """

    ax = solution.spec.plan.plot(show=False, save=False)

    number_of_floors = solution.spec.plan.floor_count

    for face in face_luminosity:
        level = solution.spec.plan.get_space_of_face(face).floor.level
        _ax = ax[level] if number_of_floors > 1 else ax
        if face_luminosity[face]:
            _ax.plot([face.as_sp.centroid.xy[0][0]], [face.as_sp.centroid.xy[1][0]], marker='*',
                     markersize=3, color="yellow")

    plot_save(True, False)


def radar_chart(plan_score: float, score_components: Dict[str, float], solution_id: int,
                chart_name: str = "FinalScore") -> None:
    """
    Final score radar chart
    :param plan_score
    :param score_components
    :param solution_id
    :param chart_name
    """
    # https://python-graph-gallery.com/391-radar-chart-with-several-individuals/

    # ------- PART 1: Create background
    # number of variable
    nbr_score_comp = len(score_components)
    categories = list(score_components.keys())

    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(nbr_score_comp) * 2 * math.pi for n in range(nbr_score_comp)]
    angles += angles[:1]

    # Initialise the spider plot
    ax = plt.subplot(111, polar=True)

    # If you want the first axis to be on top:
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)

    # Draw one axe per variable + add labels labels yet
    plt.xticks(angles[:-1], categories)

    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([25, 50, 75], ["25", "50", "75"], color="grey", size=7)
    plt.ylim(0, 100)

    # ------- PART 2: Add plots
    values = list(score_components.values())
    values += values[:1]
    ax.plot(angles, values, linewidth=1, linestyle='solid',
            label="Sol" + str(solution_id) + "_Score" + str(int(plan_score)))
    ax.fill(angles, values, alpha=0.1)

    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    # Add a title
    title = chart_name
    plt.title(title, size=11, color="black", y=1.1)

    link_save = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             title + '_radar_chart' + '.svg')
    plt.savefig(link_save)


if __name__ == '__main__':

    import libs.io.reader as reader
    from libs.modelers.grid import GRIDS
    from libs.modelers.seed import SEEDERS
    from libs.space_planner.space_planner import SPACE_PLANNERS


    def spec_adaptation(spec: 'Specification', archi_plan: 'Plan') -> 'Specification':
        """
        change reader specification :
        adding entrance
        readjustment of circulation area
        living + kitchen : opensOn --> livingKitchen
        area convergence
        :param spec
        :param archi_plan
        :return: 'Specification'
        """
        scoring_spec = Specification('Scoring_Specification', archi_plan)

        # entrance
        size_min = Size(area=2 * SQM)
        size_max = Size(area=5 * SQM)
        new_item = Item(SPACE_CATEGORIES["entrance"], "s", size_min, size_max)
        scoring_spec.add_item(new_item)
        living_kitchen = False

        for item in spec.items:
            if item.category.name == "circulation":
                if spec.typology > 2:
                    size_min = Size(area=(max(0, (spec.typology - 2) * 3 * SQM - 1 * SQM)))
                    size_max = Size(area=(max(0, (spec.typology - 2) * 3 * SQM + 1 * SQM)))
                    new_item = Item(SPACE_CATEGORIES["circulation"], item.variant, size_min,
                                    size_max)
                    scoring_spec.add_item(new_item)
            elif ((item.category.name != "living" or "kitchen" not in item.opens_on) and
                  (item.category.name != "kitchen" or len(item.opens_on) == 0)):
                scoring_spec.add_item(item)
            elif item.category.name == "living" and "kitchen" in item.opens_on:
                kitchens = spec.category_items("kitchen")
                for kitchen_item in kitchens:
                    if "living" in kitchen_item.opens_on:
                        size_min = Size(area=(kitchen_item.min_size.area + item.min_size.area))
                        size_max = Size(area=(kitchen_item.max_size.area + item.max_size.area))
                        # opens_on = item.opens_on.remove("kitchen")
                        new_item = Item(SPACE_CATEGORIES["livingKitchen"], item.variant, size_min,
                                        size_max, item.opens_on, item.linked_to)
                        scoring_spec.add_item(new_item)
                        living_kitchen = True

        # area
        invariant_categories = ["entrance", "wc", "bathroom", "laundry", "wardrobe"]
        # invariant_categories = ["entrance"]

        # area - without circulation
        # invariant_area = sum(item.required_area for item in space_planner_spec.items
        #                      if item.category.name in invariant_categories)
        # circulation_area = sum(item.required_area for item in space_planner_spec.items
        #                      if item.category.name == 'circulation')
        # coeff = (int(space_planner_spec.plan.indoor_area - invariant_area) / int(sum(
        #     item.required_area for item in space_planner_spec.items if
        #     (item.category.name not in invariant_categories
        #      and item.category.name != 'circulation'))))
        #
        # for item in space_planner_spec.items:
        #     if (item.category.name not in invariant_categories
        #             and item.category.name != 'circulation'):
        #         item.min_size.area = round(item.min_size.area * coeff)
        #         item.max_size.area = round(item.max_size.area * coeff)
        # area - with circulation
        invariant_area = sum(item.required_area for item in scoring_spec.items
                             if item.category.name in invariant_categories)
        coeff = (int(scoring_spec.plan.indoor_area - invariant_area) / int(sum(
            item.required_area for item in scoring_spec.items if
            item.category.name not in invariant_categories)))

        for item in scoring_spec.items:
            if living_kitchen:
                if "living" in item.opens_on:
                    item.opens_on.remove("living")
                    item.opens_on.append("livingKitchen")
            if item.category.name not in invariant_categories:
                item.min_size.area = round(item.min_size.area * coeff)
                item.max_size.area = round(item.max_size.area * coeff)

        logging.debug("Scoring - PLAN AREA %f ", int(scoring_spec.plan.indoor_area))
        logging.debug("Scoring - Setup AREA %f", int(sum(item.required_area
                                                        for item in scoring_spec.items)))
        logging.debug("Scoring - Initial Setup AREA %f", int(sum(item.required_area
                                                                for item in spec.items)))
        return scoring_spec

    def create_item_dict(spec: 'Specification', plan: 'Plan') -> Dict['Space', 'Item']:
        """
        Creates a 1-to-1 dict between spaces and items of the specification
        :param spec
        :param plan
        :return:
        """
        output = {}
        spec_items = spec.items[:]
        for space in plan.mutable_spaces():
            corresponding_items = list(filter(lambda i: i.category is space.category, spec_items))
            # Note : corridors have no corresponding spec item
            best_item = min(corresponding_items, key=lambda i:
            math.fabs(i.required_area - space.cached_area()), default=None)
            if space.category is not SPACE_CATEGORIES["circulation"]:
                assert best_item, "Score: Each space should have a corresponding item in the spec"
            output[space] = best_item
            if best_item is not None and space.category is not SPACE_CATEGORIES["circulation"]:
                spec_items.remove(best_item)
        return output

    def scoring_test():
        """
        Test
        :return:
        """
        logging.getLogger().setLevel(logging.INFO)

        input_blueprint_list = ["vernouillet_A108_blueprint.json",
                                "soisy-bailly_116_blueprint.json",
                                "soisy-bailly_105_blueprint.json",
                                "saint-maur-raspail_H05_blueprint.json",
                                "paris-mon18_A613_blueprint.json",
                                "paris-mon18_A1604_blueprint.json",
                                "paris-mon18_902_blueprint.json",
                                "paris-18_A501_blueprint.json",
                                "noisy-le-grand_A543_blueprint.json",
                                "levallois-zelmis_A2-305_blueprint.json",
                                "nantes-unile_B813_blueprint.json",
                                "levallois-zelmis_A3-502_blueprint.json",
                                "levallois-zelmis_A2-404_blueprint.json",
                                "groslay-nordwood_A-03-05_blueprint.json",
                                "grenoble-cambridge_222_blueprint.json",
                                "draveil-barbusse_A2-310_blueprint.json",
                                "draveil-barbusse_A1-302_blueprint.json",
                                "draveil-barbusse_A1-301_blueprint.json",
                                "bagneux-petit_B222_blueprint.json"]

        input_blueprint_list = ["noisy-le-grand_A543_blueprint.json"]

        for input_file in input_blueprint_list:
            input_file_setup = input_file[:-14] + "setup.json"
            print(input_file)
            plan = reader.create_plan_from_file(input_file)

            GRIDS['002'].apply_to(plan)
            SEEDERS["directional_seeder"].apply_to(plan)
            plan.plot()
            input_spec = reader.create_specification_from_file(input_file_setup)
            input_spec.plan = plan
            input_spec.plan.remove_null_spaces()
            print("PLAN AREA : %i", int(plan.indoor_area))
            print("Setup AREA : %i",
                  int(sum(item.required_area for item in input_spec.items)))
            print("input_spec :", input_spec)
            space_planner = SPACE_PLANNERS["standard_space_planner"]
            best_solutions = space_planner.apply_to(input_spec, 3)
            print("best_solutions", best_solutions)

            # architect plan
            architect_input_plan = input_file[:-14] + "plan.json"
            architect_plan = reader.create_plan_from_file(architect_input_plan)
            architect_plan.remove_null_spaces()
            architect_plan.plot()
            architect_plan_spec = spec_adaptation(input_spec, architect_plan)
            architect_plan_spec.plan = architect_plan
            architect_space_item = create_item_dict(architect_plan_spec, architect_plan)
            space_planner.solutions_collector.add_plan(architect_plan_spec, architect_space_item)
            architect_plan.plot()
            architect_final_score, architect_final_score_components = final_scoring(
                space_planner.solutions_collector.architect_plans[0])
            print(architect_final_score_components)
            radar_chart(architect_final_score,
                        architect_final_score_components,
                        000, plan.name + "Archi_FinalScore")
            plt.close()

            for sol in best_solutions:
                final_score, final_score_components = final_scoring(sol)
                sol.final_score = final_score
                radar_chart(final_score, final_score_components, sol.id,
                            sol.spec.plan.name + "_FinalScore")
                if space_planner.solutions_collector.architect_plans:
                    dist_to_architect_plan = sol.distance(
                        space_planner.solutions_collector.architect_plans[0])
                    print("Solution : ", sol.id, " - Distance to architect plan : ",
                          dist_to_architect_plan)
            plt.close()


    scoring_test()
