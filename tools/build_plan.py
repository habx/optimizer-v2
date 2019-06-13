import libs.io.reader as reader
from libs.modelers.seed import Seeder, GROWTH_METHODS, FILL_METHODS
from libs.space_planner.space_planner import SpacePlanner
from libs.operators.selector import SELECTORS
from libs.modelers.grid import GRIDS
from libs.plan.plan import Plan


def temporary_union(plan: Plan):
    # merges mutable spaces that have same category when are adjacent
    # TODO : TO BE REMOVED, useless when the space_planner deals with cells union
    category_list = []

    for space in plan.spaces:
        if space.mutable:
            cat = space.category.name
            if cat not in category_list:
                category_list.append(cat)

    for cat in category_list:
        cat_space = []
        for space in plan.get_spaces(cat):
            cat_space.append(space)
        if len(cat_space) > 1:
            space_ini = cat_space[0]
            i = 0
            while (len(cat_space) > 1) and i < len(cat_space) * len(cat_space):
                for space in cat_space[1:]:
                    if space.adjacent_to(space_ini):
                        space_ini._merge(space)
                        plan.remove_null_spaces()
                        cat_space.remove(space)
                i += 1


def build_plan(input_file: str) -> Plan:
    """
    Test - builds plan
    :return:
    """

    plan = reader.create_plan_from_file(input_file)

    GRIDS['ortho_grid'].apply_to(plan)
    seeder = Seeder(plan, GROWTH_METHODS).add_condition(SELECTORS['seed_duct'], 'duct')
    plan.plot()
    (seeder.plant()
     .grow()
     .fill(FILL_METHODS, (SELECTORS["farthest_couple_middle_space_area_min_100000"],
                          "empty"))
     .fill(FILL_METHODS, (SELECTORS["single_edge"], "empty"), recursive=True)
     .simplify(SELECTORS["fuse_small_cell"]))

    input_file_setup = input_file[:-5] + "_setup.json"
    spec = reader.create_specification_from_file(input_file_setup)
    spec.plan = plan
    space_planner = SpacePlanner('test', spec)
    space_planner.add_spaces_constraints()
    space_planner.add_item_constraints()
    space_planner.rooms_building()

    plan.plot(show=True)

    temporary_union(plan)

    return plan
