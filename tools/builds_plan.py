import libs.reader as reader

from libs.space_planner import SpacePlanner
from libs.seed import Seeder, Filler, GROWTH_METHODS
from libs.shuffle import few_corner_shuffle
from libs.plan import Plan
from libs.grid import GRIDS
from libs.selector import SELECTORS


def temporary_union(plan: Plan):
    # merges mutable spaces that have same category when are adjacent
    # TODO : TO BE REMOVED, useless when the space_planner deals with cells union
    category_list = []
    for space in plan.mutable_spaces():
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


def build_plan(input_file) -> Plan:
    """
    builds a plan using functions from libs module - used for test purpose, one can control the level to which
    the plan is built : grid, space filling, space attribution...
    :return: plan
    """
    plan = reader.create_plan_from_file(input_file)

    seeder = Seeder(plan, GROWTH_METHODS)
    seeder.add_condition(SELECTORS['seed_duct'], 'duct')
    GRIDS['ortho_grid'].apply_to(plan)

    seeder.plant()
    seeder.grow()
    few_corner_shuffle.run(plan, show=False)

    plan.remove_null_spaces()
    plan.make_space_seedable("empty")

    seed_empty_furthest_couple = SELECTORS['seed_empty_furthest_couple']
    seed_empty_area_max_100000 = SELECTORS['area_max=100000']
    seed_methods = [
        (
            seed_empty_furthest_couple,
            GROWTH_METHODS,
            "empty"
        ),
        (
            seed_empty_area_max_100000,
            GROWTH_METHODS,
            "empty"
        )
    ]

    filler = Filler(plan, seed_methods)
    filler.apply_to(plan)

    plan.remove_null_spaces()
    fuse_selector = SELECTORS['fuse_small_cell']

    filler.fusion(fuse_selector)

    ax = plan.plot(save=False, options=('fill', 'border', 'face'))
    seeder.plot_seeds(ax)

    plan.remove_null_spaces()

    input_file_setup = input_file[:-5] + "_setup.json"
    spec = reader.create_specification_from_file(input_file_setup)
    spec.plan = plan
    space_planner = SpacePlanner('test', spec)
    space_planner.add_spaces_constraints()
    space_planner.add_item_constraints()
    space_planner.rooms_building()

    temporary_union(plan)

    plan.plot(show=True)
    assert spec.plan.check()

    return plan
