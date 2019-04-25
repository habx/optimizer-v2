"""
Refiner Test Module
"""

import logging
from libs.refiner.refiner import REFINERS
from libs.refiner import evaluation


def main():
    """ test function """
    import time
    import tools.cache

    logging.getLogger().setLevel(logging.DEBUG)

    spec, plan = tools.cache.get_plan("029")  # 052
    if plan:
        plan.name = "original"
        plan.plot()
        plan.store_meshes_globally()

        start = time.time()

        hof = REFINERS["simple"].run(plan, spec, processes=1, with_hof=False)

        sols = sorted(hof, key=lambda i: i.fitness.value, reverse=True)
        end = time.time()
        for n, ind in enumerate(sols):
            ind.name = str(n)
            ind.plot()
            print("Fitness: {} - {}".format(ind.fitness.value, ind.fitness.values))
        print("Time elapsed: {}".format(end - start))
        best = sols[0]
        item_dict = evaluation.create_item_dict(spec)
        for space in best.mutable_spaces():
            print("• Area {} : {} -> [{}, {}]".format(space.category.name,
                                                      round(space.cached_area()),
                                                      item_dict[space.id].min_size.area,
                                                      item_dict[space.id].max_size.area))


if __name__ == '__main__':
    main()
