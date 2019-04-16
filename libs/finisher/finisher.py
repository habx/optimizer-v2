"""
Finisher module :
Applies a genetic algorithm to improve the plan according to several constraints :
• rooms sizes
• rooms shapes
• circulation

The module is inspired by the deap library (global toolbox for genetic algorithms) and
implements a simple version of the NSGA-II algorithm:
    [Deb2002] Deb, Pratab, Agarwal, and Meyarivan, "A fast elitist
    non-dominated sorting genetic algorithm for multi-objective
    optimization: NSGA-II", 2002.

TODO :
    • better match between specifications and plan via an item <-> space dictionary
    • refine grid prior to genetic search
    • guide mutation to refine choices and speed-up search
    • improve evaluation method : create the method from separate functions corresponding
    to each objective
    • forbid the swap of a face close to a needed component (eg. duct for a bathroom)
"""
import random
import math

from deap import base, creator, tools
from typing import TYPE_CHECKING, Optional, Tuple, Generator
from libs.plan.plan import Plan
from libs.operators.mutation import MUTATIONS
from libs.operators.selector import SELECTORS

if TYPE_CHECKING:
    from libs.plan.plan import Space
    from libs.mesh.mesh import Edge, Face
    from libs.specification.specification import Specification


def simple_ga(plan: 'Plan', spec: 'Specification'):
    """
    A simple implementation of a genetic algorithm.
    :param plan:
    :param spec:
    :return: the best plan
    """
    for floor in plan.floors.values():
        floor.mesh.compute_cache()

    NGEN = 100
    MU = 4 * 10  # Must be a multiple of 4 for tournament selection of NSGA-II
    CXPB = 0.2

    creator.create("Fitness", base.Fitness, weights=(-1.0, -10.0, -8.0))
    creator.create("Individual", Plan, fitness=creator.Fitness)
    toolbox = base.Toolbox()
    toolbox.unregister("clone")
    toolbox.register("clone", clone)

    def _create_individual():
        ind = creator.Individual()
        ind.copy(plan)
        return ind

    toolbox.register("evaluate", evaluate, spec=spec)
    toolbox.register("population", tools.initRepeat, list, _create_individual)
    toolbox.register("select", tools.selNSGA2)
    toolbox.register("mate", mate)
    toolbox.register("mutate", mutate_simple)

    pop = toolbox.population(MU)

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # This is just to assign the crowding distance to the individuals
    # no actual selection is done
    pop = toolbox.select(pop, len(pop))

    # Begin the generational process
    for gen in range(1, NGEN):
        # Vary the population
        offspring = tools.selTournamentDCD(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]

        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            if random.random() <= CXPB:
                toolbox.mate(ind1, ind2)
            toolbox.mutate(ind1)
            toolbox.mutate(ind2)
            del ind1.fitness.values, ind2.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Select the next generation population
        pop = toolbox.select(pop + offspring, MU)

    return pop


def clone(ind):
    """
    Clones an individual
    :param ind:
    :return:
    """
    new_ind = creator.Individual()
    new_ind.copy(ind)
    return new_ind


def mutate_aligned(ind: 'Plan'):
    """
    Mutates the plan.
    1. select a random space
    2. select a random mutable edge
    3. mutate the edge
    :param ind:
    :return: a single element tuple containing the mutated individual
    """
    space = random_space(ind)
    edge = random_edge(space)
    MUTATIONS["swap_aligned_face"].apply_to(edge, space)
    return ind,


def mutate_simple(ind: 'Plan'):
    """
    Mutates the plan.
    1. select a random space
    2. select a random mutable edge
    3. mutate the edge
    :param ind:
    :return: a single element tuple containing the mutated individual
    """
    space = random_space(ind)
    edge = random_edge(space)
    if edge:
        if space.corner_stone(edge.face) or ind.get_space_of_edge(edge) is not space:
            return ind,
        MUTATIONS["remove_face"].apply_to(edge, space)
    return ind,


def random_space(plan: 'Plan') -> Optional['Space']:
    """
    Returns a random mutable space of the plan
    :param plan:
    :return:
    """
    mutable_spaces = list(plan.mutable_spaces())
    if not mutable_spaces:
        return None
    return random.choice(mutable_spaces)


def random_edge(space: 'Space') -> Optional['Edge']:
    """
    Returns a random edge of the space
    :param space:
    :return:
    """
    mutable_edges = list(SELECTORS["is_mutable"].yield_from(space))
    if not mutable_edges:
        return None
    return random.choice(mutable_edges)


def mate(ind_1: 'Plan', ind_2: 'Plan'):
    """
    Blends the two plans.
    For each floor :
    1. Finds every face that is different between the two individuals.
    2. Clusters faces in adjacent groups
    3. Pick randomly a cluster and swap the corresponding faces from each plan
    :param ind_1:
    :param ind_2:
    :return: a tuple of the blended individual
    """
    if not ind_1 or not ind_2:
        return ind_1, ind_2
    for floor in ind_1.floors.values():
        clusters = []
        differences = [f for f in floor.mesh.faces
                       if ind_1.get_space_of_face(f).id != ind_2.get_space_of_face(f).id]

        if len(differences) <= 1:
            # nothing to do
            continue

        clusters.append([differences[0]])
        for face in differences[1::]:
            found = False
            for cluster in clusters:
                for other in cluster:
                    if face in other.siblings:
                        cluster.append(face)
                        found = True
                        break
                if found:
                    break
            else:
                clusters.append([face])

        if len(clusters) == 1:
            # do nothing
            continue

        swap_cluster = random.choice(clusters)

        modified_spaces = []

        for face in swap_cluster:
            space_1 = ind_1.get_space_of_face(face)
            space_2 = ind_2.get_space_of_face(face)
            other_1 = ind_1.get_space_from_id(space_2.id)
            other_2 = ind_2.get_space_from_id(space_1.id)

            space_1.remove_face_id(face.id)
            other_1.add_face_id(face.id)

            space_2.remove_face_id(face.id)
            other_2.add_face_id(face.id)

            for space in [space_1, space_2, other_1, other_2]:
                if space not in modified_spaces:
                    modified_spaces.append(space)

        # make sure the plan structure is correct
        for space in modified_spaces:
            space.set_edges()

        return ind_1, ind_2


def evaluate(ind, spec: Optional['Specification'] = None) -> Tuple[float, float, float]:
    """
    Evaluates the fitness of an individual.
    Example :
    1. sizes of rooms
    2. shapes of rooms: min_corners
    3. aspect_ratio of rooms
    :param ind:
    :param spec
    :return:
    """
    # Sizes score
    area_score = 0.0
    if spec is not None:
        for space in ind.spaces:
            if not space.category.mutable:
                continue
            corresponding_items = (item for item in spec.items if item.category is space.category)
            space_score = min((math.fabs((item.required_area - space.cached_area())/item.required_area)
                              for item in corresponding_items), default=100)
            area_score += space_score

    # Corners score
    min_corners = 4
    corner_score = 0.0
    for space in ind.spaces:
        if not space.category.mutable or space.category.name == "living":
            continue
        corner_score += (space.number_of_corners() - min_corners)/min_corners

    # aspect ratio score
    aspect_score = 0.0
    min_aspect_ratio = 16
    for space in ind.spaces:
        if not space.category.mutable or space.category.name == "living":
            continue
        if space.cached_area() == 0:
            aspect_score += 1000
            continue
        box = space.bounding_box()
        box_area = box[0]*box[1]
        space_score = math.fabs((space.cached_area() - box_area)/box_area)
        # space_score = math.fabs(space.perimeter**2/space.cached_area()/min_aspect_ratio - 1.0)
        aspect_score += space_score

    return area_score, corner_score, aspect_score


if __name__ == '__main__':
    """
    1. get a plan
    2. get some specifications
    3. define mutations
    4. define fitness functions
    5. run algorithm
    """
    from libs.modelers.shuffle import SHUFFLES

    def get_plan(plan_name: str = "001",
                 spec_name: str = "0",
                 solution_number: int = 0) -> Tuple['Specification', Optional['Plan']]:
        """
        Returns a solution plan
        """
        import logging

        import libs.io.reader as reader
        import libs.io.writer as writer
        from libs.modelers.grid import GRIDS
        from libs.modelers.seed import SEEDERS
        from libs.space_planner.space_planner import SpacePlanner
        from libs.plan.plan import Plan
        # logging.getLogger().setLevel(logging.DEBUG)

        try:
            new_serialized_data = reader.get_plan_from_json(plan_name + "_solution_"
                                                            + str(solution_number))
            plan = Plan(plan_name).deserialize(new_serialized_data)
            spec = reader.create_specification_from_file(plan_name + "_setup"
                                                         + spec_name + ".json")
            spec.plan = plan
            return spec, plan

        except FileNotFoundError:
            plan = reader.create_plan_from_file(plan_name + ".json")
            GRIDS['optimal_grid'].apply_to(plan)
            SEEDERS["simple_seeder"].apply_to(plan)
            spec = reader.create_specification_from_file(plan_name + "_setup"
                                                         + spec_name + ".json")
            spec.plan = plan
            space_planner = SpacePlanner("test", spec)
            best_solutions = space_planner.solution_research()

            if best_solutions:
                solution = best_solutions[solution_number]
                plan = solution.plan
                writer.save_plan_as_json(plan.serialize(), plan_name + "_solution_"
                                         + str(solution_number))
                return spec, plan
            else:
                logging.info("No solution for this plan")
                return spec, None

    spec, plan = get_plan("004")
    if plan:
        SHUFFLES["bedrooms_corner"].apply_to(plan)
        plan.plot()
        pop = simple_ga(plan, spec)
        best = max(pop, key=lambda i: i.fitness)
        best.plot()
