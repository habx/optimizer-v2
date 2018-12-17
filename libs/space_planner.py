# coding=utf-8
"""
Space Planner

A space planner attributes the spaces of the plan created by the seeder to the items.
The spaces are allocated according to constraints using constraint programming

OR-Tools : google constraint programing solver
    https://developers.google.com/optimization/
    https://acrogenesis.com/or-tools/documentation/user_manual/index.html

"""
from typing import List, Dict, Callable, Optional
import logging
import matplotlib.pyplot as plt
from libs.specification import Specification, Item
from ortools.constraint_solver import pywrapcp as ortools

WINDOW_ROOMS = ('living', 'kitchen', 'office', 'dining', 'bedroom')

DRESSING_NEIGHBOUR_ROOMS = ('entrance', 'bedroom', 'wc', 'bathroom')

CIRCULATION_ROOMS = ('living', 'dining', 'entrance')

DAY_ROOMS = ('living', 'dining', 'kitchen', 'cellar')
PRIVATE_ROOMS = ('bedroom', 'bathroom', 'laundry', 'dressing', 'entrance', 'circulationSpace')

WINDOW_CATEGORY = ('window', 'doorWindow')

BIG_VARIANTS = ('m', 'l', 'xl')

SMALL_VARIANTS = ('xs', 's')


class ConstraintSolver:
    """
    Constraint Solver
    """

    def __init__(self, items_nbr: int, spaces_nbr: int):
        self.items_nbr = items_nbr
        self.spaces_nbr = spaces_nbr
        # Create the solver
        self.solver = ortools.Solver('SpacePlanner')
        # Declare variables
        self.positions = {}  # List[List[ortools.IntVar]] = [[]]
        # For the decision builder
        self.positions_flat: List[ortools.IntVar] = []
        self.init_positions()
        self.solutions: Dict[int] = {}  # classe Solution : scoring

    def init_positions(self) -> None:
        """
        variables initialization
        :return: None
        """
        self.positions = {(i_item, j_space): self.solver.IntVar(0, 1, 'positions[{0},{1}]'.format(i_item, j_space)) for i_item in range(self.items_nbr) for j_space in range(self.spaces_nbr)}

        self.positions_flat = [self.positions[i_item, j_space] for i_item in range(self.items_nbr)
                               for
                               j_space in range(self.spaces_nbr)]

    def add_constraint(self, ct: ortools.Constraint) -> None:
        """
        add constraint
        :param ct: ortools.Constraint
        :return: None
        """
        print(ct)
        if ct is not None:
            self.solver.Add(ct)

    def solve(self) -> None:
        """
        search and solution
        :return: None
        """
        # Decision builder
        db = self.solver.Phase(self.positions_flat, self.solver.INT_VAR_DEFAULT,
                               self.solver.ASSIGN_RANDOM_VALUE)

        self.solver.NewSearch(db)

        # Maximum number of solutions
        max_num_sol = 50
        nbr_solutions = 0
        while self.solver.NextSolution():
            sol_positions = []
            for i_item in range(self.items_nbr):  # Rooms
                logging.debug("{0}: {1}".format(i_item, [self.positions[i_item, j].Value() for j in
                                                         range(self.spaces_nbr)]))
                sol_positions.append([])
                for j_space in range(self.spaces_nbr):  # empty and seed spaces
                    sol_positions[i_item].append(self.positions[i_item, j_space].Value())
                    self.solutions[nbr_solutions] = sol_positions

            # Number of solutions
            nbr_solutions += 1
            if nbr_solutions >= max_num_sol:
                break

        self.solver.EndSearch()

        logging.debug('Statistics')
        logging.debug("num_solutions: {0}".format(nbr_solutions))
        logging.debug("failures: {0}".format(self.solver.Failures()))
        logging.debug("branches: {0}".format(self.solver.Branches()))
        logging.debug("WallTime: {0}".format(self.solver.WallTime()))


class ConstraintsManager:
    """
    Space planner constraint Class
    """

    def __init__(self, sp: 'SpacePlanner', name: str = ''):
        self.name = name
        self.sp = sp

        self.constraint_solver = ConstraintSolver(len(self.sp.spec.items), len(self.sp.seed_spaces))
        self.symmetry_breaker_memo = {}
        self.windows_length = {}
        self.init_windows_length()

    def init_windows_length(self) -> None:
        for item in self.sp.spec.items:
            length = 0
            for j, space in enumerate(self.sp.seed_spaces):
                for component in space.components_associated():
                    if component.category.name == 'window' \
                            or component.category.name == 'doorWindow':
                        length += self.constraint_solver.positions[item.id, j] * int(
                            component.length)
            self.windows_length[str(item.id)] = length

    def add_item_constraint(self, item: Item, constraint_func: Callable, **kwargs) -> None:
        """
        add item constraint
        :param item: Item
        :param constraint_func: Callable
        :return: None
        """
        if kwargs is not {}:
            kwargs = {'item': item, **kwargs}
        else:
            kwargs = {'item': item}
        self.constraint_solver.add_constraint(constraint_func(self, **kwargs))

    def or_(self, ct1: ortools.Constraint, ct2: ortools.Constraint) -> ortools.Constraint:
        """
        Or between two constraints
        :param ct1: ortools.Constraint
        :param ct2: ortools.Constraint
        :return: ct: ortools.Constraint
        """
        ct = (self.constraint_solver.solver.Max(ct1, ct2) == 1)
        return ct

    def and_(self, ct1: ortools.Constraint, ct2: ortools.Constraint) -> ortools.Constraint:
        """
        And between two constraints
        :param ct1: ortools.Constraint
        :param ct2: ortools.Constraint
        :return: ct: ortools.Constraint
        """
        ct = (self.constraint_solver.solver.Min(ct1, ct2) == 1)
        return ct


def space_attribution_constraint(constraints_manager: 'ConstraintsManager',
                                 j_space: int) -> ortools.Constraint:
    """
    Each space has to be associated with an item and one time only
    :param constraints_manager: 'ConstraintsManager'
    :param j_space: int
    :return: ct: ortools.Constraint
    """
    ct = (constraints_manager.constraint_solver.solver.Sum(
        constraints_manager.constraint_solver.positions[i, j_space]
        for i in range(len(constraints_manager.sp.spec.items))) == 1)
    return ct


def area_constraint(constraints_manager: 'ConstraintsManager', item: Item,
                    min_max: str) -> ortools.Constraint:
    """
    Maximum area constraint
    :param constraints_manager: 'ConstraintsManager'
    :param item: Item
    :param min_max: str
    :return: ct: ortools.Constraint
    """
    ct = None
    max_area_coeff = 4 / 3
    min_area_coeff = 2 / 3
    if min_max == 'max':
        ct = (constraints_manager.constraint_solver.solver.Sum(
            constraints_manager.constraint_solver.positions[item.id, j] * int(space.area) for
            j, space in
            enumerate(constraints_manager.sp.seed_spaces)) <= int(
            item.max_size.area * max_area_coeff))
    elif min_max == 'min':
        ct = (constraints_manager.constraint_solver.solver.Sum(
            constraints_manager.constraint_solver.positions[item.id, j] * int(space.area) for
            j, space in
            enumerate(constraints_manager.sp.seed_spaces)) >= int(
            item.min_size.area * min_area_coeff))
    else:
        ValueError('AreaConstraint')

    return ct


def windows_constraint(constraints_manager: 'ConstraintsManager', item: Item) -> Optional[bool]:
    """
    Windows length constraint
    :param constraints_manager: 'ConstraintsManager'
    :param item: Item
    :return: ct: ortools.Constraint
    """
    ct = None
    for j_item in constraints_manager.sp.spec.items:
        if item.required_area < j_item.required_area:
            if ct is None:
                ct = (constraints_manager.windows_length[str(item.id)] <=
                      constraints_manager.windows_length[str(j_item.id)])
            else:
                new_ct = (constraints_manager.windows_length[str(item.id)] <=
                          constraints_manager.windows_length[str(j_item.id)])
                ct = constraints_manager.constraint_solver.solver.Min(ct, new_ct)

    if ct is None:
        return ct
    else:
        return ct == 1


def symmetry_breaker_constraint(constraints_manager: 'ConstraintsManager',
                                item: Item) -> ortools.Constraint:
    """
    Symmetry Breaker constraint
    :param constraints_manager: 'ConstraintsManager'
    :param item: Item
    :return: ct: ortools.Constraint
    """
    ct = None
    if not (item.category.name in constraints_manager.symmetry_breaker_memo):
        constraints_manager.symmetry_breaker_memo[item.category.name] = item.id
    else:
        for j in range(len(constraints_manager.sp.seed_spaces)):
            for k in range(len(constraints_manager.sp.seed_spaces)):
                if k < j:
                    ct = (constraints_manager.constraint_solver.positions[
                              constraints_manager.symmetry_breaker_memo[item.category.name], j] *
                          constraints_manager.constraint_solver.positions[item.id, k] == 0)
                constraints_manager.symmetry_breaker_memo[item.category.name] = item.id

    return ct


def inside_adjacency_constraint(constraints_manager: 'ConstraintsManager',
                                item: Item) -> ortools.Constraint:
    """
    Space adjacency constraint inside a given item
    :param constraints_manager: 'ConstraintsManager'
    :param item: Item
    :return: ct: ortools.Constraint
    """
    nbr_spaces_in_i_item = constraints_manager.constraint_solver.solver.Sum(
        constraints_manager.constraint_solver.positions[item.id, j] for j in
        range(len(constraints_manager.sp.seed_spaces)))
    spaces_adjacency = constraints_manager.constraint_solver.solver.Sum(
        constraints_manager.constraint_solver.solver.Sum(
            int(j_space.adjacent_to(k_space)) *
            constraints_manager.constraint_solver.positions[item.id, j] *
            constraints_manager.constraint_solver.positions[item.id, k] for
            j, j_space in enumerate(constraints_manager.sp.seed_spaces) if j > k)
        for k, k_space in enumerate(constraints_manager.sp.seed_spaces))
    ct1 = (spaces_adjacency >= nbr_spaces_in_i_item - 1)

    ct2 = None
    for k, k_space in enumerate(constraints_manager.sp.seed_spaces):
        A = constraints_manager.constraint_solver.positions[
                item.id, k] * constraints_manager.constraint_solver.solver.Sum(
            int(j_space.adjacent_to(k_space)) *
            constraints_manager.constraint_solver.positions[item.id, j] for
            j, j_space in enumerate(constraints_manager.sp.seed_spaces) if k != j)
        if ct2 is None:
            ct2 = constraints_manager.constraint_solver.solver.Max(
                A >= constraints_manager.constraint_solver.positions[item.id, k],
                nbr_spaces_in_i_item == 1)
        else:
            ct2 = constraints_manager.constraint_solver.solver.Min(ct2,
                                                                   constraints_manager.constraint_solver.solver.Max(
                                                                       A >=
                                                                       constraints_manager.constraint_solver.positions[
                                                                           item.id, k],
                                                                       nbr_spaces_in_i_item == 1))

    ct = (constraints_manager.constraint_solver.solver.Min(ct1, ct2) == 1)

    return ct


def item_adjacency_constraint(constraints_manager: 'ConstraintsManager', item: Item,
                              item_category: List[str], adj: bool = True,
                              addition_rule: str = '') -> ortools.Constraint:
    """
    Item adjacency constraint :
    :param constraints_manager: 'ConstraintsManager'
    :param item: Item
    :param item_category: List[str]
    :param adj: bool
    :param addition_rule: str
    :return: ct: ortools.Constraint
    """
    ct = None
    for cat in item_category:
        adjacency_sum = 0
        for num, num_item in enumerate(constraints_manager.sp.spec.items):
            if num_item.category.name == cat:
                adjacency_sum += constraints_manager.constraint_solver.solver.Sum(
                    constraints_manager.constraint_solver.solver.Sum(
                        int(j_space.adjacent_to(k_space)) *
                        constraints_manager.constraint_solver.positions[item.id, j] *
                        constraints_manager.constraint_solver.positions[num, k] for
                        j, j_space in enumerate(constraints_manager.sp.seed_spaces))
                    for k, k_space in enumerate(constraints_manager.sp.seed_spaces))
        if adjacency_sum is not 0:
            if ct is None:
                if adj:
                    ct = (adjacency_sum >= 1)
                else:
                    ct = (adjacency_sum == 0)
            else:
                if adj:
                    if addition_rule == 'Or':
                        ct = constraints_manager.or_(ct, (adjacency_sum >= 1))
                    elif addition_rule == 'And':
                        ct = constraints_manager.and_(ct, (adjacency_sum >= 1))
                    else:
                        ValueError('ComponentsAdjacencyConstraint')
                else:
                    if addition_rule == 'Or':
                        ct = constraints_manager.or_(ct, (adjacency_sum == 0))
                    elif addition_rule == 'And':
                        ct = constraints_manager.and_(ct, (adjacency_sum == 0))
                    else:
                        ValueError('ComponentsAdjacencyConstraint')

    return ct


def components_adjacency_constraint(constraints_manager: 'ConstraintsManager', item: Item,
                                    category: List[str], adj: bool = True,
                                    addition_rule: str = '') -> ortools.Constraint:
    """
    Components adjacency constraint
    :param constraints_manager: 'ConstraintsManager'
    :param item: Item
    :param category: List[str]
    :param adj: bool
    :param addition_rule: str
    :return: ct: ortools.Constraint
    """
    ct = None
    for c, cat in enumerate(category):
        adjacency_sum = constraints_manager.constraint_solver.solver.Sum(
            constraints_manager.constraint_solver.positions[item.id, j] for j, space in
            enumerate(constraints_manager.sp.seed_spaces) if
            cat in space.components_category_associated())
        if c == 0:
            if adj:
                ct = (adjacency_sum >= 1)
            else:
                ct = (adjacency_sum == 0)
        else:
            if adj:
                if addition_rule == 'Or':
                    ct = constraints_manager.or_(ct, (adjacency_sum >= 1))
                elif addition_rule == 'And':
                    ct = constraints_manager.and_(ct, (adjacency_sum >= 1))
                else:
                    ValueError('ComponentsAdjacencyConstraint')
            else:
                if addition_rule == 'Or':
                    ct = constraints_manager.or_(ct, (adjacency_sum == 0))
                elif addition_rule == 'And':
                    ct = constraints_manager.and_(ct, (adjacency_sum == 0))
                else:
                    ValueError('ComponentsAdjacencyConstraint')

    return ct


class SpacePlanner:
    """
    Space planner Class
    """

    def __init__(self, name: str, spec: 'Specification'):
        self.name = name
        self.spec = spec
        logging.debug(spec)
        self.seed_spaces = []
        self.init_spaces_list()

        self.constraints_manager = ConstraintsManager(self)
        self.item_constraints = {}
        self.init_item_constraints_list()

    def __repr__(self):
        # TODO
        output = 'SpacePlanner' + self.name
        return output

    def init_spaces_list(self) -> None:
        """
        Spaces list initialization
        :return: None
        """
        for space in self.spec.plan.get_spaces():  # empty and seed spaces
            if space.mutable and space.edge is not None:
                self.seed_spaces.append(space)
                logging.debug(self.seed_spaces)
                logging.debug(space.components_associated())

    def init_item_constraints_list(self) -> None:
        """
        Constraints list initialization
        :return: None
        """
        self.item_constraints = GENERAL_ITEMS_CONSTRAINTS
        if self.spec.typology >= 3:
            for item in self.spec.items:
                for constraint in T3_MORE_ITEMS_CONSTRAINTS[item.category.name]:
                    self.item_constraints[item.category.name].append(constraint)

        logging.debug('CONSTRAINTS', self.item_constraints)

    def add_spaces_constraints(self) -> None:
        """
        add spaces constraints
        :return: None
        """
        for j_space in range(len(self.seed_spaces)):
            self.constraints_manager.constraint_solver.add_constraint(
                space_attribution_constraint(self.constraints_manager, j_space))

    def add_item_constraints(self) -> None:
        """
        add items constraints
        :return: None
        """
        for item in self.spec.items:
            for constraint in self.item_constraints['all']:
                print('item', item.category.name)
                print('constraint', constraint[0])
                self.constraints_manager.add_item_constraint(item, constraint[0], **constraint[1])
            for constraint in self.item_constraints[item.category.name]:
                self.constraints_manager.add_item_constraint(item, constraint[0], **constraint[1])

    def rooms_building(self):  # -> Plan:
        """
        Rooms building
        :return: None
        """
        # plan_solution = copy.deepcopy(self.plan)
        self.constraints_manager.constraint_solver.solve()

        if len(self.constraints_manager.constraint_solver.solutions) >= 1:
            for j_space, space in enumerate(self.seed_spaces):  # empty and seed spaces
                for i_item, item in enumerate(self.spec.items):  # Rooms
                    if self.constraints_manager.constraint_solver.solutions[0][i_item][
                        j_space] == 1:
                        space.category = item.category


GENERAL_ITEMS_CONSTRAINTS = {
    'all': [
        [inside_adjacency_constraint, {}],
        [windows_constraint, {}],
        [area_constraint, {'min_max': 'min'}]
    ],
    'entrance': [
        [components_adjacency_constraint, {'category': ['frontDoor'], 'adj': True}],
        [area_constraint, {'min_max': 'max'}]
    ],
    'wc': [
        [components_adjacency_constraint, {'category': ['duct'], 'adj': True}],
        [components_adjacency_constraint,
         {'category': WINDOW_CATEGORY, 'adj': False, 'addition_rule': 'And'}],
        [area_constraint, {'min_max': 'max'}],
        [symmetry_breaker_constraint, {}]
    ],
    'bathroom': [
        [components_adjacency_constraint, {'category': ['duct'], 'adj': True}],
        [components_adjacency_constraint, {'category': ['doorWindow'], 'adj': False}],
        [area_constraint, {'min_max': 'max'}],
        [symmetry_breaker_constraint, {}]
    ],
    'living': [
        [components_adjacency_constraint,
         {'category': WINDOW_CATEGORY, 'adj': True, 'addition_rule': 'Or'}],
        [item_adjacency_constraint,
         {'item_category': ('kitchen', 'dining'), 'adj': True, 'addition_rule': 'Or'}]
    ],
    'dining': [
        [components_adjacency_constraint,
         {'category': WINDOW_CATEGORY, 'adj': True, 'addition_rule': 'Or'}],
        [item_adjacency_constraint, {'item_category': 'kitchen'}]
    ],
    'kitchen': [
        [components_adjacency_constraint,
         {'category': WINDOW_CATEGORY, 'adj': True, 'addition_rule': 'Or'}],
        [components_adjacency_constraint, {'category': ['duct'], 'adj': True}],
        # [area_constraint, {'min_max': 'max'}],
        [item_adjacency_constraint,
         {'item_category': ('living', 'dining'), 'adj': True, 'addition_rule': 'Or'}]
    ],
    'bedroom': [
        [components_adjacency_constraint,
         {'category': WINDOW_CATEGORY, 'adj': True, 'addition_rule': 'Or'}],
        [area_constraint, {'min_max': 'max'}],
        [symmetry_breaker_constraint, {}]
    ],
    'office': [
        [components_adjacency_constraint,
         {'category': WINDOW_CATEGORY, 'adj': True, 'addition_rule': 'Or'}],
        [area_constraint, {'min_max': 'max'}],
        [symmetry_breaker_constraint, {}]
    ],
    'dressing': [
        [components_adjacency_constraint,
         {'category': WINDOW_CATEGORY, 'adj': False, 'addition_rule': 'And'}],
        [area_constraint, {'min_max': 'max'}],
        [symmetry_breaker_constraint, {}]
    ],
    'laundry': [
        [components_adjacency_constraint, {'category': ['duct'], 'adj': True}],
        [components_adjacency_constraint,
         {'category': WINDOW_CATEGORY, 'adj': False, 'addition_rule': 'And'}],
        [area_constraint, {'min_max': 'max'}],
        [symmetry_breaker_constraint, {}]
    ]
}

T3_MORE_ITEMS_CONSTRAINTS = {
    'all': [

    ],
    'entrance': [

    ],
    'wc': [
        [item_adjacency_constraint,
         {'item_category': PRIVATE_ROOMS, 'adj': True, 'addition_rule': 'Or'}]
    ],
    'bathroom': [
        [item_adjacency_constraint,
         {'item_category': PRIVATE_ROOMS, 'adj': True, 'addition_rule': 'Or'}]
    ],
    'living': [

    ],
    'dining': [

    ],
    'kitchen': [

    ],
    'bedroom': [

    ],
    'office': [

    ],
    'dressing': [
        [item_adjacency_constraint,
         {'item_category': PRIVATE_ROOMS, 'adj': True, 'addition_rule': 'Or'}]
    ],
    'laundry': [
        [item_adjacency_constraint,
         {'item_category': PRIVATE_ROOMS, 'adj': True, 'addition_rule': 'Or'}]
    ]
}

if __name__ == '__main__':
    import libs.reader as reader
    import libs.seed
    from libs.selector import SELECTORS
    from libs.grid import GRIDS
    from libs.shuffle import SHUFFLES

    logging.getLogger().setLevel(logging.DEBUG)


    def space_planning():
        """
        Test
        :return:
        """

        input_file = 'Levallois_Letourneur.json'  # 5 Levallois_Letourneur / Antony_A22
        plan = reader.create_plan_from_file(input_file)

        seeder = libs.seed.Seeder(plan, libs.seed.GROWTH_METHODS)
        seeder.add_condition(SELECTORS['seed_duct'], 'duct')
        GRIDS['ortho_grid'].apply_to(plan)

        seeder.plant()
        seeder.grow(show=True)
        SHUFFLES['square_shape'].run(plan, show=True)

        logging.debug(plan)
        logging.debug(seeder)

        ax = plan.plot(save=True)
        # seeder.plot_seeds(ax)
        plt.show()
        seed_empty_furthest_couple_middle = SELECTORS[
            'seed_empty_furthest_couple_middle_space_area_min_100000']
        seed_empty_area_max_100000 = SELECTORS['area_max=100000']
        seed_methods = [
            (
                seed_empty_furthest_couple_middle,
                libs.seed.GROWTH_METHODS_FILL,
                "empty"
            ),
            (
                seed_empty_area_max_100000,
                libs.seed.GROWTH_METHODS_SMALL_SPACE_FILL,
                "empty"
            )
        ]

        filler = libs.seed.Filler(plan, seed_methods)
        filler.apply_to(plan)
        plan.remove_null_spaces()
        fuse_selector = SELECTORS['fuse_small_cell']

        logging.debug("num_mutable_spaces before merge: {0}".format(plan.count_mutable_spaces()))

        filler.fusion(fuse_selector)

        logging.debug("num_mutable_spaces after merge: {0}".format(plan.count_mutable_spaces()))

        SHUFFLES['square_shape'].run(plan, show=True)


        input_file = 'Levallois_Letourneur_setup.json'
        spec = reader.create_specification_from_file(input_file)
        spec.plan = plan

        space_planner = SpacePlanner('test', spec)
        space_planner.add_spaces_constraints()
        space_planner.add_item_constraints()
        space_planner.rooms_building()

        ax = plan.plot(save=True)
        # seeder.plot_seeds(ax)
        plt.show()
        assert plan.check()


    space_planning()
