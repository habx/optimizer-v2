"""
Core module for the genetic algorithm
Inspired from DEAP https://github.com/DEAP

A custom fitness class must be created for the corresponding pb:
ex. fitness = Fitness.new((-1.0, -1.0, -3.0))



"""
import logging
import copy
from functools import partial

from typing import Optional, Tuple, List, Callable, Sequence, Type, Any, Iterator, Dict, Set
from libs.plan.plan import Plan, Floor


class Fitness:
    """
    A fitness class
    """
    _weights: Optional[Tuple[float, ...]] = None
    cache: Dict = {}  # a class attribute to store cached values needed for fitness calculation

    @classmethod
    def new(cls, alias: str, weights: Optional[Sequence[float]]) -> type:
        """
        Creates a new Fitness subclass
        :param alias
        :param weights:
        :return:
        """
        if alias in globals():
            logging.warning("A class named '{0}' has already been created and it "
                            "will be overwritten. Consider deleting previous "
                            "creation of that class or rename it.".format(alias))

        custom_class = type(alias, (Fitness,), {"_weights": weights})
        # the custom class must be added to the global namespace for pickling
        globals()[alias] = custom_class
        return custom_class

    """The weights are used in the fitness comparison. They are shared among
       all fitnesses of the same type. When subclassing :class:`Fitness`, the
       weights must be defined as a tuple where each element is associated to an
       objective. A negative weight element corresponds to the minimization of
       the associated objective and positive weight to the maximization."""

    def __init__(self):
        # dict containing the fitness values of each objective for each space
        self._spvalues: Dict[int, Tuple[float, ...]] = {}
        # tuple containing the summed fitness values of each space for each objective
        self._values = self.compute_values(self._spvalues)

    @staticmethod
    def compute_values(spvalues: Dict[int, Tuple[float, ...]]) -> Tuple[float, ...]:
        """
        Sums the values of each space
        :param spvalues:
        :return:
        """
        return tuple(sum(t) for t in zip(*spvalues.values()))

    @property
    def wvalues(self):
        """ property returns the weighted values"""
        return tuple(x * y for x, y in zip(self._values, self._weights))

    @property
    def wvalue(self) -> float:
        """ property : returns the arithmetic sum of the weighted values of the fitness
        """
        return sum(self.wvalues)

    @property
    def sp_wvalue(self) -> Dict[int, float]:
        """ property: returns the arithmetic sum of the weighted values for each space
        """
        return {i: sum(x * y for x, y in zip(v, self._weights)) for i, v in self._spvalues.items()}

    @property
    def sp_wvalue_t(self) -> Tuple[float, ...]:
        """ property: returns a Tuple containing each fitness weighted value for each space,
        ordered by space id to preserve order. Useful for nsga algorithm adaptation.
        """
        output = list(((i, sum(x * y for x, y in zip(v, self._weights)))
                       for i, v in self._spvalues.items()))
        output.sort(key=lambda e: e[0])
        return tuple(e[1] for e in output)

    @property
    def weights(self) -> Optional[Tuple[float, ...]]:
        """ property : returns the class attribute _weights"""
        return self._weights

    @property
    def values(self) -> Tuple[float, ...]:
        """
        property
        :return:
        """
        return self._values

    @property
    def sp_values(self) -> Dict[int, Tuple[float, ...]]:
        """
        property
        :return:
        """
        return self._spvalues

    @sp_values.setter
    def sp_values(self, values_dict: Dict[int, Tuple[float, ...]]) -> None:
        """
        The values dict contains UNWEIGHTED values of the fitness.
        The values for every space are expected as the whole object will
        be replaced. If you want to only update certain spaces value, use the
        update method.
        Note : the length of each tuple must be the same as the cls._weights tuple
        """
        if not values_dict:
            return
        self._spvalues = values_dict
        self._values = self.compute_values(self._spvalues)

    def update(self, values_dict: Dict[int, Tuple[float, ...]]) -> None:
        """
        Updates the values of the fitness with the ones contained in the specified dict.
        If a value is None, the initial value of self._spvalues is kept.
        :param values_dict:
        :return:
        """
        for k, t in values_dict.items():
            self._spvalues[k] = tuple(t[i] if t[i] is not None else self._spvalues[k][i]
                                      for i in range(len(t)))
        self._values = self.compute_values(self._spvalues)

    def clear(self):
        """ Clears the values of the fitness """
        self._spvalues = {}
        self._values = ()

    def dominates(self, other: 'Fitness', obj: slice = slice(None)):
        """Return true if each objective of *self* is not strictly worse than
        the corresponding objective of *other* and at least one objective is
        strictly better.
        :param other: Fitness to be compared
        :param obj: Slice indicating on which objectives the domination is
                    tested. The default value is `slice(None)`, representing
                    every objectives.
        """
        dominates = False
        for self_wvalue, other_wvalue in zip(self.wvalues[obj], other.wvalues[obj]):
            if self_wvalue > other_wvalue:
                dominates = True
            elif self_wvalue < other_wvalue:
                return False
        return dominates

    def space_dominates(self, other: 'Fitness'):
        """Return true if each space of *self* has not a strictly worse fitness value than
        the corresponding space of *other* and at least one space is
        strictly better.
        :param other: Fitness to be compared
        """
        dominates = False
        for i, self_sp_wvalue in self.sp_wvalue.items():
            other_sp_wvalue = other.sp_wvalue[i]
            if self_sp_wvalue > other_sp_wvalue:
                dominates = True
            elif self_sp_wvalue < other_sp_wvalue:
                return False
        return dominates

    @property
    def valid(self):
        """Assess if a fitness is valid or not."""
        return len(self._values) != 0

    def __hash__(self):
        return hash(self._values)

    def __gt__(self, other: 'Fitness'):
        return not self.__le__(other)

    def __ge__(self, other: 'Fitness'):
        return not self.__lt__(other)

    def __le__(self, other: 'Fitness'):
        return self.wvalues <= other.wvalues

    def __lt__(self, other: 'Fitness'):
        return self.wvalues < other.wvalues

    def __eq__(self, other: 'Fitness'):
        return self.wvalues == other.wvalues

    def __ne__(self, other: 'Fitness'):
        return not self.__eq__(other)

    def __deepcopy__(self, memo):
        """Replace the basic deepcopy function with a faster one.

        It assumes that the elements in the :attr:`_values` tuple are
        immutable and the fitness does not contain any other object
        than :attr:``_values`, :attr:`_spvalues` and :attr:`weights`.
        """
        copy_ = self.__class__()
        copy_._spvalues = self._spvalues.copy()
        copy_._values = self._values
        return copy_

    def __str__(self):
        """Return the values of the Fitness object."""
        return str(self.values if self.valid else tuple())

    def __repr__(self):
        """Return a name."""
        return "%s.%s(%r)" % (self.__module__, self.__class__.__name__,
                              self.values if self.valid else tuple())


class UnwatchedFloor(Floor):
    """
    A Floor that does add a watcher to its mesh.
    This is needed to prevent a memory leak where each clone plan adds a watcher to its
    reference mesh resulting in thousands of watchers added to the mesh.
    """
    def add_watcher(self):
        """
        Do nothing
        :return:
        """
        return


class Individual(Plan):
    """
    An individual
    """
    _fitness_class = Fitness
    FloorType = UnwatchedFloor

    @classmethod
    def new(cls, alias: str, fitness_class: Type[Fitness]) -> type:
        """
        Creates a new Fitness subclass
        :param alias
        :param fitness_class:
        :return:
        """
        if alias in globals():
            logging.warning("A class named '{0}' has already been created and it "
                            "will be overwritten. Consider deleting previous "
                            "creation of that class or rename it.".format(alias))

        custom_class = type(alias, (Individual,), {"_fitness_class": fitness_class})
        globals()[alias] = custom_class  # needed for pickling
        return custom_class

    def __init__(self, plan: Optional[Plan] = None):
        super().__init__()
        self.fitness = self._fitness_class()
        self.modified_spaces: Set[int] = set()  # used to store the set of modified spaces_id
        if plan:
            self.copy(plan)

    def clone(self, name: str = "") -> 'Individual':
        """
        Creates a clone copy of *self*
        :param name:
        :return:
        """
        new_plan = super().clone()
        new_ind = type(self)(new_plan)
        new_ind.fitness = copy.deepcopy(self.fitness)
        new_ind.modified_spaces = self.modified_spaces.copy()
        return new_ind

    def all_spaces_modified(self) -> 'Individual':
        """
        Flag all spaces as modified
        Useful when you want to force evaluation of the fitness
        :return:
        """
        self.modified_spaces = {s.id for s in self.mutable_spaces()}
        return self

    def __getstate__(self) -> dict:
        data = self.serialize(embedded_mesh=False)
        data["fitness"] = self.fitness
        data["modified_spaces"] = self.modified_spaces.copy()
        return data

    def __setstate__(self, state: dict):
        self.deserialize(state, embedded_mesh=False)
        self.fitness = state["fitness"]
        self.modified_spaces = state["modified_spaces"]

    def __deepcopy__(self, memo) -> 'Individual':
        """
        Creates a clone copy of *self*.
        Used to preserve the copy.deepcopy() interface.
        :param memo:
        :return:
        """
        return self.clone()


CloneFunc = Callable[['Individual'], 'Individual']
MapFunc = Callable[[Callable[[Any], Any], Iterator[Any], int], Iterator[Any]]
SelectFunc = Callable[[List['Individual'], int], List['Individual']]
MateFunc = Callable[['Individual', 'Individual'], Tuple['Individual', 'Individual']]
EvaluateFunc = Callable[['Individual'], Dict[int, Tuple[float, ...]]]
MutateFunc = Callable[['Individual'], 'Individual']
PopulateFunc = Callable[[Optional['Individual'], int], List['Individual']]
MateMutateFunc = Callable[[Tuple['Individual', 'Individual']], Tuple['Individual', 'Individual']]


def _standard_clone(i: Individual) -> Individual:
    """
    Clones and individual
    :param i:
    :return:
    """
    return i.clone()


class Toolbox:
    """
    A toolbox with all the genetic operators :
    • clone
    • mutate
    • etc.
    """
    op_list = (
        "map",
        "clone",
        "mate",
        "select",
        "elite_select",
        "mutate",
        "populate",
        "evaluate",
        "mate_and_mutate"
    )
    class_list = ("individual", "fitness")
    classes = {"fitness": Fitness, "individual": Individual}

    def __init__(self):
        # operators
        self.clone: CloneFunc = _standard_clone
        self.map: MapFunc = lambda f, p, _: map(f, p)
        self.mate:  Optional[MateFunc] = None
        self.select: Optional[SelectFunc] = None
        self.elite_select: Optional[SelectFunc] = None
        self.evaluate: Optional[EvaluateFunc] = None
        self.mutate: Optional[MutateFunc] = None
        self.populate: Optional[PopulateFunc] = None
        self.mate_and_mutate: Optional[MateMutateFunc] = None

        # base class
        self.individual: Optional[Type['Individual']] = None
        self.fitness: Optional[Type['Fitness']] = None

    def configure(self, class_name: str, alias: str = "", *args, **kwargs):
        """
        Creates the customized subclass and stores it in the toolbox
            ex.: toolbox.configure("fitness", (-1.0, -2.0, -3.0))
                 toolbox.configure("individual", toolbox.fitness)
        :param alias:
        :param class_name:
        :param args:
        :param kwargs:
        :return:
        """
        alias = alias or class_name + "Custom"
        assert class_name in Toolbox.class_list, ("Toolbox: the class name is incorrect: "
                                                  "{}".format(class_name))

        setattr(self, class_name, Toolbox.classes[class_name].new(alias, *args, **kwargs))

    def register(self, operator_name: str, func: Callable, *args, **kwargs):
        """
        Register a genetic operator in the toolbox
        :param operator_name:
        :param func:
        :param args
        :param kwargs
        :return:
        """

        assert operator_name in Toolbox.op_list, ("Toolbox: Incorrect operator name: "
                                                  "{}".format(operator_name))

        pfunc = partial(func, *args, **kwargs)
        pfunc.__name__ = operator_name
        pfunc.__doc__ = func.__doc__

        if hasattr(func, "__dict__") and not isinstance(func, type):
            # Some functions don't have a dictionary, in these cases
            # simply don't copy it. Moreover, if the function is actually
            # a class, we do not want to copy the dictionary.
            pfunc.__dict__.update(func.__dict__.copy())

        setattr(self, operator_name, pfunc)

    def unregister(self, alias):
        """Unregister *alias* from the toolbox.


        :param alias: The name of the operator to remove from the toolbox.
        """
        delattr(self, alias)

    @staticmethod
    def evaluate_pop(map_func: MapFunc,
                     eval_func: EvaluateFunc,
                     pop: Sequence['Individual'],
                     chunk_size: int) -> None:
        """
        Evaluates the fitness of a specified population. Note: the method has to be made static
        for multiprocessing purposes.
        :param map_func: a mapping function
        :param eval_func: an evaluation function (NOTE: we cannot refer to self.evaluate for
               multiprocessing concerns)
        :param pop: a list of individuals
        :param chunk_size: size of chunks for ma processes
        :return:
        """
        fitnesses = map_func(eval_func, pop, chunk_size)
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.update(fit)
            ind.modified_spaces = set()
