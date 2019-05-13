"""
Core module for the genetic algorithm
Inspired from DEAP https://github.com/DEAP

A custom fitness class must be created for the corresponding pb:
ex. fitness = Fitness.new((-1.0, -1.0, -3.0))



"""
import logging
import copy
from functools import partial

from typing import Optional, Tuple, List, Callable, Sequence, Type, Any, Iterator
from libs.plan.plan import Plan, Floor


class Fitness:
    """
    A fitness class
    """
    _weights: Optional[Sequence[float]] = None

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
        globals()[alias] = custom_class  # needed for pickling
        return custom_class

    """The weights are used in the fitness comparison. They are shared among
       all fitnesses of the same type. When subclassing :class:`Fitness`, the
       weights must be defined as a tuple where each element is associated to an
       objective. A negative weight element corresponds to the minimization of
       the associated objective and positive weight to the maximization."""

    def __init__(self):
        self._wvalues = ()

    @property
    def value(self) -> float:
        """ property : returns the arithmetic sum of the values
        """
        return sum(self._wvalues)

    @property
    def weights(self):
        """ property : returns the class attribute _weights"""
        return self._weights

    @property
    def values(self):
        """
        property
        :return:
        """
        return tuple(map(lambda x, y: x / y, self._wvalues, self._weights))

    @values.setter
    def values(self, values: Sequence[float]):
        if not values:
            return
        assert len(values) == len(self._weights), ("Refiner: Fitness, the values provided are "
                                                   "incoherent with the fitness weights {} - "
                                                   "{}".format(values, self._weights))

        self._wvalues = tuple(map(lambda x, y: x * y, values, self._weights))

    def clear(self):
        """ Clears the values of the fitness """
        self._wvalues = ()

    def dominates(self, other: 'Fitness', obj: slice = slice(None)):
        """Return true if each objective of *self* is not strictly worse than
        the corresponding objective of *other* and at least one objective is
        strictly better.
        :param other: Fitness to be compared
        :param obj: Slice indicating on which objectives the domination is
                    tested. The default value is `slice(None)`, representing
                    every objectives.
        """
        not_equal = False
        for self_wvalue, other_wvalue in zip(self._wvalues[obj], other._wvalues[obj]):
            if self_wvalue > other_wvalue:
                not_equal = True
            elif self_wvalue < other_wvalue:
                return False
        return not_equal

    @property
    def valid(self):
        """Assess if a fitness is valid or not."""
        return len(self._wvalues) != 0

    def __hash__(self):
        return hash(self._wvalues)

    def __gt__(self, other: 'Fitness'):
        return not self.__le__(other)

    def __ge__(self, other: 'Fitness'):
        return not self.__lt__(other)

    def __le__(self, other: 'Fitness'):
        return self._wvalues <= other._wvalues

    def __lt__(self, other: 'Fitness'):
        return self._wvalues < other._wvalues

    def __eq__(self, other: 'Fitness'):
        return self._wvalues == other._wvalues

    def __ne__(self, other: 'Fitness'):
        return not self.__eq__(other)

    def __deepcopy__(self, memo):
        """Replace the basic deepcopy function with a faster one.

        It assumes that the elements in the :attr:`values` tuple are
        immutable and the fitness does not contain any other object
        than :attr:`values` and :attr:`weights`.
        """
        copy_ = self.__class__()
        copy_._wvalues = self._wvalues
        return copy_

    def __str__(self):
        """Return the values of the Fitness object."""
        return str(self.values if self.valid else tuple())

    def __repr__(self):
        """Return the Python code to build a copy of the object."""
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
    __slots__ = 'fitness'
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
        return new_ind

    def __getstate__(self) -> dict:
        data = self.serialize(embedded_mesh=False)
        data["fitness"] = self.fitness
        return data

    def __setstate__(self, state: dict):
        self.deserialize(state, embedded_mesh=False)
        self.fitness = state["fitness"]

    def __deepcopy__(self, memo) -> 'Individual':
        """
        Creates a clone copy of *self*.
        Used to preserve the copy.deepcopy() interface.
        :param memo:
        :return:
        """
        return self.clone()


cloneFunc = Callable[['Individual'], 'Individual']
mapFunc = Callable[[Callable[[Any], Any], Iterator[Any]], Iterator[Any]]
selectFunc = Callable[[List['Individual']], List['Individual']]
mateFunc = Callable[['Individual', 'Individual'], Tuple['Individual', 'Individual']]
evaluateFunc = Callable[['Individual'], Sequence[float]]
mutateFunc = Callable[['Individual'], 'Individual']
populateFunc = Callable[[Optional['Individual'], int], List['Individual']]
mateMutateFunc = Callable[[Tuple['Individual', 'Individual']], Tuple['Individual', 'Individual']]


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
        "mutate",
        "populate",
        "evaluate",
        "mate_and_mutate"
    )

    class_list = ("individual", "fitness")

    __slots__ = op_list + class_list

    classes = {"fitness": Fitness, "individual": Individual}

    def __init__(self):
        # operators
        self.clone: cloneFunc = _standard_clone
        self.map: mapFunc = map
        self.mate:  Optional[mateFunc] = None
        self.select: Optional[selectFunc] = None
        self.evaluate: Optional[evaluateFunc] = None
        self.mutate: Optional[mutateFunc] = None
        self.populate: Optional[populateFunc] = None
        self.mate_and_mutate: Optional[mateMutateFunc] = None

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
    def evaluate_pop(map_func,
                     eval_func,
                     pop: Sequence['Individual'],
                     refresh: bool = False) -> None:
        """
        Evaluates the fitness of a specified population. Note: the method has to be made static
        for multiprocessing purposes.
        :param map_func: a mapping function
        :param eval_func: an evaluation function (NOTE: we cannot refer to self.evaluate for
               multiprocessing concerns
        :param pop: a list of individuals
        :param refresh: whether to refresh the fitness if it is still valid
        :return:
        """
        invalid_fit = [ind for ind in pop if not ind.fitness.valid or refresh]
        fitnesses = map_func(eval_func, invalid_fit)
        for ind, fit in zip(invalid_fit, fitnesses):
            ind.fitness.values = fit
