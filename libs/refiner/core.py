"""
Core module for the genetic algorithm
Inspired from DEAP https://github.com/DEAP

A custom fitness class must be created for the corresponding pb:
ex. fitness = Fitness.new((-1.0, -1.0, -3.0))
A custom individual class must than be created
Individual.new(


"""
from typing import Optional, Tuple, List, Callable, Sequence, Type, Any, Iterator
from libs.plan.plan import Plan


class Fitness:
    """
    A fitness class
    """
    _weights: Optional[Sequence[float]] = None

    __slots__ = "_wvalues"

    @classmethod
    def new(cls, weights: Sequence[float]) -> Type['Fitness']:
        """
        Creates a new Fitness subclass
        :param weights:
        :return:
        """

        class CustomFitness(Fitness):
            """
            A customized sub-class of Fitness with the desired weights
            """
            _weights = weights

        return CustomFitness

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


class Individual(Plan):
    """
    An individual
    """
    __slots__ = 'fitness'
    _fitness_class = Fitness

    @classmethod
    def new(cls, fitness_class: Type[Fitness]) -> Type['Individual']:
        """
        Creates a new Fitness subclass
        :param fitness_class:
        :return:
        """
        class CustomIndividual(Individual):
            """
            A customized sub-class of Individual with the desired fitness
            """
            _fitness_class = fitness_class

        return CustomIndividual

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
        return type(self)(new_plan)

    def __deepcopy__(self, memo) -> 'Individual':
        """
        Creates a clone copy of *self*.
        Used to preserve the copy.deepcopy() interface.
        :param memo:
        :return:
        """
        return self.clone()


cloneFunc = Callable[['Individual'], 'Individual']
mapFunc = Callable[[Callable[['Individual'], Any], Sequence['Individual']], Iterator[Any]]
selectFunc = Callable[[List['Individual']], List['Individual']]
mateFunc = Callable[['Individual', 'Individual'], Tuple['Individual', 'Individual']]
evaluateFunc = Callable[['Individual'], Sequence[float]]
mutateFunc = Callable[['Individual'], 'Individual']
populateFunc = Callable[[Optional['Individual'], int], List['Individual']]


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

    __slots__ = ("_map", "_clone", "_mate", "_select", "_evaluate", "_mutate", "_populate",
                 "_individual_class", "_fitness_class")

    classes = {"fitness": Fitness, "individual": Individual}

    def __init__(self):
        # operators
        self._clone: cloneFunc = _standard_clone
        self._map: mapFunc = map
        self._mate:  Optional[mateFunc] = None
        self._select: Optional[selectFunc] = None
        self._evaluate: Optional[evaluateFunc] = None
        self._mutate: Optional[mutateFunc] = None
        self._populate: Optional[populateFunc] = None

        # base class
        self._individual_class: Optional[Type['Individual']] = None
        self._fitness_class: Optional[Type['Fitness']] = None

    def configure(self, class_name: str, *args, **kwargs):
        """
        Creates the customized subclass and stores it in the toolbox
            ex.: toolbox.configure("fitness", (-1.0, -2.0, -3.0))
                 toolbox.configure("individual", toolbox.fitness)
        :param class_name:
        :param args:
        :param kwargs:
        :return:
        """
        class_dict = {
            "individual": "_individual_class",
            "fitness": "_fitness_class"
        }

        assert class_name in class_dict, ("Toolbox: the class name is incorrect: "
                                          "{}".format(class_name))

        setattr(self, class_dict[class_name], Toolbox.classes[class_name].new(*args, **kwargs))

    @property
    def fitness(self) -> Type['Fitness']:
        """ property : returns the fitness class"""
        assert self._fitness_class, "Toolbox: the fitness class has not been implemented"
        return self._fitness_class

    @property
    def individual(self) -> Type['Individual']:
        """ property : returns the individual class"""
        assert self._individual_class, "Toolbox: the individual class has not been implemented"
        return self._individual_class

    def register(self, operator_name: str, func: Callable):
        """
        Register a genetic operator in the toolbox
        :param operator_name:
        :param func:
        :return:
        """
        op_dict = {
            "map": "_map",
            "clone": "_clone",
            "mate": "_mate",
            "select": "_select",
            "mutate": "_mutate",
            "populate": "_populate",
            "evaluate": "_evaluate"
        }

        assert operator_name in op_dict, ("Toolbox: Incorrect operator name: "
                                          "{}".format(operator_name))

        setattr(self, op_dict[operator_name], func)

    def map(self,
            func: Callable[['Individual'], Any], pop: Sequence['Individual']) -> Iterator[Any]:
        """
        Maps a function on an individual Sequence
        :param func:
        :param pop:
        :return:
        """
        assert self._map, "Toolbox: the map function has not been implemented"
        return self._map(func, pop)

    def clone(self, ind: 'Individual') -> 'Individual':
        """
        Clones an individual
        :param ind:
        :return:
        """
        assert self._clone, "Toolbox: the clone function has not been implemented"
        return self._clone(ind)

    def select(self, pop: List['Individual'], *args, **kwargs) -> List['Individual']:
        """
        Clones an individual
        :param pop:
        :return: the selected population
        """
        assert self._select, "Toolbox: the select function has not been implemented"
        return self._select(pop, *args, **kwargs)

    def mate(self, ind_1: 'Individual', ind_2: 'Individual') -> Tuple['Individual', 'Individual']:
        """
        Clones an individual
        :param ind_1:
        :param ind_2:
        :return:
        """
        assert self._mate, "Toolbox: the crossover function has not been implemented"
        return self._mate(ind_1, ind_2)

    def mutate(self, ind: 'Individual') -> 'Individual':
        """
        Mutates an individual in place
        :param ind:
        :return:
        """
        assert self._mutate, "Toolbox: the mutation function has not been implemented"
        return self._mutate(ind)

    def evaluate(self, ind: 'Individual') -> Sequence[float]:
        """
        Clones an individual
        :param ind:
        :return:
        """
        assert self._evaluate, "Toolbox: the evaluate function has not been implemented"
        return self._evaluate(ind)

    def populate(self, ind: Optional['Individual'], size: int) -> List['Individual']:
        """
        Creates a population of individual
        :return:
        """
        assert self._populate, "Toolbox: the populate function has not been implemented"
        return self._populate(ind, size)

    def evaluate_pop(self, pop: Sequence['Individual'], refresh: bool = False) -> None:
        """
        Evaluates the fitness of a specified population
        :param pop: a list of individuals
        :param refresh: whether to refresh the fitness if it is still valid
        :return:
        """
        invalid_fit = [ind for ind in pop if not ind.fitness.valid and not refresh]
        fitnesses = self.map(self.evaluate, invalid_fit)
        for ind, fit in zip(invalid_fit, fitnesses):
            ind.fitness.values = fit
