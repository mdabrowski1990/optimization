"""
Adaptive evolutionary algorithms.

Evolutionary algorithms that performs two level optimization of:
 - optimization problem - main problem to solve
 - adaptation problem - searching for optimal settings of evolutionary settings
"""

__all__ = ["AdaptiveEvolutionaryAlgorithm"]


from typing import Any, Union, Optional, Tuple, Iterable, Dict, Callable
from enum import Enum
from abc import abstractmethod
from collections import OrderedDict

from .evolutionary_algorithm import EvolutionaryAlgorithm
from ...problem import OptimizationProblem, AbstractSolution, OptimizationType, \
    IntegerVariable, DiscreteVariable, FloatVariable, ChoiceVariable
from ...stop_conditions import StopConditions
from ...logging import AbstractLogger
from .selection import SelectionType
from .crossover import CrossoverType
from .mutation import MutationType


class AdaptationType(Enum):
    """
    Adaptation type of algorithms.

    Determines how algorithms settings are compared with following options available:
     - BestSolution - algorithm effectiveness is considered to be equal objective value of the best solution found
     - BestSolutions - algorithm effectiveness is considered to be equal a sum of N best solutions objective values
     - BestSolutionsPercentile - algorithm effectiveness is considered to be equal an average of
            P% best solutions objective values
     - BestSolutionsProgress - algorithm effectiveness is considered to be equal SUM_END - SUM_START, where:
            SUM_END - a sum of N best solutions objective values in the iteration 0
            SUM_START - a sum of N best solutions objective values in the last iteration
     - BestSolutionsPercentileProgress - algorithm effectiveness is considered to be equal AVE_END - AVE_START, where:
            AVE_END - an average of P% best solutions in the iteration 0
            AVE_START - an average of P% best solutions in the last iteration
    """

    BestSolution = "BestSolution"
    BestSolutions = "BestSolutions"
    BestSolutionsPercentile = "BestSolutionsPercentile"
    BestSolutionsProgress = "BestSolutionsProgress"
    BestSolutionsPercentileProgress = "BestSolutionsPercentileProgress"


class EvolutionaryAlgorithmAdaptationProblem(OptimizationProblem):
    """
    Evolutionary Algorithm adaptation problem.

    Problem of Evolutionary Algorithm adaptation during optimization process of 'AdaptiveEvolutionaryAlgorithm'.
    """

    def __init__(self,
                 adaptation_type: AdaptationType,
                 population_size: Tuple[int, int],
                 selection_types: Iterable[SelectionType],
                 crossover_types: Iterable[CrossoverType],
                 mutation_types: Iterable[MutationType],
                 mutation_chance: Tuple[float, float] = (0.01, 0.2),
                 apply_elitism: Iterable[bool] = (True, False),
                 **optional_params: Tuple[Union[float, int], Union[float, int]]):
        """
        Definition of Evolutionary Algorithm Adaptation problem.

        :param adaptation_type: Determines how to assess effectiveness of Evolutionary Algorithms.
        :param population_size: Tuple with minimal and maximal population size of Evolutionary Algorithm.
        :param selection_types: Possible selection types to be chosen by Evolutionary Algorithm.
        :param crossover_types: Possible crossover types to be chosen by Evolutionary Algorithm.
        :param mutation_types: Possible mutation types to be chosen by Evolutionary Algorithm.
        :param mutation_chance: Tuple with minimal and maximal mutation chance of Evolutionary Algorithm.
        :param apply_elitism: Possible elitism values.
        :param optional_params: Definition of other (optional) parameters that depends on
            selection/crossover/mutation type.
        """
        decision_variables = OrderedDict(
            population_size=DiscreteVariable(min_value=population_size[0], max_value=population_size[1], step=2),
            selection_types=ChoiceVariable(possible_values=selection_types),
            crossover_types=ChoiceVariable(possible_values=crossover_types),
            mutation_types=ChoiceVariable(possible_values=mutation_types),
            mutation_chance=FloatVariable(min_value=mutation_chance[0], max_value=mutation_chance[1]),
            apply_elitism=ChoiceVariable(possible_values=apply_elitism)
        )
        super().__init__(decision_variables=decision_variables,
                         constraints={},  # no constraints are needed
                         penalty_function=lambda **x: 0,  # penalty function is not used
                         objective_function=self._create_objective_function(adaptation_type=adaptation_type),
                         optimization_type=OptimizationType.Maximize)  # default value (updated later on)
        # additional parameters
        # todo: find additional parameters, check if they are correct, and create 'additional_decision_variable' dict
        self.additional_decision_variable = {}

    @staticmethod
    def _create_objective_function(adaptation_type) -> Callable:
        ...


class LowerAdaptiveEvolutionaryAlgorithm(EvolutionaryAlgorithm, AbstractSolution):
    """
    Definition of lower level adaptive evolutionary algorithm.

    These algorithms (there are many of them during optimization process) search for optimal solution of
    main optimization problem.
    """

    @property
    @abstractmethod
    def optimization_problem(self) -> EvolutionaryAlgorithmAdaptationProblem:
        """Evolutionary Algorithm adaptation problem for which this class is able to create solutions (as objects)."""
        ...

    def __init__(self, upper_iteration: int, index: int,  # pylint: disable=too-many-arguments
                 problem: OptimizationProblem,
                 stop_conditions: StopConditions,
                 population_size: int,
                 selection_type: Union[SelectionType, str],
                 crossover_type: Union[CrossoverType, str],
                 mutation_type: Union[MutationType, str],
                 mutation_chance: float,
                 apply_elitism: bool,
                 logger: Optional[AbstractLogger] = None,
                 **other_params: Any) -> None:
        """
        Configuration of Lower Evolutionary Algorithm.

        :param upper_iteration: Iteration index of Upper Algorithm.
        :param index: Unique index of this Lower Evolutionary Algorithm in this Upper Algorithm iteration.
        :param kwargs: Other parameters:
            - initial_population - (optional) starting population to be set.
            - same parameters as in EvolutionaryAlgorithm.__init__
        """
        self.upper_iteration = upper_iteration
        self.index = index
        # init as evolutionary algorithm
        initial_population = other_params.pop("initial_population", [])
        EvolutionaryAlgorithm.__init__(self=self, problem=problem, stop_conditions=stop_conditions,
                                       population_size=population_size, selection_type=selection_type,
                                       crossover_type=crossover_type, mutation_type=mutation_type,
                                       mutation_chance=mutation_chance, apply_elitism=apply_elitism, logger=logger,
                                       **other_params)
        self._population = initial_population
        # init as solution
        AbstractSolution.__init__(self=self, population_size=population_size, selection_type=self.selection_type,
                                  crossover_type=self.crossover_type, mutation_type=self.mutation_type,
                                  mutation_chance=mutation_chance, apply_elitism=apply_elitism)
        self.additional_decision_variables_values = other_params

    def _log_iteration(self, iteration_index: int) -> None:
        """
        Logs population data in given algorithm's iteration.

        :param iteration_index: Index number (counted from 0) of optimization algorithm iteration.

        :return: None
        """
        if self.logger is not None:
            self.logger.log_lower_level_iteration(upper_iteration=self.upper_iteration,
                                                  lower_algorithm_index=self.index,
                                                  lower_iteration=iteration_index,
                                                  solutions=self._population)

    def get_log_data(self) -> Dict[str, Any]:
        """
        Gets data for logging purposes.

        :return: Dictionary with this Solution crucial data.
        """
        data = EvolutionaryAlgorithm.get_log_data(self=self)
        data.update(objective_value_with_penalty=self.get_objective_value_with_penalty())
        return data


class AdaptiveEvolutionaryAlgorithm(EvolutionaryAlgorithm):
    """
    Adaptive Evolutionary Algorithm definition.

    This algorithm uses mechanisms inspired by biological evolution.
    It indirectly searches of optimal solution (_LowerAdaptiveEvolutionaryAlgorithm does that), while optimization
    optimization process effectiveness (it optimizes settings of _LowerAdaptiveEvolutionaryAlgorithm).
    """

    MIN_POPULATION_SIZE = 5
    MAX_POPULATION_SIZE = 100

    def __init__(self, problem: OptimizationProblem,  # pylint: disable=too-many-arguments
                 adaptation_problem: EvolutionaryAlgorithmAdaptationProblem,
                 stop_conditions: StopConditions,
                 population_size: int,
                 selection_type: Union[SelectionType, str],
                 crossover_type: Union[CrossoverType, str],
                 mutation_type: Union[MutationType, str],
                 mutation_chance: float,
                 logger: Optional[AbstractLogger] = None,
                 **other_params: Any) -> None:
        """
        Configuration of Self-adaptive Evolutionary Algorithm.

        :param problem: Optimization problem to be solved by the algorithm.
        :param adaptation_problem: Evolutionary algorithm adaptation problem to be optimized by the algorithm.
        :param stop_conditions: Conditions when optimization algorithm shall be stopped.
        :param population_size: Size of the algorithm's population (number of LowerAdaptiveEvolutionaryAlgorithm).
        :param selection_type: Type of selection function to use.
        :param crossover_type: Type of crossover function to use.
        :param mutation_type: Type of mutation function to use.
        :param mutation_chance: Probability of a single decision variable (gene) mutation.
        :param logger: Logger used for optimization process recording.
        :param other_params: Parameter related to selected selection, crossover and mutation type.
        """
        adaptation_problem.optimization_type = problem.optimization_type
        self.adaptation_problem = adaptation_problem
        super().__init__(problem=problem, stop_conditions=stop_conditions, population_size=population_size,
                         selection_type=selection_type, crossover_type=crossover_type, mutation_type=mutation_type,
                         mutation_chance=mutation_chance, apply_elitism=False, logger=logger, **other_params)

        class AdaptiveAESolution(LowerAdaptiveEvolutionaryAlgorithm):
            """Solution class for given evolutionary algorithm adaptation problem."""

            optimization_problem = adaptation_problem

        self.SolutionClass = AdaptiveAESolution  # type: ignore

    # todo: a few methods must be updated
