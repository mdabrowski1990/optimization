"""
Adaptive evolutionary algorithms.

Evolutionary algorithms that performs two level optimization of:
 - optimization problem - main problem to solve
 - adaptation problem - searching for optimal settings of evolutionary settings
"""

__all__ = ["AdaptationType", "EvolutionaryAlgorithmAdaptationProblem", "AdaptiveEvolutionaryAlgorithm"]


from typing import Any, Union, Optional, Tuple, Iterable, Dict, Callable
from enum import Enum
from abc import abstractmethod
from collections import OrderedDict

from .evolutionary_algorithm import EvolutionaryAlgorithm
from ...problem import OptimizationProblem, AbstractSolution, OptimizationType, \
    IntegerVariable, DiscreteVariable, FloatVariable, ChoiceVariable
from ...stop_conditions import StopConditions
from ...logging import AbstractLogger
from .selection import SelectionType, SELECTION_ADDITIONAL_PARAMS_LIMITS
from .crossover import CrossoverType
from .mutation import MutationType
# mypy: ignore-errors


MIN_POPULATION_SIZE: int = 10
MAX_POPULATION_SIZE: int = 1000
MIN_MUTATION_CHANCE: float = 0.
MAX_MUTATION_CHANCE: float = 0.2
DEFAULT_SOLUTIONS_PERCENTILE: float = 0.1
DEFAULT_SOLUTIONS_NUMBER: int = 3


class AdaptationType(Enum):
    """
    Adaptation type of algorithms.

    Determines how algorithms settings are compared with following options available:
     - BestSolution - algorithm effectiveness is considered to be equal objective value of the best solution found
     - BestSolutions - algorithm effectiveness is considered to be equal a sum of N best solutions objective values
     - BestSolutionsPercentile - algorithm effectiveness is considered to be equal an average of
            P% best solutions objective values
    """

    BestSolution = "BestSolution"
    BestSolutions = "BestSolutions"
    BestSolutionsPercentile = "BestSolutionsPercentile"


class EvolutionaryAlgorithmAdaptationProblem(OptimizationProblem):
    """
    Evolutionary Algorithm adaptation problem.

    Problem of Evolutionary Algorithm adaptation during optimization process of 'AdaptiveEvolutionaryAlgorithm'.
    """

    def __init__(self,
                 adaptation_type: AdaptationType,
                 population_size_boundaries: Tuple[int, int],
                 selection_types: Iterable[SelectionType],
                 crossover_types: Iterable[CrossoverType],
                 mutation_types: Iterable[MutationType],
                 mutation_chance_boundaries: Tuple[float, float] = (0.01, 0.2),
                 apply_elitism_options: Iterable[bool] = (True, False),
                 **optional_params: Union[Tuple[Union[float, int], Union[float, int]], int, float]) -> None:
        """
        Definition of Evolutionary Algorithm Adaptation problem.

        :param adaptation_type: Determines how to assess effectiveness of Evolutionary Algorithms.
        :param population_size_boundaries: Tuple with minimal and maximal population size of Evolutionary Algorithm.
        :param selection_types: Possible selection types to be chosen by Evolutionary Algorithm.
        :param crossover_types: Possible crossover types to be chosen by Evolutionary Algorithm.
        :param mutation_types: Possible mutation types to be chosen by Evolutionary Algorithm.
        :param mutation_chance_boundaries: Tuple with minimal and maximal mutation chance of Evolutionary Algorithm.
        :param apply_elitism_options: Possible elitism values.
        :param optional_params: Configuration of other parameters such as:
            - solutions_percentile - determines number of solutions (percentile of population size) to be taken
                into consideration when assessment of LowerAdaptiveEvolutionaryAlgorithm is performed.
                Float value: 0 < solutions_percentile < 1
                Applicable only if adaptation_type == AdaptationType.BestSolutionsPercentile.
            - solutions_number - determines number of solutions to be taken into consideration when assessment of
                LowerAdaptiveEvolutionaryAlgorithm is performed.
                Int value: 2 < solutions_number <= population_size_boundaries[0]
                Applicable only if adaptation_type == AdaptationType.BestSolutions.
        """
        # pylint: disable=too-many-locals
        if not isinstance(adaptation_type, AdaptationType):
            raise TypeError(f"Value of 'adaptation_type' parameter is not AdaptationType type. "
                            f"Actual value: '{adaptation_type}'.")
        if not isinstance(population_size_boundaries, tuple):
            raise TypeError(f"Value of 'population_size_boundaries' parameter is not tuple type. "
                            f"Actual value: '{population_size_boundaries}'.")
        if len(population_size_boundaries) != 2 or not isinstance(population_size_boundaries[0], int) \
                or not isinstance(population_size_boundaries[1], int):
            raise ValueError(f"Parameter 'population_size_boundaries' has invalid value. "
                             f"Expected two element tuple with int values. Actual value: {population_size_boundaries}.")
        if population_size_boundaries[0] < MIN_POPULATION_SIZE or population_size_boundaries[1] > MAX_POPULATION_SIZE:
            raise ValueError(f"Parameter 'population_size_boundaries' has invalid value. "
                             f"Expected two element tuple with population_size_boundaries[0] >= {MIN_POPULATION_SIZE} "
                             f"and population_size_boundaries[1] <= {MAX_POPULATION_SIZE}. "
                             f"Actual value: {population_size_boundaries}.")
        if not isinstance(selection_types, Iterable):
            raise TypeError(f"Value of 'selection_types' parameter is not Iterable type. "
                            f"Actual value: '{selection_types}'.")
        if any([not isinstance(value, SelectionType) for value in selection_types]):
            raise ValueError(f"Parameter 'selection_types' has invalid value. "
                             f"Expected iterable with values of SelectionType. Actual value: {selection_types}.")
        if not isinstance(crossover_types, Iterable):
            raise TypeError(f"Value of 'crossover_types' parameter is not Iterable type. "
                            f"Actual value: '{crossover_types}'.")
        if any([not isinstance(value, CrossoverType) for value in crossover_types]):
            raise ValueError(f"Parameter 'crossover_types' has invalid value. "
                             f"Expected iterable with values of CrossoverType. Actual value: {crossover_types}.")
        if not isinstance(mutation_types, Iterable):
            raise TypeError(f"Value of 'mutation_types' parameter is not Iterable type. "
                            f"Actual value: '{mutation_types}'.")
        if any([not isinstance(value, MutationType) for value in mutation_types]):
            raise ValueError(f"Parameter 'mutation_types' has invalid value. "
                             f"Expected iterable with values of MutationType. Actual value: {mutation_types}.")
        if not isinstance(mutation_chance_boundaries, tuple):
            raise TypeError(f"Value of 'mutation_chance_boundaries' parameter is not tuple type. "
                            f"Actual value: '{mutation_chance_boundaries}'.")
        if len(mutation_chance_boundaries) != 2 or not isinstance(mutation_chance_boundaries[0], float) \
                or not isinstance(mutation_chance_boundaries[1], float):
            raise ValueError(f"Parameter 'mutation_chance_boundaries' has invalid value. "
                             f"Expected two element tuple with float values. "
                             f"Actual value: {mutation_chance_boundaries}.")
        if mutation_chance_boundaries[0] < MIN_MUTATION_CHANCE or mutation_chance_boundaries[1] > MAX_MUTATION_CHANCE:
            raise ValueError(f"Parameter 'mutation_chance_boundaries' has invalid value. "
                             f"Expected two element tuple with mutation_chance_boundaries[0] >= {MIN_MUTATION_CHANCE} "
                             f"and mutation_chance_boundaries[1] <= {MAX_MUTATION_CHANCE}. "
                             f"Actual value: {mutation_chance_boundaries}.")
        if not isinstance(apply_elitism_options, Iterable):
            raise TypeError(f"Value of 'apply_elitism_options' parameter is not Iterable type. "
                            f"Actual value: '{apply_elitism_options}'.")
        if not set(apply_elitism_options).issubset({True, False}):
            raise ValueError(f"Parameter 'apply_elitism_options' has invalid value. "
                             f"Expected iterable with bool values. Actual value: {apply_elitism_options}.")
        decision_variables = OrderedDict(
            population_size=DiscreteVariable(min_value=population_size_boundaries[0],
                                             max_value=population_size_boundaries[1], step=2),
            selection_type=ChoiceVariable(possible_values=selection_types),
            crossover_type=ChoiceVariable(possible_values=crossover_types),
            mutation_type=ChoiceVariable(possible_values=mutation_types),
            mutation_chance=FloatVariable(min_value=mutation_chance_boundaries[0],
                                          max_value=mutation_chance_boundaries[1]),
            apply_elitism=ChoiceVariable(possible_values=apply_elitism_options)
        )
        _solutions_percentile = optional_params.get("solutions_percentile", DEFAULT_SOLUTIONS_PERCENTILE) \
            if adaptation_type == AdaptationType.BestSolutionsPercentile else None
        if isinstance(_solutions_percentile, float) and not (0 < _solutions_percentile < 1):
            raise ValueError(f"Value of 'solutions_percentile' is not in range: 0 < solutions_percentile < 1. "
                             f"solutions_percentile = {_solutions_percentile}")
        _solutions_number = optional_params.get("solutions_number", DEFAULT_SOLUTIONS_NUMBER) \
            if adaptation_type == AdaptationType.BestSolutions else None
        if isinstance(_solutions_number, int) and not (2 < _solutions_number < population_size_boundaries[0]):
            raise ValueError(f"Value of 'solutions_number' is not in range: "
                             f"2 < solutions_number < {population_size_boundaries[0]}. "
                             f"solutions_number = {_solutions_number}")
        _objective_function = self._create_objective_function(adaptation_type=adaptation_type,
                                                              percentile=_solutions_percentile,
                                                              number=_solutions_number)
        super().__init__(decision_variables=decision_variables,
                         constraints={},  # no constraints are needed
                         penalty_function=lambda **x: 0,  # penalty function is not used here
                         objective_function=_objective_function,
                         optimization_type=OptimizationType.Maximize)  # default value (updated later on)
        # additional parameters
        _min_group_size, _max_group_size = \
            optional_params.get("tournament_group_size", SELECTION_ADDITIONAL_PARAMS_LIMITS["tournament_group_size"])
        _min_roulette_bias, _max_roulette_bias = \
            optional_params.get("roulette_bias", SELECTION_ADDITIONAL_PARAMS_LIMITS["roulette_bias"])
        _min_ranking_bias, _max_ranking_bias = \
            optional_params.get("ranking_bias", SELECTION_ADDITIONAL_PARAMS_LIMITS["ranking_bias"])
        # temporary values for _max_crossover_points, _max_crossover_pattern and _max_mutation_points - updated later on
        _min_crossover_points, _max_crossover_points = optional_params.get("crossover_points_number", [2, 3])
        _min_crossover_pattern, _max_crossover_pattern = 0, 1
        _min_mutation_points, _max_mutation_points = optional_params.get("mutation_points_number", [2, 3])
        self.additional_decision_variable = {
            # selection
            "tournament_group_size":
                IntegerVariable(min_value=_min_group_size, max_value=_max_group_size),
            "roulette_bias": FloatVariable(min_value=_min_roulette_bias, max_value=_max_roulette_bias),
            "ranking_bias": FloatVariable(min_value=_min_ranking_bias, max_value=_max_ranking_bias),
            # crossover
            "crossover_points_number":
                IntegerVariable(min_value=_min_crossover_points, max_value=_max_crossover_points),
            "crossover_patter": IntegerVariable(min_value=_min_crossover_pattern, max_value=_max_crossover_pattern),
            # mutation
            "mutation_points_number":
                IntegerVariable(min_value=_min_mutation_points, max_value=_max_mutation_points),
        }

    @staticmethod
    def _create_objective_function(adaptation_type: AdaptationType,
                                   percentile: Optional[float],
                                   number: Optional[int]) -> Callable:
        """
        Creates objective function for adaptation problem.

        :param adaptation_type: Determines how to assess effectiveness of Evolutionary Algorithms.
        :param percentile: Relevant for 'BestSolutionsPercentile', determines percentile of solutions to be considered.
        :param number: Relevant for 'BestSolutions', determines number of solutions to be considered.

        :return: Objective function for adaptation problem.
        """
        if adaptation_type == AdaptationType.BestSolution:
            def adaptation_objective_function(best_solution: AbstractSolution, **_: Any) -> float:
                return best_solution.get_objective_value_with_penalty()
        elif adaptation_type == AdaptationType.BestSolutions:
            if not isinstance(number, int):
                raise TypeError

            def adaptation_objective_function(solutions: list, **_: Any) -> float:
                return sum([solution.get_objective_value_with_penalty()
                            for solution in sorted(solutions, reverse=True)[:number]])
        elif adaptation_type == AdaptationType.BestSolutionsPercentile:
            if not isinstance(percentile, float):
                raise TypeError

            def adaptation_objective_function(solutions: list, population_size: int, **_: Any) -> float:
                considered_number = int(population_size*percentile // 100)
                considered_number = max(considered_number, 1)
                return sum([solution.get_objective_value_with_penalty()
                            for solution in sorted(solutions, reverse=True)[:considered_number]])
        else:
            raise NotImplemented(f"Value of 'adaptation_type' parameter is not supported. Value: {adaptation_type}")
        return adaptation_objective_function


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
        # TODO: protect from problem.variables_number < 4 and adaptation_problem uses 'crossover_points_number'
        #  or 'mutation_points_number'
        adaptation_problem.optimization_type = problem.optimization_type
        adaptation_problem.additional_decision_variable["crossover_points_number"].max_value \
            = min(adaptation_problem.additional_decision_variable["crossover_points_number"].max_value,
                  problem.variables_number // 2)
        adaptation_problem.additional_decision_variable["crossover_patter"].max_value \
            = (1 << problem.variables_number) - 1
        adaptation_problem.additional_decision_variable["mutation_points_number"].max_value \
            = min(adaptation_problem.additional_decision_variable["mutation_points_number"].max_value,
                  problem.variables_number // 2)
        self.adaptation_problem = adaptation_problem
        super().__init__(problem=problem, stop_conditions=stop_conditions, population_size=population_size,
                         selection_type=selection_type, crossover_type=crossover_type, mutation_type=mutation_type,
                         mutation_chance=mutation_chance, apply_elitism=False, logger=logger, **other_params)

        class AdaptiveAESolution(LowerAdaptiveEvolutionaryAlgorithm):
            """Solution class for given evolutionary algorithm adaptation problem."""

            optimization_problem = adaptation_problem

        self.SolutionClass = AdaptiveAESolution

    # todo: a few methods must be updated
