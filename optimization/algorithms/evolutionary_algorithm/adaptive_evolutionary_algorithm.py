"""
Adaptive evolutionary algorithms.

Evolutionary algorithms that performs two level optimization of:
 - optimization problem - main problem to solve
 - adaptation problem - searching for optimal settings of evolutionary settings
"""

__all__ = ["AdaptationType", "EvolutionaryAlgorithmAdaptationProblem", "AdaptiveEvolutionaryAlgorithm"]


from typing import Any, Union, Optional, Tuple, Iterable, Dict, Callable
from typing import OrderedDict as OrderedDictTyping
from enum import Enum
from abc import abstractmethod
from collections import OrderedDict
from copy import deepcopy

from ...utilities import shuffled
from .evolutionary_algorithm import EvolutionaryAlgorithm
from ...problem import OptimizationProblem, AbstractSolution, OptimizationType, \
    DecisionVariable, IntegerVariable, DiscreteVariable, FloatVariable, ChoiceVariable
from ...stop_conditions import StopConditions
from ...logging import AbstractLogger
from .selection import SelectionType, SELECTION_ADDITIONAL_PARAMS_LIMITS, SELECTION_ADDITIONAL_PARAMS
from .crossover import CrossoverType, CROSSOVER_ADDITIONAL_PARAMS, ChildrenValuesTyping
from .mutation import MutationType, MUTATION_ADDITIONAL_PARAMS
from .limits import MIN_EA_POPULATION_SIZE, MAX_EA_POPULATION_SIZE, MIN_EA_MUTATION_CHANCE, MAX_EA_MUTATION_CHANCE
from .defaults import DEFAULT_SOLUTIONS_NUMBER, DEFAULT_SOLUTIONS_PERCENTILE


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

    MIN_POPULATION_SIZE: int = MIN_EA_POPULATION_SIZE
    MAX_POPULATION_SIZE: int = MAX_EA_POPULATION_SIZE
    MIN_MUTATION_CHANCE: float = MIN_EA_MUTATION_CHANCE
    MAX_MUTATION_CHANCE: float = MAX_EA_MUTATION_CHANCE

    def __init__(self,
                 adaptation_type: AdaptationType,
                 population_size_boundaries: Tuple[int, int],
                 selection_types: Iterable[SelectionType],
                 crossover_types: Iterable[CrossoverType],
                 mutation_types: Iterable[MutationType],
                 mutation_chance_boundaries: Tuple[float, float] = (MIN_MUTATION_CHANCE, MAX_MUTATION_CHANCE),
                 apply_elitism_options: Iterable[bool] = (True, False),
                 **optional_params: Any) -> None:
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
            - :param solutions_percentile: float - determines number of solutions (percentile of population size)
                to be taken into consideration when assessment of LowerAdaptiveEvolutionaryAlgorithm is performed.
                Float value: 0 < solutions_percentile < 1
                Applicable only if adaptation_type == AdaptationType.BestSolutionsPercentile.
            - :param solutions_number: int - determines number of solutions to be taken into consideration when
                assessment of LowerAdaptiveEvolutionaryAlgorithm is performed.
                Int value: 2 < solutions_number <= population_size_boundaries[0]
                Applicable only if adaptation_type == AdaptationType.BestSolutions.
            - :param min_tournament_group_size: int - determines minimal group size for tournament selection
                Default value is used if not provided.
            - :param max_tournament_group_size: int - determines maximal group size for tournament selection
                Default value is used if not provided.
            - :param min_roulette_bias: float - determines minimal value of roulette selection bias
                Default value is used if not provided.
            - :param max_roulette_bias: float - determines maximal value of roulette selection bias
                Default value is used if not provided.
            - :param min_ranking_bias: float - determines minimal value of ranking selection bias
                Default value is used if not provided.
            - :param max_ranking_bias: float - determines maximal value of ranking selection bias
                Default value is used if not provided.
        """
        self._validate_mandatory_parameters(adaptation_type=adaptation_type,
                                            population_size_boundaries=population_size_boundaries,
                                            selection_types=selection_types,
                                            crossover_types=crossover_types,
                                            mutation_types=mutation_types,
                                            mutation_chance_boundaries=mutation_chance_boundaries,
                                            apply_elitism_options=apply_elitism_options)
        decision_variables = self._get_main_decision_variables(population_size_boundaries=population_size_boundaries,
                                                               selection_types=selection_types,
                                                               crossover_types=crossover_types,
                                                               mutation_types=mutation_types,
                                                               mutation_chance_boundaries=mutation_chance_boundaries,
                                                               apply_elitism_options=apply_elitism_options)
        _solutions_percentile = optional_params.pop("solutions_percentile", DEFAULT_SOLUTIONS_PERCENTILE) \
            if adaptation_type == AdaptationType.BestSolutionsPercentile else None
        _solutions_number = optional_params.pop("solutions_number", DEFAULT_SOLUTIONS_NUMBER) \
            if adaptation_type == AdaptationType.BestSolutions else None
        _objective_function = self._get_objective_function(adaptation_type=adaptation_type,
                                                           solutions_percentile=_solutions_percentile,
                                                           solutions_number=_solutions_number,
                                                           population_size_boundaries=population_size_boundaries)
        super().__init__(decision_variables=decision_variables,
                         constraints=self._get_constraints(),
                         penalty_function=self._get_penalty_function(),
                         objective_function=_objective_function,
                         optimization_type=OptimizationType.Maximize)  # default value (updated later on)
        self.additional_decision_variable = self._get_additional_decision_variables(**optional_params)

    def _validate_mandatory_parameters(self,
                                       adaptation_type: AdaptationType,
                                       population_size_boundaries: Tuple[int, int],
                                       selection_types: Iterable[SelectionType],
                                       crossover_types: Iterable[CrossoverType],
                                       mutation_types: Iterable[MutationType],
                                       mutation_chance_boundaries: Tuple[float, float],
                                       apply_elitism_options: Iterable[bool]) -> None:
        """
        Validates all mandatory parameters of __init__ method.

        :param adaptation_type: Determines how to assess effectiveness of Evolutionary Algorithms.
        :param population_size_boundaries: Tuple with minimal and maximal population size of Evolutionary Algorithm.
        :param selection_types: Possible selection types to be chosen by Evolutionary Algorithm.
        :param crossover_types: Possible crossover types to be chosen by Evolutionary Algorithm.
        :param mutation_types: Possible mutation types to be chosen by Evolutionary Algorithm.
        :param mutation_chance_boundaries: Tuple with minimal and maximal mutation chance of Evolutionary Algorithm.
        :param apply_elitism_options: Possible elitism values.

        :return: None
        """
        self._validate_adaptation_type(adaptation_type=adaptation_type)
        self._validate_population_size_boundaries(population_size_boundaries=population_size_boundaries)
        self._validate_selection_types(selection_types=selection_types)
        self._validate_crossover_types(crossover_types=crossover_types)
        self._validate_mutation_types(mutation_types=mutation_types)
        self._validate_mutation_chance_boundaries(mutation_chance_boundaries=mutation_chance_boundaries)
        self._validate_apply_elitism_options(apply_elitism_options=apply_elitism_options)

    @staticmethod
    def _validate_adaptation_type(adaptation_type: AdaptationType) -> None:
        """
        Validates value of 'adaptation_type' parameter.

        :param adaptation_type: Determines how to assess effectiveness of Evolutionary Algorithms.

        :raise TypeError: Parameter 'adaptation_type' is not AdaptationType.
        :return: None
        """
        if not isinstance(adaptation_type, AdaptationType):
            raise TypeError(f"Value of 'adaptation_type' parameter is not AdaptationType type. "
                            f"Actual value: '{adaptation_type}'.")

    def _validate_population_size_boundaries(self, population_size_boundaries: Tuple[int, int]) -> None:
        """
        Validates value of 'population_size_boundaries' parameter.

        :param population_size_boundaries: Tuple with minimal and maximal population size of Evolutionary Algorithm.

        :raise TypeError: Parameter 'population_size_boundaries' is not tuple type.
        :raise ValueError: Parameter 'population_size_boundaries' does not consists of two even integer values that
            satisfies inequality:
            MIN_POPULATION_SIZE <= population_size_boundaries[0] <= population_size_boundaries[1] <= MAX_POPULATION_SIZE
        :return: None
        """
        if not isinstance(population_size_boundaries, tuple):
            raise TypeError(f"Value of 'population_size_boundaries' parameter is not tuple type. "
                            f"Actual value: '{population_size_boundaries}'.")
        if len(population_size_boundaries) != 2 or not isinstance(population_size_boundaries[0], int) \
                or not isinstance(population_size_boundaries[1], int):
            raise ValueError(f"Parameter 'population_size_boundaries' has invalid value. "
                             f"Expected two element tuple with int values. Actual value: {population_size_boundaries}.")
        if not self.MIN_POPULATION_SIZE <= population_size_boundaries[0] <= \
                population_size_boundaries[1] <= self.MAX_POPULATION_SIZE:
            raise ValueError(f"Parameter 'population_size_boundaries' has invalid value. "
                             f"Expected two element tuple with "
                             f"{self.MIN_POPULATION_SIZE} <= population_size_boundaries[0] <= "
                             f"population_size_boundaries[1] <= {self.MAX_POPULATION_SIZE}. "
                             f"Actual value: {population_size_boundaries}.")
        if population_size_boundaries[0] & 1 or population_size_boundaries[1] & 1:
            raise ValueError(f"Both limits (min and max) of 'population_size_boundaries' parameters must be even. "
                             f"Actual values: {population_size_boundaries}")

    @staticmethod
    def _validate_selection_types(selection_types: Iterable[SelectionType]) -> None:
        """
        Validates value of 'selection_types' parameter.

        :param selection_types: Possible selection types to be chosen by Evolutionary Algorithm.

        :raise TypeError: Parameter 'selection_types' is not iterable.
        :raise ValueError: Elements of 'selection_types' are not SelectionType.
        :return: None
        """
        if not isinstance(selection_types, Iterable):  # pylint: disable=isinstance-second-argument-not-valid-type
            raise TypeError(f"Value of 'selection_types' parameter is not Iterable type. "
                            f"Actual value: '{selection_types}'.")
        if any([not isinstance(value, SelectionType) for value in selection_types]):
            raise ValueError(f"Parameter 'selection_types' has invalid value. "
                             f"Expected iterable with values of SelectionType. Actual value: {selection_types}.")

    @staticmethod
    def _validate_crossover_types(crossover_types: Iterable[CrossoverType]) -> None:
        """
        Validates value of 'crossover_types' parameter.

        :param crossover_types: Possible crossover types to be chosen by Evolutionary Algorithm.

        :raise TypeError: Parameter 'crossover_types' is not iterable.
        :raise ValueError: Elements of 'crossover_types' are not CrossoverType.
        :return: None
        """
        if not isinstance(crossover_types, Iterable):  # pylint: disable=isinstance-second-argument-not-valid-type
            raise TypeError(f"Value of 'crossover_types' parameter is not Iterable type. "
                            f"Actual value: '{crossover_types}'.")
        if any([not isinstance(value, CrossoverType) for value in crossover_types]):
            raise ValueError(f"Parameter 'crossover_types' has invalid value. "
                             f"Expected iterable with values of CrossoverType. Actual value: {crossover_types}.")

    @staticmethod
    def _validate_mutation_types(mutation_types: Iterable[MutationType]) -> None:
        """
        Validates value of 'mutation_types' parameter.

        :param mutation_types: Possible mutation types to be chosen by Evolutionary Algorithm.

        :raise TypeError: Parameter 'mutation_types' is not iterable.
        :raise ValueError: Elements of 'mutation_types' are not MutationType.
        :return: None
        """
        if not isinstance(mutation_types, Iterable):  # pylint: disable=isinstance-second-argument-not-valid-type
            raise TypeError(f"Value of 'mutation_types' parameter is not Iterable type. "
                            f"Actual value: '{mutation_types}'.")
        if any([not isinstance(value, MutationType) for value in mutation_types]):
            raise ValueError(f"Parameter 'mutation_types' has invalid value. "
                             f"Expected iterable with values of MutationType. Actual value: {mutation_types}.")

    def _validate_mutation_chance_boundaries(self, mutation_chance_boundaries: Tuple[float, float]) -> None:
        """
        Validates value of 'mutation_chance_boundaries' parameter.

        :param mutation_chance_boundaries: Tuple with minimal and maximal mutation chance of Evolutionary Algorithm.

        :raise TypeError: Parameter 'mutation_chance_boundaries' is not tuple type.
        :raise ValueError: Parameter 'mutation_chance_boundaries' does not consists of two float values that
            satisfies inequality:
            MIN_MUTATION_CHANCE <= mutation_chance_boundaries[0] <= mutation_chance_boundaries[1] <= MAX_MUTATION_CHANCE
        :return: None
        """
        if not isinstance(mutation_chance_boundaries, tuple):
            raise TypeError(f"Value of 'mutation_chance_boundaries' parameter is not tuple type. "
                            f"Actual value: '{mutation_chance_boundaries}'.")
        if len(mutation_chance_boundaries) != 2 or not isinstance(mutation_chance_boundaries[0], float) \
                or not isinstance(mutation_chance_boundaries[1], float):
            raise ValueError(f"Parameter 'mutation_chance_boundaries' has invalid value. "
                             f"Expected two element tuple with float values. "
                             f"Actual value: {mutation_chance_boundaries}.")
        if not self.MIN_MUTATION_CHANCE <= mutation_chance_boundaries[0] <= \
                mutation_chance_boundaries[1] <= self.MAX_MUTATION_CHANCE:
            raise ValueError(f"Parameter 'mutation_chance_boundaries' has invalid value. "
                             f"Expected two element tuple with "
                             f"{self.MIN_MUTATION_CHANCE} <= mutation_chance_boundaries[0] <= "
                             f"mutation_chance_boundaries[1] <= {self.MAX_MUTATION_CHANCE}. "
                             f"Actual value: {mutation_chance_boundaries}.")

    @staticmethod
    def _validate_apply_elitism_options(apply_elitism_options: Iterable[bool]) -> None:
        """
        Validates value of 'apply_elitism_options' parameter.

        :param apply_elitism_options: Possible elitism values.

        :raise TypeError: Parameter 'apply_elitism_options' is not iterable.
        :raise ValueError: Elements of 'apply_elitism_options' are not bool type or no elements.
        :return: None
        """
        if not isinstance(apply_elitism_options, Iterable):  # pylint: disable=isinstance-second-argument-not-valid-type
            raise TypeError(f"Value of 'apply_elitism_options' parameter is not Iterable type. "
                            f"Actual value: '{apply_elitism_options}'.")
        if not (apply_elitism_options and set(apply_elitism_options).issubset({True, False})):
            raise ValueError(f"Parameter 'apply_elitism_options' has invalid value. "
                             f"Expected iterable with one or both bool values. Actual value: {apply_elitism_options}.")

    @staticmethod
    def _create_objective_function(adaptation_type: AdaptationType,
                                   solutions_percentile: Optional[float],
                                   solutions_number: Optional[int]) -> Callable:
        """
        Creates objective function for adaptation problem.

        :param adaptation_type: Determines how to assess effectiveness of Evolutionary Algorithms.
        :param solutions_percentile: Relevant for 'BestSolutionsPercentile' value of 'adaptation_type'.
            Determines percentile of solutions to be considered when assessing effectiveness of Evolutionary Algorithms.
        :param solutions_number: Relevant for 'BestSolutions' value of 'adaptation_type'.
            Determines number of solutions to be considered when assessing effectiveness of Evolutionary Algorithms.

        :raise NotImplementedError: Unsupported value of 'adaptation_type' is provided.
        :return: Objective function for adaptation problem.
        """
        if adaptation_type == AdaptationType.BestSolution:

            def adaptation_objective_function(best_solution: AbstractSolution, **_: Any) -> float:  # type: ignore
                return best_solution.get_objective_value_with_penalty()
        elif adaptation_type == AdaptationType.BestSolutions:

            def adaptation_objective_function(solutions: list, **_: Any) -> float:  # type: ignore
                return sum([solution.get_objective_value_with_penalty()
                            for solution in sorted(solutions, reverse=True)[:solutions_number]])
        elif adaptation_type == AdaptationType.BestSolutionsPercentile:

            def adaptation_objective_function(solutions: list, population_size: int, **_: Any) -> float:  # type: ignore
                considered_number = int(population_size*solutions_percentile // 100)  # type: ignore
                considered_number = max(considered_number, 1)
                return sum([solution.get_objective_value_with_penalty()
                            for solution in sorted(solutions, reverse=True)[:considered_number]])
        else:
            raise NotImplementedError(f"This value of 'adaptation_type' parameter is not supported. "
                                      f"Actual value: {adaptation_type}")
        return adaptation_objective_function

    def _get_objective_function(self,
                                adaptation_type: AdaptationType,
                                solutions_percentile: Optional[float],
                                solutions_number: Optional[int],
                                population_size_boundaries: Tuple[int, int]) -> Callable:
        """
        Provides objective function for Evolutionary Algorithm Adaptation problem.

        :param adaptation_type: Determines how to assess effectiveness of Evolutionary Algorithms.
        :param solutions_percentile: Relevant for 'BestSolutionsPercentile' value of 'adaptation_type'.
            Determines percentile of solutions to be considered when assessing effectiveness of Evolutionary Algorithms.
        :param solutions_number: Relevant for 'BestSolutions' value of 'adaptation_type'.
            Determines number of solutions to be considered when assessing effectiveness of Evolutionary Algorithms.
        :param population_size_boundaries: Tuple with minimal and maximal population size of Evolutionary Algorithm.

        :raise TypeError: Invalid type of 'solutions_percentile' or 'solutions_number' parameter was provided.
        :raise ValueError: Invalid value of 'solutions_percentile' or 'solutions_number' parameter was provided.
        :return: Objective function for adaptation problem.
        """
        if adaptation_type == AdaptationType.BestSolutionsPercentile:
            if not isinstance(solutions_percentile, float):
                raise TypeError(f"Value of 'solutions_percentile' is not float type for "
                                f"'BestSolutionsPercentile' adaptation type. "
                                f"Actual value: {solutions_percentile}")
            if not (0 < solutions_percentile <= 1):
                raise ValueError(f"Value of 'solutions_percentile' is not in range: 0 < solutions_percentile <= 1. "
                                 f"Actual value: {solutions_percentile}")
        elif adaptation_type == AdaptationType.BestSolutions:
            if not isinstance(solutions_number, int):
                raise TypeError(f"Value of 'solutions_number' is not int type for 'BestSolutions' adaptation type. "
                                f"Actual value: {solutions_number}")
            if not 2 <= solutions_number <= population_size_boundaries[0]:
                raise ValueError(f"Value of 'solutions_number' is not in range "
                                 f"2 <= solutions_number <= population_size_boundaries[0]. "
                                 f"Actual value: {solutions_number}")
        elif adaptation_type == AdaptationType.BestSolution:
            pass
        else:
            raise NotImplementedError(f"This value of 'adaptation_type' parameter is not supported.  "
                                      f"Actual value: {adaptation_type}")
        return self._create_objective_function(adaptation_type=adaptation_type,
                                               solutions_percentile=solutions_percentile,
                                               solutions_number=solutions_number)

    @staticmethod
    def _get_main_decision_variables(population_size_boundaries: Tuple[int, int],
                                     selection_types: Iterable[SelectionType],
                                     crossover_types: Iterable[CrossoverType],
                                     mutation_types: Iterable[MutationType],
                                     mutation_chance_boundaries: Tuple[float, float],
                                     apply_elitism_options: Iterable[bool]) \
            -> OrderedDictTyping[str, DecisionVariable]:  # type: ignore
        """
        Creates definition of main decision variables vector.

        :param population_size_boundaries: Tuple with minimal and maximal population size of Evolutionary Algorithm.
        :param selection_types: Possible selection types to be chosen by Evolutionary Algorithm.
        :param crossover_types: Possible crossover types to be chosen by Evolutionary Algorithm.
        :param mutation_types: Possible mutation types to be chosen by Evolutionary Algorithm.
        :param mutation_chance_boundaries: Tuple with minimal and maximal mutation chance of Evolutionary Algorithm.
        :param apply_elitism_options: Possible elitism values.

        :return: OrderedDict with main decision variables definitions:
            Keys: Names of decision variables.
            Values: Definition of decision variables.
        """
        return OrderedDict(
            population_size=DiscreteVariable(min_value=population_size_boundaries[0],
                                             max_value=population_size_boundaries[1],
                                             step=2),
            selection_type=ChoiceVariable(possible_values=selection_types),
            crossover_type=ChoiceVariable(possible_values=crossover_types),
            mutation_type=ChoiceVariable(possible_values=mutation_types),
            mutation_chance=FloatVariable(min_value=mutation_chance_boundaries[0],
                                          max_value=mutation_chance_boundaries[1]),
            apply_elitism=ChoiceVariable(possible_values=apply_elitism_options)
        )

    @staticmethod
    def _get_additional_decision_variables(**optional_params: Union[int, float]) -> Dict[str, DecisionVariable]:
        """
        Creates definition of additional decision variables vector.

        Note: Additional decision variables are relevant only for certain selection/crossover/mutation type.

        :param optional_params: Parameters that overwrites default configuration of additional decision variables.

        :raise ValueError: Any of provided values is not in acceptable range.
        :return: Dictionary with additional decision variables definitions:
            Keys: Names of decision variables.
            Values: Definition of decision variables.
        """
        # pylint: disable=too-many-locals
        # get default values
        _default_min_group_size, _default_max_group_size = SELECTION_ADDITIONAL_PARAMS_LIMITS["tournament_group_size"]
        _default_min_roulette_bias, _default_max_roulette_bias = SELECTION_ADDITIONAL_PARAMS_LIMITS["roulette_bias"]
        _default_min_ranking_bias, _default_max_ranking_bias = SELECTION_ADDITIONAL_PARAMS_LIMITS["ranking_bias"]
        # set provided values (use default if not provided)
        _min_group_size: int = optional_params.pop("min_tournament_group_size", _default_min_group_size)  # type: ignore
        _max_group_size: int = optional_params.pop("max_tournament_group_size", _default_max_group_size)  # type: ignore
        _min_roulette_bias = optional_params.pop("min_roulette_bias", _default_min_roulette_bias)
        _max_roulette_bias = optional_params.pop("max_roulette_bias", _default_max_roulette_bias)
        _min_ranking_bias = optional_params.pop("min_ranking_bias", _default_min_ranking_bias)
        _max_ranking_bias = optional_params.pop("max_ranking_bias", _default_max_ranking_bias)
        # validate values
        if not _default_min_group_size <= _min_group_size <= _max_group_size <= _default_max_group_size:
            raise ValueError(f"Provided tournament groups size limits are not in range. "
                             f"Expected: {_default_min_group_size} <= min_tournament_group_size <= "
                             f"max_tournament_group_size <= {_default_max_group_size}"
                             f"Actual values: min_tournament_group_size={_min_group_size}, "
                             f"max_tournament_group_size={_max_group_size}")
        if not _default_min_roulette_bias <= _min_roulette_bias <= _max_roulette_bias <= _default_max_roulette_bias:
            raise ValueError(f"Provided roulette bias limits are not in range. "
                             f"Expected: {_default_min_roulette_bias} <= min_roulette_bias <= "
                             f"max_roulette_bias <= {_default_max_roulette_bias}"
                             f"Actual values: min_roulette_bias={_min_roulette_bias}, "
                             f"max_roulette_bias={_max_roulette_bias}")
        if not _default_min_ranking_bias <= _min_ranking_bias <= _max_ranking_bias <= _default_max_ranking_bias:
            raise ValueError(f"Provided ranking bias limits are not in range. "
                             f"Expected: {_default_min_ranking_bias} <= min_ranking_bias <= "
                             f"max_ranking_bias <= {_default_max_ranking_bias}"
                             f"Actual values: min_ranking_bias={_min_ranking_bias}, "
                             f"max_ranking_bias={_max_ranking_bias}")
        if optional_params:
            raise ValueError(f"Unexpected value(s) was/were provided to 'optional_params': {optional_params}")
        # temporary values for _max_crossover_points, _max_crossover_pattern and _max_mutation_points - updated later on
        _min_crossover_points, _max_crossover_points = 2, 3
        _min_crossover_pattern, _max_crossover_pattern = 0, 1
        _min_mutation_points, _max_mutation_points = 2, 3
        return {
            # selection
            "tournament_group_size": IntegerVariable(min_value=_min_group_size, max_value=_max_group_size),
            "roulette_bias": FloatVariable(min_value=_min_roulette_bias, max_value=_max_roulette_bias),
            "ranking_bias": FloatVariable(min_value=_min_ranking_bias, max_value=_max_ranking_bias),
            # crossover
            "crossover_points_number":
                IntegerVariable(min_value=_min_crossover_points, max_value=_max_crossover_points),
            "crossover_patter": IntegerVariable(min_value=_min_crossover_pattern, max_value=_max_crossover_pattern),
            # mutation
            "mutation_points_number": IntegerVariable(min_value=_min_mutation_points, max_value=_max_mutation_points),
        }

    @staticmethod
    def _get_constraints() -> Dict[str, Callable]:
        """
        Creates constraints for adaptation problem.

        :return: Dictionary with constraints.
        """
        return {}  # no constraints are needed

    @staticmethod
    def _get_penalty_function() -> Callable:
        """
        Creates penalty function for adaptation problem.

        :return: Penalty function.
        """
        return lambda **decision_variables_values: 0  # penalty function is not used here


class LowerAdaptiveEvolutionaryAlgorithm(EvolutionaryAlgorithm, AbstractSolution):
    """
    Definition of Lower (Slave) Adaptive Evolutionary Algorithm.

    These algorithms (there are many of them during optimization process) search for optimal solution of
    main optimization problem while Upper (Master) Adaptive Evolutionary Algorithm optimizes settings (configurations)
    of Lower Adaptive Evolutionary Algorithms.
    """

    @property
    @abstractmethod
    def optimization_problem(self) -> EvolutionaryAlgorithmAdaptationProblem:
        """Evolutionary Algorithm adaptation problem for which this class is able to create solutions (as objects)."""
        ...

    def __init__(self,  # pylint: disable=too-many-arguments
                 upper_iteration: int,
                 index: int,
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
        :param other_params: Parameter related to initial algorithm conditions, selected selection, crossover
            or mutation type such as:
            - :param initial_population - (optional) starting population to be set.
            - :param tournament_group_size: int - determines group size used in tournament
                and double tournament selections
            - :param roulette_bias: float - bias towards promoting better adopted individuals in roulette selection
            - :param ranking_bias: float - represents selection pressure (the higher the value the higher the pressure)
                in ranking selection
            - :param crossover_points_number: int - number of crossover points to use in multipoint crossover
            - :param crossover_pattern: int - pattern of crossover to be used in adaptive crossover
            - :param mutation_points_number: int - number of mutation points to be used in multipoint mutation.
        """
        self.upper_iteration = upper_iteration
        self.index = index
        # init as evolutionary algorithm
        initial_population = other_params.pop("initial_population", [])
        EvolutionaryAlgorithm.__init__(self_ea=self, problem=problem, stop_conditions=stop_conditions,
                                       population_size=population_size, selection_type=selection_type,
                                       crossover_type=crossover_type, mutation_type=mutation_type,
                                       mutation_chance=mutation_chance, apply_elitism=apply_elitism, logger=logger,
                                       **other_params)
        self._population = initial_population
        # init as solution
        AbstractSolution.__init__(self_solution=self, population_size=population_size, selection_type=selection_type,
                                  crossover_type=crossover_type, mutation_type=mutation_type,
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
        data = EvolutionaryAlgorithm.get_log_data(self_ea=self)
        data.update(AbstractSolution.get_log_data(self_solution=self))
        data["additional_decision_variables_values"] = self.additional_decision_variables_values
        return data


class AdaptiveEvolutionaryAlgorithm(EvolutionaryAlgorithm):
    """
    Adaptive Evolutionary Algorithm definition.

    This algorithm uses mechanisms inspired by biological evolution.
    It indirectly searches of optimal solution (_LowerAdaptiveEvolutionaryAlgorithm does that), while optimization
    optimization process effectiveness (it optimizes settings of _LowerAdaptiveEvolutionaryAlgorithm).
    """

    ParentsTyping = Tuple[LowerAdaptiveEvolutionaryAlgorithm, LowerAdaptiveEvolutionaryAlgorithm]

    def __init__(self,  # pylint: disable=too-many-arguments
                 problem: OptimizationProblem,
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
        :param other_params: Parameter related to selected selection, crossover and mutation type further
            described in parent class.
        """
        if not isinstance(adaptation_problem, EvolutionaryAlgorithmAdaptationProblem):
            raise TypeError(f"Value of 'adaptation_problem' parameter is not EvolutionaryAlgorithmAdaptationProblem "
                            f"type. Actual value: '{adaptation_problem}'.")
        adaptation_problem.optimization_type = problem.optimization_type
        dv_crossover_points_number = adaptation_problem.additional_decision_variable["crossover_points_number"]
        dv_crossover_points_number.max_value = problem.variables_number // 2  # type: ignore
        dv_crossover_patter = adaptation_problem.additional_decision_variable["crossover_patter"]
        dv_crossover_patter.max_value = (1 << problem.variables_number) - 1  # type: ignore
        dv_mutation_points_number = adaptation_problem.additional_decision_variable["mutation_points_number"]
        dv_mutation_points_number.max_value = problem.variables_number // 2  # type: ignore
        self.adaptation_problem = adaptation_problem
        super().__init__(problem=problem, stop_conditions=stop_conditions, population_size=population_size,
                         selection_type=selection_type, crossover_type=crossover_type, mutation_type=mutation_type,
                         mutation_chance=mutation_chance, apply_elitism=False, logger=logger, **other_params)

        class AdaptiveAESolution(LowerAdaptiveEvolutionaryAlgorithm):
            """Solution class for given evolutionary algorithm adaptation problem."""

            optimization_problem = adaptation_problem

        self.SolutionClass = AdaptiveAESolution  # type: ignore

    @staticmethod
    def _assign_additional_params_to_children(parents: ParentsTyping,
                                              child_1_main_values: OrderedDictTyping,  # type: ignore
                                              child_2_main_values: OrderedDictTyping) -> Tuple[dict, dict]:  # noqa
        """
        Performs crossover on additional decision variables.

        Additional parameters values are selected basing on parents genes and already assigned main decision variables.

        :param parents: Solution objects with values of selected parents.
        :param child_1_main_values: Already assigned (to first child) main decision variables.
        :param child_2_main_values: Already assigned (to second child) main decision variables.

        :return: Tuple with additional decision variables assignment for following children.
        """
        child_1_side_values, child_2_side_values = {}, {}
        for main_function, additional_params in (("selection_type", SELECTION_ADDITIONAL_PARAMS),
                                                 ("crossover_type", CROSSOVER_ADDITIONAL_PARAMS),
                                                 ("mutation_type", MUTATION_ADDITIONAL_PARAMS)):
            val1, val2 = child_1_main_values[main_function], child_2_main_values[main_function]  # type: ignore
            child_1_params, child_2_params = additional_params[val1.value], additional_params[val2.value]
            if not child_1_params and not child_2_params:
                continue
            if child_1_params == child_2_params:
                for param in child_1_params:
                    child_1_side_values[param], child_2_side_values[param] = \
                        shuffled([parents[0].additional_decision_variables_values[param],
                                  parents[1].additional_decision_variables_values[param]])
                continue
            for param in child_1_params:
                if param in parents[0].additional_decision_variables_values:
                    child_1_side_values[param] = parents[0].additional_decision_variables_values[param]
                elif param in parents[1].additional_decision_variables_values:
                    child_1_side_values[param] = parents[1].additional_decision_variables_values[param]
                else:
                    raise ValueError("Unexpected additional decision variable value of child 1.")
            for param in child_2_params:
                if param in parents[0].additional_decision_variables_values:
                    child_2_side_values[param] = parents[0].additional_decision_variables_values[param]
                elif param in parents[1].additional_decision_variables_values:
                    child_2_side_values[param] = parents[1].additional_decision_variables_values[param]
                else:
                    raise ValueError("Unexpected additional decision variable value of child 2.")
        return child_1_side_values, child_2_side_values

    def _update_additional_params(self,
                                  values_before_main_mutation: OrderedDictTyping,  # type: ignore
                                  values_after_main_mutation: OrderedDictTyping) -> OrderedDictTyping:  # type: ignore
        """
        Adjusts additional decision variable values.

        If any of main function (selection_type, crossover_type, mutation_type) was changed changed (mutated), then
        side parameters must be adjusted.

        :Note Example:
        After crossover, SelectionType.Tournament was selection type assigned.
        It required 'tournament_group_size' as additional parameter.
        After mutation selection_type was switched from SelectionType.Tournament to SelectionType.Ranking.
        Ranking selection requires 'ranking_bias' as additional parameter.
        It is necessary to remove 'tournament_group_size' value and assign some 'ranking_bias' value
        to additional params.


        :param values_before_main_mutation: All decision variables values before mutation.
        :param values_after_main_mutation: Decision variables values after mutation of main parameters.

        :return: Final version of all decision variables (both main and additional) after mutation.
        """
        final_values = deepcopy(values_after_main_mutation)
        for main_function, additional_params in (("selection_type", SELECTION_ADDITIONAL_PARAMS),
                                                 ("crossover_type", CROSSOVER_ADDITIONAL_PARAMS),
                                                 ("mutation_type", MUTATION_ADDITIONAL_PARAMS)):
            value_before_mutation = values_before_main_mutation[main_function]  # type: ignore
            value_after_mutation = values_after_main_mutation[main_function]  # type: ignore
            if value_before_mutation != value_after_mutation:
                old_side_values = set(additional_params[value_before_mutation])
                required_side_values = set(additional_params[value_before_mutation])
                for var_name_no_longer_needed in old_side_values.difference(required_side_values):
                    final_values.pop(var_name_no_longer_needed)  # type: ignore
                for var_name_new in required_side_values.difference(old_side_values):
                    value = self.adaptation_problem.additional_decision_variable[var_name_new].generate_random_value()
                    final_values[var_name_new] = value  # type: ignore
        return final_values

    def _perform_crossover(self, parents: ParentsTyping) -> ChildrenValuesTyping:  # type: ignore
        """
        Performs crossover of two parents.

        :param parents: Solution objects with values of selected parents.

        :return: Values of children decision variables (genes).
        """
        child_1_main_values, child_2_main_values = super()._perform_crossover(parents=parents)
        child_1_side_values, child_2_side_values = self._assign_additional_params_to_children(
            parents=parents, child_1_main_values=child_1_main_values, child_2_main_values=child_2_main_values)
        return OrderedDict(**child_1_main_values, **child_1_side_values), \
            OrderedDict(**child_2_main_values, **child_2_side_values)

    def _perform_mutation(self, individual_values: OrderedDictTyping[str, Any]) -> None:  # type: ignore
        """
        Performs mutation on individual decision variables values.

        :param individual_values: Individual values to be mutated.

        :return: None
        """
        initial_value = deepcopy(individual_values)
        super()._perform_mutation(individual_values=individual_values)
        final_values = self._update_additional_params(values_before_main_mutation=initial_value,
                                                      values_after_main_mutation=individual_values)
        individual_values.clear()  # type: ignore
        individual_values.update(final_values)  # type: ignore
