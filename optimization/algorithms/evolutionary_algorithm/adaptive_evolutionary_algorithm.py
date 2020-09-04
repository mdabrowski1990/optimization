"""
Adaptive evolutionary algorithms.

Evolutionary algorithms that performs two level optimization of:
 - optimization problem - main problem to solve
 - adaptation problem - searching for optimal settings of evolutionary settings
"""

__all__ = ["AdaptiveEvolutionaryAlgorithm"]


from typing import Any, Union, Optional, Tuple, Iterable
from enum import Enum

from .evolutionary_algorithm import EvolutionaryAlgorithm
from ...problem import OptimizationProblem, AbstractSolution
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
        super(EvolutionaryAlgorithmAdaptationProblem, self).__init__()


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
        self.adaptation_problem = adaptation_problem
        super(AdaptiveEvolutionaryAlgorithm, self).__init__(problem=problem, stop_conditions=stop_conditions,
                                                            population_size=population_size,
                                                            selection_type=selection_type,
                                                            crossover_type=crossover_type, mutation_type=mutation_type,
                                                            mutation_chance=mutation_chance, apply_elitism=False,
                                                            logger=logger, **other_params)
        # TODO: update SolutionClass (_LowerAdaptiveEvolutionaryAlgorithm)


class LowerAdaptiveEvolutionaryAlgorithm(EvolutionaryAlgorithm, AbstractSolution):
    """
    Definition of lower level adaptive evolutionary algorithm.

    These algorithms (there are many of them during optimization process) search for optimal solution of
    main optimization problem.
    """

    def __init__(self, upper_iteration: int, index: int, **kwargs: Any) -> None:
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
        initial_population = kwargs.pop("initial_population", [])
        super(LowerAdaptiveEvolutionaryAlgorithm, self).__init__(**kwargs)
        self._population = initial_population

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
