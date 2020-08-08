"""Random optimization algorithm."""

__all__ = ["RandomAlgorithm"]


from typing import Optional, Dict, Union

from .abstract_algorithm import AbstractOptimizationAlgorithm
from ..problem import OptimizationProblem, AbstractSolution
from ..stop_conditions import StopConditions
from ..logging import AbstractLogger


class RandomAlgorithm(AbstractOptimizationAlgorithm):
    """
    Random optimization algorithm implementation.

    This algorithm is very ineffective and it is advised to use it only for research (comparison with other algorithms).
    """

    def __init__(self, problem: OptimizationProblem,
                 stop_conditions: StopConditions,
                 population_size: int = 1000,
                 logger: Optional[AbstractLogger] = None) -> None:
        """
        Configuration of Random Algorithm.

        :param problem: Optimization problem to be solved by the algorithm.
        :param stop_conditions: Conditions when optimization algorithm shall be stopped.
        :param population_size: Number of solutions generated in one iteration.
            Note: Setting big value might cause big memory usage and delay in stopping the optimization process.
        :param logger: Logger used for optimization process recording.
        """
        if not isinstance(population_size, int):
            raise TypeError(f"Parameter 'population_size' value is not int type. Actual value: {population_size}.")
        if population_size <= 0:
            raise ValueError(f"Parameter 'population_size' value must be greater than 0. "
                             f"Actual value: {population_size}.")
        super().__init__(problem=problem, stop_conditions=stop_conditions, logger=logger)
        self.population_size = population_size

        class Solution(AbstractSolution):
            """Solution class for given optimization problem."""

            optimization_problem = problem

        self.SolutionClass = Solution

    def _perform_iteration(self, iteration_index: int) -> None:
        """
        Perform optimization of algorithm iteration.

        :param iteration_index: Index number (counted from 0) of random algorithm iteration.
        """
        solutions = [self.SolutionClass() for _ in range(self.population_size)]
        best_in_iter = max(solutions)
        self._best_solution = best_in_iter if self._best_solution is None else max(best_in_iter, self._best_solution)
        if self.logger is not None:
            self.logger.log_iteration(iteration=iteration_index, solutions=solutions)

    def get_log_data(self) -> Dict[str, Union[str, int]]:
        """
        Gets data for logging purposes.

        :return: Dictionary with this Random Algorithm crucial data.
        """
        log_data = super().get_log_data()
        log_data.update(population_size=self.population_size)
        return log_data
