"""Common implementation fo all optimization algorithms."""

__all__ = ["AbstractOptimizationAlgorithm"]


from typing import Optional, Iterable, List, Dict, Any
from abc import ABC, abstractmethod
from datetime import datetime

from ..problem import OptimizationProblem, AbstractSolution
from ..stop_conditions import StopConditions
from ..logging import AbstractLogger


class AbstractOptimizationAlgorithm(ABC):
    """Abstract definition of Optimization Algorithm."""

    @abstractmethod
    def __init__(self, problem: OptimizationProblem,
                 stop_conditions: StopConditions,
                 logger: Optional[AbstractLogger] = None) -> None:
        """
        Common initialization of all optimization algorithms.

        :param problem: Optimization problem to be solved by the algorithm.
        :param stop_conditions: Conditions when optimization algorithm shall be stopped.
        :param logger: Logger used for optimization process recording.
        """
        if not isinstance(problem, OptimizationProblem):
            raise TypeError(f"Parameter 'problem' value is not OptimizationProblem type. Actual value: {problem}.")
        if not isinstance(stop_conditions, StopConditions):
            raise TypeError(f"Parameter 'stop_conditions' value is not StopConditions type. "
                            f"Actual value: {stop_conditions}.")
        if logger is not None and not isinstance(logger, AbstractLogger):
            raise TypeError(f"Parameter 'logger' value is not AbstractLogger type. Actual value: {logger}.")
        self.problem = problem
        self.stop_conditions = stop_conditions
        self.logger = logger
        self._start_time: Optional[datetime] = None
        self._end_time: Optional[datetime] = None
        self._best_solution: Optional[AbstractSolution] = None

        class Solution(AbstractSolution):
            """Solution class for given optimization problem."""

            optimization_problem = problem

        self.SolutionClass = Solution

    @abstractmethod
    def _perform_iteration(self, iteration_index: int) -> None:
        """
        Executes following iteration of optimization algorithm.

        Note: Following actions must be defined inside this method:
            - update of '_best_solution' attribute so it contains the best solution found after each iteration
            - execution of 'self.logger.log_iteration' method after each iteration of algorithm
            - (only if this is self adaptation algorithm) execution of 'self.logger.log_lower_level_iteration'
                method after each iteration of lower algorithm

        :param iteration_index: Index number (counted from 0) of optimization algorithm iteration.
        """
        ...

    @abstractmethod
    def get_log_data(self) -> Dict[str, Any]:
        """
        Gets data for logging purposes.

        :return: Dictionary with this Optimization Algorithm crucial data.
        """
        return {
            "type": self.__class__.__name__,
        }

    def _is_stop_achieved(self) -> bool:
        """
        Checks whether stop conditions of optimization process were achieved.

        :return: True if stop conditions are achieved, False otherwise.
        """
        return self.stop_conditions.is_achieved(start_time=self._start_time,  # type: ignore
                                                best_solution=self._best_solution)  # type: ignore

    @staticmethod
    def sorted_solutions(solutions: Iterable[AbstractSolution], descending: bool = True) -> List[AbstractSolution]:
        """
        Sorts solutions by objective value (with penalty) according to optimization problem definition.

        :param solutions: Iterable with solutions of a single optimization problem.
        :param descending: Determines whether solutions to be sorted in descending order. Possible values:
            - True - from the best to the worst.
            - False - form the worst to the best.

        :return: Sorted list with optimization solution.
        """
        return sorted(solutions, reverse=descending)

    def perform_optimization(self) -> AbstractSolution:
        """
        Executes optimization process.

        :return: The best solution that was found by the optimization algorithm.
        """
        # pre start
        if self.logger is not None:
            self.logger.log_at_start(algorithm=self, stop_conditions=self.stop_conditions, problem=self.problem)
        self._start_time = datetime.now()
        # optimization process
        iteration_index = 0
        self._perform_iteration(iteration_index=iteration_index)
        while not self._is_stop_achieved():
            iteration_index += 1
            self._perform_iteration(iteration_index=iteration_index)
        # after stop
        self._end_time = datetime.now()
        if self.logger is not None:
            self.logger.log_at_end(best_solution=self._best_solution, optimization_time=self._end_time-self._start_time)
        return self._best_solution  # type: ignore
