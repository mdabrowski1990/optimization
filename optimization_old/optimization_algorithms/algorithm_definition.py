from typing import Optional, List
from abc import ABC, abstractmethod
from datetime import datetime

from optimization_old.logging import Logger
from optimization_old.optimization_problem import OptimizationProblem, OptimizationType, AbstractSolution
from optimization_old.optimization_algorithms.stop_conditions import StopCondition


__all__ = ["OptimizationAlgorithm"]


class OptimizationAlgorithm(ABC):
    """Abstract definition of Optimization Algorithm."""

    @abstractmethod
    def __init__(self, optimization_problem: OptimizationProblem, stop_condition: StopCondition,
                 logger: Optional[Logger]) -> None:
        """
        Common initialization of optimization_old algorithm.

        :param optimization_problem: Definition of optimization_old problem to solve.
        :param stop_condition: Definition of condition when optimization_old should be stopped.
        :param logger: Configured logger that would report optimization_old process.
        :raise TypeError: When 'optimization_problem' parameter is not instance of 'OptimizationProblem' class or
            'stop_condition' parameter is not instance of 'StopCondition' class.

        :return Optimization algorithm ready for the optimization_old process.
        """
        if not isinstance(optimization_problem, OptimizationProblem):
            raise TypeError(f"Provided value of 'optimization_problem' parameter has unexpected type. "
                            f"Expected: {OptimizationProblem}. Actual: {type(optimization_problem)}.")
        if not isinstance(stop_condition, StopCondition):
            raise TypeError(f"Provided value of 'stop_condition' parameter has unexpected type. "
                            f"Expected: {StopCondition}. Actual: {type(stop_condition)}.")
        self.optimization_problem = optimization_problem
        self.stop_condition = stop_condition
        self.solution_type = optimization_problem.spawn_solution_definition()
        self.logger = logger
        self.start_time = None
        self.end_time = None

    @abstractmethod
    def _initial_iteration(self) -> List[AbstractSolution]:
        """
        Abstract definition of a method that perform initial iteration of optimization_old process and creates initial
        population of solutions.

        :raise NotImplementedError: Abstract method was called.
        """
        raise NotImplementedError("You have called abstract method '_initial_iteration' of 'OptimizationAlgorithm' "
                                  "abstract class.")

    @abstractmethod
    def _following_iteration(self) -> List[AbstractSolution]:
        """
        Abstract definition of a method that perform following iteration of optimization_old process.

        :raise NotImplementedError: Abstract method was called.
        """
        raise NotImplementedError("You have called abstract method '_following_iteration' of 'OptimizationAlgorithm' "
                                  "abstract class.")

    def _is_stop_condition_achieved(self, solutions: list) -> bool:
        """
        Checks whether stop condition was achieved in this iteration of the optimization_old process.

        :param solutions: List of solutions that were created in last iteration of optimization_old process.

        :return: True when time limit restriction is exceeded, False if it is not.
        """
        return self.stop_condition.is_achieved(start_time=self.start_time, solutions=solutions)

    def perform_optimization(self) -> AbstractSolution:
        """
        Start and executed optimization_old process.

        :return: Best solution that was found for the optimization_old problem.
        """
        # pre start
        if self.logger is not None:
            self.logger.log_at_start(optimization_algorithm=self)
        # initial iteration
        i = 0
        self.start_time = datetime.now()
        solutions = self._initial_iteration()
        best_solution = solutions[0]
        if self.logger is not None:
            self.logger.log_solutions(iteration=i, solutions=solutions)
        # following iterations
        while not self._is_stop_condition_achieved(solutions=solutions):
            i += 1
            solutions = self._following_iteration()
            _tmp = [solutions[0], best_solution]
            self.sort_solutions(_tmp)
            best_solution = _tmp[0]
            if self.logger is not None:
                self.logger.log_solutions(iteration=i, solutions=solutions)
        # end
        self.end_time = datetime.now()
        if self.logger is not None:
            self.logger.log_at_end(best_solution=best_solution)
        return best_solution

    @abstractmethod
    def get_data_for_logging(self) -> dict:
        """
        Method which prepares data of the instance of this class for logging.

        :return: Crucial data of this object.
        """
        return {
            "algorithm_type": self.__class__.__name__,
            "stop_condition": self.stop_condition.get_data_for_logging(),
        }

    def sort_solutions(self, solutions: List[AbstractSolution]) -> None:
        """
        Sorts list with solution by objective value (with penalty).

        :param solutions: List with solutions.
        """
        _reverse = self.optimization_problem.optimization_type == OptimizationType.Maximize
        solutions.sort(key=lambda solution: solution.get_objective_value_with_penalty(), reverse=_reverse)
