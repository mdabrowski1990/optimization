from typing import List
from abc import ABC, abstractmethod
from datetime import datetime

from optimization.optimization_algorithms.stop_conditions import StopCondition
from optimization.optimization_problem.problem import OptimizationProblem, Solution


class OptimizationAlgorithm(ABC):
    """Abstract definition of Optimization Algorithm."""

    @property
    @abstractmethod
    def algorithm_type(self) -> str:
        """
        Abstract definition of a property that stores type of stop condition.

        :raise NotImplementedError: Abstract method was called.
        """
        raise NotImplementedError("You have called abstract property 'algorithm_type' of 'OptimizationAlgorithm' "
                                  "abstract class.")

    @abstractmethod
    def __init__(self, optimization_problem: OptimizationProblem, stop_condition: StopCondition) -> None:
        """
        Common initialization of optimization algorithm.

        :param optimization_problem: Definition of optimization problem to solve.
        :param stop_condition: Definition of condition when optimization should be stopped.
        :raise TypeError: When 'optimization_problem' parameter is not instance of 'OptimizationProblem' class or
            'stop_condition' parameter is not instance of 'StopCondition' class.
        """
        if not isinstance(optimization_problem, OptimizationProblem):
            raise TypeError(f"Provided value of 'optimization_problem' parameter has unexpected type. "
                            f"Expected: {OptimizationProblem}. Actual: {type(optimization_problem)}.")
        if not isinstance(stop_condition, StopCondition):
            raise TypeError(f"Provided value of 'stop_condition' parameter has unexpected type. "
                            f"Expected: {StopCondition}. Actual: {type(stop_condition)}.")
        self.optimization_problem = optimization_problem
        self.stop_condition = stop_condition
        self.start_time = None
        self.end_time = None

    @abstractmethod
    def _initial_iteration(self) -> list:
        # todo
        pass

    @abstractmethod
    def _is_stop_condition_achieved(self) -> bool:
        # todo
        return True

    @abstractmethod
    def _find_new_solutions(self) -> list:
        # todo: new iteration?
        pass

    def perform_optimization(self) -> None:
        # todo
        self.start_time = datetime.now()
        solutions = self._initial_iteration()
        while self.stop_condition.is_condition_achieved(start_time=self.start_time, solutions=solutions):
            solutions = self._find_new_solutions()
        self.end_time = datetime.now()

    @abstractmethod
    def get_data_for_logging(self) -> dict:
        """
        Method which prepares data of the instance of this class for logging.

        :return: Crucial data of this object.
        """
        return {
            "algorithm_type": self.algorithm_type,
            "stop_condition": self.stop_condition.get_data_for_logging(),
        }

    @staticmethod
    def sort_solution_by_objective_value(solutions: list) -> list:
        return sorted(solutions, key=lambda solution: solution.get_objective_value_with_penalty(), reverse=True)
