from typing import Tuple
from abc import ABC, abstractmethod

from optimization.optimization_algorithms.stop_conditions import StopCondition
from optimization.optimization_problem.problem import OptimizationProblem, Solution


class OptimizationAlgorithm(ABC):
    """Abstract definition of Optimization Algorithm."""

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
        self.stop_condition = stop_condition  # todo: rethink stop_condition implementation

    def get_data_for_logging(self) -> dict:
        # todo
        pass

    def start_optimization(self) -> None:
        # todo
        while not self.is_stop_condition_achieved():
            solutions = self.find_new_solutions()

    @abstractmethod
    def is_stop_condition_achieved(self) -> bool:
        # todo: think how to implement different stop conditions
        pass

    @abstractmethod
    def find_new_solutions(self) -> Tuple[Solution, ...]:
        # todo: new iteration?
        pass


