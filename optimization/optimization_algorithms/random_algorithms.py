from typing import List, Dict, Optional

from optimization.optimization_algorithms.algorithm_definition import OptimizationAlgorithm
from optimization.optimization_algorithms.stop_conditions import StopCondition
from optimization.optimization_problem import OptimizationProblem, AbstractSolution


__all__ = ["RandomAlgorithm"]


class RandomAlgorithm(OptimizationAlgorithm):
    """Optimization algorithm that uses random values of decision variables to search for optimal solution."""

    def __init__(self, optimization_problem: OptimizationProblem, stop_condition: StopCondition,
                 population_size, logger: Optional[object] = None) -> None:
        """
        Initialization of random algorithm.

        :param optimization_problem: Definition of optimization problem to solve.
        :param stop_condition: Definition of condition when optimization should be stopped.
        :param population_size: Number of solutions generated in one iteration.
        :param logger: Configured logger that would report optimization process.

        :return Random optimization algorithm ready for the optimization process.
        """
        super().__init__(optimization_problem=optimization_problem, stop_condition=stop_condition, logger=logger)
        self.population_size = population_size

    def _initial_iteration(self) -> List[AbstractSolution]:
        """
        Searches for solutions in first iterations of random algorithm.

        :return: List of solution sorted by objective value (calculation includes penalty).
        """
        solutions = [self.solution_type() for _ in range(self.population_size)]
        self.sort_solutions(solutions=solutions)
        return solutions

    def _following_iteration(self) -> List[AbstractSolution]:
        """
        Searches for solutions in following iterations of random algorithm.

        :return: List of solution sorted by objective value (calculation includes penalty).
        """
        return self._initial_iteration()

    def get_data_for_logging(self) -> Dict[str, str]:
        """
        Method which prepares data of the instance of this class for logging.

        :return: Crucial data of this object.
        """
        data = super().get_data_for_logging()
        data["population_size"] = repr(self.population_size)
        return data
