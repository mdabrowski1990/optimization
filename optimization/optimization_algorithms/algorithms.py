from abc import ABC, abstractmethod

from optimization.optimization_algorithms.stop_conditions import StopCondition
from optimization.optimization_problem.problem import OptimizationProblem


class OptimizationAlgorithm(ABC):
    """Abstract definition of Optimization Algorithm."""

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
        self.solution_type = optimization_problem.spawn_solution_definition()
        self.start_time = None
        self.end_time = None

    @abstractmethod
    def initial_iteration(self) -> list:
        """
        Abstract definition of a method that perform initial iteration of optimization process and creates initial
        population of solutions.

        :raise NotImplementedError: Abstract method was called.
        """
        raise NotImplementedError("You have called abstract method '_initial_iteration' of 'OptimizationAlgorithm' "
                                  "abstract class.")

    @abstractmethod
    def following_iteration(self) -> list:
        """
        Abstract definition of a method that perform following iteration of optimization process.

        :raise NotImplementedError: Abstract method was called.
        """
        raise NotImplementedError("You have called abstract method '_following_iteration' of 'OptimizationAlgorithm' "
                                  "abstract class.")

    def is_stop_condition_achieved(self, solutions: list) -> bool:
        """
        Checks whether stop condition was achieved in this iteration of the optimization process.

        :param solutions: List of solutions that were created in last iteration of optimization process.

        :return: True when time limit restriction is exceeded, False if it is not.
        """
        return self.stop_condition.is_achieved(start_time=self.start_time, solutions=solutions)

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

    @staticmethod
    def sort_solution_by_objective_value(solutions: list) -> None:
        """
        Sorts list with solution by objective value (with penalty).

        :param solutions: List with solutions.
        """
        solutions.sort(key=lambda solution: solution.get_objective_value_with_penalty(), reverse=True)


class RandomOptimizationAlgorithm(OptimizationAlgorithm):
    """Optimization algorithm that uses random values of decision variables to search for optimal solution."""

    def __init__(self, optimization_problem: OptimizationProblem, stop_condition: StopCondition,
                 population_size=1000) -> None:
        """
        Initialization of random algorithm.

        :param optimization_problem: Definition of optimization problem to solve.
        :param stop_condition: Definition of condition when optimization should be stopped.
        :param population_size: Number of solution generated in one iteration.
        """
        super().__init__(optimization_problem=optimization_problem, stop_condition=stop_condition)
        self.population_size = population_size

    def initial_iteration(self) -> list:
        """
        Searches for solutions in first iterations of random algorithm.

        :return: List of solution sorted by objective value (calculation includes penalty).
        """
        solutions = [self.solution_type() for _ in range(self.population_size)]
        self.sort_solution_by_objective_value(solutions=solutions)
        return solutions

    def following_iteration(self) -> list:
        """
        Searches for solutions in following iterations of random algorithm.

        :return: List of solution sorted by objective value (calculation includes penalty).
        """
        return self.initial_iteration()

    def get_data_for_logging(self) -> dict:
        """
        Method which prepares data of the instance of this class for logging.

        :return: Crucial data of this object.
        """
        data = super().get_data_for_logging()
        data["population_size"] = self.population_size
        return data
