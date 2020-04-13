from typing import Optional, Iterable
from enum import Enum
from os import path, mkdir
from yaml import dump as yaml_dump
from datetime import datetime

from optimization.optimization_algorithms.base_optimization_algorithm import OptimizationAlgorithm
from optimization.optimization_problem.problem import Solution


class LoggingVerbosity(Enum):
    """
    Enum with possible logging levels that represent verbosity of the log.

    Options:
        - Nothing: Logging functionality is turned off.
        - BestSolution: Contains all logs from lower levels. Additionally, best solution of optimization problem
            will be logged.
        - ProblemDefinition: Contains all logs from lower levels. Additionally, definition of optimization problem
            will be logged.
        - AlgorithmConfiguration: Contains all logs from lower levels. Additionally, configuration of used optimization
            algorithm will be logged.
        - AllSolutions: Contains all logs from lower levels. Additionally, information about all solution found
            during optimization process will be logged.
        - Full: All possible information about optimization process will be logged.
    """
    Nothing = 0
    BestSolution = 10
    ProblemDefinition = 20
    AlgorithmConfiguration = 30
    AllSolutions = 40
    Full = 50

    def __lt__(self, other):
        if not isinstance(other, LoggingVerbosity):
            raise TypeError
        return self.value < other.value

    def __le__(self, other):
        if not isinstance(other, LoggingVerbosity):
            raise TypeError
        return self.value <= other.value

    def __eq__(self, other):
        if not isinstance(other, LoggingVerbosity):
            raise TypeError
        return self.value == other.value

    def __ne__(self, other):
        if not isinstance(other, LoggingVerbosity):
            raise TypeError
        return self.value != other.value

    def __ge__(self, other):
        if not isinstance(other, LoggingVerbosity):
            raise TypeError
        return self.value >= other.value

    def __gt__(self, other):
        if not isinstance(other, LoggingVerbosity):
            raise TypeError
        return self.value > other.value


class Logger:
    """
    Logger of optimization process.
    """
    def __init__(self, logging_verbosity: LoggingVerbosity = LoggingVerbosity.Nothing,
                 logs_location: Optional[str] = None) -> None:
        """
        Defines how logging should be executed.

        :param logging_verbosity: Defines how much details of optimization process should be logged.
        :param logs_location: Path of directory in which log files will be created.
        :raise TypeError: When 'logging_verbosity' parameter is not instance of 'LoggingVerbosity' class.
        """
        if not isinstance(logging_verbosity, LoggingVerbosity):
            raise TypeError(f"Provided value of 'logging_verbosity' parameter has unexpected type. "
                            f"Expected: {LoggingVerbosity}. Actual: {type(logging_verbosity)}.")
        self.logging_verbosity = logging_verbosity
        if logging_verbosity > LoggingVerbosity.Nothing:
            if not path.isdir(logs_location):
                raise ValueError(f"Provided value of 'logs_location' parameter is not an existing directory. "
                                 f"Received: {logs_location}.")
            directory_name = datetime.now().strftime("optimization_%Y_%m_%d_%H_%M_%S")
            self.logs_location = path.join(logs_location, directory_name)
            mkdir(self.logs_location)
        else:
            self.logs_location = None

    def log_at_start(self, optimization_algorithm: OptimizationAlgorithm) -> None:
        """
        Logging method that should be executed at the start of the optimization process.
        It logs information available before optimization process such as problem definition,
        optimization algorithm configuration, etc.

        :param optimization_algorithm: Optimization algorithm used during optimization process.
        :raise TypeError: When 'optimization_algorithm' parameter is not instance of 'OptimizationAlgorithm' class.
        """
        if not isinstance(optimization_algorithm, OptimizationAlgorithm):
            raise TypeError(f"Provided value of 'optimization_algorithm' parameter has unexpected type. "
                            f"Expected: {OptimizationAlgorithm}. Actual: {type(optimization_algorithm)}.")
        if self.logging_verbosity >= LoggingVerbosity.ProblemDefinition:
            with open(file=path.join(self.logs_location, "problem.yaml"), mode="w") as problem_file:
                yaml_dump(data=optimization_algorithm.optimization_problem.get_data_for_logging(), stream=problem_file)
        if self.logging_verbosity >= LoggingVerbosity.AlgorithmConfiguration:
            with open(file=path.join(self.logs_location, "algorithm_configuration.yaml"), mode="w") as alg_config_file:
                yaml_dump(data=optimization_algorithm.get_data_for_logging(), stream=alg_config_file)

    def log_found_solutions(self, iteration: int, solutions: Iterable) -> None:
        """
        Logging method that should be executed after each iteration of optimization algorithm.
        It logs information available about found solution.

        :parameter iteration: Index of the optimization process iteration.
        :param solutions: Solutions of the optimization problem that were found during last iteration
            of the optimization process.
        """
        if self.logging_verbosity >= LoggingVerbosity.AllSolutions:
            with open(file=path.join(self.logs_location, f"solutions_{iteration}.yaml"), mode="w") as solutions_file:
                yaml_dump(data=[solution.get_data_for_logging() for solution in solutions], stream=solutions_file)

    def log_at_end(self, best_solution: Solution) -> None:
        """
        Logging method that should be executed at the end of the optimization process.
        It logs information available after optimization process is finished such as best solution found.

        :param best_solution: Best solution that was found by the optimization algorithm.
        """
        if not isinstance(best_solution, Solution):
            raise TypeError(f"Provided value of 'best_solution' parameter has unexpected type. "
                            f"Expected: {Solution}. Actual: {type(best_solution)}.")
        if self.logging_verbosity >= LoggingVerbosity.ProblemDefinition:
            with open(file=path.join(self.logs_location, "best_solution.yaml"), mode="w") as best_solution_file:
                yaml_dump(data=best_solution.get_data_for_logging(), stream=best_solution_file)

