from typing import Iterable
from enum import Enum
from os import path, mkdir
from yaml import dump as yaml_dump
from yamlordereddictloader import Dumper
from datetime import datetime


__all__ = ["LoggingVerbosity", "Logger", "SEALogger"]


class LoggingVerbosity(Enum):
    """
    Enum with possible logging levels that represent verbosity of the log.

    Options:
        - BestSolution: Only the best solution of optimization_old problem will be logged.
        - ProblemDefinition: Contains all logs from lower levels. Additionally, definition of optimization_old problem
            will be logged.
        - AlgorithmConfiguration: Contains all logs from lower levels. Additionally, configuration of used optimization_old
            algorithm will be logged.
        - AllSolutions: Contains all logs from lower levels. Additionally, information about all solution found
            during optimization_old process will be logged.
    """
    BestSolution = 0
    ProblemDefinition = 10
    AlgorithmConfiguration = 20
    AllSolutions = 30

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
    Logger of optimization_old process.
    """
    LOG_DIRECTORY_PATTERN = "optimization_%Y-%m-%d_%H-%M-%S"

    def __init__(self, logs_location: str, logging_verbosity: LoggingVerbosity = LoggingVerbosity.AllSolutions) -> None:
        """
        Defines how logging should be executed.

        :param logging_verbosity: Defines how much details of optimization_old process should be logged.
        :param logs_location: Path of directory in which log files will be created.

        :raise TypeError: When 'logging_verbosity' parameter is not instance of 'LoggingVerbosity' class
            or 'logs_location' parameter is not str type.
        :raise ValueError: If value of 'logs_location' parameter is not an existing directory.
        """
        if not isinstance(logging_verbosity, LoggingVerbosity):
            raise TypeError(f"Provided value of 'logging_verbosity' parameter has unexpected type. "
                            f"Expected: {LoggingVerbosity}. Actual: {type(logging_verbosity)}.")
        if not isinstance(logs_location, str):
            raise TypeError(f"Provided value of 'logs_location' parameter has unexpected type. "
                            f"Expected: {str}. Actual: {type(logs_location)}.")
        if not path.isdir(logs_location):
            raise ValueError(f"Provided value of 'logs_location' parameter is not an existing directory. "
                             f"Received: {logs_location}.")
        self.logging_verbosity = logging_verbosity
        logs_directory_name = datetime.now().strftime(self.LOG_DIRECTORY_PATTERN)
        self.logs_location = path.join(logs_location, logs_directory_name)
        mkdir(self.logs_location)

    def log_at_start(self, optimization_algorithm: "Algorithm") -> None:
        """
        Logging method that should be executed at the start of the optimization_old process.
        It logs information available before optimization_old process such as problem definition,
        optimization_old algorithm configuration, etc.

        :param optimization_algorithm: Optimization algorithm used during optimization_old process.
        """
        if self.logging_verbosity >= LoggingVerbosity.ProblemDefinition:
            with open(file=path.join(self.logs_location, "problem.yaml"), mode="w") as problem_file:
                yaml_dump(data=optimization_algorithm.optimization_problem.get_data_for_logging(), stream=problem_file,
                          Dumper=Dumper)
        if self.logging_verbosity >= LoggingVerbosity.AlgorithmConfiguration:
            with open(file=path.join(self.logs_location, "algorithm_configuration.yaml"), mode="w") as alg_config_file:
                yaml_dump(data=optimization_algorithm.get_data_for_logging(), stream=alg_config_file, Dumper=Dumper)

    def log_solutions(self, iteration: int, solutions: Iterable["Solution"]) -> None:
        """
        Logging method that should be executed after each iteration of optimization_old algorithm.
        It logs information available about found solutions.

        :param iteration: Index of the optimization_old process iteration.
        :param solutions: Solutions of the optimization_old problem that were found during last iteration
            of the optimization_old process.
        """
        if self.logging_verbosity >= LoggingVerbosity.AllSolutions:
            _data_to_dump = {f"Iteration {iteration}": [solution.get_data_for_logging() for solution in solutions]}
            _mode = "a" if iteration else "w"
            with open(file=path.join(self.logs_location, "solutions.yaml"), mode=_mode) as solutions_file:
                yaml_dump(data=_data_to_dump, stream=solutions_file, Dumper=Dumper)

    def log_at_end(self, best_solution: "Solution") -> None:
        """
        Logging method that should be executed at the end of the optimization_old process.
        It logs information available after optimization_old process is finished such as best solution found.

        :param best_solution: Best solution that was found by the optimization_old algorithm.
        """
        if self.logging_verbosity >= LoggingVerbosity.BestSolution:
            with open(file=path.join(self.logs_location, "best_solution.yaml"), mode="w") as best_solution_file:
                yaml_dump(data=best_solution.get_data_for_logging(), stream=best_solution_file, Dumper=Dumper)


class SEALogger(Logger):
    """
    Logger of optimization_old process for Self-adaptive Evolutionary Algorithm.
    """
    LOG_SAE_LOWER_DIRECTORY_PATTERN = "SAE_Iteration_{}"

    def log_at_start(self, sea: "SEA") -> None:
        """
        Logging method that should be executed at the start of the optimization_old process.
        It logs information available before optimization_old process such as problem definition,
        optimization_old algorithm configuration, etc.

        :param sea: Self-adaptive Evolutionary Algorithm used during optimization_old process.
        """
        if self.logging_verbosity >= LoggingVerbosity.ProblemDefinition:
            with open(file=path.join(self.logs_location, "adaptation_problem.yaml"), mode="w") as problem_file:
                yaml_dump(data=sea.optimization_problem.get_data_for_logging(), stream=problem_file, Dumper=Dumper)
            with open(file=path.join(self.logs_location, "problem.yaml"), mode="w") as problem_file:
                yaml_dump(data=sea.optimization_problem.optimization_problem.get_data_for_logging(),
                          stream=problem_file, Dumper=Dumper)
        if self.logging_verbosity >= LoggingVerbosity.AlgorithmConfiguration:
            with open(file=path.join(self.logs_location, "algorithm_configuration.yaml"), mode="w") as alg_config_file:
                yaml_dump(data=sea.get_data_for_logging(), stream=alg_config_file, Dumper=Dumper)

    def log_solutions(self, iteration: int, solutions: Iterable["SEALower"]) -> None:
        """
        Logging method that should be executed after each iteration of SEA optimization_old algorithm.
        It logs information available about found solutions (SEA Lower).

        :param iteration: Index of the SEA optimization_old process iteration.
        :param solutions: Solutions of the adaptation problem that were found during last iteration of the
            optimization_old process.
        """
        if self.logging_verbosity >= LoggingVerbosity.AllSolutions:
            _data_to_dump = {f"Iteration {iteration}": [solution.get_data_for_logging() for solution in solutions]}
            _mode = "a" if iteration else "w"
            with open(file=path.join(self.logs_location, "sea_lower.yaml"), mode=_mode) as solutions_file:
                yaml_dump(data=_data_to_dump, stream=solutions_file, Dumper=Dumper)

    def log_sea_lower_solutions(self, sea_upper_iteration: int, sea_lower_index: int, sea_lower_iteration: int,
                                solutions: Iterable["Solution"]) -> None:
        """
        Logging method that should be executed after each iteration of SEA Lower optimization_old algorithm.
        It logs information available about found solutions (optimized problem).

        :param sea_upper_iteration: Index of the SEA optimization_old process iteration.
        :param sea_lower_index: Ordinal number of the SEA Lower algorithm for which solutions are logged.
        :param sea_lower_iteration: Index of the SEA Lower optimization_old process iteration.
        :param solutions: Solutions of the optimization_old problem that were found during last iteration of the
            optimization_old process.
        """
        if self.logging_verbosity >= LoggingVerbosity.AllSolutions:
            directory_path = path.join(self.logs_location, self.LOG_SAE_LOWER_DIRECTORY_PATTERN.format(sea_upper_iteration))
            if not path.isdir(directory_path):
                mkdir(directory_path)
            _data_to_dump = {f"Iteration {sea_lower_iteration}":
                                 [solution.get_data_for_logging() for solution in solutions]}
            _mode = "a" if sea_lower_iteration else "w"
            with open(file=path.join(directory_path, f"solution_of_sea_lower_{sea_lower_index}.yaml"), mode=_mode) \
                    as solutions_file:
                yaml_dump(data=_data_to_dump, stream=solutions_file, Dumper=Dumper)