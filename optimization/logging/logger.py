"""
Optimization process logger.

In this file you can find:
- AbstractLogger - abstract definition of Logger that you can use to create your own logger
- Logger - build-in logger for recording optimization process
- LoggingVerbosity - enum with available 'Logger' verbosity level
- LoggingFormat - enum with available 'Logger' output files formats
"""

__all__ = ["AbstractLogger", "LoggingVerbosity", "LoggingFormat", "Logger"]


from typing import Iterable, Optional
from abc import ABC, abstractmethod
from enum import IntEnum, Enum
from os import path, mkdir
from datetime import datetime

from yaml import dump as yaml_dump
from yamlordereddictloader import Dumper as YamlDumper
from json import dump as json_dump


class LoggingVerbosity(IntEnum):
    """
    All possible logging levels that represents different verbosity levels of the logger.

    Options:
        - BestSolution: The best solution of the optimization problem will be reported to separate file 'best_solution'.
        - ProblemDefinition: Reports optimization problem definition to separate file 'optimization_problem'.
            Contains all logs from lower levels.
        - AlgorithmConfiguration: Reports optimization algorithm configuration to separate file 'algorithm'.
            Contains all logs from lower levels.
        - AllSolutions: Reports all solution found during optimization search to 'solutions' file.
            Contains all logs from lower levels.
    """

    BestSolution = 0
    ProblemDefinition = 10
    AlgorithmConfiguration = 20
    AllSolutions = 30

    def __eq__(self, other: object) -> bool:
        """
        :raise TypeError: Compared object is not LoggingVerbosity type.

        :return: True if values are the same, False otherwise.
        """
        if not isinstance(other, LoggingVerbosity):
            raise TypeError(f"Parameter 'other' is not LoggingVerbosity type. Actual value: {other}.")
        return self.value == other.value  # pylint: disable=comparison-with-callable

    def __ne__(self, other: object) -> bool:
        """
        :raise TypeError: Compared object is not LoggingVerbosity type.

        :return: True if values are different, False otherwise.
        """
        return not self.__eq__(other)

    def __lt__(self, other: object) -> bool:
        """
        :raise TypeError: Compared object is not LoggingVerbosity type.

        :return: True if this is less than other, False otherwise.
        """
        if not isinstance(other, LoggingVerbosity):
            raise TypeError(f"Parameter 'other' is not LoggingVerbosity type. Actual value: {other}.")
        return self.value < other.value  # pylint: disable=comparison-with-callable

    def __le__(self, other: object) -> bool:
        """
        :raise TypeError: Compared object is not LoggingVerbosity type.

        :return: True if this is less or equal than other, False otherwise.
        """
        if not isinstance(other, LoggingVerbosity):
            raise TypeError(f"Parameter 'other' is not LoggingVerbosity type. Actual value: {other}.")
        return self.value <= other.value  # pylint: disable=comparison-with-callable

    def __ge__(self, other: object) -> bool:
        """
        :raise TypeError: Compared object is not LoggingVerbosity type.

        :return: True if this is greater or equal than other, False otherwise.
        """
        return not self.__lt__(other)

    def __gt__(self, other: object) -> bool:
        """
        :raise TypeError: Compared object is not LoggingVerbosity type.

        :return: True if this is greater than other, False otherwise.
        """
        return not self.__le__(other)


class LoggingFormat(Enum):
    """Enum with currently supported log formats."""

    YAML = "YAML"
    JSON = "JSON"


class AbstractLogger(ABC):
    """
    Abstract definition of optimization process Logger.

    Basing on this abstract definition, you can create your own custom logger!
    All you need is to inherit after this class and implement all abstract methods.
    """

    @abstractmethod
    def log_at_start(self, algorithm, stop_conditions, problem):
        """
        Logging method that will be called before the start of the optimization process.

        :param algorithm: Optimization algorithm configuration that will be used during the optimization process.
        :param stop_conditions: Stop conditions of the optimization process.
        :param problem: Optimization problem to solve.
        """
        ...

    @abstractmethod
    def log_iteration(self, iteration: int, solutions: Iterable):
        """
        Logging method that will be called at each iteration of main optimization algorithm.

        :param iteration: Number of iteration of main optimization algorithm.
        :param solutions: Solutions found in this iteration.
        """
        ...

    @abstractmethod
    def log_lower_level_iteration(self, upper_iteration: int,
                                  lower_algorithm_index: int,
                                  lower_iteration: int,
                                  solutions: Iterable):
        """
        Logging method that will be called at each iteration of lower level optimization algorithms.

        Note: This method will only be called by adaptive algorithm!

        :param upper_iteration: Upper algorithm iteration.
        :param lower_algorithm_index: Lower algorithm index.
        :param lower_iteration: Lower algorithm iteration.
        :param solutions: Solutions found in this iteration of lower algorithm.
        """
        ...

    @abstractmethod
    def log_at_end(self, best_solution):
        """
        Logging method that will be called at the end of optimization process.

        :param best_solution: The best solution found by the optimization algorithm.
        """
        ...


class Logger(AbstractLogger):
    """Build-in logger for reporting optimization data."""

    LOG_DIRECTORY_PATTERN = "{}_%Y-%m-%d_%H-%M-%S"

    def __init__(self, logs_dir: str,
                 verbosity: LoggingVerbosity = LoggingVerbosity.BestSolution,
                 log_format: LoggingFormat = LoggingFormat.YAML) -> None:
        """
        Creates optimization process logger.

        :param logs_dir: Directory where optimization process logs to be created.
        :param verbosity: Level of logger verbosity (how much information to be provided).
        :param log_format: Format in which log files to be presented.
        """
        if not isinstance(logs_dir, str):
            raise TypeError(f"Parameter 'logs_dir' is not str type. Actual value: {logs_dir}.")
        if not path.isdir(logs_dir):
            raise ValueError(f"Directory '{logs_dir}' is not a valid path to existing directory.")
        if isinstance(verbosity, LoggingVerbosity):
            pass
        elif isinstance(verbosity, str):
            verbosity = getattr(LoggingVerbosity, verbosity)
        else:
            raise TypeError(f"Parameter 'verbosity' is not LoggingVerbosity or str type. Actual value: {verbosity}.")
        if isinstance(log_format, LoggingFormat):
            pass
        elif isinstance(log_format, str):
            log_format = getattr(LoggingFormat, log_format)
        else:
            raise TypeError(f"Parameter 'log_format' is not LoggingFormat or str type. Actual value: {log_format}.")
        self.main_dir = logs_dir
        self.verbosity = verbosity
        self.log_format = log_format
        self.optimization_process_dir: Optional[str] = None

    def _dump_algorithm_data(self, algorithm_data: dict, stop_conditions_data: dict) -> None:
        """
        Dumps optimization algorithm data to proper file.

        :param algorithm_data: Configuration data of optimization algorithm used.
        :param stop_conditions_data: Configuration data of optimization process stop conditions.
        """
        if self.log_format == LoggingFormat.YAML:
            algorithm_file_path = path.join(self.optimization_process_dir, "algorithm.yaml")  # type: ignore
            stop_conditions_file_path = path.join(self.optimization_process_dir, "stop_conditions.yaml")  # type: ignore
            with open(algorithm_file_path, "w") as yaml_file:
                yaml_dump(algorithm_data, yaml_file, YamlDumper)
            with open(stop_conditions_file_path, "w") as yaml_file:
                yaml_dump(stop_conditions_data, yaml_file, YamlDumper)
        elif self.log_format == LoggingFormat.JSON:
            algorithm_file_path = path.join(self.optimization_process_dir, "algorithm.json")  # type: ignore
            stop_conditions_file_path = path.join(self.optimization_process_dir, "stop_conditions.json")  # type: ignore
            with open(algorithm_file_path, "w") as json_file:
                json_dump(algorithm_data, json_file)
            with open(stop_conditions_file_path, "w") as json_file:
                json_dump(stop_conditions_data, json_file)

    def _dump_problem_data(self, problem_data: dict) -> None:
        """
        Dumps optimization problem data to proper file.

        :param problem_data: Optimization problem definition.
        """
        if self.log_format == LoggingFormat.YAML:
            file_path = path.join(self.optimization_process_dir, "problem.yaml")  # type: ignore
            with open(file_path, "w") as yaml_file:
                yaml_dump(problem_data, yaml_file, YamlDumper)
        elif self.log_format == LoggingFormat.JSON:
            file_path = path.join(self.optimization_process_dir, "problem.json")  # type: ignore
            with open(file_path, "w") as json_file:
                json_dump(problem_data, json_file)

    def log_at_start(self, algorithm, stop_conditions, problem) -> None:
        """
        Logging method that will be called before the start of the optimization process.

        Note: Asynchronous optimization with the same logger does not make sense, so please do not do it as
        it will cause some pain!

        :param algorithm: Optimization algorithm configuration that will be used during the optimization process.
        :param stop_conditions: Stop conditions of the optimization process.
        :param problem: Optimization problem to solve.
        """
        # prepare separate directory for this optimization process logging
        _top_level_dir_name = datetime.now().strftime(self.LOG_DIRECTORY_PATTERN)
        self.optimization_process_dir = path.join(self.main_dir, _top_level_dir_name)
        mkdir(self.optimization_process_dir)
        # create log files and dump data according to verbosity level
        if self.verbosity >= LoggingVerbosity.ProblemDefinition:
            self._dump_problem_data(problem_data=problem.get_log_data())
        if self.verbosity >= LoggingVerbosity.AlgorithmConfiguration:
            self._dump_algorithm_data(algorithm_data=algorithm.get_log_data(),
                                      stop_conditions_data=stop_conditions.get_log_data())

    def log_iteration(self, iteration: int, solutions: Iterable) -> None:
        """
        Logging method that will be called at each iteration of main optimization algorithm.

        :param iteration: Number of iteration of main optimization algorithm.
        :param solutions: Solutions found in this iteration.
        """
        if self.verbosity >= LoggingVerbosity.AllSolutions:
            if iteration > 0:
                mode = "a"
            else:
                mode = "w"
            data_to_log = {f"Iteration {iteration}": [solution.get_log_data() for solution in solutions]}
            if self.log_format == LoggingFormat.YAML:
                file_path = path.join(self.optimization_process_dir, "solutions.yaml")  # type: ignore
                with open(file_path, mode) as yaml_file:
                    yaml_dump(data_to_log, yaml_file, YamlDumper)
            elif self.log_format == LoggingFormat.JSON:
                file_path = path.join(self.optimization_process_dir, "solutions.json")  # type: ignore
                with open(file_path, mode) as json_file:
                    json_dump(data_to_log, json_file)

    def log_lower_level_iteration(self, upper_iteration: int,
                                  lower_algorithm_index: int,
                                  lower_iteration: int,
                                  solutions: Iterable) -> None:
        """
        Logging method that will be called at each iteration of lower level optimization algorithms.

        Log files naming convention: solution_iter_X_alg_Y.E,
        where:
        - X - iteration of main algorithm
        - Y - index (order number) of lower algorithm in main algorithm
        - E - extension (according to log_format attribute)

        Note: This method will only be called by adaptive algorithm!

        :param upper_iteration: Upper algorithm iteration.
        :param lower_algorithm_index: Lower algorithm index.
        :param lower_iteration: Lower algorithm iteration.
        :param solutions: Solutions found in this iteration of lower algorithm.
        """
        if self.verbosity >= LoggingVerbosity.AllSolutions:
            if lower_iteration > 0:
                mode = "a"
            else:
                mode = "w"
            data_to_log = {f"Iteration {lower_iteration}": [solution.get_log_data() for solution in solutions]}
            if self.log_format == LoggingFormat.YAML:
                file_path = path.join(self.optimization_process_dir,  # type: ignore
                                      f"solutions_iter_{upper_iteration}_alg_{lower_algorithm_index}.yaml")
                with open(file_path, mode) as yaml_file:
                    yaml_dump(data_to_log, yaml_file, YamlDumper)
            elif self.log_format == LoggingFormat.JSON:
                file_path = path.join(self.optimization_process_dir,  # type: ignore
                                      f"solutions_iter_{upper_iteration}_alg_{lower_algorithm_index}.json")
                with open(file_path, mode) as json_file:
                    json_dump(data_to_log, json_file)

    def log_at_end(self, best_solution) -> None:
        """
        Logging method that will be called at the end of optimization process.

        :param best_solution: The best solution found by the optimization algorithm.
        """
        if self.verbosity >= LoggingVerbosity.BestSolution:
            if self.log_format == LoggingFormat.YAML:
                file_path = path.join(self.optimization_process_dir, "best_solution.yaml")  # type: ignore
                with open(file_path, "w") as yaml_file:
                    yaml_dump(best_solution.get_log_data(), yaml_file, YamlDumper)
            elif self.log_format == LoggingFormat.JSON:
                file_path = path.join(self.optimization_process_dir, "best_solution.json")  # type: ignore
                with open(file_path, "w") as json_file:
                    json_dump(best_solution.get_log_data(), json_file)
