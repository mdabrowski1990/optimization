from typing import List, Optional
from datetime import datetime, timedelta

from optimization.optimization_problem.problem import OptimizationType, AbstractSolution


class StopCondition:
    """
    Definition of Stop Condition for Optimization Algorithm.
    It always contains definition of time restriction (time after which, optimization process should be stopped).
    Additionally, it may stop optimization process under following conditions:
      - satisfying solution found (with objective value better than predefined value)
      - no progress for some number of iteration (better solution was not found for predefined number of iterations)
      - no progress for some time (better solution was not found for some predefined time)
    """

    def __init__(self, time_limit: timedelta,
                 satisfying_objective_value: Optional[float] = None,
                 max_iterations_without_progress: Optional[int] = None,
                 max_time_without_progress: Optional[timedelta] = None) -> None:
        """
        Creates definition of a stop condition.

        :param time_limit: Time limit after which optimization shall be stopped.
        :param satisfying_objective_value: Boundary value of the objective function.
            When solution with better or equal objective value is found, then optimization process is stopped.
            It is not taken into account when equals None
        :param max_iterations_without_progress: Maximal number of iterations without progress.
            After the number of iterations without progress is exceeded, then optimization process is stopped.
            It is not taken into account when equals None
        :param max_time_without_progress: Maximal time without progress (no better solution found).
            After the number of iterations without progress is exceeded, then optimization process is stopped.
            It is not taken into account when equals None
        """
        if not isinstance(time_limit, timedelta):
            raise TypeError(f"Parameter 'time_limit' must be timedelta type. "
                            f"Received: {time_limit} ({type(time_limit)}).")
        if timedelta() >= time_limit:
            raise ValueError(f"Parameter 'time_limit' value must be greater than 0 seconds. "
                             f"Received: {time_limit}")
        if satisfying_objective_value is not None and not isinstance(satisfying_objective_value, float):
            raise TypeError(f"Parameter 'satisfying_objective_value' must be None or float type. "
                            f"Received: {satisfying_objective_value} ({type(satisfying_objective_value)}).")
        if max_iterations_without_progress is not None:
            if not isinstance(max_iterations_without_progress, int):
                raise TypeError(f"Parameter 'max_iterations_without_progress' must be None or int type. "
                                f"Received: {max_iterations_without_progress} "
                                f"({type(max_iterations_without_progress)}).")
            elif 0 > max_iterations_without_progress:
                raise ValueError(f"Parameter 'max_iterations_without_progress' value must be greater than 0. "
                                 f"Received: {max_iterations_without_progress}")
        if max_time_without_progress is not None:
            if not isinstance(max_time_without_progress, timedelta):
                raise TypeError(f"Parameter 'max_iterations_without_progress' must be None or timedelta type. "
                                f"Received: {max_time_without_progress} ({type(max_time_without_progress)}).")
            elif timedelta() >= max_time_without_progress:
                raise ValueError(f"Parameter 'max_time_without_progress' value must be greater than 0 seconds. "
                                 f"Received: {max_time_without_progress}")
        self.time_limit = time_limit
        self.satisfying_objective_value = satisfying_objective_value
        self.max_iterations_without_progress = max_iterations_without_progress
        self.max_time_without_progress = max_time_without_progress
        self._iterations_without_progress = None
        self._best_solution = None
        self._last_progress_time = None

    def _is_time_exceeded(self, start_time: datetime) -> bool:
        """
        Check if optimization process lasts for longer than time limit.

        :param start_time: Time when optimization process was started.

        :return: True if time exceeded, otherwise False.
        """
        return start_time - datetime.now() >= self.time_limit

    def _is_satisfying_solution_found(self, best_solution: AbstractSolution) -> bool:
        """
        Check if satisfying solution was found.

        :param best_solution: Instance of AbstractSolution class with the best solution found in this iteration.
        :raise ValueError: Unexpected value of 'optimization_type' in 'optimization_problem' was found.

        :return: True if satisfying solution found, otherwise False.
        """
        optimization_type = best_solution.optimization_problem.optimization_type
        if optimization_type == OptimizationType.Minimize:
            return best_solution.get_objective_value_with_penalty() <= self.satisfying_objective_value
        elif optimization_type == OptimizationType.Maximize:
            return best_solution.get_objective_value_with_penalty() >= self.satisfying_objective_value
        raise ValueError("Unexpected value of 'optimization_type' attribute.")

    def _is_max_iteration_without_progress_exceeded(self, best_solution: AbstractSolution) -> bool:
        """
        Check if exceeded maximal number of iteration without finding a better solution.

        :param best_solution: Instance of AbstractSolution class with the best solution found in this iteration.

        :return: True if exceeded maximal number of iteration without progress, otherwise False.
        """
        if self._best_solution is None or self._best_solution <= best_solution:
            self._iterations_without_progress = 0
            self._best_solution = best_solution
        else:
            self._iterations_without_progress += 1
        return self._iterations_without_progress > self.max_iterations_without_progress

    def _is_max_time_without_progress_exceeded(self, best_solution: AbstractSolution) -> bool:
        """
        Check if exceeded maximal time without finding a better solution.

        :param best_solution: Instance of AbstractSolution class with the best solution found in this iteration.

        :return: True if exceeded maximal number of iteration without progress, otherwise False.
        """
        if self._best_solution is None or self._best_solution <= best_solution:
            self._last_progress_time = datetime.now()
            self._best_solution = best_solution
        return datetime.now() - self._last_progress_time > self.max_time_without_progress

    def is_achieved(self, start_time: datetime, solutions: List) -> bool:
        """
        Checks whether stop condition was achieved and optimization process should be stopped.

        :param start_time: Time when optimization process was started.
        :param solutions: List of solutions that were created in last iteration of optimization process.

        :return: True when time limit restriction is exceeded, False if it is not.
        """
        if self._is_time_exceeded(start_time=start_time):
            return True
        elif self.satisfying_objective_value is not None and \
                self._is_satisfying_solution_found(best_solution=solutions[0]):
            return True
        elif self.max_iterations_without_progress is not None and \
                self._is_max_iteration_without_progress_exceeded(best_solution=solutions[0]):
            return True
        elif self.max_time_without_progress is not None and \
                self._is_max_time_without_progress_exceeded(best_solution=solutions[0]):
            return True
        return False

    def get_data_for_logging(self) -> dict:
        """
        Method which prepares data of the instance of this class for logging.

        :return: Crucial data of this object.
        """
        return {
            "time_limit": self.time_limit,
            "satisfying_objective_value": self.satisfying_objective_value,
            "max_iterations_without_progress": self.max_iterations_without_progress,
            "max_time_without_progress": self.max_time_without_progress,
        }

