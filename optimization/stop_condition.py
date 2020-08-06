"""Definition of stop condition for optimization process."""

__all__ = ["StopCondition"]

from typing import Optional, Union
from datetime import timedelta, datetime

from .problem.solution import AbstractSolution


class StopCondition:
    """Definition of stop condition which causes the end of optimization process."""

    def __init__(self, time_limit: timedelta, satisfying_objective_value: Optional[Union[float, int]] = None,
                 max_iter_without_progress: Optional[int] = None,
                 max_time_without_progress: Optional[timedelta] = None) -> None:
        """
        Create definition of certain stop conditions.

        :param time_limit: Time that optimization process might last.
            After this time it will be stopped (sooner than later).
            Note: Passing time restriction is mandatory to avoid infinity loop!
        :param satisfying_objective_value: Boundary value of the objective.
            When solution with better or equal objective value is found, then optimization process is stopped.
            Note: It is not taken into account if equal None.
        :param max_iter_without_progress: Maximal number of optimization algorithm iterations without progress
            (finding a better solution). After the number of iterations without progress is higher than this value,
            then optimization process is stopped.
            Note: It is not taken into account if equal None.
        :param max_time_without_progress: Maximal time that optimization process might last without progress
            (finding a better solution). After this time is exceeded, then optimization process is stopped.
            Note: It is not taken into account if equal None.
        """
        if not isinstance(time_limit, timedelta):
            raise TypeError(f"Parameter 'time_limit' value is not timedelta type. Actual value: '{time_limit}'.")
        if time_limit <= timedelta():
            raise ValueError(f"Parameter 'time_limit' value less or equal 0s. Actual value: '{time_limit}'.")
        if satisfying_objective_value is not None and not isinstance(satisfying_objective_value, (float, int)):
            raise TypeError(f"Parameter 'satisfying_objective_value' value is not float, int nor None type. "
                            f"Actual value: '{satisfying_objective_value}'.")
        if max_iter_without_progress is not None and not isinstance(max_iter_without_progress, int):
            raise TypeError(f"Parameter 'max_iter_without_progress' value is not int nor None type. "
                            f"Actual value: '{max_iter_without_progress}'.")
        if max_time_without_progress is not None and not isinstance(max_time_without_progress, timedelta):
            raise TypeError(f"Parameter 'max_time_without_progress' value is not timedelta nor None type. "
                            f"Actual value: '{max_time_without_progress}'.")
        self.time_limit = time_limit
        self.satisfying_objective_value = satisfying_objective_value
        self.max_iter_without_progress = max_iter_without_progress
        self.max_time_without_progress = max_time_without_progress
        # internal variables for assessing if test condition were achieved
        self._best_objective_found: Optional[float] = None
        self._last_objective_progress_datetime: Optional[datetime] = None
        self._iter_without_progress: Optional[int] = None

    def _is_time_exceeded(self, start_time: datetime) -> bool:
        """
        Check if optimization process lasts at least time limit.

        :param start_time: Time when optimization process was started.

        :return: True if time limit exceeded, otherwise False.
        """
        return datetime.now() - start_time >= self.time_limit

    def _is_satisfying_solution_found(self, best_solution: AbstractSolution) -> bool:
        """
        Check if satisfying solution was found.

        :param best_solution: Instance of AbstractSolution class with the best solution found in this iteration.
        :raise ValueError: Unexpected value of 'optimization_type' in 'optimization_problem' was found.

        :return: True if satisfying solution found, otherwise False.
        """
        return best_solution <= self.satisfying_objective_value

    def _is_max_iteration_without_progress_exceeded(self, best_solution: AbstractSolution) -> bool:
        """
        Check if exceeded maximal number of iteration without finding a better solution.

        :param best_solution: Instance of AbstractSolution class with the best solution found in this optimization
            algorithm iteration.

        :return: True if exceeded maximal number of iterations without progress, otherwise False.
        """
        if self._best_solution is None or best_solution > self._best_objective_found:
            self._iter_without_progress = 0
            self._best_objective_found = best_solution.get_objective_value_with_penalty()
        else:
            self._iter_without_progress += 1
        return self._iter_without_progress > self.max_iter_without_progress

    # todo: continue here