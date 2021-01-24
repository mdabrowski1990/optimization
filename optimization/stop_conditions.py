"""Definition of stop condition for optimization process."""

__all__ = ["StopConditions"]


from typing import Optional, Union, Dict
from datetime import timedelta, datetime

from .problem.solution import AbstractSolution


class StopConditions:
    """Definition of stop condition which causes the end of optimization process."""

    def __init__(self,
                 time_limit: timedelta,
                 satisfying_objective_value: Optional[Union[float, int]] = None,
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
        if max_iter_without_progress is not None:
            if not isinstance(max_iter_without_progress, int):
                raise TypeError(f"Parameter 'max_iter_without_progress' value is not int nor None type. "
                                f"Actual value: '{max_iter_without_progress}'.")
            if max_iter_without_progress <= 0:
                raise ValueError(f"Parameter 'max_iter_without_progress' must be positive integer. "
                                 f"Actual value: {max_iter_without_progress}.")
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
        return best_solution.get_objective_value_with_penalty() >= self.satisfying_objective_value

    def _is_limit_without_progress_exceeded(self, best_solution: AbstractSolution) -> bool:
        """
        Check if exceeded maximal number of iteration or time without finding a better solution.

        :param best_solution: Instance of AbstractSolution class with the best solution found in this optimization
            algorithm iteration.

        :return: True if exceeded maximal number of iterations or time without progress, otherwise False.
        """
        # skip if no checks are performed
        if self.max_iter_without_progress is None and self.max_time_without_progress is None:
            return False
        # update helping variables
        if self._best_objective_found is None or \
                self._best_objective_found < best_solution.get_objective_value_with_penalty():
            self._iter_without_progress = 0
            self._last_objective_progress_datetime = datetime.now()
            self._best_objective_found = best_solution.get_objective_value_with_penalty()
        else:
            self._iter_without_progress += 1  # type: ignore
        # assess status
        return self._is_iter_without_progress_exceeded() or self._is_time_without_progress_exceeded()

    def _is_iter_without_progress_exceeded(self) -> bool:
        """
        !Warning! This method returns only assessed status! Must be called by '_is_limit_without_progress_exceeded'.

        :return: True if maximal number of iteration without progress exceeded, otherwise False.
        """
        return self.max_iter_without_progress is not None \
            and self._iter_without_progress > self.max_iter_without_progress  # type: ignore

    def _is_time_without_progress_exceeded(self) -> bool:
        """
        !Warning! This method returns only assessed status! Must be called by '_is_limit_without_progress_exceeded'.

        :return: True if maximal time without progress exceeded, otherwise False.
        """
        return self.max_time_without_progress is not None \
            and datetime.now() - self._last_objective_progress_datetime > self.max_time_without_progress  # type: ignore

    def is_achieved(self, start_time: datetime, best_solution: AbstractSolution) -> bool:
        """
        Checks whether stop condition was achieved and optimization process should be stopped.

        :param start_time: Time when optimization process was started.
        :param best_solution: Instance of AbstractSolution class with the best solution found in this optimization
            algorithm iteration.

        :return: True if stop conditions were achieved, False otherwise.
        """
        return any([self._is_time_exceeded(start_time=start_time),
                    self._is_satisfying_solution_found(best_solution=best_solution),
                    self._is_limit_without_progress_exceeded(best_solution=best_solution)])

    def get_log_data(self) -> Dict[str, Union[str, int, float, None]]:
        """
        Gets data for logging purposes.

        :return: Dictionary with this Stop Conditions crucial data.
        """
        return {
            "time_limit": str(self.time_limit),
            "satisfying_objective_value": self.satisfying_objective_value,
            "max_iter_without_progress": self.max_iter_without_progress,
            "max_time_without_progress": str(self.max_time_without_progress)
        }
