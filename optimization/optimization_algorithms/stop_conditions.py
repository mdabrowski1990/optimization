from typing import List
from datetime import datetime, timedelta

from optimization.optimization_problem.problem import OptimizationType


class StopCondition:
    """
    Base definition of Stop Condition for Optimization Algorithm due to exceeding time limit.
    Each Stop Condition must also have time limit defined (avoid infinite optimization).
    """

    def __init__(self, time_limit: timedelta) -> None:
        """
        Creates definition of basic stop condition.

        :param time_limit: Time limit after which optimization should be stopped.
        """
        self.time_limit = time_limit

    def is_achieved(self, start_time: datetime, solutions: List) -> bool:
        """
        Checks whether stop condition was achieved and optimization process should be stopped.

        :param start_time: Time when optimization process was started.
        :param solutions: List of solutions that were created in last iteration of optimization process.

        :return: True when time limit restriction is exceeded, False if it is not.
        """
        return start_time - datetime.now() >= self.time_limit

    def get_data_for_logging(self) -> dict:
        """
        Method which prepares data of the instance of this class for logging.

        :return: Crucial data of this object.
        """
        return {
            "time_limit": repr(self.time_limit)
        }


class StopDueSatisfyingSolutionFound(StopCondition):
    """Definition of Stop Condition for Optimization Algorithm due to finding satisfying solution."""

    def __init__(self, time_limit: timedelta, satisfying_objective_value: float) -> None:
        """
        Creates definition of the stop condition.

        :param time_limit: Time limit after which optimization should be stopped.
        :param satisfying_objective_value: Boundary value of the objective function. When solution with better or equal
            objective value is found, then optimization process is stopped.
        """
        super().__init__(time_limit)
        self.satisfying_objective_value = satisfying_objective_value

    def is_achieved(self, start_time: datetime, solutions: List) -> bool:
        """
        Checks whether stop condition was achieved and optimization process should be stopped.

        :param start_time: Time when optimization process was started.
        :param solutions: List of solutions that were created in last iteration of optimization process.

        :return: True when time restriction or satisfying solution is found, otherwise False.
        """
        optimization_type = solutions[0].optimization_problem.optimization_type
        if optimization_type == OptimizationType.Minimize:
            return super().is_achieved(start_time=start_time, solutions=solutions) or \
                   solutions[0].get_objective_value_with_penalty() <= self.satisfying_objective_value
        elif optimization_type == OptimizationType.Maximize:
            return super().is_achieved(start_time=start_time, solutions=solutions) or \
                   solutions[0].get_objective_value_with_penalty() >= self.satisfying_objective_value
        else:
            raise ValueError

    def get_data_for_logging(self) -> dict:
        """
        Method which prepares data of the instance of this class for logging.

        :return: Crucial data of this object.
        """
        base_data = super().get_data_for_logging()
        base_data.update({"satisfying_objective_value": self.satisfying_objective_value})
        return base_data


class StopDueNoProgressInFewIterations(StopCondition):
    """
    Definition of Stop Condition for Optimization Algorithm due to no progress of optimization process in a few
    iterations.
    """
    # todo
    pass


class StopDueNoProgressForSomeTime(StopCondition):
    """
    Definition of Stop Condition for Optimization Algorithm due to no progress of optimization process for some time.
    """
    # todo
    pass

