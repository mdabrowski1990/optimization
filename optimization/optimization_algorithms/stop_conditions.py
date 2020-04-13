from abc import ABC

# todo: design stop conditions


class StopCondition(ABC):
    pass


class StopDueTimeExceeding(StopCondition):
    pass


class StopDueSatisfyingSolutionFound(StopCondition):
    pass


class StopDueNoProgress(StopCondition):
    pass

