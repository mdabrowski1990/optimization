"""Classic evolutionary algorithm."""

# __all__ = ["EvolutionaryAlgorithm"]
#
#
# from typing import Optional
#
# from ..abstract_algorithm import AbstractOptimizationAlgorithm
# from ...problem import OptimizationProblem
# from ...stop_conditions import StopConditions
# from ...logging import AbstractLogger
#
#
# class EvolutionaryAlgorithm(AbstractOptimizationAlgorithm):
#     """
#     Evolutionary optimization algorithm implementation.
#
#     This algorithm uses mechanisms inspired by biological evolution in searches of optimal solution
#     for given optimization problem.
#     """
#
#     def __init__(self, problem: OptimizationProblem,
#                  stop_conditions: StopConditions,
#                  logger: Optional[AbstractLogger] = None) -> None:
#         super().__init__(problem=problem, stop_conditions=stop_conditions, logger=logger)
