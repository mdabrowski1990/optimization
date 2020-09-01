"""
Adaptive evolutionary algorithms.

Evolutionary algorithms that performs two level optimization of:
 - optimization problem - main problem to solve
 - adaptation problem - searching for optimal settings of evolutionary settings
"""

__all__ = ["AdaptiveEvolutionaryAlgorithm"]


from typing import Any

from .evolutionary_algorithm import EvolutionaryAlgorithm


class AdaptiveEvolutionaryAlgorithm(EvolutionaryAlgorithm):
    """
    Adaptive Evolutionary Algorithm definition.

    This algorithm uses mechanisms inspired by biological evolution.
    It indirectly searches of optimal solution (_LowerAdaptiveEvolutionaryAlgorithm does that), while optimization
    optimization process effectiveness (it optimizes settings of _LowerAdaptiveEvolutionaryAlgorithm).
    """


class _LowerAdaptiveEvolutionaryAlgorithm(EvolutionaryAlgorithm):
    """
    Definition of lower level adaptive evolutionary algorithm.

    These algorithms (there are many of them during optimization process) search for optimal solution of
    main optimization problem.
    """

    def __init__(self, upper_iteration: int, index: int, **kwargs: Any) -> None:
        """
        Configuration of Lower Evolutionary Algorithm.

        :param upper_iteration: Iteration index of Upper Algorithm.
        :param index: Unique index of this Lower Evolutionary Algorithm in this Upper Algorithm iteration.
        :param kwargs: Other parameters:
            - initial_population - (optional) starting population to be set.
            - same parameters as in EvolutionaryAlgorithm.__init__
        """
        self.upper_iteration = upper_iteration
        self.index = index
        initial_population = kwargs.pop("initial_population", [])
        super().__init__(**kwargs)
        self._population = initial_population

    def _log_iteration(self, iteration_index: int) -> None:
        """
        Logs population data in given algorithm's iteration.

        :param iteration_index: Index number (counted from 0) of optimization algorithm iteration.

        :return: None
        """
        if self.logger is not None:
            self.logger.log_lower_level_iteration(upper_iteration=self.upper_iteration,
                                                  lower_algorithm_index=self.index,
                                                  lower_iteration=iteration_index,
                                                  solutions=self._population)
