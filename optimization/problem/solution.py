"""Optimization problem solution implementation."""

__all__ = ["AbstractSolution"]


from abc import ABC, abstractmethod

from .problem import OptimizationProblem


class AbstractSolution(ABC):
    """Abstract definition of optimization problem solution."""

    @property
    @abstractmethod
    def optimization_problem(self) -> OptimizationProblem:
        """Optimization problem for which this class is able to create solutions (as objects)."""
        ...

    # todo: finish implementation and prepare tests
