from typing import List, Generator, Any
from enum import Enum

from optimization.optimization_algorithms.algorithm_definition import OptimizationAlgorithm
from optimization.optimization_algorithms.stop_conditions import StopCondition
from optimization.optimization_problem import OptimizationProblem, AbstractSolution
from optimization.utilities import choose_random_pair


class EvolutionaryAlgorithm(OptimizationAlgorithm):
    """Optimization algorithm that uses mechanism inspired by biological evolution."""

    def __init__(self, population_size: int, selection_function, crossover_function, mutation_type, elitism: bool, **other_params: Any) -> None:
        # todo: use super
        self.population_size = population_size

    # selection methods

    # todo: normal methods, only population is passed?
    @staticmethod
    def _selection_stochastic(population_size: int, population: List[AbstractSolution]) -> Generator[List[AbstractSolution]]:
        return (choose_random_pair(values_pool=population) for _ in range(population_size//2))

    @staticmethod
    def _selection_tournament():
        # todo
        pass

    @staticmethod
    def _selection_ranking(population_size: int, population: list) -> list:
        # todo
        pass

    @staticmethod
    def _selection_ranking_with_bias():
        # todo
        pass

    # crossover

    # mutation
