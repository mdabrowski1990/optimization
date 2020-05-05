from enum import Enum

from optimization.utilities import generate_random_int, choose_random_value
from optimization.optimization_problem import AbstractSolution


class CrossoverFunctions(Enum):
    """Enum with crossover functions that are available for Evolutionary Algorithm"""

    @staticmethod
    def single(individual: AbstractSolution, mutation_chance: float) -> None:
        """
        Mutation function that provides random gene.
        Crossover is performed according to single point crossover.

        :param individual: Individual that may be mutated.
        :param mutation_chance: Probability of the mutation.

        :return: Individual after mutation process.
        """
        if generate_random_int(0, 1) <= mutation_chance:
            mutation_point = choose_random_value(individual.decision_variables_values.keys())
            individual.decision_variables_values[mutation_point] = \
                individual.optimization_problem.decision_variables[mutation_point].generate_random_value()

