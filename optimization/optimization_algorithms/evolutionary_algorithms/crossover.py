from typing import Tuple
from enum import Enum

from optimization.utilities import generate_random_int, choose_random_values
from optimization.optimization_problem import AbstractSolution


IndividualsPair = Tuple[AbstractSolution, AbstractSolution]


class CrossoverFunctions(Enum):
    """Enum with crossover functions that are available for Evolutionary Algorithm"""

    @staticmethod
    def single_point(parents: IndividualsPair, variables_number: int, solution_class: type) -> IndividualsPair:
        """
        Crossover function that mixes genes of two parents and returns new children.
        Crossover is performed according to single point crossover.

        :param parents: Pair of parents that provides genes.
        :param variables_number: Number of decision variables (genes).
        :param solution_class: Definition of solution (class of which objects must be returned).

        :return: Children solutions pair.
        """
        crossover_point = generate_random_int(1, variables_number-1)
        parent_1_values = list(parents[0].decision_variables_values.items())
        parent_2_values = list(parents[1].decision_variables_values.items())
        child_1_values = dict(parent_1_values[:crossover_point])
        child_1_values.update(parent_2_values[crossover_point:])
        child_2_values = dict(parent_2_values[:crossover_point])
        child_2_values.update(parent_1_values[crossover_point:])
        return solution_class(**child_1_values), solution_class(**child_2_values)

    @staticmethod
    def multi_point(parents: IndividualsPair, variables_number: int, solution_class: type, number_of_points) \
            -> IndividualsPair:
        """
        Crossover function that mixes genes of two parents and returns new children.
        Crossover is performed according to multi point crossover.

        :param parents: Pair of parents that provides genes.
        :param variables_number: Number of decision variables (genes).
        :param solution_class: Definition of solution (class of which objects must be returned).
        :param number_of_points: Number of crossover points.

        :return: Children solutions pair.
        """
        crossover_points = choose_random_values(range(1, variables_number-1), number_of_points)
        parent_1_values = list(parents[0].decision_variables_values.items())
        parent_2_values = list(parents[1].decision_variables_values.items())
        child_1_values = dict(parent_1_values[:crossover_points[0]])
        child_2_values = dict(parent_2_values[:crossover_points[0]])
        for i, current_point in enumerate(crossover_points[1:]):
            previous_point = crossover_points[i]
            if i & 1:
                child_1_values.update(parent_1_values[previous_point:current_point])
                child_2_values.update(parent_2_values[previous_point:current_point])
            else:
                child_1_values.update(parent_2_values[previous_point:current_point])
                child_2_values.update(parent_1_values[previous_point:current_point])
        return solution_class(**child_1_values), solution_class(**child_2_values)

    @staticmethod
    def uniform(parents: IndividualsPair, variables_number: int, solution_class: type) -> IndividualsPair:
        """
        Crossover function that mixes genes of two parents and returns new children.
        Crossover is performed according to uniform crossover.

        :param parents: Pair of parents that provides genes.
        :param variables_number: Number of decision variables (genes).
        :param solution_class: Definition of solution (class of which objects must be returned).

        :return: Children solutions pair.
        """
        parent_1_values = list(parents[0].decision_variables_values.items())
        parent_2_values = list(parents[1].decision_variables_values.items())
        child_1_values = dict(parent_1_values[:1])
        child_2_values = dict(parent_2_values[:1])
        for i in range(1, variables_number):
            if generate_random_int(0, 1):
                child_1_values.update(parent_1_values[i:i+1])
                child_2_values.update(parent_2_values[i:i+1])
            else:
                child_1_values.update(parent_2_values[i:i+1])
                child_2_values.update(parent_1_values[i:i+1])
        return solution_class(**child_1_values), solution_class(**child_2_values)

    @staticmethod
    def pattern_based(parents: IndividualsPair, variables_number: int, solution_class: type, pattern: int) \
            -> IndividualsPair:
        """
        Crossover function that mixes genes of two parents and returns new children.
        Crossover is performed according to pattern based crossover.

        :param parents: Pair of parents that provides genes.
        :param variables_number: Number of decision variables (genes).
        :param solution_class: Definition of solution (class of which objects must be returned).
        :param pattern: Pattern of crossover.

        :return: Children solutions pair.
        """
        parent_1_values = list(parents[0].decision_variables_values.items())
        parent_2_values = list(parents[1].decision_variables_values.items())
        child_1_values = {}
        child_2_values = {}
        for i in range(variables_number):
            if (pattern << i) & 1:
                child_1_values.update(parent_1_values[i:i+1])
                child_2_values.update(parent_2_values[i:i+1])
            else:
                child_1_values.update(parent_2_values[i:i+1])
                child_2_values.update(parent_1_values[i:i+1])
        return solution_class(**child_1_values), solution_class(**child_2_values)
