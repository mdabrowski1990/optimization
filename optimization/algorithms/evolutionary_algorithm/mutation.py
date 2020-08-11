"""Mutation functions implementation that are used by Evolutionary Algorithms."""

__all__ = ["MutationType"]


from typing import List
from enum import Enum

from ...utilities import generate_random_float, choose_random_value, choose_random_values


MutationPointsTyping = List[int]


class MutationType(Enum):
    """
    Enum with available mutation function types.

    Options:
        - SinglePoint - Chooses a single position of decision variable (gene) for mutation.
        - MultiPoint - Chooses multiple positions of decision variables (genes) for mutation.
        - Probabilistic - Each decision variable (gene) for mutation is chosen independently with the same probability.
    """

    SinglePoint = "SinglePoint"
    MultiPoint = "MultiPoint"
    Probabilistic = "Probabilistic"


def single_point_mutation(variables_number: int,
                          mutation_chance: float) -> MutationPointsTyping:
    """
    Single point mutation function.

    Having in mind 'mutation_chance', selects a single (or none) decision variable to be mutated.

    :param variables_number: Number of decision variables (genes).
    :param mutation_chance: Probability of single decision variable (gene) mutation.

    :return: List of decision variables (genes) positions to be mutated.
    """
    scaled_mutation_chance = variables_number * mutation_chance
    if generate_random_float(0, 1) <= scaled_mutation_chance:
        return [choose_random_value(values_pool=range(variables_number))]
    return []


def multi_point_mutation(variables_number: int,
                         mutation_chance: float,
                         mutation_points_number: int) -> MutationPointsTyping:
    """
    Multi point mutation function.

    Having in mind 'mutation_chance', selects [mutation_points_number] (or none) decision variables to be mutated.

    :param variables_number: Number of decision variables (genes).
    :param mutation_chance: Probability of single decision variable (gene) mutation.
    :param mutation_points_number: Number of mutation points to select.

    :return: List of decision variables (genes) positions to be mutated.
    """
    scaled_mutation_chance = variables_number * mutation_chance / mutation_points_number
    if generate_random_float(0, 1) <= scaled_mutation_chance:
        return choose_random_values(values_pool=range(variables_number), values_number=mutation_points_number)
    return []


def probabilistic_mutation(variables_number: int,
                           mutation_chance: float) -> MutationPointsTyping:
    """
    Probabilistic mutation function.

    Having in mind 'mutation_chance', selects decision variables to be mutated with the same probability.
    Number of selected decision variables can be any value between 0 and [variables_number].

    :param variables_number: Number of decision variables (genes).
    :param mutation_chance: Probability of single decision variable (gene) mutation.

    :return: List of decision variables (genes) positions to be mutated.
    """
    mutation_points = []
    for var_index in range(variables_number):
        if generate_random_float(0, 1) <= mutation_chance:
            mutation_points.append(var_index)
    return mutation_points
