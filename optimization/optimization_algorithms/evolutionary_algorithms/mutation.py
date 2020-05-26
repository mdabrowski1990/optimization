from typing import List
from enum import Enum

from optimization.utilities import generate_random_float, choose_random_values


__all__ = ["MutationType", "ADDITIONAL_MUTATION_PARAMETERS", "MUTATION_FUNCTIONS"]


MUTATION_POINTS = List[int]


class MutationType(Enum):
    SINGLE_POINT = "Single point"
    MULTI_POINT = "Multi point"
    PROBABILISTIC = "Probabilistic"

    def __repr__(self):
        return self.value


ADDITIONAL_MUTATION_PARAMETERS = {
    MutationType.SINGLE_POINT.value: {},
    MutationType.MULTI_POINT.value: {"mutation_points_number"},
    MutationType.PROBABILISTIC.value: {},
}


def single_point_mutation(variables_number: int, mutation_chance: float) -> MUTATION_POINTS:
    """
    Mutation function that changes gene(s).
    Mutation is performed according to single point mutation.

    :param variables_number: Number of decision variables (genes).
    :param mutation_chance: Probability of the mutation.

    :return: List of genes indices to be mutated.
    """
    scaled_mutation_chance = variables_number * mutation_chance
    if generate_random_float(0, 1) <= scaled_mutation_chance:
        return choose_random_values(values_pool=range(variables_number), values_number=1)
    return []


def multi_point_mutation(variables_number: int, mutation_chance: float, mutation_points_number: int) -> MUTATION_POINTS:
    """
    Mutation function that changes gene(s).
    Mutation is performed according to multi points mutation.

    :param variables_number: Number of decision variables (genes).
    :param mutation_chance: Probability of the mutation.
    :param mutation_points_number: Number of mutation points

    :return: List of genes indices to be mutated.
    """
    scaled_mutation_chance = variables_number * mutation_chance / mutation_points_number
    if generate_random_float(0, 1) <= scaled_mutation_chance:
        return choose_random_values(values_pool=range(variables_number), values_number=mutation_points_number)
    return []


def mutation_probabilistic_mutation(variables_number: int, mutation_chance: float) -> MUTATION_POINTS:
    """
    Mutation function that changes gene(s).
    Mutation is performed according to probabilistic mutation.

    :param variables_number: Number of decision variables (genes).
    :param mutation_chance: Probability of the mutation to be taking place.

    :return: List of genes indices to be mutated.
    """
    mutation_points = []
    for var_index in range(variables_number):
        if generate_random_float(0, 1) <= mutation_chance:
            mutation_points.append(var_index)
    return mutation_points


MUTATION_FUNCTIONS = {
    MutationType.SINGLE_POINT.value: single_point_mutation,
    MutationType.MULTI_POINT.value: multi_point_mutation,
    MutationType.PROBABILISTIC.value: mutation_probabilistic_mutation,
}