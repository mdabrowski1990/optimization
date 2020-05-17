from enum import Enum

from optimization.utilities import generate_random_int, choose_random_value, choose_random_values
from optimization.optimization_problem import AbstractSolution


__all__ = ["MutationType", "ADDITIONAL_MUTATION_PARAMETERS", "MUTATION_FUNCTIONS"]


class MutationType(Enum):
    SINGLE_POINT = "Single point"
    MULTI_POINT = "Multi point"
    PROBABILISTIC = "Probabilistic"


ADDITIONAL_MUTATION_PARAMETERS = {
    MutationType.SINGLE_POINT.value: {},
    MutationType.MULTI_POINT.value: {"mutation_points_number"},
    MutationType.PROBABILISTIC.value: {},
}


def single_point_mutation(individual: AbstractSolution, mutation_chance: float) -> None:
    """
    Mutation function that changes gene(s).
    Mutation is performed according to single point mutation.

    :param individual: Individual that may be mutated.
    :param mutation_chance: Probability of the mutation.

    :return: None
    """
    if generate_random_int(0, 1) <= mutation_chance:
        mutation_point = choose_random_value(individual.decision_variables_values.keys())
        individual.decision_variables_values[mutation_point] = \
            individual.optimization_problem.decision_variables[mutation_point].generate_random_value()


def multi_point_mutation(individual: AbstractSolution, mutation_chance: float, mutation_points_number: int) -> None:
    """
    Mutation function that changes gene(s).
    Mutation is performed according to multi points mutation.

    :param individual: Individual that may be mutated.
    :param mutation_chance: Probability of the mutation.
    :param mutation_points_number: Number of mutation points

    :return: None
    """
    scaled_mutation_chance = mutation_chance/mutation_points_number
    if generate_random_int(0, 1) <= scaled_mutation_chance:
        mutation_points = choose_random_values(individual.decision_variables_values.keys(), mutation_points_number)
        for mutation_point in mutation_points:
            individual.decision_variables_values[mutation_point] = \
                individual.optimization_problem.decision_variables[mutation_point].generate_random_value()


def mutation_probabilistic_mutation(individual: AbstractSolution, mutation_chance: float) -> None:
    """
    Mutation function that changes gene(s).
    Mutation is performed according to probabilistic mutation.

    :param individual: Individual that may be mutated.
    :param mutation_chance: Probability of the mutation to be taking place.

    :return: None
    """
    # todo: update scaling formula
    scaled_mutation_chance = pow(mutation_chance, len(individual.decision_variables_values))
    for variable_name in individual.decision_variables_values.keys():
        if generate_random_int(0, 1) <= scaled_mutation_chance:
            individual.decision_variables_values[variable_name] = \
                individual.optimization_problem.decision_variables[variable_name].generate_random_value()


MUTATION_FUNCTIONS = {
    MutationType.SINGLE_POINT.value: single_point_mutation,
    MutationType.MULTI_POINT.value: multi_point_mutation,
    MutationType.PROBABILISTIC.value: mutation_probabilistic_mutation,
}