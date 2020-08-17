"""Mutation functions implementation that are used by Evolutionary Algorithms."""

__all__ = ["MutationType", "MUTATION_FUNCTIONS", "MUTATION_ADDITIONAL_PARAMS", "check_mutation_parameters"]


from typing import List, Dict, Callable, Tuple, Any
from enum import Enum

from ...utilities import generate_random_float, choose_random_value, choose_random_values


# mutation utilities


MutationPointsTyping = List[int]


def check_mutation_points_number(variables_number: int, mutation_points_number: int) -> None:
    """
    Check if 'mutation_points_number' has proper value.

    :param variables_number: Number of decision variables (genes).
    :param mutation_points_number: Number of mutation points to select in 'multi_point_mutation'.

    :raise TypeError: Value of parameter is not int type.
    :raise ValueError: Value of parameter is not in proper range.
    """
    if not isinstance(mutation_points_number, int):
        raise TypeError(f"Parameter 'mutation_points_number' is not int type. Actual value: {mutation_points_number}.")
    if not 2 <= mutation_points_number < variables_number:
        raise ValueError(f"Parameter 'mutation_points_number' has invalid value. "
                         f"Expected value: 2 <= mutation_points_number < {variables_number}. "
                         f"Actual value: {mutation_points_number}.")

# mutation functions


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


# outputs (visible outside)


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


MUTATION_FUNCTIONS: Dict[str, Callable] = {
    # mutation type: mutation function
    MutationType.SinglePoint.value: single_point_mutation,
    MutationType.MultiPoint.value: multi_point_mutation,
    MutationType.Probabilistic.value: probabilistic_mutation,
}


MUTATION_ADDITIONAL_PARAMS: Dict[str, Tuple[str, ...]] = {
    # mutation type: (parameter 1 name, parameter 2 name, ...)
    MutationType.SinglePoint.value: (),
    MutationType.MultiPoint.value: ("mutation_points_number", ),
    MutationType.Probabilistic.value: (),
}


def check_mutation_parameters(variables_number: int, **mutation_params: Any) -> None:
    """
    Checks whether additional crossover parameters (crossover function specific) have proper value.

    :param variables_number: Number of decision variables (genes).
    :param mutation_params: Values of additional mutation parameters.
    """
    if "mutation_points_number" in mutation_params:
        check_mutation_points_number(variables_number, mutation_params["mutation_points_number"])
