from typing import Tuple
from enum import Enum
from collections import OrderedDict

from optimization.utilities import generate_random_int, choose_random_values
from optimization.optimization_problem import AbstractSolution


__all__ = ["CrossoverType", "ADDITIONAL_CROSSOVER_PARAMETERS", "CROSSOVER_FUNCTIONS"]


Parents = Tuple[AbstractSolution, AbstractSolution]
Children = Tuple[OrderedDict, OrderedDict]


class CrossoverType(Enum):
    SINGLE_POINT = "Single point"
    MULTI_POINT = "Multi point"
    UNIFORM = "Uniform"
    ADAPTIVE = "Adaptive"


ADDITIONAL_CROSSOVER_PARAMETERS = {
    CrossoverType.SINGLE_POINT.value: {},
    CrossoverType.MULTI_POINT.value: {"crossover_points_number"},
    CrossoverType.UNIFORM.value: {},
    CrossoverType.ADAPTIVE.value: {"crossover_pattern"},
}


def single_point_crossover(parents: Parents, variables_number: int) -> Children:
    """
    Crossover function that mixes genes of two parents and returns new children.
    Crossover is performed according to single point crossover.

    :param parents: Pair of parents that provides genes.
    :param variables_number: Number of decision variables (genes).

    :return: Values of children solutions pair.
    """
    crossover_point = generate_random_int(1, variables_number-1)
    parent_1_values = list(parents[0].decision_variables_values.items())
    parent_2_values = list(parents[1].decision_variables_values.items())
    child_1_values = OrderedDict(parent_1_values[:crossover_point])
    child_1_values.update(parent_2_values[crossover_point:])
    child_2_values = OrderedDict(parent_2_values[:crossover_point])
    child_2_values.update(parent_1_values[crossover_point:])
    return child_1_values, child_2_values


def multi_point_crossover(parents: Parents, variables_number: int, crossover_points_number: int) -> Children:
    """
    Crossover function that mixes genes of two parents and returns new children.
    Crossover is performed according to multi point crossover.

    :param parents: Pair of parents that provides genes.
    :param variables_number: Number of decision variables (genes).
    :param crossover_points_number: Number of crossover points.

    :return: Values of children solutions pair.
    """
    crossover_points = choose_random_values(range(1, variables_number), crossover_points_number)
    parent_1_values = list(parents[0].decision_variables_values.items())
    parent_2_values = list(parents[1].decision_variables_values.items())
    child_1_values = OrderedDict(parent_1_values[:crossover_points[0]])
    child_2_values = OrderedDict(parent_2_values[:crossover_points[0]])
    for i, current_point in enumerate(crossover_points[1:] + [variables_number]):
        previous_point = crossover_points[i]
        if i & 1:
            child_1_values.update(parent_1_values[previous_point:current_point])
            child_2_values.update(parent_2_values[previous_point:current_point])
        else:
            child_1_values.update(parent_2_values[previous_point:current_point])
            child_2_values.update(parent_1_values[previous_point:current_point])
    return child_1_values, child_2_values


def uniform_crossover(parents: Parents, variables_number: int) -> Children:
    """
    Crossover function that mixes genes of two parents and returns new children.
    Crossover is performed according to uniform crossover.

    :param parents: Pair of parents that provides genes.
    :param variables_number: Number of decision variables (genes).

    :return: Values of children solutions pair.
    """
    parent_1_values = list(parents[0].decision_variables_values.items())
    parent_2_values = list(parents[1].decision_variables_values.items())
    child_1_values = OrderedDict(parent_1_values[:1])
    child_2_values = OrderedDict(parent_2_values[:1])
    for i in range(1, variables_number):
        if generate_random_int(0, 1):
            child_1_values.update(parent_1_values[i:i+1])
            child_2_values.update(parent_2_values[i:i+1])
        else:
            child_1_values.update(parent_2_values[i:i+1])
            child_2_values.update(parent_1_values[i:i+1])
    return child_1_values, child_2_values


def adaptive_crossover(parents: Parents, variables_number: int, crossover_pattern: int) -> Children:
    """
    Crossover function that mixes genes of two parents and returns new children.
    Crossover is performed according to pattern based crossover.

    :param parents: Pair of parents that provides genes.
    :param variables_number: Number of decision variables (genes).
    :param crossover_pattern: Pattern of crossover.

    :return: Values of children solutions pair.
    """
    parent_1_values = list(parents[0].decision_variables_values.items())
    parent_2_values = list(parents[1].decision_variables_values.items())
    child_1_values = OrderedDict()
    child_2_values = OrderedDict()
    for i in range(variables_number):
        if (crossover_pattern << i) & 1:
            child_1_values.update(parent_1_values[i:i+1])
            child_2_values.update(parent_2_values[i:i+1])
        else:
            child_1_values.update(parent_2_values[i:i+1])
            child_2_values.update(parent_1_values[i:i+1])
    return child_1_values, child_2_values


CROSSOVER_FUNCTIONS = {
    CrossoverType.SINGLE_POINT.value: single_point_crossover,
    CrossoverType.MULTI_POINT.value: multi_point_crossover,
    CrossoverType.UNIFORM.value: uniform_crossover,
    CrossoverType.ADAPTIVE.value: adaptive_crossover,
}

