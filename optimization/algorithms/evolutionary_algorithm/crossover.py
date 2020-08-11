"""Crossover functions implementation that are used by Evolutionary Algorithms."""

__all__ = ["CrossoverType"]


from typing import Tuple
from enum import Enum
from collections import OrderedDict

from ...problem import AbstractSolution
from ...utilities import generate_random_int, choose_random_values


ChildrenValuesTyping = Tuple[OrderedDict, OrderedDict]


class CrossoverType(Enum):
    """
    Enum with available crossover function types.

    Options:
        - SinglePoint - Mixes genes of two parents with a single point of crossover.
        - MultiPoint - Mixes genes of two parents with many points of crossover.
        - Uniform - Each gene is independently picked from randomly picked parent.
        - Adaptive - Genes are mixed according to crossover pattern.
    """

    SinglePoint = "SinglePoint"
    MultiPoint = "MultiPoint"
    Adaptive = "Adaptive"
    Uniform = "Uniform"


def single_point_crossover(parents: Tuple[AbstractSolution, AbstractSolution],
                           variables_number: int) -> ChildrenValuesTyping:
    """
    Single point crossover function.

    Randomly picks a single crossover point, then mixes parents genes (decision variables values) in this point
    to produce a pair of children genes sets.

    :param parents: Pair of parent solution that provides genes for a new pair of children.
    :param variables_number: Number of decision variables (genes).

    :return: Pair of children data sets.
    """
    crossover_point = generate_random_int(1, variables_number-1)
    parent_1_values = list(parents[0].decision_variables_values.items())
    parent_2_values = list(parents[1].decision_variables_values.items())
    child_1_values = OrderedDict(parent_1_values[:crossover_point] + parent_2_values[crossover_point:])
    child_2_values = OrderedDict(parent_2_values[:crossover_point] + parent_1_values[crossover_point:])
    return child_1_values, child_2_values


def multi_point_crossover(parents: Tuple[AbstractSolution, AbstractSolution],
                          variables_number: int,
                          crossover_points_number: int) -> ChildrenValuesTyping:
    """
    Multi point crossover function.

    Randomly picks points of crossover, then mixes parents genes (decision variables values) in this points
    to produce a pair of children genes sets.

    :param parents: Pair of parent solution that provides genes for a new pair of children.
    :param variables_number: Number of decision variables (genes).
    :param crossover_points_number: Number of crossover points to use.

    :return: Pair of children data sets.
    """
    crossover_points = choose_random_values(values_pool=range(1, variables_number),
                                            values_number=crossover_points_number)
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


def adaptive_crossover(parents: Tuple[AbstractSolution, AbstractSolution],
                       variables_number: int,
                       crossover_pattern: int) -> ChildrenValuesTyping:
    """
    Adaptive crossover function.

    Crossover is performed according to a pattern that determines which gene (decision variable) value
    should be picked from which parent.

    :param parents: Pair of parent solution that provides genes for a new pair of children.
    :param variables_number: Number of decision variables (genes).
    :param crossover_pattern: Pattern of crossover to be used.

    :return: Pair of children data sets.
    """
    parents_values = list(parents[0].decision_variables_values.items()), \
        list(parents[1].decision_variables_values.items())
    child_1_values: OrderedDict = OrderedDict()
    child_2_values: OrderedDict = OrderedDict()
    for i in range(variables_number):
        pattern_value = (crossover_pattern >> i) & 1
        child_1_values.update([parents_values[pattern_value][i]])
        child_2_values.update([parents_values[pattern_value ^ 1][i]])
    return child_1_values, child_2_values


def uniform_crossover(parents: Tuple[AbstractSolution, AbstractSolution],
                      variables_number: int) -> ChildrenValuesTyping:
    """
    Uniform crossover function.

    Each gene (decision variables) on each position is picked from randomly chosen parent.
    Each decision is independent from other each other.

    :param parents: Pair of parent solution that provides genes for a new pair of children.
    :param variables_number: Number of decision variables (genes).

    :return: Pair of children data sets.
    """
    random_pattern = generate_random_int(0, (1 << variables_number) - 1)
    return adaptive_crossover(parents=parents, variables_number=variables_number, crossover_pattern=random_pattern)
