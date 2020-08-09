"""Crossover functions implementation that are used by Evolutionary Algorithms."""

__all__ = ["CrossoverType"]


from typing import Tuple
from enum import Enum
from collections import OrderedDict

from ...problem import AbstractSolution
from ...utilities import generate_random_int


ChildrenValues = Tuple[OrderedDict, OrderedDict]


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
    Uniform = "Uniform"
    Adaptive = "Adaptive"


def single_point_crossover(parents: Tuple[AbstractSolution, AbstractSolution], variables_number: int) -> ChildrenValues:
    """
    Single point crossover function.

    Randomly picks a single crossover point, then mixes parent genes (decision variables vales) in this points
    to produce a pair of children genes sets.

    :param parents: Pair of parent solution that provides genes for a new pair of children.
    :param variables_number: Number of decision variables.

    :return: Pair of children data sets.
    """
    crossover_point = generate_random_int(1, variables_number-1)
    parent_1_values = list(parents[0].decision_variables_values.items())
    parent_2_values = list(parents[1].decision_variables_values.items())
    child_1_values = OrderedDict(parent_1_values[:crossover_point])
    child_1_values.update(parent_2_values[crossover_point:])
    child_2_values = OrderedDict(parent_2_values[:crossover_point])
    child_2_values.update(parent_1_values[crossover_point:])
    return child_1_values, child_2_values
