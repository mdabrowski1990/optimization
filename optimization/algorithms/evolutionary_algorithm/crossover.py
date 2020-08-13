"""Crossover functions implementation that are used by Evolutionary Algorithms."""

__all__ = ["CrossoverType", "CROSSOVER_FUNCTIONS", "CROSSOVER_ADDITIONAL_PARAMS", "check_crossover_parameters"]


from typing import Tuple, Dict, Callable, Any
from enum import Enum
from collections import OrderedDict

from ...problem import AbstractSolution
from ...utilities import generate_random_int, choose_random_values


# crossover utilities


ChildrenValuesTyping = Tuple[OrderedDict, OrderedDict]


def check_crossover_points_number(variables_number: int, crossover_points_number: int) -> None:
    """
    Check if 'crossover_points_number' has proper value.

    :param variables_number: Number of decision variables (genes).
    :param crossover_points_number: Number of crossover points to use in 'multi_point_crossover'.

    :raise TypeError: Value of parameter is not int type.
    :raise ValueError: Value of parameter is not in proper range.
    """
    if not isinstance(crossover_points_number, int):
        raise TypeError(f"Parameter 'crossover_points_number' is not int type. "
                        f"Actual value: {crossover_points_number}.")
    if not 2 <= crossover_points_number < variables_number:
        raise ValueError(f"Parameter 'crossover_points_number' has invalid value. "
                         f"Expected value: 2 <= crossover_points_number < {variables_number}. "
                         f"Actual value: {crossover_points_number}.")


def check_crossover_pattern(variables_number: int, crossover_pattern: int) -> None:
    """
    Check if 'crossover_pattern' has proper value.

    :param variables_number: Number of decision variables (genes).
    :param crossover_pattern: Pattern of crossover to be used in 'adaptive_crossover'.

    :raise TypeError: Value of parameter is not int type.
    :raise ValueError: Value of parameter is not in proper range.
    """
    if not isinstance(crossover_pattern, int):
        raise TypeError(f"Parameter 'crossover_pattern' is not int type. Actual value: {crossover_pattern}.")
    max_pattern_value = (1 << variables_number) - 1
    if not 0 < crossover_pattern < max_pattern_value:
        raise ValueError(f"Parameter 'crossover_pattern' has invalid value. "
                         f"Expected value: 0 < crossover_pattern < {max_pattern_value}. "
                         f"Actual value: {crossover_pattern}.")

# crossover functions


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


# outputs (visible outside)


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


CROSSOVER_FUNCTIONS: Dict[str, Callable] = {
    # crossover type: crossover function
    CrossoverType.SinglePoint.value: single_point_crossover,
    CrossoverType.MultiPoint.value: multi_point_crossover,
    CrossoverType.Adaptive.value: adaptive_crossover,
    CrossoverType.Uniform.value: uniform_crossover,
}


CROSSOVER_ADDITIONAL_PARAMS: Dict[str, Tuple[str, ...]] = {
    # crossover type: (parameter 1 name, parameter 2 name, ...)
    CrossoverType.SinglePoint.value: (),
    CrossoverType.MultiPoint.value: ("crossover_points_number", ),
    CrossoverType.Adaptive.value: ("crossover_pattern", ),
    CrossoverType.Uniform.value: (),
}


def check_crossover_parameters(variables_number: int, **crossover_params: Any) -> None:
    """
    Checks whether additional crossover parameters (crossover function specific) have proper value.

    :param variables_number: Number of decision variables (genes).
    :param crossover_params: Values of additional crossover parameters.
    """
    if "crossover_points_number" in crossover_params:
        check_crossover_points_number(variables_number, crossover_params["crossover_points_number"])
    if "crossover_pattern" in crossover_params:
        check_crossover_pattern(variables_number, crossover_params["crossover_pattern"])
