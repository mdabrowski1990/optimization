"""
Selection functions implementation that are used by Evolutionary Algorithms.

Code is basing on  information from:
'Introduction to Evolutionary Computing. Second edition.' Eiben, A.E., Smith, James E.
"""

__all__ = ["SelectionType", "SELECTION_FUNCTIONS", "SELECTION_ADDITIONAL_PARAMS", "SELECTION_ADDITIONAL_PARAMS_LIMITS",
           "check_selection_parameters", "SelectionOutput"]


from typing import List, Iterator, Union, Tuple, Dict, Callable, Any
from enum import Enum

from ...problem import AbstractSolution
from ...utilities import choose_random_values, choose_random_value_with_weights


# selection utilities


SelectionOutput = Iterator[Tuple[AbstractSolution, AbstractSolution]]

MIN_TOURNAMENT_GROUP_SIZE = 3
MAX_TOURNAMENT_GROUP_SIZE = 8
MIN_ROULETTE_BIAS = 1.
MAX_ROULETTE_BIAS = 100.
MIN_RANKING_BIAS = 1.
MAX_RANKING_BIAS = 2.


def calculate_roulette_scaling(best_solution: AbstractSolution,
                               worst_solution: AbstractSolution,
                               roulette_bias: Union[float, int]) -> Tuple[Union[float, int], Union[float, int]]:
    """
    Calculates factor and offset for scaling of objective value according to 'roulette_bias' value.

    :param best_solution: The best solution in this iteration.
    :param worst_solution: The worst solution in this iteration.
    :param roulette_bias: Bias towards promoting better adopted individuals.
        [probability of picking the best adopted individual] = \
            roulette_bias * [probability of picking the worst adopted individual]

    :return: Tuple with factor and offset needed to rescaling of objective value.
    """
    factor = (roulette_bias - 1) / \
             (best_solution.get_objective_value_with_penalty() - worst_solution.get_objective_value_with_penalty())
    offset = 1 - factor * worst_solution.get_objective_value_with_penalty()
    return factor, offset


def get_scaled_objective(solution: AbstractSolution,
                         factor: Union[float, int],
                         offset: Union[float, int]) -> Union[float, int]:
    """
    Calculates scaled value of solution objective.

    :param solution: Solution for which objective value to be scaled.
    :param factor: Factor used during scaling.
    :param offset: Offset used during scaling.

    :return: Scaled value of solution objective.
    """
    return factor*solution.get_objective_value_with_penalty() + offset


def get_scaled_ranking(rank: int, population_size: int, ranking_bias: Union[float, int]) -> Union[float, int]:
    """
    Calculates scaled value of ranking.

    Algorithm used is proposed by Eiben, A.E., Smith, James E in
    'Introduction to Evolutionary Computing. Second edition.' (page 82).
    Note: Division by 'population_size' was skipped to reduce code complexity.

    :param rank: Rank of the solution in population (0: the worst, 1: second worst, ..., population_size-1: the best).
    :param population_size: Size of the population for which ranking scaling is performed.
    :param ranking_bias: Bias that represents selection pressure (the higher the value the higher the pressure).
        When ranking_bias == 1, then the function acts the same effect as uniform selection.
        When ranking_bias == 2, then the function acts like classic ranking selection.
        Expected value: 1 < ranking_bias <= 2

    :return: Scaled value of the solution ranking.
    """
    return 2 - ranking_bias + (2 * rank * (ranking_bias - 1) / (population_size - 1))


def check_tournament_group_size(tournament_group_size: int) -> None:
    """
    Checks if 'tournament_group_size' has proper value.

    :param tournament_group_size: Size of a random groups for
        'tournament_selection' or 'double_tournament_selection' function.

    :raise TypeError: Value of parameter is not int type.
    :raise ValueError: Value of parameter is not in proper range.
    """
    if not isinstance(tournament_group_size, int):
        raise TypeError(f"Parameter 'tournament_group_size' is not int type. Actual value: {tournament_group_size}.")
    if not MIN_TOURNAMENT_GROUP_SIZE <= tournament_group_size <= MAX_TOURNAMENT_GROUP_SIZE:
        raise ValueError(f"Parameter 'tournament_group_size' has invalid value. "
                         f"Expected value: 3 <= tournament_group_size <= 8. Actual value: {tournament_group_size}.")


def check_roulette_bias(roulette_bias: Union[float, int]) -> None:
    """
    Checks if 'roulette_bias' has proper value.

    :param roulette_bias: Bias towards promoting better adopted individuals for 'roulette_selection' function.

    :raise TypeError: Value of parameter is not int nor float type.
    :raise ValueError: Value of parameter is not in proper range.
    """
    if not isinstance(roulette_bias, (int, float)):
        raise TypeError(f"Parameter 'roulette_bias' is not int nor float type. Actual value: {roulette_bias}.")
    if not MIN_ROULETTE_BIAS < roulette_bias <= MAX_ROULETTE_BIAS:
        raise ValueError(f"Parameter 'roulette_bias' has invalid value. Expected value: 1 < roulette_bias <= 100. "
                         f"Actual value: {roulette_bias}.")


def check_ranking_bias(ranking_bias: float) -> None:
    """
    Checks if 'ranking_bias' has proper value.

    :param ranking_bias: Bias that represents selection pressure (the higher the value the higher the pressure)
        for 'ranking_Selection' function.

    :raise TypeError: Value of parameter is not float type.
    :raise ValueError: Value of parameter is not in proper range.
    """
    if not isinstance(ranking_bias, float):
        raise TypeError(f"Parameter 'ranking_bias' is not float type. Actual value: {ranking_bias}.")
    if not MIN_RANKING_BIAS < ranking_bias <= MAX_RANKING_BIAS:
        raise ValueError(f"Parameter 'ranking_bias' has invalid value. Expected value: 1 < ranking_bias <= 2. "
                         f"Actual value: {ranking_bias}.")


# selection functions


def uniform_selection(population_size: int,
                      population: List[AbstractSolution]) -> SelectionOutput:
    """
    Uniform selection function that picks each individual with the same probability.

    :param population_size: Size of the population (number of parents to pick).
    :param population: List with individuals (solutions) from which parents to be selected.

    :return: Generator producing individual pairs.
    """
    for _ in range(population_size // 2):
        yield tuple(choose_random_values(values_pool=population, values_number=2))  # type: ignore


def tournament_selection(population_size: int,
                         population: List[AbstractSolution],
                         tournament_group_size: int) -> SelectionOutput:
    """
    Tournament selection function that picks the two best individuals from a small random groups as new parents pair.

    :param population_size: Size of the population (number of parents to pick).
    :param population: List with individuals (solutions) from which parents to be selected.
    :param tournament_group_size: Size of a random groups.

    :return: Generator producing individual pairs.
    """
    for _ in range(population_size // 2):
        random_group = choose_random_values(values_pool=population, values_number=tournament_group_size)
        best = max(random_group)
        random_group.remove(best)
        second_best = max(random_group)
        yield best, second_best


def double_tournament_selection(population_size: int,
                                population: List[AbstractSolution],
                                tournament_group_size: int) -> SelectionOutput:
    """
    Double tournament selection function similar to tournament selection.

    Double tournament differs from classic tournament selection due to the fact that only one parent is picked
    from a single tournament group. Second parent is determined in similar way as the first one (process reoccurs).

    :param population_size: Size of the population (number of parents to pick).
    :param population: List with individuals (solutions) from which parents to be selected.
    :param tournament_group_size: Size of a random groups.

    :return: Generator producing individual pairs.
    """

    def _get_individual():
        return max(choose_random_values(values_pool=population, values_number=tournament_group_size))

    for _ in range(population_size // 2):
        yield _get_individual(), _get_individual()


def roulette_selection(population_size: int,
                       population: List[AbstractSolution],
                       roulette_bias: Union[float, int]) -> SelectionOutput:
    """
    Roulette selection function.

    Each parent provided by this function is picked with probability proportional to their
    fitness (level of adaptation) value.

    Note: Scaling of objective value is needed. More information in the topic can be found in the book:
    'Introduction to Evolutionary Computing. Second edition.' Eiben, A.E., Smith, James E., pages 80-81.
    Proposed scaling algorithm is different than 'windowing' and 'sigma scaling' in order to reduce code complexity.

    :param population_size: Size of the population (number of parents to pick).
    :param population: List with individuals (solutions) from which parents to be selected.
    :param roulette_bias: Bias towards promoting better adopted individuals.
        [probability of picking the best adopted individual] = \
            roulette_bias * [probability of picking the worst adopted individual]

    :return: Generator producing individual pairs.
    """
    best_solution = max(population)
    worst_solution = min(population)
    # if all solution values are the same, then we can use uniform selection - there will be the same result
    if best_solution == worst_solution:
        for pair in uniform_selection(population_size=population_size, population=population):
            yield pair
    else:
        # determine scaling values
        factor, offset = calculate_roulette_scaling(best_solution=best_solution, worst_solution=worst_solution,
                                                    roulette_bias=roulette_bias)
        weights = [get_scaled_objective(solution=solution, factor=factor, offset=offset) for solution in population]

        def _get_individual():
            return choose_random_value_with_weights(values_pool=population, weights=weights)

        for _ in range(population_size // 2):
            yield _get_individual(), _get_individual()


def ranking_selection(population_size: int,
                      population: List[AbstractSolution],
                      ranking_bias: float) -> SelectionOutput:
    """
    Ranking selection function which picks new parent with proportional probability to position in the ranking.

    Note: More information about 'ranking_bias' can be found in the book:
    'Introduction to Evolutionary Computing. Second edition.' A.E. Eiben, J.E. Smith., pages 81-82

    :param population_size: Size of the population (number of parents to pick).
    :param population: List with individuals (solutions) from which parents to be selected.
    :param ranking_bias: Bias that represents selection pressure (the higher the value the higher the pressure).
        When ranking_bias == 1, then the function acts the same effect as uniform selection.
        When ranking_bias == 2, then the function acts like classic ranking selection.
        Expected value: 1 < ranking_bias <= 2

    :return: Generator producing individual pairs.
    """
    population.sort()
    weights = [get_scaled_ranking(rank=rank, population_size=population_size, ranking_bias=ranking_bias)
               for rank in range(population_size)]

    def _get_individual():
        return choose_random_value_with_weights(values_pool=population, weights=weights)

    for _ in range(population_size // 2):
        yield _get_individual(), _get_individual()


# outputs (visible outside)


class SelectionType(Enum):
    """
    Enum with available selection functions types.

    Options:
        - Uniform - Each individual has the same chance to be selected as parent.
        - Tournament - Each parents pair is the best best adapted individuals from a small random group.
        - Tournament - Each parent is the best best adapted individual from a small random group.
    """

    Uniform = "Uniform"
    Tournament = "Tournament"
    DoubleTournament = "DoubleTournament"
    Roulette = "Roulette"
    Ranking = "Ranking"


SELECTION_FUNCTIONS: Dict[str, Callable] = {
    # selection type: selection function
    SelectionType.Uniform.value: uniform_selection,
    SelectionType.Tournament.value: tournament_selection,
    SelectionType.DoubleTournament.value: double_tournament_selection,
    SelectionType.Roulette.value: roulette_selection,
    SelectionType.Ranking.value: ranking_selection,
}


SELECTION_ADDITIONAL_PARAMS: Dict[str, Tuple[str, ...]] = {
    # selection type: (parameter 1 name, parameter 2 name, ...)
    SelectionType.Uniform.value: (),
    SelectionType.Tournament.value: ("tournament_group_size", ),
    SelectionType.DoubleTournament.value: ("tournament_group_size", ),
    SelectionType.Roulette.value: ("roulette_bias", ),
    SelectionType.Ranking.value: ("ranking_bias", ),
}

SELECTION_ADDITIONAL_PARAMS_LIMITS: Dict[str, Tuple[Union[float, int], Union[float, int]]] = {
    # parameter name: (min value, max value)
    "tournament_group_size": (MIN_TOURNAMENT_GROUP_SIZE, MAX_TOURNAMENT_GROUP_SIZE),
    "roulette_bias": (MIN_ROULETTE_BIAS, MAX_ROULETTE_BIAS),
    "ranking_bias": (MIN_RANKING_BIAS, MAX_RANKING_BIAS),
}


def check_selection_parameters(**selection_params: Any) -> None:
    """
    Checks whether additional selection parameters (selection function specific) have proper value.

    :param selection_params: Values of additional selection parameters.
    """
    if "tournament_group_size" in selection_params:
        check_tournament_group_size(selection_params["tournament_group_size"])
    if "roulette_bias" in selection_params:
        check_roulette_bias(selection_params["roulette_bias"])
    if "ranking_bias" in selection_params:
        check_ranking_bias(selection_params["ranking_bias"])
