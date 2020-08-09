"""
Selection functions implementation that are used by Evolutionary Algorithms.

Code is basing on  information from:
'Introduction to Evolutionary Computing. Second edition.' Eiben, A.E., Smith, James E.
"""

__all__ = ["SelectionType"]


from typing import List, Iterator, Union, Tuple
from enum import Enum

from ...problem import AbstractSolution
from ...utilities import choose_random_values, choose_random_value_with_weights


# selection utilities


SelectionOutput = Iterator[Tuple[AbstractSolution, AbstractSolution]]


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


# selection implementation


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
                       roulette_bias: float) -> SelectionOutput:
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
