from typing import Tuple, Iterator, List
from enum import Enum
from itertools import accumulate

from optimization.utilities import choose_random_values, binary_search, generate_random_float, generate_random_int
from optimization.optimization_problem import AbstractSolution, OptimizationType


__all__ = ["SelectionType", "ADDITIONAL_SELECTION_PARAMETERS", "SELECTION_FUNCTIONS"]


SelectionOutput = Iterator[Tuple[AbstractSolution, AbstractSolution]]


class SelectionType(Enum):
    STOCHASTIC = "Stochastic"
    TOURNAMENT = "Tournament"
    ROULETTE = "Roulette"
    RANKING = "Ranking"

    def __repr__(self):
        return self.value


ADDITIONAL_SELECTION_PARAMETERS = {
    SelectionType.STOCHASTIC.value: {},
    SelectionType.TOURNAMENT.value: {"tournament_group_size"},
    SelectionType.ROULETTE.value: {"roulette_bias"},
    SelectionType.RANKING.value: {"ranking_bias"},
}


def stochastic_selection(population_size: int, population: List[AbstractSolution]) -> SelectionOutput:
    """
    Creates generator that can be used as selection function.
    Selection is performed according to stochastic selection.

    :param population_size: Size of the population.
    :param population: Sorted list with solutions.

    :return: Generator of solutions pairs.
    """
    for _ in range(population_size // 2):
        yield choose_random_values(values_pool=population, values_number=2)


def tournament_selection(population_size: int, population: List[AbstractSolution],
                         tournament_group_size: int) -> SelectionOutput:
    """
    Creates generator that can be used as selection function.
    Selection is performed according to tournament selection.

    :param population_size: Size of the population.
    :param population: Sorted list with solutions.
    :param tournament_group_size: Number os individuals (solutions) in one group.

    :return: Generator of solutions pairs.
    """
    if population[0].optimization_problem.optimization_type == OptimizationType.Minimize:
        func = min
    else:
        func = max

    def _get_individual():
        return func(choose_random_values(values_pool=population, values_number=tournament_group_size),
                    key=lambda solution: solution.get_objective_value_with_penalty())

    for _ in range(population_size // 2):
        yield _get_individual(), _get_individual()


def roulette_selection(population_size: int, population: List[AbstractSolution],
                       roulette_bias: float) -> SelectionOutput:
    """
    Creates generator that can be used as selection function.
    Selection is performed according to roulette selection.

    :param population_size: Size of the population.
    :param population: Sorted list with solutions.
    :param roulette_bias: Value of bias used during scaling of objective values.
        Bias represents how much often the best solution would be picked than the worst.
        Expected value: 1 < roulette_bias <= 100
        [scaled_best_solution_objective] = roulette_bias * [scaled_worst_solution_objective]

    :return: Generator of solutions pairs.
    """
    best_solution_obj = population[0].get_objective_value_with_penalty()
    worst_solution_obj = population[-1].get_objective_value_with_penalty()
    # if best solution has similar objective value as worst, then roulette works the same as stochastic selection
    if best_solution_obj == worst_solution_obj:
        for _ in range(population_size // 2):
            yield choose_random_values(values_pool=population, values_number=2)
    else:
        factor = (roulette_bias - 1) / (best_solution_obj - worst_solution_obj)
        offset = 1 - factor*worst_solution_obj

        def _get_scaled_objective(solution: AbstractSolution) -> float:
            return factor*solution.get_objective_value_with_penalty() + offset

        roulette_wheel = list(accumulate((_get_scaled_objective(solution) for solution in population)))

        def _get_individual():
            rand_value = generate_random_float(0, roulette_wheel[-1])
            return population[binary_search(sorted_list=roulette_wheel, value=rand_value, list_size=population_size)]

        for _ in range(population_size // 2):
            yield _get_individual(), _get_individual()


def ranking_selection(population_size: int, population: List[AbstractSolution], ranking_bias: float) -> SelectionOutput:
    """
    Creates generator that can be used as selection function.
    Selection is performed according to ranking selection with bias

    More information about this selection can be found in 'Introduction to Evolutionary Computing. Second edition.'
    A.E. Eiben, J.E. Smith.

    :param population_size: Size of the population.
    :param population: Sorted list with solutions.
    :param ranking_bias: Bias that represents selection pressure (the higher the value the higher the pressure).
        When ranking_bias == 1, then it has the same effect as stochastic selection.
        When ranking_bias == 2, then it is classic ranking selection.
        Expected value: 1 < ranking_bias <= 2

    :return: Generator of solutions pairs.
    """
    def _get_scaled_rank(rank: int) -> float:
        # if value is additionally divided by population_size, we will received probability of selection
        return (2 - ranking_bias) + (2*rank*(ranking_bias-1) / (population_size-1))

    roulette_wheel = list(accumulate((_get_scaled_rank(rank) for rank in range(population_size))))[::-1]

    def _get_individual():
        rand_value = generate_random_float(0., roulette_wheel[-1])
        return population[binary_search(sorted_list=roulette_wheel, value=rand_value, list_size=population_size)]

    for _ in range(population_size // 2):
        yield _get_individual(), _get_individual()


SELECTION_FUNCTIONS = {
    SelectionType.STOCHASTIC.value: stochastic_selection,
    SelectionType.TOURNAMENT.value: tournament_selection,
    SelectionType.ROULETTE.value: roulette_selection,
    SelectionType.RANKING.value: ranking_selection,
}
