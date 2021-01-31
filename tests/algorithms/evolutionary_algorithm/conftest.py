import pytest

from optimization import SelectionType, CrossoverType, MutationType, AdaptationType
from optimization.algorithms.evolutionary_algorithm.limits import MIN_EA_MUTATION_CHANCE, MAX_EA_MUTATION_CHANCE, \
    MIN_EA_POPULATION_SIZE, MAX_EA_POPULATION_SIZE, MIN_TOURNAMENT_GROUP_SIZE, MAX_TOURNAMENT_GROUP_SIZE, \
    MIN_ROULETTE_BIAS, MAX_ROULETTE_BIAS, MIN_RANKING_BIAS, MAX_RANKING_BIAS


@pytest.fixture
def example_population_size():
    return 10


@pytest.fixture
def example_apply_elitism():
    return False


@pytest.fixture
def example_mutation_chance():
    return 0.05


@pytest.fixture
def example_adaptation_type():
    return AdaptationType.BestSolution


@pytest.fixture
def example_population_size_boundaries():
    return MIN_EA_POPULATION_SIZE, MAX_EA_POPULATION_SIZE


@pytest.fixture
def example_selection_types():
    return list(SelectionType)


@pytest.fixture
def example_crossover_types():
    return list(CrossoverType)


@pytest.fixture
def example_mutation_types():
    return list(MutationType)


@pytest.fixture
def example_mutation_chance_boundaries():
    return MIN_EA_MUTATION_CHANCE, MAX_EA_MUTATION_CHANCE


@pytest.fixture
def example_apply_elitism_options():
    return True, False


@pytest.fixture
def default_min_tournament_group_size():
    return MIN_TOURNAMENT_GROUP_SIZE


@pytest.fixture
def default_max_tournament_group_size():
    return MAX_TOURNAMENT_GROUP_SIZE


@pytest.fixture
def default_min_ranking_bias():
    return MIN_RANKING_BIAS


@pytest.fixture
def default_max_ranking_bias():
    return MAX_RANKING_BIAS


@pytest.fixture
def default_min_roulette_bias():
    return MIN_ROULETTE_BIAS


@pytest.fixture
def default_max_roulette_bias():
    return MAX_ROULETTE_BIAS