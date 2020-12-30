import pytest

from optimization import SelectionType, CrossoverType, MutationType, AdaptationType
from optimization.algorithms.evolutionary_algorithm.limits import MIN_MUTATION_CHANCE, MAX_MUTATION_CHANCE, \
    MIN_POPULATION_SIZE, MAX_POPULATION_SIZE


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
    return MIN_POPULATION_SIZE, MAX_POPULATION_SIZE


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
    return MIN_MUTATION_CHANCE, MAX_MUTATION_CHANCE


@pytest.fixture
def example_apply_elitism_options():
    return True, False
