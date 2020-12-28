import pytest

from optimization import SelectionType, CrossoverType, MutationType, AdaptationType


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
def example_selection_types():
    return list(SelectionType)


@pytest.fixture
def example_crossover_types():
    return list(CrossoverType)


@pytest.fixture
def example_mutation_types():
    return list(MutationType)
