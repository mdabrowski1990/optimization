import pytest


@pytest.fixture
def example_population_size():
    return 10


@pytest.fixture
def example_apply_elitism():
    return False


@pytest.fixture
def example_mutation_chance():
    return 0.05
