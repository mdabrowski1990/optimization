import pytest
from copy import deepcopy
from string import printable

from optimization.utilities.random_values import generate_random_int, generate_random_float, choose_random_value, \
    choose_random_values, shuffle, shuffled, choose_random_value_with_weights


class TestRandomFunctions:
    """
    Tests for random functions.

    TODO: It is worth considering whether randomness test would be beneficial here:
        https://en.wikipedia.org/wiki/Randomness_tests
    """

    @pytest.mark.parametrize("min_value, max_value, samples", [(1, 10, 100), (-100, 100, 2000)])
    def test_generate_random_int__value_in_range(self, min_value, max_value, samples):
        """
        Check that 'generate_random_int' function return integer value in given range.

        :param min_value: Minimal possible random value.
        :param max_value: Maximal possible random value.
        :param samples: Number of test repetitions.
        """
        for _ in range(samples):
            value = generate_random_int(min_value, max_value)
            assert isinstance(value, int) and min_value <= value <= max_value

    @pytest.mark.parametrize("min_value, max_value, samples", [(1., 10., 100), (-100., 100., 2000)])
    def test_generate_random_float__value_in_range(self, min_value, max_value, samples):
        """
        Check that 'generate_random_float' function return float value in given range.

        :param min_value: Minimal possible random value.
        :param max_value: Maximal possible random value.
        :param samples: Number of test repetitions.
        """
        for _ in range(samples):
            value = generate_random_float(min_value, max_value)
            assert isinstance(value, float) and min_value <= value <= max_value

    @pytest.mark.parametrize("values_pool, samples", [
        ({"white", "red", "black", "green", "blue", "gray", "brown", "purple", "pink", "yellow"}, 100),
        (set(range(-10, 11)), 200),
    ])
    def test_choose_random_value__value_in_pool(self, values_pool, samples):
        """
        Check that 'choose_random_value' function returns value from given pool.

        :param values_pool: Minimal possible random value.
        :param samples: Number of test repetitions.
        """
        for _ in range(samples):
            value = choose_random_value(values_pool)
            assert value in values_pool

    @pytest.mark.parametrize("values_pool, weights, samples", [
        (["white", "red", "black", "green", "blue", "gray", "brown", "purple", "pink", "yellow"], list(range(10)), 100),
        (range(-10, 11), [1, 2, 3] * 7, 200),
    ])
    def test_choose_random_value_with_weights__value_in_pool(self, values_pool, weights, samples):
        """
        Check that 'choose_random_value_with_weights' function returns value from given pool.

        :param values_pool: Minimal possible random value.
        :param weights: Examples weights values.
        :param samples: Number of test repetitions.
        """
        for _ in range(samples):
            value = choose_random_value_with_weights(values_pool=values_pool, weights=weights)
            assert value in values_pool

    @pytest.mark.parametrize("values_pool, samples", [
        ({"white", "red", "black", "green", "blue", "gray", "brown", "purple", "pink", "yellow"}, 100),
        (set(range(-10, 11)), 200),
    ])
    @pytest.mark.parametrize("values_number", [2, 3, 5])
    def test_choose_random_values__values_in_pool(self, values_pool, values_number, samples):
        """
        Check that 'choose_random_values' function returns values from given pool.

        :param values_pool: Minimal possible random value.
        :poram values_number: Number of values to be picked.
        :param samples: Number of test repetitions.
        """
        for _ in range(samples):
            values = choose_random_values(values_pool, values_number)
            assert isinstance(values, list) and len(values) == values_number \
                and all([value in values_pool for value in values])

    @pytest.mark.random
    @pytest.mark.parametrize("values", [list(range(1000)), list(printable)])
    def test_shuffle__values(self, values):
        """
        Check that 'shuffle' function changes value in place (inside the list).

        :param values: List of values to shuffle.
        """
        copy_input_values = deepcopy(values)
        shuffle(values)
        assert set(copy_input_values) == set(values) and type(copy_input_values) == type(values) \
            and copy_input_values != values

    @pytest.mark.random
    @pytest.mark.parametrize("values", [list(range(1000)), list(printable)])
    def test_shuffle__values(self, values):
        """
        Check that 'shuffle' function changes value in place (inside the list).

        :param values: List of values to shuffle.
        """
        copy_input_values = deepcopy(values)
        output_values = shuffled(values)
        assert copy_input_values == values, "Input were unchanged"
        assert set(values) == set(output_values) and type(values) == type(output_values) and values != output_values
