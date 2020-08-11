import pytest
from mock import patch, Mock, call

from optimization.algorithms.evolutionary_algorithm.mutation import single_point_mutation, multi_point_mutation, \
    probabilistic_mutation


class TestMutationFunctions:
    """Tests for mutation function."""

    SCRIPT_LOCATION = "optimization.algorithms.evolutionary_algorithm.mutation"

    def setup(self):
        # patching
        self._patcher_generate_random_float = patch(f"{self.SCRIPT_LOCATION}.generate_random_float")
        self.mock_generate_random_float = self._patcher_generate_random_float.start()
        self._patcher_choose_random_value = patch(f"{self.SCRIPT_LOCATION}.choose_random_value")
        self.mock_choose_random_value = self._patcher_choose_random_value.start()
        self._patcher_choose_random_values = patch(f"{self.SCRIPT_LOCATION}.choose_random_values")
        self.mock_choose_random_values = self._patcher_choose_random_values.start()

    def teardown(self):
        self._patcher_generate_random_float.stop()
        self._patcher_choose_random_value.stop()
        self._patcher_choose_random_values.stop()

    # single_point_mutation

    @pytest.mark.parametrize("variables_number, mutation_chance, random_value", [
        (2, 0.1, 1),
        (2, 0.1, 0.2000001),
        (5, 0.01, 0.0500001),
        (5, 0.01, 0.55),
    ])
    def test_single_point_mutation__no_mutation(self, variables_number, mutation_chance, random_value):
        """
        Test 'single_point_mutation' selects no points of mutation if random_value outside scaled probability chance.

        :param variables_number: Example value of 'variables_number'.
        :param mutation_chance: Example value of 'mutation_chance'.
        :param random_value: Value simulated as return from 'generate_random_float'.
        """
        self.mock_generate_random_float.return_value = random_value
        assert single_point_mutation(variables_number=variables_number, mutation_chance=mutation_chance) == []
        self.mock_generate_random_float.assert_called_once_with(0, 1)
        self.mock_choose_random_value.assert_not_called()

    @pytest.mark.parametrize("variables_number, mutation_chance, random_value", [
        (2, 0.1, 0),
        (2, 0.1, 0.2),
        (5, 0.01, 0.05),
        (5, 0.01, 0.034561),
    ])
    def test_single_point_mutation__mutation(self, variables_number, mutation_chance, random_value):
        """
        Test 'single_point_mutation' selects a single point of mutation returned by 'choose_random_value' if
        random_value in scaled probability chance.

        :param variables_number: Example value of 'variables_number'.
        :param mutation_chance: Example value of 'mutation_chance'.
        :param random_value: Value simulated as return from 'generate_random_float'.
        """
        self.mock_generate_random_float.return_value = random_value
        assert single_point_mutation(variables_number=variables_number, mutation_chance=mutation_chance) \
            == [self.mock_choose_random_value.return_value]
        self.mock_generate_random_float.assert_called_once_with(0, 1)
        self.mock_choose_random_value.assert_called_once_with(values_pool=range(variables_number))

    # multi_point_mutation

    @pytest.mark.parametrize("variables_number, mutation_chance, mutation_points_number, random_value", [
        (2, 0.1, 1, 1),
        (2, 0.1, 1, 0.2000001),
        (5, 0.01, 2, 0.0250001),
        (5, 0.01, 2, 0.55),
        (5, 0.01, 3, 0.0166667),
        (10, 0.02, 4, 0.050001),
    ])
    def test_multi_point_mutation__no_mutation(self, variables_number, mutation_chance, mutation_points_number,
                                               random_value):
        """
        Test 'multi_point_mutation' selects no points of mutation if random_value outside scaled probability chance.

        :param variables_number: Example value of 'variables_number'.
        :param mutation_chance: Example value of 'mutation_chance'.
        :param mutation_points_number: Example value of 'mutation_points_number'.
        :param random_value: Value simulated as return from 'generate_random_float'.
        """
        self.mock_generate_random_float.return_value = random_value
        assert multi_point_mutation(variables_number=variables_number, mutation_chance=mutation_chance,
                                    mutation_points_number=mutation_points_number) == []
        self.mock_generate_random_float.assert_called_once_with(0, 1)
        self.mock_choose_random_values.assert_not_called()

    @pytest.mark.parametrize("variables_number, mutation_chance, mutation_points_number, random_value", [
        (2, 0.1, 1, 0),
        (2, 0.1, 1, 0.2),
        (5, 0.01, 2, 0.025),
        (5, 0.01, 2, 0.012345),
        (5, 0.01, 3, 0.016666),
        (10, 0.02, 4, 0.05),
    ])
    def test_multi_point_mutation__mutation(self, variables_number, mutation_chance, mutation_points_number,
                                            random_value):
        """
        Test 'multi_point_mutation' selects a [mutation_points_number] of mutation returned by
        'mock_choose_random_values' if random_value in scaled probability chance.

        :param variables_number: Example value of 'variables_number'.
        :param mutation_chance: Example value of 'mutation_chance'.
        :param mutation_points_number: Example value of 'mutation_points_number'.
        :param random_value: Value simulated as return from 'generate_random_float'.
        """
        self.mock_generate_random_float.return_value = random_value
        assert multi_point_mutation(variables_number=variables_number, mutation_chance=mutation_chance,
                                    mutation_points_number=mutation_points_number) \
            == self.mock_choose_random_values.return_value
        self.mock_generate_random_float.assert_called_once_with(0, 1)
        self.mock_choose_random_values.assert_called_once_with(values_pool=range(variables_number),
                                                               values_number=mutation_points_number)

    # probabilistic_mutation

    @pytest.mark.parametrize("mutation_chance", [0.01, 0.25, 0.5])
    @pytest.mark.parametrize("variables_number, random_values", [
        (3, [0.25, 0.5, 0.75]),
        (5, [0.654, 0.4999, 0.025, 0.12345, 0.009999999]),
        (4, [0., 0.01, 0.9999, 0.876]),
    ])
    def test_probabilistic_mutation(self, variables_number, mutation_chance, random_values):
        """
        Test 'probabilistic_mutation'.

        :param variables_number: Example value of 'variables_number'.
        :param mutation_chance: Example value of 'mutation_chance'.
        :param random_values: Values simulated as return from 'generate_random_float'.
        """
        self.mock_generate_random_float.side_effect = random_values
        assert probabilistic_mutation(variables_number=variables_number, mutation_chance=mutation_chance) \
            == [i for i, random_value in enumerate(random_values) if random_value <= mutation_chance]
        self.mock_generate_random_float.assert_has_calls([call(0, 1)] * variables_number)
