import pytest
from copy import deepcopy
from mock import Mock, patch, call
from types import GeneratorType

from optimization.algorithms.evolutionary_algorithm.selection import uniform_selection, tournament_selection, \
    double_tournament_selection, roulette_selection, ranking_selection, \
    get_scaled_ranking, get_scaled_objective, calculate_roulette_scaling, \
    check_selection_parameters, \
    MIN_TOURNAMENT_GROUP_SIZE, MAX_TOURNAMENT_GROUP_SIZE, MIN_ROULETTE_BIAS, MAX_ROULETTE_BIAS, \
    MIN_RANKING_BIAS, MAX_RANKING_BIAS


class TestUtilities:
    """Tests for utilities functions"""

    def setup(self):
        self.mock_get_objective_value_with_penalty = Mock()
        self.mock_solution_object = Mock(get_objective_value_with_penalty=self.mock_get_objective_value_with_penalty)

    # calculate_roulette_scaling

    @pytest.mark.parametrize("best_objective, worst_objective, roulette_bias", [
        (100, 1, 100),
        (99, 0, 100),
        (5.25, 1.25, 3.25),
        (-10, 10, 1.11),
        (-6.543, -32.1, 2.278),
    ])
    def test_calculate_roulette_scaling(self, best_objective, worst_objective, roulette_bias):
        """
        Test 'calculate_roulette_scaling' method calculates factor and offset according

        :param best_objective: Example objective value of the best solution.
        :param worst_objective: Example objective value of the worst solution.
        :param roulette_bias: Example value of 'roulette_bias'.
        """
        get_worst_solution_objective = Mock(return_value=worst_objective)
        worst_solution = Mock(get_objective_value_with_penalty=get_worst_solution_objective)
        get_best_solution_objective = Mock(return_value=best_objective)
        best_solution = Mock(get_objective_value_with_penalty=get_best_solution_objective)
        factor, offset = calculate_roulette_scaling(best_solution=best_solution, worst_solution=worst_solution,
                                                    roulette_bias=roulette_bias)
        assert round((factor * best_objective + offset), 8) == round(roulette_bias * (factor * worst_objective + offset), 8)
        get_worst_solution_objective.assert_called()
        get_best_solution_objective.assert_called()

    # get_scaled_objective

    @pytest.mark.parametrize("objective", [-2, 0, 1.25])
    @pytest.mark.parametrize("factor", [-1.23, 5.2, 3])
    @pytest.mark.parametrize("offset", [-1.23, 0, 5.2])
    def test_get_scaled_objective(self, objective, factor, offset):
        """
        Calculates scaled value of objective.

        :param objective: Example value of solution objective.
        :param factor: Example value of 'factor' parameter.
        :param offset: Example value of 'offset' parameter.
        """
        self.mock_get_objective_value_with_penalty.return_value = objective
        assert get_scaled_objective(solution=self.mock_solution_object, factor=factor, offset=offset) == objective*factor + offset
        self.mock_get_objective_value_with_penalty.assert_called_once()

    # get_scaled_ranking

    @pytest.mark.parametrize("rank, population_size, ranking_bias, expected_result", [
        (0, 3, 2, 0),
        (0, 10, 1, 1),
        (0, 100, 1, 1),
        (2, 3, 1.5, 1.5),
        (2, 3, 2, 2),
        (9, 10, 2, 2),
    ])
    def test_get_scaled_ranking(self, rank, population_size, ranking_bias, expected_result):
        """
        Test for 'get_scaled_ranking' function.

        :param rank: Ranking of the solution.
        :param population_size: Example value of 'population_size' parameter.
        :param ranking_bias: Example value of 'ranking_bias' parameter.
        :param expected_result: Expected result ffrom the function.
        """
        assert get_scaled_ranking(rank=rank, population_size=population_size, ranking_bias=ranking_bias) == expected_result

    # check_selection_parameters

    @pytest.mark.parametrize("selection_params", [
        {},
        {"tournament_group_size": MIN_TOURNAMENT_GROUP_SIZE},
        {"tournament_group_size": MAX_TOURNAMENT_GROUP_SIZE},
        {"tournament_group_size": int((MIN_TOURNAMENT_GROUP_SIZE + MAX_TOURNAMENT_GROUP_SIZE) // 2)},
        {"roulette_bias": MIN_ROULETTE_BIAS},
        {"roulette_bias": MAX_ROULETTE_BIAS},
        {"roulette_bias": (MIN_ROULETTE_BIAS + MAX_ROULETTE_BIAS) // 2},
        {"ranking_bias": MIN_RANKING_BIAS},
        {"ranking_bias": MAX_RANKING_BIAS},
        {"ranking_bias": (MIN_RANKING_BIAS + MAX_RANKING_BIAS) // 2},
        {"tournament_group_size": 4, "roulette_bias": 4.232, "ranking_bias": 1.5}
    ])
    def test_check_selection_parameters__valid(self, selection_params):
        """
        Test that 'check_selection_parameters' raises no exception when valid parameter values are provided.

        :param selection_params: Example valid values of selection parameters.
        """
        assert check_selection_parameters(**selection_params) is None

    @pytest.mark.parametrize("selection_params", [
        {"tournament_group_size": 4.2},
        {"tournament_group_size": "4"},
        {"roulette_bias": "some bias"},
        {"roulette_bias": [0, 1]},
        {"ranking_bias": False},
        {"ranking_bias": "some incorrect value"},
        {"tournament_group_size": 6., "roulette_bias": "something stupid", "ranking_bias": 1}
    ])
    def test_check_selection_parameters__invalid_type(self, selection_params):
        """
        Test that 'check_selection_parameters' raises TypeError when value of incorrect type is provided.

        :param selection_params: Example invalid values (wrong type) of selection parameters.
        """
        with pytest.raises(TypeError):
            check_selection_parameters(**selection_params)

    @pytest.mark.parametrize("selection_params", [
        {"tournament_group_size": MIN_TOURNAMENT_GROUP_SIZE - 1},
        {"tournament_group_size": MAX_TOURNAMENT_GROUP_SIZE + 1},
        {"roulette_bias": MIN_ROULETTE_BIAS - 0.00001},
        {"roulette_bias": MAX_ROULETTE_BIAS + 0.00001},
        {"ranking_bias": MIN_RANKING_BIAS - 0.00001},
        {"ranking_bias": MAX_RANKING_BIAS + 0.00001},
        {"tournament_group_size": 0, "roulette_bias": 0, "ranking_bias": 0}
    ])
    def test_check_selection_parameters__invalid_value(self, selection_params):
        """
        Test that 'check_selection_parameters' raises ValueError when invalid value is provided.

        :param selection_params: Example invalid values (wrong value) of selection parameters.
        """
        with pytest.raises(ValueError):
            check_selection_parameters(**selection_params)


class TestSelectionFunctions:
    """Tests for selection function."""

    SCRIPT_LOCATION = "optimization.algorithms.evolutionary_algorithm.selection"

    def setup(self):
        # patching
        self._patcher_choose_random_values = patch(f"{self.SCRIPT_LOCATION}.choose_random_values")
        self.mock_choose_random_values = self._patcher_choose_random_values.start()
        self._patcher_choose_random_value_with_weights = patch(f"{self.SCRIPT_LOCATION}.choose_random_value_with_weights")
        self.mock_choose_random_value_with_weights = self._patcher_choose_random_value_with_weights.start()
        self._patcher_get_scaled_ranking = patch(f"{self.SCRIPT_LOCATION}.get_scaled_ranking")
        self.mock_get_scaled_ranking = self._patcher_get_scaled_ranking.start()
        self._patcher_uniform_selection = patch(f"{self.SCRIPT_LOCATION}.uniform_selection")
        self.mock_uniform_selection = self._patcher_uniform_selection.start()
        self._patcher_calculate_roulette_scaling = patch(f"{self.SCRIPT_LOCATION}.calculate_roulette_scaling")
        self.mock_calculate_roulette_scaling = self._patcher_calculate_roulette_scaling.start()
        self._patcher_get_scaled_objective = patch(f"{self.SCRIPT_LOCATION}.get_scaled_objective")
        self.mock_get_scaled_objective = self._patcher_get_scaled_objective.start()

    def teardown(self):
        self._patcher_choose_random_values.stop()
        self._patcher_choose_random_value_with_weights.stop()
        self._patcher_get_scaled_ranking.stop()
        self._patcher_uniform_selection.stop()
        self._patcher_calculate_roulette_scaling.stop()
        self._patcher_get_scaled_objective.stop()

    # uniform_selection

    @pytest.mark.parametrize("population_size, random_output", [
        (2, [("a", "b")]),
        (4, [("1 value 1", "1 value 2"), ("2 value 1", "2 value 2")]),
        (10, [(1, 11), (2, 22), (3, 33), (4, 44), (5, 55)]),
    ])
    @pytest.mark.parametrize("population", ["some population", range(10)])
    def test_uniform_selection(self, population_size, population, random_output):
        """
        Test that 'uniform_selection' uses 'choose_random_values' to return generator of solution pairs.

        :param population_size: Example value of 'population_size' parameter.
        :param population: Example value of 'population' parameter.
        :param random_output: Values simulated as returned by 'choose_random_values' function.
        """
        self.mock_choose_random_values.side_effect = random_output
        output = uniform_selection(population_size=population_size, population=population)
        assert isinstance(output, GeneratorType)
        output_list = list(output)
        assert len(output_list) == population_size // 2
        assert output_list == random_output
        self.mock_choose_random_values.assert_has_calls([call(values_pool=population, values_number=2)] * (population_size//2))

    # tournament_selection

    @pytest.mark.parametrize("population_size, random_output", [
        (2, [[1, 2, 3, 4, 5]]),
        (4, [[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]]),
        (10, [[1, 11], [2, 22], [3, 33], [4, 44], [5, 55]]),
    ])
    @pytest.mark.parametrize("population", ["some population", range(10)])
    @pytest.mark.parametrize("tournament_group_size", [2, 4, 7])
    def test_tournament_selection(self, population_size, population, tournament_group_size, random_output):
        """
        Test that 'tournament_selection' uses 'choose_random_values' and picks two best solution from each group.

        :param population_size: Example value of 'population_size' parameter.
        :param population: Example value of 'population' parameter.
        :param tournament_group_size: Example value of 'tournament_group_size' parameter.
        :param random_output: Values simulated as returned by 'choose_random_values' function.
        """
        self.mock_choose_random_values.side_effect = deepcopy(random_output)
        output = tournament_selection(population_size=population_size, population=population,
                                      tournament_group_size=tournament_group_size)
        assert isinstance(output, GeneratorType)
        output_list = list(output)
        assert len(output_list) == population_size // 2
        assert [set(sorted(group, reverse=True)[:2]) for group in random_output] == [set(pair) for pair in output_list]
        self.mock_choose_random_values.assert_has_calls([call(values_pool=population,
                                                              values_number=tournament_group_size)] * (population_size // 2))

    # double_tournament_selection

    @pytest.mark.parametrize("population_size, random_output", [
        (2, [[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]]),
        (4, [range(2, 6), range(9, 5, -2), range(100, 130, 5), range(200, 120, -15)]),
        (6, [[1, 11], [22, 2], [3, 33], [44, 4], [5, 55], [66, 6]]),
    ])
    @pytest.mark.parametrize("population", ["some population", range(10)])
    @pytest.mark.parametrize("tournament_group_size", [2, 4, 7])
    def test_double_tournament_selection(self, population_size, population, tournament_group_size, random_output):
        """
        Test that 'double_tournament_selection' uses 'choose_random_values' and picks the best solution from each group.

        :param population_size: Example value of 'population_size' parameter.
        :param population: Example value of 'population' parameter.
        :param tournament_group_size: Example value of 'tournament_group_size' parameter.
        :param random_output: Values simulated as returned by 'choose_random_values' function.
        """
        self.mock_choose_random_values.side_effect = deepcopy(random_output)
        double_tournament_selection(population_size=population_size, population=population,
                                    tournament_group_size=tournament_group_size)
        output = double_tournament_selection(population_size=population_size, population=population,
                                             tournament_group_size=tournament_group_size)
        assert isinstance(output, GeneratorType)
        output_list = list(output)
        assert len(output_list) == population_size // 2
        assert all([(max(random_output[2*i]), max(random_output[2*i+1])) == pair for i, pair in enumerate(output_list)])
        self.mock_choose_random_values.assert_has_calls([call(values_pool=population,
                                                              values_number=tournament_group_size)] * population_size)

    # roulette_selection

    @pytest.mark.parametrize("population_size", [2, 4, 50])
    @pytest.mark.parametrize("solution_values", [-1.23, 2.5, 13])
    @pytest.mark.parametrize("roulette_bias", [1, 66, 100])
    def test_roulette_selection__all_the_same(self, population_size, solution_values, roulette_bias):
        """
        Tests 'roulette_selection' when the best and the worst solution have the same objective.

        :param population_size: Example value of 'population_size' parameter.
        :param solution_values: Values that simulates all solution objects.
        :param roulette_bias: Example value of 'roulette_bias' parameter.
        """
        self.mock_uniform_selection.return_value = [(solution_values, solution_values) for _ in range(population_size//2)]
        output = roulette_selection(population_size=population_size, population=[solution_values]*population_size,
                                    roulette_bias=roulette_bias)
        assert isinstance(output, GeneratorType)
        output_list = list(output)
        assert self.mock_uniform_selection.return_value == output_list

    @pytest.mark.parametrize("factor", [-1.1, 2.2])
    @pytest.mark.parametrize("offset", [0, 2.5])
    @pytest.mark.parametrize("roulette_bias", [1.987, 98.42])
    @pytest.mark.parametrize("population_size, population, weights, expected_output", [
        (2, [1, 2], [0.25, 0.75], [2, 1]),
        (4, [1, 2, 3, 4], [0.1, 0.11, 0.05, 0.77], [3, 4, 1, 2]),
    ])
    def test_roulette_selection__different(self, population_size, population, roulette_bias, factor, offset, weights,
                                           expected_output):
        """
        Tests 'roulette_selection' when population is not homogeneous.

        :param population_size: Example value of 'population_size' parameter.
        :param population: Example value of 'population' parameter.
        :param roulette_bias: Example value of 'roulette_bias' parameter.
        """
        self.mock_calculate_roulette_scaling.return_value = [factor, offset]
        self.mock_get_scaled_objective.side_effect = weights
        self.mock_choose_random_value_with_weights.side_effect = expected_output
        output = roulette_selection(population_size=population_size, population=population, roulette_bias=roulette_bias)
        assert isinstance(output, GeneratorType)
        output_list = list(output)
        assert len(output_list) == population_size // 2
        assert all([(expected_output[2 * i], expected_output[2 * i + 1]) == pair for i, pair in enumerate(output_list)])
        self.mock_get_scaled_objective.assert_has_calls([call(solution=solution, factor=factor, offset=offset)
                                                         for solution in population])
        self.mock_choose_random_value_with_weights.assert_has_calls([call(values_pool=population, weights=weights)] * population_size)

    # ranking_selection

    @pytest.mark.parametrize("population_size, population, weights, expected_output", [
        (2, [1, 2], [0.1, 0.9], [0, 1]),
        (4, [99, -1, 0.25, 3.14], [0.25, 0.22, 0.25, 0.28], [2, 1, 0, 3]),
    ])
    @pytest.mark.parametrize("ranking_bias", [1.25, 1.5, 2])
    def test_ranking_selection(self, population_size, population, ranking_bias, weights, expected_output):
        """
        Test that 'test_ranking_selection' uses 'get_scaled_ranking' and 'choose_random_value_with_weights' functions.

        :param population_size: Example value of 'population_size' parameter.
        :param population: Example value of 'population' parameter.
        :param ranking_bias: Example value of 'ranking_bias' parameter.
        :param weights: Values to be simulated as returned by 'get_scaled_ranking' function.
        """
        self.mock_get_scaled_ranking.side_effect = weights
        self.mock_choose_random_value_with_weights.side_effect = expected_output
        output = ranking_selection(population_size=population_size, population=population, ranking_bias=ranking_bias)
        assert isinstance(output, GeneratorType)
        output_list = list(output)
        assert len(output_list) == population_size // 2
        assert all([(expected_output[2*i], expected_output[2*i+1]) == pair for i, pair in enumerate(output_list)])
        self.mock_get_scaled_ranking.assert_has_calls([
            call(rank=i, population_size=population_size, ranking_bias=ranking_bias) for i in range(population_size)
        ])
        self.mock_choose_random_value_with_weights.assert_has_calls([call(values_pool=population,
                                                                          weights=weights)] * population_size)
