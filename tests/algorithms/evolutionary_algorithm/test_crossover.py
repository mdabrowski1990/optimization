import pytest
from mock import Mock, patch, call

from optimization.algorithms.evolutionary_algorithm.crossover import single_point_crossover, multi_point_crossover, \
    adaptive_crossover, uniform_crossover, check_crossover_parameters


class TestUtilities:
    """Tests for utilities functions"""

    # check_crossover_parameters

    @pytest.mark.parametrize("variables_number, crossover_params", [
        (1, {}),
        (8, {}),
        (3, {"crossover_points_number": 2}),
        (8, {"crossover_points_number": 2}),
        (8, {"crossover_points_number": 5}),
        (8, {"crossover_points_number": 7}),
        (3, {"crossover_pattern": 1}),
        (3, {"crossover_pattern": 6}),
        (8, {"crossover_pattern": 1}),
        (8, {"crossover_pattern": 254}),
    ])
    def test_check_crossover_parameters__valid(self, variables_number, crossover_params):
        """
        Test that 'check_crossover_parameters' raises no exception when valid parameter values are provided.

        :param variables_number: Example value of 'variables_number'.
        :param crossover_params: Example valid values of crossover parameters.
        """
        assert check_crossover_parameters(variables_number=variables_number, **crossover_params) is None

    @pytest.mark.parametrize("variables_number, crossover_params", [
        (3, {"crossover_points_number": 2.1}),
        (8, {"crossover_points_number": "3"}),
        (8, {"crossover_points_number": None}),
        (3, {"crossover_pattern": 4.3}),
        (3, {"crossover_pattern": None}),
        (100, {"crossover_points_number": [], "crossover_pattern": ()}),
    ])
    def test_check_crossover_parameters__invalid_type(self, variables_number, crossover_params):
        """
        Test that 'check_crossover_parameters' raises TypeError when value of incorrect type is provided.

        :param variables_number: Example value of 'variables_number'.
        :param crossover_params: Example invalid values (wrong type) of crossover parameters.
        """
        with pytest.raises(TypeError):
            assert check_crossover_parameters(variables_number=variables_number, **crossover_params) is None

    @pytest.mark.parametrize("variables_number, crossover_params", [
        (3, {"crossover_points_number": 1}),
        (3, {"crossover_points_number": 3}),
        (8, {"crossover_points_number": 1}),
        (8, {"crossover_points_number": 8}),
        (3, {"crossover_pattern": 0}),
        (3, {"crossover_pattern": 7}),
        (8, {"crossover_pattern": 0}),
        (8, {"crossover_pattern": 255}),
        (10, {"crossover_points_number": -1, "crossover_pattern": -1}),
    ])
    def test_check_crossover_parameters__invalid_value(self, variables_number, crossover_params):
        """
        Test that 'check_crossover_parameters' raises ValueError when invalid value is provided.

        :param variables_number: Example value of 'variables_number'.
        :param crossover_params: Example invalid values (wrong value) of crossover parameters.
        """
        with pytest.raises(ValueError):
            assert check_crossover_parameters(variables_number=variables_number, **crossover_params) is None


class TestCrossoverFunctions:
    """Tests for crossover function."""

    SCRIPT_LOCATION = "optimization.algorithms.evolutionary_algorithm.crossover"

    def setup(self):
        self.mock_parent_1_decision_variables_values_items = Mock()
        self.mock_parent_2_decision_variables_values_items = Mock()
        self.mock_parent_1_decision_variables_values = Mock(items=self.mock_parent_1_decision_variables_values_items)
        self.mock_parent_2_decision_variables_values = Mock(items=self.mock_parent_2_decision_variables_values_items)
        self.mock_parent_1 = Mock(decision_variables_values=self.mock_parent_1_decision_variables_values)
        self.mock_parent_2 = Mock(decision_variables_values=self.mock_parent_2_decision_variables_values)
        self.mock_parents = [self.mock_parent_1, self.mock_parent_2]
        # patching
        self._patcher_OrderedDict = patch(f"{self.SCRIPT_LOCATION}.OrderedDict")
        self.mock_OrderedDict = self._patcher_OrderedDict.start()
        self._patcher_generate_random_int = patch(f"{self.SCRIPT_LOCATION}.generate_random_int")
        self.mock_generate_random_int = self._patcher_generate_random_int.start()
        self._patcher_choose_random_values = patch(f"{self.SCRIPT_LOCATION}.choose_random_values")
        self.mock_choose_random_values = self._patcher_choose_random_values.start()
        self._patcher_adaptive_crossover = patch(f"{self.SCRIPT_LOCATION}.adaptive_crossover")
        self.mock_adaptive_crossover = self._patcher_adaptive_crossover.start()

    def teardown(self):
        self._patcher_generate_random_int.stop()
        self._patcher_OrderedDict.stop()
        self._patcher_choose_random_values.stop()
        self._patcher_adaptive_crossover.stop()

    # single_point_crossover

    @pytest.mark.parametrize("output", [("child 1 data set", "child 2 data set"), ("12345", "54321")])
    @pytest.mark.parametrize("variables_number, crossover_point, parent_1_data_list, parent_2_data_list", [
        (5, 1, [1, 2, 3, 4, 5], [6, 7, 8, 9, 10]),
        (5, 2, [1, 2, 3, 4, 5], [6, 7, 8, 9, 10]),
        (5, 4, [1, 2, 3, 4, 5], [6, 7, 8, 9, 10]),
        (3, 1, ["a", "b", "c"], ["x", "y", "z"]),
        (3, 2, ["a", "b", "c"], ["x", "y", "z"]),
    ])
    def test_single_point_crossover(self, variables_number, crossover_point, parent_1_data_list, parent_2_data_list,
                                    output):
        """
        Test for 'single_point_crossover' function.

        :param variables_number: Example value of 'variables_number'.
        :param crossover_point: Simulated value of crossover point.
        :param parent_1_data_list: Simulated value of parents decision variables values.
        :param parent_2_data_list: Simulated value of parents decision variables values.
        :param output: Simulated output.
        """
        self.mock_generate_random_int.return_value = crossover_point
        self.mock_parent_1_decision_variables_values_items.return_value = parent_1_data_list
        self.mock_parent_2_decision_variables_values_items.return_value = parent_2_data_list
        self.mock_OrderedDict.side_effect = output
        assert output == single_point_crossover(parents=self.mock_parents, variables_number=variables_number)
        self.mock_generate_random_int.assert_called_once_with(1, variables_number-1)
        self.mock_OrderedDict.assert_has_calls([
            call(parent_1_data_list[:crossover_point] + parent_2_data_list[crossover_point:]),
            call(parent_2_data_list[:crossover_point] + parent_1_data_list[crossover_point:])
        ])

    # multi_point_crossover

    @pytest.mark.parametrize("variables_number, crossover_points, parent_1_data_list, parent_2_data_list, child1_snips, child2_snips", [
        (5, [1], [1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [[1], [7, 8, 9, 10]], [[6], [2, 3, 4, 5]]),
        (5, [1, 4], [1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [[1], [7, 8, 9], [5]], [[6], [2, 3, 4], [10]]),
        (5, [2, 3], [1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [[1, 2], [8], [4, 5]], [[6, 7], [3], [9, 10]]),
        (3, [1], ["a", "b", "c"], ["x", "y", "z"], [["a"], ["y", "z"]], [["x"], ["b", "c"]]),
        (3, [1, 2], ["a", "b", "c"], ["x", "y", "z"], [["a"], ["y"], ["c"]], [["x"], ["b"], ["z"]]),
    ])
    def test_multi_point_crossover(self, variables_number, crossover_points, parent_1_data_list, parent_2_data_list,
                                   child1_snips, child2_snips):
        """
        Test for 'multi_point_crossover' function.

        :param variables_number: Example value of 'variables_number'.
        :param crossover_points: Simulated values of returned crossover points.
        :param parent_1_data_list: Simulated value of parents decision variables values.
        :param parent_2_data_list: Simulated value of parents decision variables values.
        """
        crossover_points_number = len(crossover_points)
        self.mock_choose_random_values.return_value = crossover_points
        self.mock_parent_1_decision_variables_values_items.return_value = parent_1_data_list
        self.mock_parent_2_decision_variables_values_items.return_value = parent_2_data_list
        mock_child1 = Mock()
        mock_child2 = Mock()
        self.mock_OrderedDict.side_effect = [mock_child1, mock_child2]
        assert (mock_child1, mock_child2) == multi_point_crossover(parents=self.mock_parents,
                                                                   variables_number=variables_number,
                                                                   crossover_points_number=crossover_points_number)
        self.mock_choose_random_values.assert_called_once_with(values_pool=range(1, variables_number),
                                                               values_number=crossover_points_number)
        self.mock_OrderedDict.assert_has_calls([call(child1_snips[0]), call(child2_snips[0])])
        mock_child1.update.assert_has_calls([call(snip) for snip in child1_snips[1:]])
        mock_child2.update.assert_has_calls([call(snip) for snip in child2_snips[1:]])

    # adaptive_crossover

    @pytest.mark.parametrize("variables_number, crossover_pattern, parent_1_data_list, parent_2_data_list, child_1_output, child_2_output", [
        (5, 0b00000, [1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [1, 2, 3, 4, 5], [6, 7, 8, 9, 10]),
        (5, 0b11111, [1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [6, 7, 8, 9, 10], [1, 2, 3, 4, 5]),
        (5, 0b01101, [1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [6, 2, 8, 9, 5], [1, 7, 3, 4, 10]),
        (3, 0b101, ["a", "b", "c"], ["x", "y", "z"], "xbz", "ayc"),
        (3, 0b010, ["a", "b", "c"], ["x", "y", "z"], "ayc", "xbz"),
    ])
    def test_adaptive_crossover(self, variables_number, crossover_pattern, parent_1_data_list,
                                parent_2_data_list, child_1_output, child_2_output):
        self.mock_parent_1_decision_variables_values_items.return_value = parent_1_data_list
        self.mock_parent_2_decision_variables_values_items.return_value = parent_2_data_list
        mock_child1 = Mock()
        mock_child2 = Mock()
        self.mock_OrderedDict.side_effect = [mock_child1, mock_child2]
        assert (mock_child1, mock_child2) == adaptive_crossover(parents=self.mock_parents,
                                                                variables_number=variables_number,
                                                                crossover_pattern=crossover_pattern)
        self.mock_OrderedDict.assert_has_calls([call(), call()])
        mock_child1.update.assert_has_calls([call([snip]) for snip in child_1_output])
        mock_child2.update.assert_has_calls([call([snip]) for snip in child_2_output])

    # uniform_crossover

    @pytest.mark.parametrize("variables_number, max_pattern_value", [(1, 1), (2, 3), (3, 7), (8, 255)])
    def test_uniform_crossover(self, variables_number, max_pattern_value):
        """
        Test for 'uniform_crossover' function.

        :param variables_number: Example value of 'variables_number'.
        :param max_pattern_value: Maximal number of pattern for given number of decision variables.
        """
        assert uniform_crossover(parents=self.mock_parents, variables_number=variables_number) \
            == self.mock_adaptive_crossover.return_value
        self.mock_generate_random_int.assert_called_once_with(0, max_pattern_value)
        self.mock_adaptive_crossover.aasert_called_once_with(parents=self.mock_parents,
                                                             variables_number=variables_number,
                                                             crossover_pattern=self.mock_generate_random_int.return_value)
