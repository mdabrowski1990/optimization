import pytest
from mock import Mock, patch
from datetime import timedelta

from optimization.optimization_algorithms.stop_conditions import StopCondition, OptimizationType
from .conftest import EXAMPLE_VALUE_TYPES, EXAMPLE_VALID_MAX_ITERATIONS_WITHOUT_PROGRESS_VALUES, \
    EXAMPLE_VALID_MAX_TIME_WITHOUT_PROGRESS_VALUES, EXAMPLE_INVALID_MAX_ITERATIONS_WITHOUT_PROGRESS_VALUES, \
    EXAMPLE_VALID_SATISFYING_OBJECTIVE_VALUES


class TestStopConditionInit:
    """Tests for '__init__' method of 'StopCondition' class."""
    def test_valid_no_params(self, random_positive_timedelta):
        """
        Check that 'StopCondition' class can be initialized if only example valid value of 'time_limit' parameter
        is passed.

        :param random_positive_timedelta: Valid value for 'time_limit' parameter.
        """
        stop_condition = StopCondition(time_limit=random_positive_timedelta)
        assert stop_condition.time_limit == random_positive_timedelta
        assert stop_condition.satisfying_objective_value is None
        assert stop_condition.max_iterations_without_progress is None
        assert stop_condition.max_time_without_progress is None
        assert stop_condition._iterations_without_progress is None
        assert stop_condition._best_solution is None
        assert stop_condition._last_progress_time is None

    @pytest.mark.parametrize("valid_satisfying_objective_value", EXAMPLE_VALID_SATISFYING_OBJECTIVE_VALUES)
    @pytest.mark.parametrize("valid_max_iterations_without_progress",
                             EXAMPLE_VALID_MAX_ITERATIONS_WITHOUT_PROGRESS_VALUES)
    @pytest.mark.parametrize("valid_max_time_without_progress", EXAMPLE_VALID_MAX_TIME_WITHOUT_PROGRESS_VALUES)
    def test_valid(self, random_positive_timedelta, valid_satisfying_objective_value,
                   valid_max_iterations_without_progress, valid_max_time_without_progress):
        """
        Check that 'StopCondition' class can be initialized with example valid values for all parameters.

        :param random_positive_timedelta: Valid value for 'time_limit' parameter.
        :param valid_satisfying_objective_value: Valid value for 'satisfying_objective_value' parameter.
        :param valid_max_iterations_without_progress: Valid value for 'max_iterations_without_progress' parameter.
        :param valid_max_time_without_progress: Valid value for '_max_time_without_progress' parameter.
        """
        stop_condition = StopCondition(time_limit=random_positive_timedelta,
                                       satisfying_objective_value=valid_satisfying_objective_value,
                                       max_iterations_without_progress=valid_max_iterations_without_progress,
                                       max_time_without_progress=valid_max_time_without_progress)
        assert stop_condition.time_limit == random_positive_timedelta
        assert stop_condition.satisfying_objective_value == valid_satisfying_objective_value
        assert stop_condition.max_iterations_without_progress == valid_max_iterations_without_progress
        assert stop_condition.max_time_without_progress == valid_max_time_without_progress
        assert stop_condition._iterations_without_progress is None
        assert stop_condition._best_solution is None
        assert stop_condition._last_progress_time is None

    # time_limit error

    @pytest.mark.parametrize("example_value", EXAMPLE_VALUE_TYPES.difference({timedelta}), indirect=True)
    @pytest.mark.parametrize("valid_satisfying_objective_value", EXAMPLE_VALID_SATISFYING_OBJECTIVE_VALUES)
    @pytest.mark.parametrize("valid_max_iterations_without_progress",
                             EXAMPLE_VALID_MAX_ITERATIONS_WITHOUT_PROGRESS_VALUES)
    @pytest.mark.parametrize("valid_max_time_without_progress", EXAMPLE_VALID_MAX_TIME_WITHOUT_PROGRESS_VALUES)
    def test_invalid_time_limit_type(self, example_value, valid_satisfying_objective_value,
                                     valid_max_iterations_without_progress, valid_max_time_without_progress):
        """
        Check that TypeError will be raised if 'time_limit' parameter is not 'timedelta' type.

        :param example_value: Value for 'time_limit' parameter that is invalid type.
        :param valid_satisfying_objective_value: Valid value for 'satisfying_objective_value' parameter.
        :param valid_max_iterations_without_progress: Valid value for 'max_iterations_without_progress' parameter.
        :param valid_max_time_without_progress: Valid value for '_max_time_without_progress' parameter.
        """
        with pytest.raises(TypeError):
            StopCondition(time_limit=example_value, satisfying_objective_value=valid_satisfying_objective_value,
                          max_iterations_without_progress=valid_max_iterations_without_progress,
                          max_time_without_progress=valid_max_time_without_progress)

    def test_time_limit_value_0(self):
        """
        Check that ValueError will be raised if 'time_limit' parameter is equal 0 timedelta.
        """
        with pytest.raises(ValueError):
            StopCondition(time_limit=timedelta())

    @pytest.mark.parametrize("valid_satisfying_objective_value", EXAMPLE_VALID_SATISFYING_OBJECTIVE_VALUES)
    @pytest.mark.parametrize("valid_max_iterations_without_progress",
                             EXAMPLE_VALID_MAX_ITERATIONS_WITHOUT_PROGRESS_VALUES)
    @pytest.mark.parametrize("valid_max_time_without_progress", EXAMPLE_VALID_MAX_TIME_WITHOUT_PROGRESS_VALUES)
    def test_time_limit_negative_value(self, random_negative_timedelta, valid_satisfying_objective_value,
                                       valid_max_iterations_without_progress, valid_max_time_without_progress):
        """
        Check that ValueError will be raised if 'time_limit' parameter has negative value of 'timedelta' type.

        :param random_negative_timedelta: Value for 'time_limit' parameter that has invalid value.
        :param valid_satisfying_objective_value: Valid value for 'satisfying_objective_value' parameter.
        :param valid_max_iterations_without_progress: Valid value for 'max_iterations_without_progress' parameter.
        :param valid_max_time_without_progress: Valid value for '_max_time_without_progress' parameter.
        """
        with pytest.raises(ValueError):
            StopCondition(time_limit=random_negative_timedelta,
                          satisfying_objective_value=valid_satisfying_objective_value,
                          max_iterations_without_progress=valid_max_iterations_without_progress,
                          max_time_without_progress=valid_max_time_without_progress)

    # satisfying_objective_value error

    @pytest.mark.parametrize("example_value", EXAMPLE_VALUE_TYPES.difference({float, None}), indirect=True)
    @pytest.mark.parametrize("valid_max_iterations_without_progress",
                             EXAMPLE_VALID_MAX_ITERATIONS_WITHOUT_PROGRESS_VALUES)
    @pytest.mark.parametrize("valid_max_time_without_progress", EXAMPLE_VALID_MAX_TIME_WITHOUT_PROGRESS_VALUES)
    def test_invalid_satisfying_objective_value_type(self, random_positive_timedelta, example_value,
                                                     valid_max_iterations_without_progress,
                                                     valid_max_time_without_progress):
        """
        Check that TypeError will be raised if 'satisfying_objective_value' parameter is not None neither 'float'type.

        :param random_positive_timedelta: Valid value for 'time_limit' parameter.
        :param example_value: Value for 'satisfying_objective_value' parameter that is invalid type.
        :param valid_max_iterations_without_progress: Valid value for 'max_iterations_without_progress' parameter.
        :param valid_max_time_without_progress: Valid value for '_max_time_without_progress' parameter.
        """
        with pytest.raises(TypeError):
            StopCondition(time_limit=random_positive_timedelta, satisfying_objective_value=example_value,
                          max_iterations_without_progress=valid_max_iterations_without_progress,
                          max_time_without_progress=valid_max_time_without_progress)

    # max_iterations_without_progress error

    @pytest.mark.parametrize("valid_satisfying_objective_value", EXAMPLE_VALID_SATISFYING_OBJECTIVE_VALUES)
    @pytest.mark.parametrize("example_value", EXAMPLE_VALUE_TYPES.difference({int, None}), indirect=True)
    @pytest.mark.parametrize("valid_max_time_without_progress", EXAMPLE_VALID_MAX_TIME_WITHOUT_PROGRESS_VALUES)
    def test_invalid_max_iterations_without_progress_type(self, random_positive_timedelta,
                                                          valid_satisfying_objective_value, example_value,
                                                          valid_max_time_without_progress):
        """
        Check that TypeError will be raised if 'max_iterations_without_progress' parameter is not None nor 'int' type.

        :param random_positive_timedelta: Valid value for 'time_limit' parameter.
        :param valid_satisfying_objective_value: Valid value for 'satisfying_objective_value' parameter.
        :param example_value: Value for 'max_iterations_without_progress' parameter that is invalid type.
        :param valid_max_time_without_progress: Valid value for '_max_time_without_progress' parameter.
        """
        with pytest.raises(TypeError):
            StopCondition(time_limit=random_positive_timedelta,
                          satisfying_objective_value=valid_satisfying_objective_value,
                          max_iterations_without_progress=example_value,
                          max_time_without_progress=valid_max_time_without_progress)

    @pytest.mark.parametrize("valid_satisfying_objective_value", EXAMPLE_VALID_SATISFYING_OBJECTIVE_VALUES)
    @pytest.mark.parametrize("invalid_max_iterations_without_progress",
                             EXAMPLE_INVALID_MAX_ITERATIONS_WITHOUT_PROGRESS_VALUES)
    @pytest.mark.parametrize("valid_max_time_without_progress", EXAMPLE_VALID_MAX_TIME_WITHOUT_PROGRESS_VALUES)
    def test_invalid_max_iterations_without_progress_value(self, random_positive_timedelta,
                                                           valid_satisfying_objective_value,
                                                           invalid_max_iterations_without_progress,
                                                           valid_max_time_without_progress):
        """
        Check that ValueError will be raised if 'max_iterations_without_progress' parameter has negative value of
        int type.

        :param random_positive_timedelta: Valid value for 'time_limit' parameter.
        :param valid_satisfying_objective_value: Valid value for 'satisfying_objective_value' parameter.
        :param invalid_max_iterations_without_progress: Value for 'max_iterations_without_progress' parameter that has
            invalid value.
        :param valid_max_time_without_progress: Valid value for '_max_time_without_progress' parameter.
        """
        with pytest.raises(ValueError):
            StopCondition(time_limit=random_positive_timedelta,
                          satisfying_objective_value=valid_satisfying_objective_value,
                          max_iterations_without_progress=invalid_max_iterations_without_progress,
                          max_time_without_progress=valid_max_time_without_progress)

    # max_time_without_progress error

    @pytest.mark.parametrize("valid_satisfying_objective_value", EXAMPLE_VALID_SATISFYING_OBJECTIVE_VALUES)
    @pytest.mark.parametrize("valid_max_iterations_without_progress",
                             EXAMPLE_VALID_MAX_ITERATIONS_WITHOUT_PROGRESS_VALUES)
    @pytest.mark.parametrize("example_value", EXAMPLE_VALUE_TYPES.difference({timedelta, None}), indirect=True)
    def test_invalid_max_time_without_progress_type(self, random_positive_timedelta, valid_satisfying_objective_value,
                                                    valid_max_iterations_without_progress, example_value):
        """
        Check that TypeError will be raised if 'max_time_without_progress' parameter is not None nor 'timedelta' type.

        :param random_positive_timedelta: Valid value for 'time_limit' parameter.
        :param valid_satisfying_objective_value: Valid value for 'satisfying_objective_value' parameter.
        :param valid_max_iterations_without_progress: Valid value for 'max_iterations_without_progress' parameter.
        :param example_value: Value for 'max_time_without_progress' parameter that is invalid type.
        """
        with pytest.raises(TypeError):
            StopCondition(time_limit=random_positive_timedelta,
                          satisfying_objective_value=valid_satisfying_objective_value,
                          max_iterations_without_progress=valid_max_iterations_without_progress,
                          max_time_without_progress=example_value)

    def test_max_time_without_progress_value_0(self, random_positive_timedelta):
        """
        Check that ValueError will be raised if 'max_time_without_progress' parameter has 0 value of timedelta type.

        :param random_positive_timedelta: Valid value for 'time_limit' parameter.
        """
        with pytest.raises(ValueError):
            StopCondition(time_limit=random_positive_timedelta, max_time_without_progress=timedelta())

    def test_max_time_without_progress_negative_value(self, random_positive_timedelta, random_negative_timedelta):
        """
        Check that ValueError will be raised if 'max_time_without_progress' parameter has negative value of
        timedelta type.

        :param random_positive_timedelta: Valid value for 'time_limit' parameter.
        :param random_negative_timedelta: Invalid value for 'max_time_without_progress' parameter.
        """
        with pytest.raises(ValueError):
            StopCondition(time_limit=random_positive_timedelta, max_time_without_progress=random_negative_timedelta)


class TestStopConditionMethods:
    """Tests for non magic methods of 'StopCondition' class."""

    # _is_time_exceeded

    @patch("optimization.optimization_algorithms.stop_conditions.datetime")
    @pytest.mark.parametrize("diff, result", [(-100, False), (-0.0000001, False),
                                              (0, True), (0.0000001, True), (100, True)])
    def test_is_time_exceeded(self, mock_datetime, random_positive_int, random_float, diff, result):
        """
        Boundary Value Analysis tests for '_is_time_exceeded' method.

        :param mock_datetime: Mocked datetime class.
        :param random_positive_int: Value for 'time_limit' attribute of StopCondition object.
        :param random_float: Value to be returned by datetime.now method.
        :param diff: Distance from boundary value.
        :param result: Expected return from '_is_time_exceeded' method.
        """
        mock_stop_condition = Mock(time_limit=random_positive_int)
        mock_datetime.now = Mock(return_value=random_float + random_positive_int + diff)
        assert StopCondition._is_time_exceeded(self=mock_stop_condition,
                                               start_time=random_float) is result
        mock_datetime.now.assert_called_once_with()

    # _is_satisfying_solution_found

    @pytest.mark.parametrize("diff, result", [(-100, False), (-0.0000001, False),
                                              (0, True), (0.0000001, True), (100, True)])
    def test_is_satisfying_solution_found_minimize(self, random_float, diff, result):
        """
        Boundary Value Analysis tests for '_is_satisfying_solution_found' method (minimization).

        :param random_float: Value to 'satisfying_objective_value' attribute of StopCondition object.
        :param diff: Distance from boundary value.
        :param result: Expected return from '_is_satisfying_solution_found' method.
        """
        mock_stop_condition = Mock(satisfying_objective_value=random_float)
        mock_best_solution = Mock(optimization_problem=Mock(optimization_type=OptimizationType.Minimize),
                                  get_objective_value_with_penalty=Mock(return_value=random_float-diff))
        assert StopCondition._is_satisfying_solution_found(self=mock_stop_condition,
                                                           best_solution=mock_best_solution) is result
        mock_best_solution.get_objective_value_with_penalty.assert_called_once_with()

    @pytest.mark.parametrize("diff, result", [(-100, True), (-0.0000001, True),
                                              (0, True), (0.0000001, False), (100, False)])
    def test_is_satisfying_solution_found_maximize(self, random_float, diff, result):
        """
        Boundary Value Analysis tests for '_is_satisfying_solution_found' method (maximization).

        :param random_float: Value to 'satisfying_objective_value' attribute of StopCondition object.
        :param diff: Distance from boundary value.
        :param result: Expected return from '_is_satisfying_solution_found' method.
        """
        mock_stop_condition = Mock(satisfying_objective_value=random_float)
        mock_best_solution = Mock(optimization_problem=Mock(optimization_type=OptimizationType.Maximize),
                                  get_objective_value_with_penalty=Mock(return_value=random_float-diff))
        assert StopCondition._is_satisfying_solution_found(self=mock_stop_condition,
                                                           best_solution=mock_best_solution) is result
        mock_best_solution.get_objective_value_with_penalty.assert_called_once_with()

    @pytest.mark.parametrize("example_value", EXAMPLE_VALUE_TYPES, indirect=True)
    def test_is_satisfying_solution_found_value_error(self, example_value):
        """
        Check that ValueError is raised if 'best_solution' has unexpected value of 'optimization_type' attribute.

        :param example_value: Invalid value for 'optimization_type' attribute of 'best_solution' parameter.
        """
        mock_stop_condition = Mock()
        mock_best_solution = Mock(optimization_problem=Mock(optimization_type=example_value))
        with pytest.raises(ValueError):
            StopCondition._is_satisfying_solution_found(self=mock_stop_condition, best_solution=mock_best_solution)

    # _is_max_iteration_without_progress_exceeded

    @pytest.mark.parametrize("max_iterations, result", [(-100, True), (-1, True), (0, False), (100, False)])
    def test_is_max_iteration_without_progress_exceeded_first(self, max_iterations, result):
        """
        Boundary Value Analysis tests for '_is_max_iteration_without_progress_exceeded' method.
        Simulated first iteration (no solution is stored).

        :param max_iterations: Value to set in 'max_iterations_without_progress' attribute of StopCondition.
        :param result: Expected return from '_is_max_iteration_without_progress_exceeded' method.
        """
        mock_stop_condition = Mock(_best_solution=None, max_iterations_without_progress=max_iterations)
        mock_best_solution = Mock()
        assert StopCondition._is_max_iteration_without_progress_exceeded(self=mock_stop_condition,
                                                                         best_solution=mock_best_solution) is result
        assert mock_stop_condition._best_solution is mock_best_solution
        assert mock_stop_condition._iterations_without_progress == 0

    @pytest.mark.parametrize("max_iterations, result", [(-100, True), (-1, True), (0, False), (100, False)])
    def test_is_max_iteration_without_progress_exceeded_better_solution_found(self, max_iterations, result):
        """
        Boundary Value Analysis tests for '_is_max_iteration_without_progress_exceeded' method.
        Simulated better solution is found.

        :param max_iterations: Value to set in 'max_iterations_without_progress' attribute of StopCondition.
        :param result: Expected return from '_is_max_iteration_without_progress_exceeded' method.
        """
        mock_stop_condition = Mock(_best_solution=Mock(),
                                   max_iterations_without_progress=max_iterations)
        mock_le = Mock(return_value=True)
        mock_stop_condition._best_solution.__le__ = mock_le
        mock_best_solution = Mock()
        assert StopCondition._is_max_iteration_without_progress_exceeded(self=mock_stop_condition,
                                                                         best_solution=mock_best_solution) is result
        mock_le.assert_called_once_with(mock_best_solution)
        assert mock_stop_condition._best_solution is mock_best_solution
        assert mock_stop_condition._iterations_without_progress == 0

    @pytest.mark.parametrize("diff, result", [(-100, True), (0, True), (1, False), (100, False)])
    def test_is_max_iteration_without_progress_exceeded_better_solution_not_found(self, random_positive_int,
                                                                                  diff, result):
        """
        Boundary Value Analysis tests for '_is_max_iteration_without_progress_exceeded' method.
        Simulated better solution is not found.

        :param random_positive_int: Value to set in 'max_iterations_without_progress' attribute of StopCondition.
        :param diff: Distance from boundary value.
        :param result: Expected return from '_is_max_iteration_without_progress_exceeded' method.
        """
        mock_previously_best_solution = Mock()
        mock_stop_condition = Mock(_best_solution=mock_previously_best_solution,
                                   _iterations_without_progress=random_positive_int,
                                   max_iterations_without_progress=random_positive_int + diff)
        mock_le = Mock(return_value=False)
        mock_stop_condition._best_solution.__le__ = mock_le
        mock_best_solution = Mock()
        assert StopCondition._is_max_iteration_without_progress_exceeded(self=mock_stop_condition,
                                                                         best_solution=mock_best_solution) is result
        mock_le.assert_called_once_with(mock_best_solution)
        assert mock_stop_condition._best_solution is mock_previously_best_solution
        assert mock_stop_condition._iterations_without_progress == random_positive_int + 1

    # _is_max_time_without_progress_exceeded

    @pytest.mark.parametrize("max_time, result", [(-100, True), (-1, True), (0, False), (100, False)])
    @patch("optimization.optimization_algorithms.stop_conditions.datetime")
    def test_is_max_time_without_progress_exceeded_first(self, mock_datetime, random_positive_int, max_time, result):
        """
        Boundary Value Analysis tests for '_is_max_time_without_progress_exceeded' method.
        Simulated first iteration (no solution is stored).

        :param mock_datetime: Mock of datetime class in tested scope.
        :param random_positive_int: Value to return by datetime.now() function.
        :param max_time: Value to set in 'max_time_without_progress' attribute of StopCondition.
        :param result: Expected return from '_is_max_time_without_progress_exceeded' method.
        """
        mock_datetime.now = Mock(return_value=random_positive_int)
        mock_stop_condition = Mock(_best_solution=None, max_time_without_progress=max_time)
        mock_best_solution = Mock()
        assert StopCondition._is_max_time_without_progress_exceeded(self=mock_stop_condition,
                                                                    best_solution=mock_best_solution) is result
        assert mock_stop_condition._best_solution is mock_best_solution
        assert mock_stop_condition._last_progress_time == random_positive_int

    @pytest.mark.parametrize("max_time, result", [(-100, True), (-1, True), (0, False), (100, False)])
    @patch("optimization.optimization_algorithms.stop_conditions.datetime")
    def test_is_max_time_without_progress_exceeded_better_solution_found(self, mock_datetime, random_positive_int,
                                                                         max_time, result):
        """
        Boundary Value Analysis tests for '_is_max_iteration_without_progress_exceeded' method.
        Simulated better solution is found.

        :param mock_datetime: Mock of datetime class in tested scope.
        :param random_positive_int: Value to return by datetime.now() function.
        :param max_time: Value to set in 'max_time_without_progress' attribute of StopCondition.
        :param result: Expected return from '_is_max_time_without_progress_exceeded' method.
        """
        mock_datetime.now = Mock(return_value=random_positive_int)
        mock_stop_condition = Mock(_best_solution=Mock(), max_time_without_progress=max_time)
        mock_le = Mock(return_value=True)
        mock_stop_condition._best_solution.__le__ = mock_le
        mock_best_solution = Mock()
        assert StopCondition._is_max_time_without_progress_exceeded(self=mock_stop_condition,
                                                                    best_solution=mock_best_solution) is result
        mock_le.assert_called_once_with(mock_best_solution)
        assert mock_stop_condition._best_solution is mock_best_solution
        assert mock_stop_condition._last_progress_time == random_positive_int

    @pytest.mark.parametrize("diff, result", [(-100, True), (-1, True), (0, False), (100, False)])
    @patch("optimization.optimization_algorithms.stop_conditions.datetime")
    def test_is_max_time_without_progress_exceeded_better_solution_not_found(self, mock_datetime, random_positive_int,
                                                                             random_int, diff, result):
        """
        Boundary Value Analysis tests for '_is_max_iteration_without_progress_exceeded' method.
        Simulated better solution is not found.

        :param mock_datetime: Mock of datetime class in tested scope.
        :param random_positive_int: Value to return by datetime.now() function.
        :param random_int: Value to set in '_last_progress_time' attribute of stop condition.
        :param diff: Distance from boundary value.
        :param result: Expected return from '_is_max_time_without_progress_exceeded' method.
        """
        mock_datetime.now = Mock(return_value=random_positive_int)
        mock_previously_best_solution = Mock()
        mock_stop_condition = Mock(_best_solution=mock_previously_best_solution, _last_progress_time=random_int,
                                   max_time_without_progress=random_positive_int - random_int + diff)
        mock_le = Mock(return_value=False)
        mock_stop_condition._best_solution.__le__ = mock_le
        mock_best_solution = Mock()
        assert StopCondition._is_max_time_without_progress_exceeded(self=mock_stop_condition,
                                                                    best_solution=mock_best_solution) is result
        mock_le.assert_called_once_with(mock_best_solution)
        assert mock_stop_condition._best_solution is mock_previously_best_solution
        assert mock_stop_condition._last_progress_time == random_int

    # is_achieved

    def test_is_achieved_all_negative(self, random_datetime):
        """
        Test for 'is_achieved' method when False is expected in return value.

        :param random_datetime: Value for 'start_time' parameter.
        """
        solutions = [Mock() for _ in range(5)]
        mock_is_time_exceeded = Mock(return_value=False)
        mock_is_satisfying_solution_found = Mock(return_value=False)
        mock_is_max_iteration_without_progress_exceeded = Mock(return_value=False)
        mock_is_max_time_without_progress_exceeded = Mock(return_value=False)
        mock_stop_condition = Mock(
            satisfying_objective_value="not none",
            max_iterations_without_progress="not none",
            max_time_without_progress="not none",
            _is_time_exceeded=mock_is_time_exceeded,
            _is_satisfying_solution_found=mock_is_satisfying_solution_found,
            _is_max_iteration_without_progress_exceeded=mock_is_max_iteration_without_progress_exceeded,
            _is_max_time_without_progress_exceeded=mock_is_max_time_without_progress_exceeded
        )
        assert StopCondition.is_achieved(self=mock_stop_condition, start_time=random_datetime, solutions=solutions) \
            is False
        mock_is_time_exceeded.assert_called_once_with(start_time=random_datetime)
        mock_is_satisfying_solution_found.assert_called_once_with(best_solution=solutions[0])
        mock_is_max_iteration_without_progress_exceeded.assert_called_once_with(best_solution=solutions[0])
        mock_is_max_time_without_progress_exceeded.assert_called_once_with(best_solution=solutions[0])

    @pytest.mark.parametrize("condition_achieved", range(4))
    def test_is_achieved_positive(self, condition_achieved, random_datetime):
        """
        Test for 'is_achieved' method when True is expected in return value due to time exceeding
        (no other checks are done).

        :param condition_achieved: Index of stop condition that was achieved.
        :param random_datetime: Value for 'start_time' parameter.
        """
        solutions = [Mock() for _ in range(5)]
        mock_is_time_exceeded = Mock(return_value=(condition_achieved == 0))
        mock_is_satisfying_solution_found = Mock(return_value=(condition_achieved == 1))
        mock_is_max_iteration_without_progress_exceeded = Mock(return_value=(condition_achieved == 2))
        mock_is_max_time_without_progress_exceeded = Mock(return_value=(condition_achieved == 3))
        mock_stop_condition = Mock(
            satisfying_objective_value="not none",
            max_iterations_without_progress="not none",
            max_time_without_progress="not none",
            _is_time_exceeded=mock_is_time_exceeded,
            _is_satisfying_solution_found=mock_is_satisfying_solution_found,
            _is_max_iteration_without_progress_exceeded=mock_is_max_iteration_without_progress_exceeded,
            _is_max_time_without_progress_exceeded=mock_is_max_time_without_progress_exceeded
        )
        assert StopCondition.is_achieved(self=mock_stop_condition, start_time=random_datetime, solutions=solutions) \
            is True
        mock_is_time_exceeded.assert_called_once_with(start_time=random_datetime)
        if condition_achieved > 0:
            mock_is_satisfying_solution_found.assert_called_once_with(best_solution=solutions[0])
        else:
            mock_is_satisfying_solution_found.assert_not_called()
        if condition_achieved > 1:
            mock_is_max_iteration_without_progress_exceeded.assert_called_once_with(best_solution=solutions[0])
        else:
            mock_is_max_iteration_without_progress_exceeded.assert_not_called()
        if condition_achieved > 2:
            mock_is_max_time_without_progress_exceeded.assert_called_once_with(best_solution=solutions[0])
        else:
            mock_is_max_time_without_progress_exceeded.assert_not_called()

    def test_is_achieved_not_checked(self, random_datetime):
        """
        Test for 'is_achieved' method when True is expected in return value due to time exceeding
        (no other checks are done).

        :param random_datetime: Value for 'start_time' parameter.
        """
        solutions = [Mock() for _ in range(5)]
        mock_is_time_exceeded = Mock(return_value=False)
        mock_is_satisfying_solution_found = Mock(return_value=True)
        mock_is_max_iteration_without_progress_exceeded = Mock(return_value=True)
        mock_is_max_time_without_progress_exceeded = Mock(return_value=True)
        mock_stop_condition = Mock(
            satisfying_objective_value=None,
            max_iterations_without_progress=None,
            max_time_without_progress=None,
            _is_time_exceeded=mock_is_time_exceeded,
            _is_satisfying_solution_found=mock_is_satisfying_solution_found,
            _is_max_iteration_without_progress_exceeded=mock_is_max_iteration_without_progress_exceeded,
            _is_max_time_without_progress_exceeded=mock_is_max_time_without_progress_exceeded
        )
        assert StopCondition.is_achieved(self=mock_stop_condition, start_time=random_datetime, solutions=solutions) \
            is False
        mock_is_time_exceeded.assert_called_once_with(start_time=random_datetime)
        mock_is_satisfying_solution_found.assert_not_called()
        mock_is_max_iteration_without_progress_exceeded.assert_not_called()
        mock_is_max_time_without_progress_exceeded.assert_not_called()

    # get_data_for_logging

    def test_get_data_for_logging(self):
        """Test for 'get_data_for_logging' method."""
        mock_stop_condition = Mock(time_limit=Mock(),
                                   satisfying_objective_value=Mock(),
                                   max_iterations_without_progress=Mock(),
                                   max_time_without_progress=Mock())
        data_for_logging = StopCondition.get_data_for_logging(self=mock_stop_condition)
        assert isinstance(data_for_logging, dict)
        assert data_for_logging["time_limit"] is mock_stop_condition.time_limit
        assert data_for_logging["satisfying_objective_value"] is mock_stop_condition.satisfying_objective_value
        assert data_for_logging["max_iterations_without_progress"] \
            is mock_stop_condition.max_iterations_without_progress
        assert data_for_logging["max_time_without_progress"] is mock_stop_condition.max_time_without_progress
