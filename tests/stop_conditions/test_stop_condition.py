import pytest
from mock import Mock, patch
from datetime import timedelta

from optimization.stop_condition import StopCondition


class TestStopConditions:
    """Tests for 'StopCondition' class and their methods."""

    SCRIPT_LOCATION = "optimization.stop_condition"

    def setup(self):
        self.mock_stop_condition_object = Mock(spec=StopCondition)
        self.mock_solution_object__less_equal = Mock()
        self.mock_solution_object = Mock(__le__=self.mock_solution_object__less_equal)
        self.mock_datetime_now = Mock()
        # patching
        self._patcher_datetime = patch(f"{self.SCRIPT_LOCATION}.datetime", Mock(now=self.mock_datetime_now))
        self.mock_datetime = self._patcher_datetime.start()

    def teardown(self):
        self._patcher_datetime.stop()

    # __init__

    def test_init__valid_time_limit_only(self):
        """Test 'StopCondition' initialization with valid value of 'time_limit' parameter only."""
        mock_time_limit = Mock(spec=timedelta, __le__=Mock(return_value=False))
        StopCondition.__init__(self=self.mock_stop_condition_object, time_limit=mock_time_limit)
        assert self.mock_stop_condition_object.time_limit == mock_time_limit
        assert self.mock_stop_condition_object.satisfying_objective_value is None
        assert self.mock_stop_condition_object.max_iter_without_progress is None
        assert self.mock_stop_condition_object.max_time_without_progress is None
        assert self.mock_stop_condition_object._best_objective_found is None
        assert self.mock_stop_condition_object._last_objective_progress_datetime is None
        assert self.mock_stop_condition_object._iter_without_progress is None

    @pytest.mark.parametrize("satisfying_objective_value", [None, 123.456, -10])
    @pytest.mark.parametrize("max_iter_without_progress", [None, 2])
    @pytest.mark.parametrize("max_time_without_progress", [None, Mock(spec=timedelta, __le__=Mock(return_value=True))])
    def test_init__valid_all(self, satisfying_objective_value, max_iter_without_progress, max_time_without_progress):
        """
        Test 'StopCondition' initialization with valid values of all parameters.

        :param satisfying_objective_value: Example value of 'satisfying_objective_value'.
        :param max_iter_without_progress: Example value of 'max_iter_without_progress'.
        :param max_time_without_progress: Example value of 'max_time_without_progress'.
        """
        mock_time_limit = Mock(spec=timedelta, __le__=Mock(return_value=False))
        StopCondition.__init__(self=self.mock_stop_condition_object, time_limit=mock_time_limit,
                               satisfying_objective_value=satisfying_objective_value,
                               max_iter_without_progress=max_iter_without_progress,
                               max_time_without_progress=max_time_without_progress)
        assert self.mock_stop_condition_object.time_limit == mock_time_limit
        assert self.mock_stop_condition_object.satisfying_objective_value == satisfying_objective_value
        assert self.mock_stop_condition_object.max_iter_without_progress == max_iter_without_progress
        assert self.mock_stop_condition_object.max_time_without_progress == max_time_without_progress
        assert self.mock_stop_condition_object._best_objective_found is None
        assert self.mock_stop_condition_object._last_objective_progress_datetime is None
        assert self.mock_stop_condition_object._iter_without_progress is None

    def test_init__time_limit_less_or_equal_zero(self):
        """
        Test 'StopCondition' initialization with invalid value of 'time_limit' parameter.
        Value of 'time_limit' simulated as less or equal zero.
        """
        mock_time_limit = Mock(spec=timedelta, __le__=Mock(return_value=True))
        with pytest.raises(ValueError):
            StopCondition.__init__(self=self.mock_stop_condition_object, time_limit=mock_time_limit)

    def test_init__time_limit_incorrect_type(self):
        """Test 'StopCondition' initialization with invalid type of 'time_limit' parameter."""
        mock_time_limit = Mock(__le__=Mock(return_value=False))
        with pytest.raises(TypeError):
            StopCondition.__init__(self=self.mock_stop_condition_object, time_limit=mock_time_limit)

    @pytest.mark.parametrize("invalid_satisfying_objective_value", ["some value", (0,)])
    def test_init__satisfying_objective_value_incorrect_type(self, invalid_satisfying_objective_value):
        """Test 'StopCondition' initialization with invalid type of 'satisfying_objective_value' parameter."""
        mock_time_limit = Mock(spec=timedelta, __le__=Mock(return_value=False))
        with pytest.raises(TypeError):
            StopCondition.__init__(self=self.mock_stop_condition_object, time_limit=mock_time_limit,
                                   satisfying_objective_value=invalid_satisfying_objective_value)

    @pytest.mark.parametrize("invalid_max_iter_without_progress", ["some value", (0,), 1.2])
    def test_init__max_iter_without_progress_incorrect_type(self, invalid_max_iter_without_progress):
        """Test 'StopCondition' initialization with invalid type of 'max_iter_without_progress' parameter."""
        mock_time_limit = Mock(spec=timedelta, __le__=Mock(return_value=False))
        with pytest.raises(TypeError):
            StopCondition.__init__(self=self.mock_stop_condition_object, time_limit=mock_time_limit,
                                   max_iter_without_progress=invalid_max_iter_without_progress)

    @pytest.mark.parametrize("invalid_max_time_without_progress", ["some value", 0, 1.1])
    def test_init__max_iter_without_progress_incorrect_type(self, invalid_max_time_without_progress):
        """Test 'StopCondition' initialization with invalid type of 'max_time_without_progress' parameter."""
        mock_time_limit = Mock(spec=timedelta, __le__=Mock(return_value=False))
        with pytest.raises(TypeError):
            StopCondition.__init__(self=self.mock_stop_condition_object, time_limit=mock_time_limit,
                                   max_time_without_progress=invalid_max_time_without_progress)

    # _is_time_exceeded

    @pytest.mark.parametrize("now, start_time, time_limit, expected_result", [
        (1, 0, 1.00001, False),
        (1, 0, 1, True),
        (1, 0, 0.99999, True),
        (987.654, 123.456, 864.199, False),
        (987.654, 123.456, 864.198, True),
        (987.654, 123.456, 864.197, True),
    ])
    def test_is_time_exceeded(self, now, start_time, time_limit, expected_result):
        """
        Test '_is_time_exceeded' method return True if time_limit exceeded ([now] - [start_time] >= [time_limit]),
        False otherwise.

        :param now: Value that represents time now.
        :param start_time: Value that represents start time.
        :param time_limit: Value that represents time limit.
        :param expected_result: Expected return from '_is_time_exceeded' method.
        """
        self.mock_datetime_now.return_value = now
        self.mock_stop_condition_object.time_limit = time_limit
        assert StopCondition._is_time_exceeded(self.mock_stop_condition_object, start_time) is expected_result
        self.mock_datetime_now.assert_called_once()

    # _is_satisfying_solution_found

    @pytest.mark.parametrize("satisfying_objective_value", [0, 2.456])
    @pytest.mark.parametrize("expected_result", [True, False])
    def test_is_satisfying_solution_found(self, satisfying_objective_value, expected_result):
        """
        Test '_is_satisfying_solution_found' method return results of comparison 'best_solution' and
        'satisfying_objective_value'.

        :param satisfying_objective_value: Example value of 'satisfying_objective_value'.
        :param expected_result: Expected return from '_is_satisfying_solution_found' method.
        """
        self.mock_stop_condition_object.satisfying_objective_value = satisfying_objective_value
        self.mock_solution_object__less_equal.return_value = expected_result
        assert StopCondition._is_satisfying_solution_found(self=self.mock_stop_condition_object,
                                                           best_solution=self.mock_solution_object) is expected_result
        self.mock_solution_object__less_equal.assert_called_once_with(satisfying_objective_value)

    # _is_max_iteration_without_progress_exceeded

    # todo: continue here
    # def test_is_max_iteration_without_progress_exceeded(self):
