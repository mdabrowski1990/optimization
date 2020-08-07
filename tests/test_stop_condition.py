import pytest
from mock import Mock, patch
from datetime import timedelta

from optimization.stop_condition import StopCondition


class TestStopConditions:
    """Tests for 'StopCondition' class and their methods."""

    SCRIPT_LOCATION = "optimization.stop_condition"

    def setup(self):
        self.mock_is_time_exceeded = Mock()
        self.mock_is_satisfying_solution_found = Mock()
        self.mock_is_iter_without_progress_exceeded = Mock()
        self.mock_is_time_without_progress_exceeded = Mock()
        self.mock_is_limit_without_progress_exceeded = Mock()
        self.mock_stop_condition_object = Mock(spec=StopCondition,
                                               _is_time_exceeded=self.mock_is_time_exceeded,
                                               _is_satisfying_solution_found=self.mock_is_satisfying_solution_found,
                                               _is_iter_without_progress_exceeded=self.mock_is_iter_without_progress_exceeded,
                                               _is_time_without_progress_exceeded=self.mock_is_time_without_progress_exceeded,
                                               _is_limit_without_progress_exceeded=self.mock_is_limit_without_progress_exceeded)
        self.mock_get_objective_value_with_penalty = Mock()
        self.mock_solution_object = Mock(get_objective_value_with_penalty=self.mock_get_objective_value_with_penalty)
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

    @pytest.mark.parametrize("invalid_max_iter_without_progress", [0, -1, -10])
    def test_init__max_iter_without_progress_incorrect_value(self, invalid_max_iter_without_progress):
        """Test 'StopCondition' initialization with invalid value of 'max_iter_without_progress' parameter."""
        mock_time_limit = Mock(spec=timedelta, __le__=Mock(return_value=False))
        with pytest.raises(ValueError):
            StopCondition.__init__(self=self.mock_stop_condition_object, time_limit=mock_time_limit,
                                   max_iter_without_progress=invalid_max_iter_without_progress)

    @pytest.mark.parametrize("invalid_max_time_without_progress", ["some value", 0, 1.1])
    def test_init__max_time_without_progress_incorrect_type(self, invalid_max_time_without_progress):
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

    @pytest.mark.parametrize("satisfying_objective_value", [-1, -32.342, 0, 0., 2.456, 564])
    @pytest.mark.parametrize("diff", [0, 0.000000000001, 1, 543.543534])
    def test_is_satisfying_solution_found__true(self, satisfying_objective_value, diff):
        """
        Test '_is_satisfying_solution_found' method return results of comparison 'best_solution' and
        'satisfying_objective_value'.

        :param satisfying_objective_value: Example value of 'satisfying_objective_value'.
        :param diff: Difference between 'satisfying_objective_value' and best)solution objective.
        """
        self.mock_stop_condition_object.satisfying_objective_value = satisfying_objective_value
        self.mock_get_objective_value_with_penalty.return_value = satisfying_objective_value + diff
        assert StopCondition._is_satisfying_solution_found(self=self.mock_stop_condition_object,
                                                           best_solution=self.mock_solution_object) is True
        self.mock_get_objective_value_with_penalty.assert_called_once_with()

    # _is_iter_without_progress_exceeded

    @pytest.mark.parametrize("max_iter", [1, 4, 5, 45343])
    @pytest.mark.parametrize("current_iter_diff", [1, 5, 32718])
    def test_is_iter_without_progress_exceeded__true(self, max_iter, current_iter_diff):
        """
        Check status returned by '_is_iter_without_progress_exceeded' is True
        if max_iter_without_progress < _iter_without_progress.

        :param max_iter: Maximal number of iterations without progress.
        :param current_iter_diff: Difference between maximal and current number of iterations.
        """
        self.mock_stop_condition_object.max_iter_without_progress = max_iter
        self.mock_stop_condition_object._iter_without_progress = max_iter + current_iter_diff
        assert StopCondition._is_iter_without_progress_exceeded(self=self.mock_stop_condition_object) is True

    @pytest.mark.parametrize("max_iter, current_iter", [(None, 4312), (2, 2), (3, 0), (423, 21)])
    def test_is_iter_without_progress_exceeded__false(self, max_iter, current_iter):
        """
        Check status returned by '_is_iter_without_progress_exceeded' is False
        if max_iter_without_progress >= _iter_without_progress.

        :param max_iter: Maximal number of iterations without progress.
        :param current_iter: Current number of iterations without progress.
        """
        self.mock_stop_condition_object.max_iter_without_progress = max_iter
        self.mock_stop_condition_object._iter_without_progress = current_iter
        assert StopCondition._is_iter_without_progress_exceeded(self=self.mock_stop_condition_object) is False

    # _is_time_without_progress_exceeded

    @pytest.mark.parametrize("max_time_without_progress, current_time, last_progress_time", [
        (1, 1.000001, 0),
        (9876.5432, 9920.4133, 43.87),
        (10, 20, 1)
    ])
    def test_is_time_without_progress_exceeded__true(self, max_time_without_progress, current_time, last_progress_time):
        """
        Check status returned by '_is_time_without_progress_exceeded' is True
        if max_time_without_progress < current_time - last_progress_time.

        :param max_time_without_progress: Maximal time without finding a better solution.
        :param current_time: Time returned by datetime.now()
        :param last_progress_time: Last time when progress in solution took place.
        """
        self.mock_datetime_now.return_value = current_time
        self.mock_stop_condition_object.max_time_without_progress = max_time_without_progress
        self.mock_stop_condition_object._last_objective_progress_datetime = last_progress_time
        assert StopCondition._is_time_without_progress_exceeded(self=self.mock_stop_condition_object) is True

    @pytest.mark.parametrize("max_time_without_progress, current_time, last_progress_time", [
        (None, 1, 0),
        (1, 1, 0),
        (9876.5432, 9920.4132, 43.87),
        (10, 20, 15)
    ])
    def test_is_time_without_progress_exceeded__false(self, max_time_without_progress, current_time, last_progress_time):
        """
        Check status returned by '_is_time_without_progress_exceeded' is False
        if max_time_without_progress >= current_time - last_progress_time.

        :param max_time_without_progress: Maximal time without finding a better solution.
        :param current_time: Time returned by datetime.now()
        :param last_progress_time: Last time when progress in solution took place.
        """
        self.mock_datetime_now.return_value = current_time
        self.mock_stop_condition_object.max_time_without_progress = max_time_without_progress
        self.mock_stop_condition_object._last_objective_progress_datetime = last_progress_time
        assert StopCondition._is_time_without_progress_exceeded(self=self.mock_stop_condition_object) is False

    # _is_limit_without_progress_exceeded

    def test_is_limit_without_progress_exceeded__limits_not_set(self):
        """
        Test '_is_limit_without_progress_exceeded' return False if 'max_iter_without_progress' and
        'max_time_without_progress' is not defined.
        """
        self.mock_stop_condition_object.max_iter_without_progress = None
        self.mock_stop_condition_object.max_time_without_progress = None
        assert StopCondition._is_limit_without_progress_exceeded(self=self.mock_stop_condition_object,
                                                                 best_solution=Mock()) is False

    @pytest.mark.parametrize("max_iter_without_progress, max_time_without_progress", [(2, None), (None, 2), (10, 10)])
    @pytest.mark.parametrize("best_objective_found, new_objective", [(None, -10), (-231, -230.999), (1.1, 2321)])
    @pytest.mark.parametrize("is_iter_exceeded, is_time_exceeded, expected_result", [
        (True, True, True),
        (True, False, True),
        (False, True, True),
        (False, False, False)
    ])
    def test_is_limit_without_progress_exceeded__better_solution_found(self, max_iter_without_progress,
                                                                       max_time_without_progress, best_objective_found,
                                                                       new_objective, is_iter_exceeded,
                                                                       is_time_exceeded, expected_result):
        """
        Test '_is_limit_without_progress_exceeded' when better solution is found in this iteration.

        :param max_iter_without_progress: Example value of 'max_iter_without_progress'.
        :param max_time_without_progress: Example value of 'max_time_without_progress'.
        :param best_objective_found: Example value of currently best solution.
        :param new_objective: Value of newly found best solution.
        :param is_iter_exceeded: Value to be returned by '_is_iter_without_progress_exceeded' method.
        :param is_time_exceeded: Value to be returned by '_is_time_without_progress_exceeded' method.
        :param expected_result: Expected result from '_is_limit_without_progress_exceeded' method.
        """
        self.mock_stop_condition_object.max_iter_without_progress = max_iter_without_progress
        self.mock_stop_condition_object.max_time_without_progress = max_time_without_progress
        self.mock_stop_condition_object._best_objective_found = best_objective_found
        self.mock_get_objective_value_with_penalty.return_value = new_objective
        self.mock_is_iter_without_progress_exceeded.return_value = is_iter_exceeded
        self.mock_is_time_without_progress_exceeded.return_value = is_time_exceeded
        assert StopCondition._is_limit_without_progress_exceeded(self=self.mock_stop_condition_object,
                                                                 best_solution=self.mock_solution_object) is expected_result
        assert self.mock_stop_condition_object._iter_without_progress == 0
        assert self.mock_stop_condition_object._last_objective_progress_datetime == self.mock_datetime_now.return_value
        assert self.mock_stop_condition_object._best_objective_found == self.mock_get_objective_value_with_penalty.return_value

    @pytest.mark.parametrize("max_iter_without_progress, max_time_without_progress", [(2, None), (None, 2), (10, 10)])
    @pytest.mark.parametrize("iter_without_progress", [0, 1, 5])
    @pytest.mark.parametrize("last_progress_time", ["some date", 0])
    @pytest.mark.parametrize("best_objective_found, new_objective", [(-231, -231), (1.1, 1.09999), (0.12345, -9387)])
    @pytest.mark.parametrize("is_iter_exceeded, is_time_exceeded, expected_result", [
        (True, True, True),
        (True, False, True),
        (False, True, True),
        (False, False, False)
    ])
    def test_is_limit_without_progress_exceeded__better_solution_not_found(self, max_iter_without_progress,
                                                                           max_time_without_progress,
                                                                           iter_without_progress, last_progress_time,
                                                                           best_objective_found, new_objective,
                                                                           is_iter_exceeded, is_time_exceeded,
                                                                           expected_result):
        """
        Test '_is_limit_without_progress_exceeded' when better solution is not found in this iteration.

        :param max_iter_without_progress: Example value of 'max_iter_without_progress'.
        :param max_time_without_progress: Example value of 'max_time_without_progress'.
        :param iter_without_progress: Example value of '_iter_without_progress'.
        :param last_progress_time: Example value of '_last_objective_progress_datetime'.
        :param best_objective_found: Example value of currently best solution.
        :param new_objective: Value of newly found best solution.
        :param is_iter_exceeded: Value to be returned by '_is_iter_without_progress_exceeded' method.
        :param is_time_exceeded: Value to be returned by '_is_time_without_progress_exceeded' method.
        :param expected_result: Expected result from '_is_limit_without_progress_exceeded' method.
        """
        self.mock_stop_condition_object.max_iter_without_progress = max_iter_without_progress
        self.mock_stop_condition_object.max_time_without_progress = max_time_without_progress
        self.mock_stop_condition_object._iter_without_progress = iter_without_progress
        self.mock_stop_condition_object._last_objective_progress_datetime = last_progress_time
        self.mock_stop_condition_object._best_objective_found = best_objective_found
        self.mock_get_objective_value_with_penalty.return_value = new_objective
        self.mock_is_iter_without_progress_exceeded.return_value = is_iter_exceeded
        self.mock_is_time_without_progress_exceeded.return_value = is_time_exceeded
        assert StopCondition._is_limit_without_progress_exceeded(self=self.mock_stop_condition_object,
                                                                 best_solution=self.mock_solution_object) is expected_result
        assert self.mock_stop_condition_object._iter_without_progress == iter_without_progress + 1
        assert self.mock_stop_condition_object._best_objective_found == best_objective_found
        assert self.mock_stop_condition_object._last_objective_progress_datetime == last_progress_time

    # is_achieved

    @pytest.mark.parametrize("is_time_exceeded, is_satisfying_solution_found, is_limit_without_progress_exceeded, "
                             "expected_result", [
                                 (True, True, True, True),
                                 (True, False, False, True),
                                 (False, True, False, True),
                                 (False, False, True, True),
                                 (False, False, False, False),
                             ])
    def test_is_achieved(self, is_time_exceeded, is_satisfying_solution_found, is_limit_without_progress_exceeded,
                         expected_result):
        """
        Test 'is_achieved' method returns outcome of '_is_time_exceeded', '_is_satisfying_solution_found'
        and '_is_limit_without_progress_exceeded' methods.

        :param is_time_exceeded: Simulated return value of '_is_time_exceeded' method.
        :param is_satisfying_solution_found: Simulated return value of '_is_satisfying_solution_found' method.
        :param is_limit_without_progress_exceeded: Simulated return value of '_is_limit_without_progress_exceeded' method.
        :param expected_result: Expected return value of 'is_achieved' method.
        """
        self.mock_is_time_exceeded.return_value = is_time_exceeded
        self.mock_is_satisfying_solution_found.return_value = is_satisfying_solution_found
        self.mock_is_limit_without_progress_exceeded.return_value = is_limit_without_progress_exceeded
        mock_start_time = Mock()
        assert StopCondition.is_achieved(self=self.mock_stop_condition_object, start_time=mock_start_time,
                                         best_solution=self.mock_solution_object) is expected_result

    @pytest.mark.parametrize("time_limit", [timedelta(days=1), timedelta(hours=5)])
    @pytest.mark.parametrize("satisfying_objective_value", [None, 0, 2.321, -453])
    @pytest.mark.parametrize("max_iter_without_progress", [None, 1, 54])
    @pytest.mark.parametrize("max_time_without_progress", [None, timedelta(seconds=5), timedelta(hours=1)])
    def test_get_log_data(self, time_limit, satisfying_objective_value, max_iter_without_progress,
                          max_time_without_progress):
        """
        Test 'log_data' return dictionary with proper data.
        """
        self.mock_stop_condition_object.time_limit = time_limit
        self.mock_stop_condition_object.satisfying_objective_value = satisfying_objective_value
        self.mock_stop_condition_object.max_iter_without_progress = max_iter_without_progress
        self.mock_stop_condition_object.max_time_without_progress = max_time_without_progress
        log_data = StopCondition.get_log_data(self.mock_stop_condition_object)
        assert isinstance(log_data, dict)
        assert "time_limit" in log_data
        assert "satisfying_objective_value" in log_data
        assert "max_iter_without_progress" in log_data
        assert "max_time_without_progress" in log_data
