import pytest
from mock import Mock, patch, call

from optimization.algorithms.abstract_algorithm import AbstractOptimizationAlgorithm, \
    OptimizationProblem, StopConditions, AbstractLogger, AbstractSolution


class TestAbstractOptimizationAlgorithm:
    """Tests for 'AbstractOptimizationAlgorithm' and their methods."""

    SCRIPT_LOCATION = "optimization.algorithms.abstract_algorithm"

    def setup(self):
        self.mock_algorithm_object_is_stop_achieved = Mock()
        self.mock_algorithm_object_perform_iteration = Mock()
        self.mock_algorithm_object_stop_conditions_is_achieved = Mock()
        self.mock_algorithm_object_stop_conditions = Mock(is_achieved=self.mock_algorithm_object_stop_conditions_is_achieved)
        self.mock_algorithm_object = Mock(spec=AbstractOptimizationAlgorithm,
                                          _is_stop_achieved=self.mock_algorithm_object_is_stop_achieved,
                                          _perform_iteration=self.mock_algorithm_object_perform_iteration,
                                          stop_conditions=self.mock_algorithm_object_stop_conditions)
        self.mock_problem_object = Mock(spec=OptimizationProblem)
        self.mock_stop_conditions_object = Mock(spec=StopConditions)
        self.mock_logger_object = Mock(spec=AbstractLogger)
        self.mock_datetime_now = Mock()
        # patching
        self._patcher_sorted = patch(f"{self.SCRIPT_LOCATION}.sorted")
        self.mock_sorted = self._patcher_sorted.start()
        self._patcher_datetime = patch(f"{self.SCRIPT_LOCATION}.datetime", Mock(now=self.mock_datetime_now))
        self.mock_datetime = self._patcher_datetime.start()

    def teardown(self):
        self._patcher_sorted.stop()
        self._patcher_datetime.stop()

    # __init__

    @pytest.mark.parametrize("invalid_problem", ["some problem", 1, None])
    def test_init__invalid_problem_type(self, invalid_problem):
        """
        Test that TypeError is raised when initialization of 'AbstractOptimizationAlgorithm' is performed with invalid
        problem type.

        :param invalid_problem: Value that is not 'OptimizationProblem' type.
        """
        with pytest.raises(TypeError):
            AbstractOptimizationAlgorithm.__init__(self=self.mock_algorithm_object, problem=invalid_problem,
                                                   stop_conditions=self.mock_stop_conditions_object)

    @pytest.mark.parametrize("invalid_stop_conditions", ["some problem", 1, None])
    def test_init__invalid_stop_conditions_type(self, invalid_stop_conditions):
        """
        Test that TypeError is raised when initialization of 'AbstractOptimizationAlgorithm' is performed with invalid
        stop conditions type.

        :param invalid_stop_conditions: Value that is not 'StopConditions' type.
        """
        with pytest.raises(TypeError):
            AbstractOptimizationAlgorithm.__init__(self=self.mock_algorithm_object,
                                                   problem=self.mock_problem_object,
                                                   stop_conditions=invalid_stop_conditions)

    @pytest.mark.parametrize("invalid_logger", ["some logger", 1, []])
    def test_init__invalid_logger_type(self, invalid_logger):
        """
        Test that TypeError is raised when initialization of 'AbstractOptimizationAlgorithm' is performed with invalid
        logger type.

        :param invalid_logger: Value that is not 'AbstractLogger' type.
        """
        with pytest.raises(TypeError):
            AbstractOptimizationAlgorithm.__init__(self=self.mock_algorithm_object, problem=self.mock_problem_object,
                                                   stop_conditions=self.mock_stop_conditions_object,
                                                   logger=invalid_logger)

    def test_init__valid(self):
        """
        Tests initialization of 'AbstractOptimizationAlgorithm' with mandatory arguments.
        """
        AbstractOptimizationAlgorithm.__init__(self=self.mock_algorithm_object, problem=self.mock_problem_object,
                                               stop_conditions=self.mock_stop_conditions_object)
        assert self.mock_algorithm_object.problem == self.mock_problem_object
        assert self.mock_algorithm_object.stop_conditions == self.mock_stop_conditions_object
        assert issubclass(self.mock_algorithm_object.SolutionClass, AbstractSolution)
        assert self.mock_algorithm_object.SolutionClass.optimization_problem == self.mock_problem_object
        assert self.mock_algorithm_object.logger is None
        assert self.mock_algorithm_object._start_time is None
        assert self.mock_algorithm_object._end_time is None
        assert self.mock_algorithm_object._best_solution is None

    def test_init__valid_all_args(self):
        """
        Tests initialization of 'AbstractOptimizationAlgorithm' with all arguments.
        """
        AbstractOptimizationAlgorithm.__init__(self=self.mock_algorithm_object, problem=self.mock_problem_object,
                                               stop_conditions=self.mock_stop_conditions_object,
                                               logger=self.mock_logger_object)
        assert self.mock_algorithm_object.problem == self.mock_problem_object
        assert self.mock_algorithm_object.stop_conditions == self.mock_stop_conditions_object
        assert issubclass(self.mock_algorithm_object.SolutionClass, AbstractSolution)
        assert self.mock_algorithm_object.SolutionClass.optimization_problem == self.mock_problem_object
        assert self.mock_algorithm_object.logger == self.mock_logger_object
        assert self.mock_algorithm_object._start_time is None
        assert self.mock_algorithm_object._end_time is None
        assert self.mock_algorithm_object._best_solution is None

    # _is_stop_achieved

    @pytest.mark.parametrize("start_time", [0, "some time"])
    @pytest.mark.parametrize("best_solution", [1, "some solution"])
    @pytest.mark.parametrize("status", [True, False])
    def test_is_stop_achieved(self, status, start_time, best_solution):
        """
        Tests '_is_stop_achieved' method return the same value as 'is_achieved' method of stop conditions.

        :param status: Status returned by 'is_achieved' method of stop_condition attribute.
        :param start_time: Value of start time currently stored in optimization algorithm.
        :param best_solution: Value of best_solution currently stored in optimization algorithm.
        """
        self.mock_algorithm_object_stop_conditions_is_achieved.return_value = status
        self.mock_algorithm_object._start_time = start_time
        self.mock_algorithm_object._best_solution = best_solution
        assert AbstractOptimizationAlgorithm._is_stop_achieved(self=self.mock_algorithm_object) is status
        self.mock_algorithm_object_stop_conditions_is_achieved.assert_called_once_with(start_time=start_time,
                                                                                       best_solution=best_solution)

    # sorted_solutions

    @pytest.mark.parametrize("solutions", [range(10), "absdefg"])
    @pytest.mark.parametrize("descending", [True, False])
    def test_sorted_solutions(self, solutions, descending):
        """
        Tests solution sorting method.

        :param solutions: Iterable with solution objects.
        :param descending: Order of solutions sorting.
        """
        assert AbstractOptimizationAlgorithm.sorted_solutions(solutions=solutions, descending=descending) \
            == self.mock_sorted.return_value
        self.mock_sorted.assert_called_once_with(solutions, reverse=descending)

    # perform_optimization

    @pytest.mark.parametrize("best_solution", [1, "some solution"])
    @pytest.mark.parametrize("start_time", ["some start time", "some other start time"])
    @pytest.mark.parametrize("end_time", ["some end time", "some other end time"])
    @pytest.mark.parametrize("last_iteration", [1, 5])
    def test_perform_optimization__without_logger(self, best_solution, start_time, end_time, last_iteration):
        """
        Test 'perform_optimization' execution without logger.

        :param best_solution: Value simulated as best solution found by the algorithm.
        :param start_time: Value simulated as start time of the optimization process.
        :param end_time: Value simulated as end time of the optimization process.
        :param last_iteration: Value determining how many simulated iteration to be performed.
        """
        self.mock_algorithm_object.logger = None
        self.mock_datetime_now.side_effect = [start_time, end_time]
        self.mock_algorithm_object_is_stop_achieved.side_effect = [False] * last_iteration + [True]
        self.mock_algorithm_object._best_solution = best_solution
        assert AbstractOptimizationAlgorithm.perform_optimization(self=self.mock_algorithm_object) == best_solution
        assert self.mock_algorithm_object._start_time == start_time
        assert self.mock_algorithm_object._end_time == end_time
        self.mock_datetime_now.assert_called()
        self.mock_algorithm_object_perform_iteration.assert_has_calls([call(iteration_index=i)
                                                                       for i in range(last_iteration+1)])

    @pytest.mark.parametrize("problem", [987, "some problem"])
    @pytest.mark.parametrize("best_solution", [1, "some solution"])
    @pytest.mark.parametrize("start_time", [10, 9])
    @pytest.mark.parametrize("end_time", [5.5, 6.6])
    @pytest.mark.parametrize("last_iteration", [1, 5])
    def test_perform_optimization__with_logger(self, best_solution, start_time, end_time, last_iteration, problem):
        """
        Test 'perform_optimization' execution with logger.

        :param best_solution: Value simulated as best solution found by the algorithm.
        :param start_time: Value simulated as start time of the optimization process.
        :param end_time: Value simulated as end time of the optimization process.
        :param last_iteration: Value determining how many simulated iteration to be performed.
        """
        mock_logger = Mock()
        self.mock_algorithm_object.logger = mock_logger
        self.mock_datetime_now.side_effect = [start_time, end_time]
        self.mock_algorithm_object_is_stop_achieved.side_effect = [False] * last_iteration + [True]
        self.mock_algorithm_object._best_solution = best_solution
        self.mock_algorithm_object.problem = problem
        assert AbstractOptimizationAlgorithm.perform_optimization(self=self.mock_algorithm_object) == best_solution
        assert self.mock_algorithm_object._start_time == start_time
        assert self.mock_algorithm_object._end_time == end_time
        self.mock_datetime_now.assert_called()
        self.mock_algorithm_object_perform_iteration.assert_has_calls([call(iteration_index=i)
                                                                       for i in range(last_iteration+1)])
        mock_logger.log_at_start.assert_called_once_with(algorithm=self.mock_algorithm_object,
                                                         stop_conditions=self.mock_algorithm_object.stop_conditions,
                                                         problem=problem)
        mock_logger.log_at_end.assert_called_once_with(best_solution=best_solution,
                                                       optimization_time=end_time-start_time)

    # get_log_data

    def test_get_log_data(self):
        """Test that 'get_log_data' return dictionary with certain keys."""
        log_data = AbstractOptimizationAlgorithm.get_log_data(self=self.mock_algorithm_object)
        assert isinstance(log_data, dict)
        assert "type" in log_data
