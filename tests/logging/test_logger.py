import pytest
from mock import Mock, patch, call

from optimization.logging.logger import Logger, LoggingVerbosity
from optimization.optimization_algorithms.algorithms import OptimizationAlgorithm
from optimization.optimization_problem.problem import AbstractSolution
from .conftest import COMPARISON_FUNCTIONS, EXAMPLE_VALUE_TYPES, \
    VALID_COMPARISON_DATA_SETS, INVALID_COMPARISON_DATA_SETS


class TestLoggingVerbosity:
    @pytest.mark.parametrize("logging_verbosity", LoggingVerbosity)
    @pytest.mark.parametrize("comparison_function", COMPARISON_FUNCTIONS)
    @pytest.mark.parametrize("example_value", EXAMPLE_VALUE_TYPES, indirect=True)
    def test_comparison_with_other_type_object(self, logging_verbosity, comparison_function, example_value):
        """
        Check that comparison of 'LoggingVerbosity' instance with object of other type will raise TypeError.

        :param logging_verbosity: Instance of 'LoggingVerbosity' enum class.
        :param comparison_function: Any of comparison functions ('<', '<=', '==', ...).
        :param example_value: An example value of python basic type.
        """
        with pytest.raises(TypeError):
            comparison_function(LoggingVerbosity.AlgorithmConfiguration, example_value)

    @pytest.mark.parametrize("logging_verbosity", LoggingVerbosity)
    @pytest.mark.parametrize("comparison_function, values_diff", VALID_COMPARISON_DATA_SETS)
    def test_comparison_valid(self, logging_verbosity, comparison_function, values_diff):
        """
        Positive tests for comparison of 'LoggingVerbosity' instance with other 'LoggingVerbosity' instance.

        :param logging_verbosity: Instance of 'LoggingVerbosity' enum class.
        :param comparison_function: Any of comparison functions ('<', '<=', '==', ...).
        :param values_diff: List of values with Difference between compared values.
        """
        assert all(comparison_function(logging_verbosity,
                                       Mock(spec=LoggingVerbosity, value=logging_verbosity.value + value_diff))
                   for value_diff in values_diff)
        assert all(comparison_function(Mock(spec=LoggingVerbosity, value=logging_verbosity.value - value_diff),
                                       logging_verbosity)
                   for value_diff in values_diff)

    @pytest.mark.parametrize("logging_verbosity", LoggingVerbosity)
    @pytest.mark.parametrize("comparison_function, values_diff", INVALID_COMPARISON_DATA_SETS)
    def test_comparison_invalid(self, logging_verbosity, comparison_function, values_diff):
        """
        Negative tests for comparison of 'LoggingVerbosity' instance with other 'LoggingVerbosity' instance.

        :param logging_verbosity: Instance of 'LoggingVerbosity' enum class.
        :param comparison_function: Any of comparison functions ('<', '<=', '==', ...).
        :param values_diff: List of values with Difference between compared values.
        """
        assert all(not comparison_function(logging_verbosity,
                                           Mock(spec=LoggingVerbosity, value=logging_verbosity.value + value_diff))
                   for value_diff in values_diff)
        assert all(not comparison_function(Mock(spec=LoggingVerbosity, value=logging_verbosity.value - value_diff),
                                           logging_verbosity)
                   for value_diff in values_diff)


class TestLoggerInit:
    @pytest.mark.parametrize("example_value", EXAMPLE_VALUE_TYPES, indirect=True)
    def test_init_invalid_logging_verbosity_type(self, example_value, random_text):
        """
        Check that 'Logger' class cannot be initialized with parameter 'logging_verbosity' of invalid type.

        :param example_value: Example value that is not 'LoggingVerbosity' type that would be used as invalid value for
            'logging_verbosity' parameter.
        :param random_text: Random 'str' type value for 'logs_location' paramter.
        """
        with pytest.raises(TypeError):
            Logger(logs_location=random_text, logging_verbosity=example_value)

    @pytest.mark.parametrize("example_value", EXAMPLE_VALUE_TYPES.difference({str}), indirect=True)
    @pytest.mark.parametrize("logging_verbosity", LoggingVerbosity)
    def test_init_invalid_logs_location_type(self, logging_verbosity, example_value):
        """
        Check that 'Logger' class cannot be initialized with parameter 'logs_location' of invalid type.

        :param logging_verbosity: Instance of 'LoggingVerbosity' enum class.
        :param example_value: Example value that is not 'str' type that would be used as invalid value for
            'logs_location' parameter.
        """
        with pytest.raises(TypeError):
            Logger(logs_location=example_value, logging_verbosity=logging_verbosity)

    @patch("optimization.logging.logger.path")
    @pytest.mark.parametrize("logging_verbosity", LoggingVerbosity)
    def test_init_invalid_logs_location_value(self, mock_path, logging_verbosity, random_text):
        """
        Check that 'Logger' class cannot be initialized with parameter 'logs_location' of invalid value.

        :param logging_verbosity: Instance of 'LoggingVerbosity' enum class.
        :param random_text: Random 'str' type value for 'logs_location' parameter.
        """
        mock_path.isdir = Mock(return_value=False)
        with pytest.raises(ValueError):
            Logger(logs_location=random_text, logging_verbosity=logging_verbosity)

    @patch("optimization.logging.logger.path")
    @patch("optimization.logging.logger.mkdir")
    @pytest.mark.parametrize("logging_verbosity", LoggingVerbosity)
    def test_init_valid(self, mock_make_directory, mock_path, random_text, logging_verbosity):
        mock_path.isdir = Mock(return_value=True)
        logger = Logger(logs_location=random_text, logging_verbosity=logging_verbosity)
        assert logger.logging_verbosity == logging_verbosity
        mock_path.isdir.assert_called_once_with(random_text)
        mock_make_directory.assert_called_once_with(logger.logs_location)


class TestLoggerMethods:
    def setup(self):
        logging_verbosity_mock = Mock(spec=LoggingVerbosity, value=0)
        logger_instance_mock = Mock(spec=Logger, logging_verbosity=logging_verbosity_mock,
                                    logs_location="some_log_location")
        self.logger_instance_mock = logger_instance_mock
        self.logging_verbosity_mock = logging_verbosity_mock

    # log_at_start

    @pytest.mark.parametrize("example_value", EXAMPLE_VALUE_TYPES, indirect=True)
    def test_log_at_start_invalid_optimization_algorithm(self, example_value):
        """
        Check that TypeError is raised when 'log_at_start' method of 'Logger' class is called with parameter
        that is not instance of 'OptimizationAlgorithm'.

        :param example_value: Example invalid value for 'optimization_algorithm' parameter that is not instance
            of 'OptimizationAlgorithm' class.
        """
        with pytest.raises(TypeError):
            Logger.log_at_start(self=self.logger_instance_mock, optimization_algorithm=example_value)

    @patch("optimization.logging.logger.open")
    @patch("optimization.logging.logger.yaml_dump")
    @pytest.mark.parametrize("verbosity_value", ["just_below_lower_value", "below_lower_value"])
    def test_log_at_start_valid_call_with_no_action(self, mock_yaml_dump, mock_open, verbosity_value):
        """
        Check that no action is performed during execution of 'log_at_start' method if verbosity level is below
        'LoggingVerbosity.ProblemDefinition' and 'LoggingVerbosity.AlgorithmConfiguration' values.

        :param mock_yaml_dump: Mock of 'yaml_dump' function used within 'log_at_start' method.
        :param mock_open: Mock of 'open' function used within 'log_at_start' method.
        :param verbosity_value: Type of 'verbosity_level' attribute value of the 'Logger' class instance.
        """
        mock_optimization_algorithm = Mock(spec=OptimizationAlgorithm)
        if verbosity_value == "just_below_lower_value":
            self.logging_verbosity_mock.value = min(LoggingVerbosity.ProblemDefinition.value,
                                                    LoggingVerbosity.AlgorithmConfiguration.value) - 0.1
        elif verbosity_value == "below_lower_value":
            self.logging_verbosity_mock.value = min(LoggingVerbosity.ProblemDefinition.value,
                                                    LoggingVerbosity.AlgorithmConfiguration.value) - 100
        else:
            raise ValueError
        Logger.log_at_start(self=self.logger_instance_mock, optimization_algorithm=mock_optimization_algorithm)
        mock_yaml_dump.assert_not_called()
        mock_open.assert_not_called()

    @patch("optimization.logging.logger.path")
    @patch("optimization.logging.logger.open")
    @patch("optimization.logging.logger.yaml_dump")
    @pytest.mark.parametrize("verbosity_value", ["lower_value", "just_above_lower_value", "middle",
                                                 "just_below_higher_value"])
    @pytest.mark.parametrize("example_value", EXAMPLE_VALUE_TYPES, indirect=True)
    def test_log_at_start_valid_call_with_one_action(self, mock_yaml_dump, mock_open, mock_path, verbosity_value,
                                                     example_value, random_text):
        """
        Check that one action (logging to one file) is performed during execution of 'log_at_start' method when
        verbosity level equal or higher than lower value of verbosity level ('LoggingVerbosity.ProblemDefinition' or
        'LoggingVerbosity.AlgorithmConfiguration') and lower than higher of these values.

        :param mock_yaml_dump: Mock of 'yaml_dump' function used within 'log_at_start' method.
        :param mock_open: Mock of 'open' function used within 'log_at_start' method.
        :param mock_path: Mock of 'path' class used within 'log_at_start' method.
        :param verbosity_value: Type of 'verbosity_level' attribute value of the 'Logger' class instance.
        :param example_value: Example value to be returned by 'get_data_for_logging' method.
        :param random_text: Random text value to be returned by 'path.join' function.
        """
        mock_path.join = Mock(return_value=random_text)
        mock_get_data_for_logging = Mock(return_value=example_value)
        mock_optimization_problem = Mock(get_data_for_logging=mock_get_data_for_logging)
        mock_optimization_algorithm = Mock(spec=OptimizationAlgorithm, optimization_problem=mock_optimization_problem)
        if verbosity_value == "lower_value":
            self.logging_verbosity_mock.value = min(LoggingVerbosity.ProblemDefinition.value,
                                                    LoggingVerbosity.AlgorithmConfiguration.value)
        elif verbosity_value == "just_above_lower_value":
            self.logging_verbosity_mock.value = min(LoggingVerbosity.ProblemDefinition.value,
                                                    LoggingVerbosity.AlgorithmConfiguration.value) + 0.1
        elif verbosity_value == "middle":
            self.logging_verbosity_mock.value = (LoggingVerbosity.ProblemDefinition.value +
                                                 LoggingVerbosity.AlgorithmConfiguration.value) / 2
        elif verbosity_value == "just_below_higher_value":
            self.logging_verbosity_mock.value = max(LoggingVerbosity.ProblemDefinition.value,
                                                    LoggingVerbosity.AlgorithmConfiguration.value) - 0.1
        else:
            raise ValueError
        Logger.log_at_start(self=self.logger_instance_mock, optimization_algorithm=mock_optimization_algorithm)
        mock_open.assert_called_once_with(file=random_text, mode="w")
        mock_yaml_dump.assert_called_once_with(data=example_value, stream=mock_open().__enter__())

    @patch("optimization.logging.logger.path")
    @patch("optimization.logging.logger.open")
    @patch("optimization.logging.logger.yaml_dump")
    @pytest.mark.parametrize("verbosity_value", ["higher_value", "above_higher_value"])
    @pytest.mark.parametrize("example_value", EXAMPLE_VALUE_TYPES, indirect=True)
    def test_log_at_start_valid_call_with_two_actions(self, mock_yaml_dump, mock_open, mock_path, verbosity_value,
                                                      example_value, random_text):
        """
        Check that two actions (logging to two files) are performed during execution of 'log_at_start' method when
        verbosity level equal or greater than higher value of verbosity level ('LoggingVerbosity.ProblemDefinition' or
        'LoggingVerbosity.AlgorithmConfiguration').

        :param mock_yaml_dump: Mock of 'yaml_dump' function used within 'log_at_start' method.
        :param mock_open: Mock of 'open' function used within 'log_at_start' method.
        :param mock_path: Mock of 'path' class used within 'log_at_start' method.
        :param verbosity_value: Type of 'verbosity_level' attribute value of the 'Logger' class instance.
        :param example_value: Example value to be returned by 'get_data_for_logging' method.
        :param random_text: Random text value to be returned by 'path.join' function.
        """
        mock_path.join = Mock(return_value=random_text)
        mock_get_data_for_logging = Mock(return_value=example_value)
        mock_optimization_problem = Mock(get_data_for_logging=mock_get_data_for_logging)
        mock_optimization_algorithm = Mock(spec=OptimizationAlgorithm, optimization_problem=mock_optimization_problem)
        if verbosity_value == "higher_value":
            self.logging_verbosity_mock.value = max(LoggingVerbosity.ProblemDefinition.value,
                                                    LoggingVerbosity.AlgorithmConfiguration.value)
        elif verbosity_value == "above_higher_value":
            self.logging_verbosity_mock.value = max(LoggingVerbosity.ProblemDefinition.value,
                                                    LoggingVerbosity.AlgorithmConfiguration.value) + 100
        else:
            raise ValueError
        Logger.log_at_start(self=self.logger_instance_mock, optimization_algorithm=mock_optimization_algorithm)
        mock_open.assert_has_calls([call(file=random_text, mode="w")])
        mock_yaml_dump.assert_has_calls([call(data=example_value, stream=mock_open().__enter__())])

    # log_found_solutions

    @pytest.mark.parametrize("example_value", EXAMPLE_VALUE_TYPES.difference({int}), indirect=True)
    def test_log_found_solutions_invalid_iteration(self, example_value):
        """
        Check that TypeError is raised when 'log_found_solutions' method of 'Logger' class is called with parameter
        'iteration' that is not int type.

        :param example_value: Example invalid value for 'iteration' parameter that is not int type.
        """
        with pytest.raises(TypeError):
            Logger.log_found_solutions(self=self.logger_instance_mock, iteration=example_value, solutions=[])

    @patch("optimization.logging.logger.open")
    @patch("optimization.logging.logger.yaml_dump")
    @pytest.mark.parametrize("verbosity_value", ["just_below", "below"])
    def test_log_found_solutions_valid_call_with_no_action(self, mock_yaml_dump, mock_open, verbosity_value,
                                                           random_int):
        """
        Check that no action is performed during execution of 'log_found_solutions' method if verbosity level is below
        'LoggingVerbosity.AllSolutions' value.

        :param mock_yaml_dump: Mock of 'yaml_dump' function used within 'log_found_solutions' method.
        :param mock_open: Mock of 'open' function used within 'log_found_solutions' method.
        :param verbosity_value: Type of 'verbosity_level' attribute value of the 'Logger' class instance.
        :param random_int: Valid value for 'iteration' parameter of 'log_found_solutions' method.
        """
        if verbosity_value == "just_below":
            self.logging_verbosity_mock.value = LoggingVerbosity.AllSolutions.value - 0.1
        elif verbosity_value == "below":
            self.logging_verbosity_mock.value = LoggingVerbosity.AllSolutions.value - 100
        else:
            raise ValueError
        Logger.log_found_solutions(self=self.logger_instance_mock, iteration=random_int, solutions=[])
        mock_yaml_dump.assert_not_called()
        mock_open.assert_not_called()

    @patch("optimization.logging.logger.path")
    @patch("optimization.logging.logger.open")
    @patch("optimization.logging.logger.yaml_dump")
    @pytest.mark.parametrize("verbosity_value", ["equal", "just_above", "above"])
    @pytest.mark.parametrize("some_values", [range(10), set("abcdef"), ["some", 1]])
    def test_log_found_solutions_valid_call_with_iteration_zero(self, mock_yaml_dump, mock_open, mock_path,
                                                                verbosity_value, some_values, random_text):
        """
        Check that action is performed during execution of 'log_found_solutions' method if verbosity level is
        greater or equal 'LoggingVerbosity.AllSolutions' value.
        Parameter 'iteration' is  equal 0 (file is created).

        :param mock_yaml_dump: Mock of 'yaml_dump' function used within 'log_found_solutions' method.
        :param mock_open: Mock of 'open' function used within 'log_found_solutions' method.
        :param mock_path: Mock of 'path' class used within 'log_found_solutions' method.
        :param verbosity_value: Type of 'verbosity_level' attribute value of the 'Logger' class instance.
        :param some_values: Values to be used in mocking solution.
        :param random_text: Random text value to be returned by 'path.join' function.
        """
        mock_path.join = Mock(return_value=random_text)
        mocked_solutions = [Mock(spec=AbstractSolution, get_data_for_logging=Mock(return_value=value)) for value in some_values]
        if verbosity_value == "equal":
            self.logging_verbosity_mock.value = LoggingVerbosity.AllSolutions.value
        elif verbosity_value == "just_above":
            self.logging_verbosity_mock.value = LoggingVerbosity.AllSolutions.value + 0.1
        elif verbosity_value == "above":
            self.logging_verbosity_mock.value = LoggingVerbosity.AllSolutions.value + 100
        else:
            raise ValueError
        Logger.log_found_solutions(self=self.logger_instance_mock, iteration=0, solutions=mocked_solutions)
        mock_open.assert_called_once_with(file=random_text, mode="w")
        mock_yaml_dump.assert_called_once_with(data={"Iteration 0": list(some_values)}, stream=mock_open().__enter__())

    @patch("optimization.logging.logger.path")
    @patch("optimization.logging.logger.open")
    @patch("optimization.logging.logger.yaml_dump")
    @pytest.mark.parametrize("verbosity_value", ["equal", "just_above", "above"])
    @pytest.mark.parametrize("some_values", [range(10), set("abcdef"), ["some", 1]])
    @pytest.mark.parametrize("iteration", [1, 100, 666, 987654321])
    def test_log_found_solutions_valid_call_with_next_iterations(self, mock_yaml_dump, mock_open, mock_path,
                                                                 verbosity_value, some_values, iteration, random_text):
        """
        Check that action is performed during execution of 'log_found_solutions' method if verbosity level is
        greater or equal 'LoggingVerbosity.AllSolutions' value.
        Parameter 'iteration' is  greater than 0 (file is appended).

        :param mock_yaml_dump: Mock of 'yaml_dump' function used within 'log_found_solutions' method.
        :param mock_open: Mock of 'open' function used within 'log_found_solutions' method.
        :param mock_path: Mock of 'path' class used within 'log_found_solutions' method.
        :param verbosity_value: Type of 'verbosity_level' attribute value of the 'Logger' class instance.
        :param some_values: Values to be used in mocking solution.
        :param iteration: Value for 'iteration' parameter of 'log_found_solutions method.
        :param random_text: Random text value to be returned by 'path.join' function.
        """
        mock_path.join = Mock(return_value=random_text)
        mocked_solutions = [Mock(spec=AbstractSolution, get_data_for_logging=Mock(return_value=value)) for value in some_values]
        if verbosity_value == "equal":
            self.logging_verbosity_mock.value = LoggingVerbosity.AllSolutions.value
        elif verbosity_value == "just_above":
            self.logging_verbosity_mock.value = LoggingVerbosity.AllSolutions.value + 0.1
        elif verbosity_value == "above":
            self.logging_verbosity_mock.value = LoggingVerbosity.AllSolutions.value + 100
        else:
            raise ValueError
        Logger.log_found_solutions(self=self.logger_instance_mock, iteration=iteration, solutions=mocked_solutions)
        mock_open.assert_called_once_with(file=random_text, mode="a")
        mock_yaml_dump.assert_called_once_with(data={f"Iteration {iteration}": list(some_values)},
                                               stream=mock_open().__enter__())

    # log_at_end

    @pytest.mark.parametrize("example_value", EXAMPLE_VALUE_TYPES.difference({int}), indirect=True)
    def test_log_at_end_invalid_best_solution(self, example_value):
        """
        Check that TypeError is raised when 'log_at_end' method of 'Logger' class is called with parameter
        'best_solution' that is not instance of 'AbstractSolution' class.

        :param example_value: Example invalid value for 'best_solution' parameter that is not instance of
            'AbstractSolution' class.
        """
        with pytest.raises(TypeError):
            Logger.log_at_end(self=self.logger_instance_mock, best_solution=example_value)

    @patch("optimization.logging.logger.open")
    @patch("optimization.logging.logger.yaml_dump")
    @pytest.mark.parametrize("verbosity_value", ["just_below", "below"])
    def test_log_at_end_valid_call_with_no_action(self, mock_yaml_dump, mock_open, verbosity_value):
        """
        Check that no action is performed during execution of 'log_at_end' method if verbosity level is below
        'LoggingVerbosity.BestSolution' value.

        :param mock_yaml_dump: Mock of 'yaml_dump' function used within 'log_at_end' method.
        :param mock_open: Mock of 'open' function used within 'log_at_end' method.
        :param verbosity_value: Type of 'verbosity_level' attribute value of the 'Logger' class instance.
        """
        mock_best_solution = Mock(spec=AbstractSolution)
        if verbosity_value == "just_below":
            self.logging_verbosity_mock.value = LoggingVerbosity.BestSolution.value - 0.1
        elif verbosity_value == "below":
            self.logging_verbosity_mock.value = LoggingVerbosity.BestSolution.value - 100
        else:
            raise ValueError
        Logger.log_at_end(self=self.logger_instance_mock, best_solution=mock_best_solution)
        mock_yaml_dump.assert_not_called()
        mock_open.assert_not_called()

    @patch("optimization.logging.logger.path")
    @patch("optimization.logging.logger.open")
    @patch("optimization.logging.logger.yaml_dump")
    @pytest.mark.parametrize("verbosity_value", ["equal", "just_above", "above"])
    @pytest.mark.parametrize("example_value", EXAMPLE_VALUE_TYPES, indirect=True)
    def test_log_at_end_valid_call_with_action(self, mock_yaml_dump, mock_open, mock_path, verbosity_value,
                                               example_value, random_text):
        """
        Check that action is performed during execution of 'log_at_end' method if verbosity level is greater or equal
        'LoggingVerbosity.BestSolution' value.

        :param mock_yaml_dump: Mock of 'yaml_dump' function used within 'log_at_end' method.
        :param mock_open: Mock of 'open' function used within 'log_at_end' method.
        :param mock_path: Mock of 'path' class used within 'log_at_end' method.
        :param verbosity_value: Type of 'verbosity_level' attribute value of the 'Logger' class instance.
        :param example_value: Example value to be used by best solution mock.
        :param random_text: Random text value to be returned by 'path.join' function.
        """
        mock_best_solution = Mock(spec=AbstractSolution, get_data_for_logging=Mock(return_value=example_value))
        mock_path.join = Mock(return_value=random_text)
        if verbosity_value == "equal":
            self.logging_verbosity_mock.value = LoggingVerbosity.BestSolution.value
        elif verbosity_value == "just_above":
            self.logging_verbosity_mock.value = LoggingVerbosity.BestSolution.value + 0.1
        elif verbosity_value == "above":
            self.logging_verbosity_mock.value = LoggingVerbosity.BestSolution.value + 100
        else:
            raise ValueError
        Logger.log_at_end(self=self.logger_instance_mock, best_solution=mock_best_solution)
        mock_open.assert_called_once_with(file=random_text, mode="w")
        mock_yaml_dump.assert_called_once_with(data=example_value, stream=mock_open().__enter__())
