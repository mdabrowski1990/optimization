import pytest
from mock import Mock, PropertyMock, patch

from optimization.logging.logger import Logger, LoggingVerbosity
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


class TestLogger:
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

    # todo: test methods: log_at_start, log_found_solutions, log_at_end

