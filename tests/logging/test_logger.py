import pytest
from mock import Mock, patch
from operator import eq, ne, lt, le, ge, gt

from optimization.logging.logger import LoggingVerbosity, Logger, LoggingFormat


class TestLoggingVerbosity:
    """Tests for 'LoggingVerbosity' Enum."""

    @pytest.mark.parametrize("value1, value2, expected_result", [
        (LoggingVerbosity.BestSolution, LoggingVerbosity.BestSolution, True),
        (LoggingVerbosity.AllSolutions, LoggingVerbosity.AllSolutions, True),
        (LoggingVerbosity.AllSolutions, LoggingVerbosity.BestSolution, False),
        (LoggingVerbosity.AlgorithmConfiguration, LoggingVerbosity.ProblemDefinition, False),
        (LoggingVerbosity.ProblemDefinition, LoggingVerbosity.AlgorithmConfiguration, False),
    ])
    def test_eq(self, value1, value2, expected_result):
        assert eq(value1, value2) is expected_result

    @pytest.mark.parametrize("value1, value2, expected_result", [
        (LoggingVerbosity.BestSolution, LoggingVerbosity.BestSolution, False),
        (LoggingVerbosity.AllSolutions, LoggingVerbosity.AllSolutions, False),
        (LoggingVerbosity.AllSolutions, LoggingVerbosity.BestSolution, True),
        (LoggingVerbosity.AlgorithmConfiguration, LoggingVerbosity.ProblemDefinition, True),
        (LoggingVerbosity.ProblemDefinition, LoggingVerbosity.AlgorithmConfiguration, True),
    ])
    def test_ne(self, value1, value2, expected_result):
        assert ne(value1, value2) is expected_result

    @pytest.mark.parametrize("value1, value2, expected_result", [
        (LoggingVerbosity.BestSolution, LoggingVerbosity.AllSolutions, True),
        (LoggingVerbosity.AlgorithmConfiguration, LoggingVerbosity.AllSolutions, True),
        (LoggingVerbosity.ProblemDefinition, LoggingVerbosity.AlgorithmConfiguration, True),
        (LoggingVerbosity.AlgorithmConfiguration, LoggingVerbosity.AlgorithmConfiguration, False),
        (LoggingVerbosity.ProblemDefinition, LoggingVerbosity.BestSolution, False),
        (LoggingVerbosity.AllSolutions, LoggingVerbosity.AlgorithmConfiguration, False),
    ])
    def test_lt(self, value1, value2, expected_result):
        assert lt(value1, value2) is expected_result

    @pytest.mark.parametrize("value1, value2, expected_result", [
        (LoggingVerbosity.BestSolution, LoggingVerbosity.AllSolutions, True),
        (LoggingVerbosity.AlgorithmConfiguration, LoggingVerbosity.AllSolutions, True),
        (LoggingVerbosity.ProblemDefinition, LoggingVerbosity.AlgorithmConfiguration, True),
        (LoggingVerbosity.AlgorithmConfiguration, LoggingVerbosity.AlgorithmConfiguration, True),
        (LoggingVerbosity.ProblemDefinition, LoggingVerbosity.BestSolution, False),
        (LoggingVerbosity.AllSolutions, LoggingVerbosity.AlgorithmConfiguration, False),
    ])
    def test_le(self, value1, value2, expected_result):
        assert le(value1, value2) is expected_result

    @pytest.mark.parametrize("value1, value2, expected_result", [
        (LoggingVerbosity.BestSolution, LoggingVerbosity.AllSolutions, False),
        (LoggingVerbosity.AlgorithmConfiguration, LoggingVerbosity.AllSolutions, False),
        (LoggingVerbosity.ProblemDefinition, LoggingVerbosity.AlgorithmConfiguration, False),
        (LoggingVerbosity.AlgorithmConfiguration, LoggingVerbosity.AlgorithmConfiguration, True),
        (LoggingVerbosity.ProblemDefinition, LoggingVerbosity.BestSolution, True),
        (LoggingVerbosity.AllSolutions, LoggingVerbosity.AlgorithmConfiguration, True),
    ])
    def test_ge(self, value1, value2, expected_result):
        assert ge(value1, value2) is expected_result

    @pytest.mark.parametrize("value1, value2, expected_result", [
        (LoggingVerbosity.BestSolution, LoggingVerbosity.AllSolutions, False),
        (LoggingVerbosity.AlgorithmConfiguration, LoggingVerbosity.AllSolutions, False),
        (LoggingVerbosity.ProblemDefinition, LoggingVerbosity.AlgorithmConfiguration, False),
        (LoggingVerbosity.AlgorithmConfiguration, LoggingVerbosity.AlgorithmConfiguration, False),
        (LoggingVerbosity.ProblemDefinition, LoggingVerbosity.BestSolution, True),
        (LoggingVerbosity.AllSolutions, LoggingVerbosity.AlgorithmConfiguration, True),
    ])
    def test_gt(self, value1, value2, expected_result):
        assert gt(value1, value2) is expected_result

    @pytest.mark.parametrize("logging_verbosity", [LoggingVerbosity.ProblemDefinition, LoggingVerbosity.AllSolutions])
    @pytest.mark.parametrize("other", [1, "BestSolution", 3.432, None])
    @pytest.mark.parametrize("operation", [eq, ne, lt, le, ge, gt])
    def test_wrong_type(self, logging_verbosity, other, operation):
        with pytest.raises(TypeError):
            operation(logging_verbosity, other)


class TestLogger:
    """Tests for 'Logger' class and their methods."""

    SCRIPT_LOCATION = "optimization.logging.logger"
    EXAMPLE_LOGS_DIR = ["C:\\logs\\", "path\\to\\some dir"]

    def setup(self):
        self.mock_logger_object = Mock(spec=Logger)
        self.mock_path_isdir = Mock()
        # patching
        self._patcher_path = patch(f"{self.SCRIPT_LOCATION}.path", Mock(isdir=self.mock_path_isdir))
        self.mock_path = self._patcher_path.start()

    def teardown(self):
        self._patcher_path.stop()

    # __init__

    @pytest.mark.parametrize("logs_dir", EXAMPLE_LOGS_DIR)
    def test_init__valid_only_dir(self, logs_dir):
        """
        Tests that 'Logger' class can be initialized with only a proper value of 'logs_dir' param.

        :param logs_dir: Example value of 'logs_dir' parameter.
        """
        self.mock_path_isdir.return_value = True
        Logger.__init__(self=self.mock_logger_object, logs_dir=logs_dir)
        assert self.mock_logger_object.main_dir == logs_dir
        assert self.mock_logger_object.verbosity == LoggingVerbosity.BestSolution
        assert self.mock_logger_object.log_format == LoggingFormat.YAML
        assert self.mock_logger_object.optimization_process_dir is None

    @pytest.mark.parametrize("logs_dir", EXAMPLE_LOGS_DIR)
    @pytest.mark.parametrize("verbosity", list(LoggingVerbosity))
    @pytest.mark.parametrize("log_format", list(LoggingFormat))
    def test_init__valid_all_params(self, logs_dir, verbosity, log_format):
        """
        Tests that 'Logger' class can be initialized with only a proper value of 'logs_dir' param.

        :param logs_dir: Example value of 'logs_dir' parameter.
        """
        self.mock_path_isdir.return_value = True
        Logger.__init__(self=self.mock_logger_object, logs_dir=logs_dir, verbosity=verbosity, log_format=log_format)
        assert self.mock_logger_object.main_dir == logs_dir
        assert self.mock_logger_object.verbosity == verbosity
        assert self.mock_logger_object.log_format == log_format
        assert self.mock_logger_object.optimization_process_dir is None

    @pytest.mark.parametrize("invalid_logs_dir", [1, None, 43., []])
    def test_init__invalid_logs_dir_type(self, invalid_logs_dir):
        """
        Tests that during init of 'Logger' class TypeError will be raise if 'logs_dir' parameter is not str type.

        :param invalid_logs_dir: Value that is not str type.
        """
        with pytest.raises(TypeError):
            Logger.__init__(self=self.mock_logger_object, logs_dir=invalid_logs_dir)

    @pytest.mark.parametrize("logs_dir", EXAMPLE_LOGS_DIR)
    @pytest.mark.parametrize("invalid_verbosity", [1, None, 43., [], "verbosity"])
    def test_init__invalid_verbosity_type(self, logs_dir, invalid_verbosity):
        """
        Tests that during init of 'Logger' class TypeError will be raise if 'verbosity' parameter is not
        LoggingVerbosity type.

        :param logs_dir: Example value of 'logs_dir' parameter.
        :param invalid_verbosity:  Value that is not LoggingVerbosity type.
        """
        with pytest.raises(TypeError):
            Logger.__init__(self=self.mock_logger_object, logs_dir=logs_dir, verbosity=invalid_verbosity)

    @pytest.mark.parametrize("logs_dir", EXAMPLE_LOGS_DIR)
    @pytest.mark.parametrize("invalid_format", [1, None, 43., [], "format"])
    def test_init__invalid_format_type(self, logs_dir, invalid_format):
        """
        Tests that during init of 'Logger' class TypeError will be raise if 'verbosity' parameter is not
        LoggingVerbosity type.

        :param logs_dir: Example value of 'logs_dir' parameter.
        :param invalid_format:  Value that is not LoggingFormat type.
        """
        with pytest.raises(TypeError):
            Logger.__init__(self=self.mock_logger_object, logs_dir=logs_dir, log_format=invalid_format)

    @pytest.mark.parametrize("logs_dir", EXAMPLE_LOGS_DIR)
    def test_init__invalid_logs_dir_value(self, logs_dir):
        """
        Tests that during init of 'Logger' class ValueError will be raise if 'logs_dir' parameter contains string
        that is non-existing path or is not a directory path.

        :param logs_dir: Example value of 'logs_dir' parameter.
        """
        self.mock_path_isdir.return_value = False
        with pytest.raises(ValueError):
            Logger.__init__(self=self.mock_logger_object, logs_dir=logs_dir)


