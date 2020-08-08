import pytest
from mock import Mock, patch

from optimization.algorithms.random_algorithm import RandomAlgorithm


class TestRandomAlgorithm:
    """Tests for 'RandomAlgorithm' class and their methods."""

    SCRIPT_LOCATION = "optimization.algorithms.random_algorithm"

    def setup(self):
        self.mock_random_algorithm_object = Mock(spec=RandomAlgorithm)
        # patching
        self._patcher_abstract_algorithm_init = patch(f"{self.SCRIPT_LOCATION}.AbstractOptimizationAlgorithm.__init__")
        self.mock_abstract_algorithm_class_init = self._patcher_abstract_algorithm_init.start()

    def teardown(self):
        self._patcher_abstract_algorithm_init.stop()

    # __init__

    @pytest.mark.parametrize("population_size", [1, 15, 965478])
    def test_init__valid(self, population_size):
        """
        Test valid case of Random Algorithm class initialization.

        :param population_size: Example value of 'population_size'.
        """
        problem = Mock()
        stop_conditions = Mock()
        logger = Mock()
        RandomAlgorithm.__init__(self=self.mock_random_algorithm_object, stop_conditions=stop_conditions,
                                 problem=problem, logger=logger, population_size=population_size)
        assert self.mock_random_algorithm_object.population_size == population_size
        assert self.mock_random_algorithm_object.SolutionClass.optimization_problem == problem
        self.mock_abstract_algorithm_class_init.assert_called_once_with(problem=problem,
                                                                        stop_conditions=stop_conditions, logger=logger)

    @pytest.mark.parametrize("invalid_population_size", ["some population", None, 3.])
    def test_init__invalid_population_size_type(self, invalid_population_size):
        """
        Test initialization of Random Algorithm class when invalid type of 'population_size' is passed.

        :param invalid_population_size: Value that is not int type.
        """
        with pytest.raises(TypeError):
            RandomAlgorithm.__init__(self=self.mock_random_algorithm_object, stop_conditions=Mock(),
                                     problem=Mock(), logger=Mock(), population_size=invalid_population_size)

    @pytest.mark.parametrize("invalid_population_size", [0, -1, -543])
    def test_init__invalid_population_size_value(self, invalid_population_size):
        """
        Test initialization of Random Algorithm class when invalid value of 'population_size' is passed.

        :param invalid_population_size: Value that is int <= 0.
        """
        with pytest.raises(ValueError):
            RandomAlgorithm.__init__(self=self.mock_random_algorithm_object, stop_conditions=Mock(),
                                     problem=Mock(), logger=Mock(), population_size=invalid_population_size)

    # _perform_iteration

    # todo

    # get_log_data

    # todo
