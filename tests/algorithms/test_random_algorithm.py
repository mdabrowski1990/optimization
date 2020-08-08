import pytest
from mock import Mock, patch

from optimization.algorithms.random_algorithm import RandomAlgorithm


class TestRandomAlgorithm:
    """Tests for 'RandomAlgorithm' class and their methods."""

    SCRIPT_LOCATION = "optimization.algorithms.random_algorithm"

    def setup(self):
        self.mock_solution_class = Mock()
        self.mock_logger = Mock()
        self.mock_random_algorithm_object = Mock(spec=RandomAlgorithm, SolutionClass=self.mock_solution_class,
                                                 logger=self.mock_logger)
        # patching
        self._patcher_abstract_algorithm_init = patch(f"{self.SCRIPT_LOCATION}.AbstractOptimizationAlgorithm.__init__")
        self.mock_abstract_algorithm_class_init = self._patcher_abstract_algorithm_init.start()
        self._patcher_abstract_algorithm_get_log_data = patch(f"{self.SCRIPT_LOCATION}.AbstractOptimizationAlgorithm.get_log_data")
        self.mock_abstract_algorithm_class_get_log_data = self._patcher_abstract_algorithm_get_log_data.start()

    def teardown(self):
        self._patcher_abstract_algorithm_init.stop()
        self._patcher_abstract_algorithm_get_log_data.stop()

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

    @pytest.mark.parametrize("iteration", [0, 1, 45])
    @pytest.mark.parametrize("best_solution", [None, -100, 100])
    @pytest.mark.parametrize("population_size", [1, 5])
    def test_perform_iteration__without_logger(self, iteration, best_solution, population_size):
        solutions = list(range(population_size))
        self.mock_solution_class.side_effect = solutions
        self.mock_random_algorithm_object.logger = None
        self.mock_random_algorithm_object.population_size = population_size
        self.mock_random_algorithm_object._best_solution = best_solution
        RandomAlgorithm._perform_iteration(self=self.mock_random_algorithm_object, iteration_index=iteration)
        if best_solution is None:
            assert self.mock_random_algorithm_object._best_solution == solutions[-1]
        else:
            assert self.mock_random_algorithm_object._best_solution == max(solutions[-1], best_solution)

    @pytest.mark.parametrize("iteration", [0, 1, 45])
    @pytest.mark.parametrize("best_solution", [None, -100, 100])
    @pytest.mark.parametrize("population_size", [1, 5])
    def test_perform_iteration__with_logger(self, iteration, best_solution, population_size):
        solutions = list(range(population_size+10, 10, -1))
        self.mock_solution_class.side_effect = solutions
        self.mock_random_algorithm_object.logger = self.mock_logger
        self.mock_random_algorithm_object.population_size = population_size
        self.mock_random_algorithm_object._best_solution = best_solution
        RandomAlgorithm._perform_iteration(self=self.mock_random_algorithm_object, iteration_index=iteration)
        if best_solution is None:
            assert self.mock_random_algorithm_object._best_solution == solutions[0]
        else:
            assert self.mock_random_algorithm_object._best_solution == max(solutions[0], best_solution)
        self.mock_logger.log_iteration.assert_called_once_with(iteration=iteration, solutions=solutions)

    # get_log_data

    @pytest.mark.parametrize("population_size", [1, 5])
    @pytest.mark.parametrize("parent_data", [{"x": "x", "1234": None}, {"a": 1, "b": 2}])
    def test_get_log_data(self, population_size, parent_data):
        self.mock_random_algorithm_object.population_size = population_size
        self.mock_abstract_algorithm_class_get_log_data.return_value = parent_data
        log_data = RandomAlgorithm.get_log_data(self=self.mock_random_algorithm_object)
        assert isinstance(log_data, dict)
        assert log_data["population_size"] == population_size
        assert all([log_data[key] == value for key, value in parent_data.items()])
        self.mock_abstract_algorithm_class_get_log_data.assert_called_once()

