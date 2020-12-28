import pytest
from mock import Mock, patch

from optimization.algorithms.evolutionary_algorithm.adaptive_evolutionary_algorithm import \
    AdaptiveEvolutionaryAlgorithm, AdaptationType, LowerAdaptiveEvolutionaryAlgorithm, \
    EvolutionaryAlgorithmAdaptationProblem, \
    SelectionType, CrossoverType, MutationType, \
    MIN_POPULATION_SIZE, MAX_POPULATION_SIZE, MIN_MUTATION_CHANCE, MAX_MUTATION_CHANCE


class TestEvolutionaryAlgorithmAdaptationProblem:

    SCRIPT_LOCATION = "optimization.algorithms.evolutionary_algorithm.adaptive_evolutionary_algorithm"

    def setup(self):
        self.mock_adaptation_problem_object = Mock(spec=EvolutionaryAlgorithmAdaptationProblem)
        # patching
        self._patcher_optimization_problem_init = patch(f"{self.SCRIPT_LOCATION}.OptimizationProblem.__init__")
        self.mock_optimization_problem_init = self._patcher_optimization_problem_init.start()

    def teardown(self):
        self._patcher_optimization_problem_init.stop()

    # __init__

    @pytest.mark.parametrize("adaptation_type, adaptation_params", [
        (AdaptationType.BestSolution, {}),
        (AdaptationType.BestSolutions, {"solutions_number": 5}),
        (AdaptationType.BestSolutionsPercentile, {"solutions_percentile": 0.1}),
    ])
    @pytest.mark.parametrize("selection_types, crossover_types, mutation_types", [
        ({SelectionType.Ranking}, {CrossoverType.SinglePoint}, {MutationType.SinglePoint}),
        (list(SelectionType), list(CrossoverType), list(MutationType)),
    ])
    def test_init__valid_input(self, adaptation_type, adaptation_params,
                               selection_types, crossover_types, mutation_types):
        EvolutionaryAlgorithmAdaptationProblem.__init__(self=self.mock_adaptation_problem_object,
                                                        adaptation_type=adaptation_type,
                                                        **adaptation_params,
                                                        selection_types=selection_types,
                                                        crossover_types=crossover_types,
                                                        mutation_types=mutation_types,
                                                        population_size_boundaries=(MIN_POPULATION_SIZE,
                                                                                    MAX_POPULATION_SIZE),
                                                        mutation_chance_boundaries=(MIN_MUTATION_CHANCE,
                                                                                    MAX_MUTATION_CHANCE))
        create_objective_params = {"percentile": adaptation_params.get("solutions_percentile", None),
                                   "number": adaptation_params.get("solutions_number", None)}
        self.mock_adaptation_problem_object._create_objective_function.assert_called_once_with(
            adaptation_type=adaptation_type, **create_objective_params)
        self.mock_optimization_problem_init.assert_called_once()
        assert isinstance(self.mock_adaptation_problem_object.additional_decision_variable, dict)

    @pytest.mark.parametrize("adaptation_type", [None, 1])
    def test_init__invalid_adaptation_type(self, adaptation_type, example_selection_types, example_crossover_types,
                                           example_mutation_types):
        with pytest.raises(TypeError):
            EvolutionaryAlgorithmAdaptationProblem.__init__(self=self.mock_adaptation_problem_object,
                                                            adaptation_type=adaptation_type,
                                                            selection_types=example_selection_types,
                                                            crossover_types=example_crossover_types,
                                                            mutation_types=example_mutation_types,
                                                            population_size_boundaries=(MIN_POPULATION_SIZE,
                                                                                        MAX_POPULATION_SIZE),
                                                            mutation_chance_boundaries=(MIN_MUTATION_CHANCE,
                                                                                        MAX_MUTATION_CHANCE))

    # todo: more invalid test cases - for all exceptions
    # def test_init__invalid_adaptation_type(self, example_adaptation_type, example_selection_types, example_crossover_types,
    #                                        example_mutation_types):
    #     with pytest.raises(TypeError):
    #         EvolutionaryAlgorithmAdaptationProblem.__init__(self=self.mock_adaptation_problem_object,
    #                                                         adaptation_type=example_adaptation_type,
    #                                                         selection_types=example_selection_types,
    #                                                         crossover_types=example_crossover_types,
    #                                                         mutation_types=example_mutation_types,
    #                                                         population_size_boundaries=(MIN_POPULATION_SIZE,
    #                                                                                     MAX_POPULATION_SIZE),
    #                                                         mutation_chance_boundaries=(MIN_MUTATION_CHANCE,
    #                                                                                     MAX_MUTATION_CHANCE))

class TestLowerAdaptiveEvolutionaryAlgorithm:

    SCRIPT_LOCATION = "optimization.algorithms.evolutionary_algorithm.adaptive_evolutionary_algorithm"

    def setup(self):
        self.mock_lower_evolutionary_algorithm_object = Mock(spec=LowerAdaptiveEvolutionaryAlgorithm)
        # patching
        self._patcher_evolutionary_algorithm_init = patch(f"{self.SCRIPT_LOCATION}.EvolutionaryAlgorithm.__init__")
        self.mock_evolutionary_algorithm_init = self._patcher_evolutionary_algorithm_init.start()

    def teardown(self):
        self._patcher_evolutionary_algorithm_init.stop()

    # __init__

    # TODO: update these
    # @pytest.mark.parametrize("upper_iteration", [5, 569])
    # @pytest.mark.parametrize("index", [0, 13])
    # @pytest.mark.parametrize("params", [{}, {"a": 1, "b": "abc", "something_other": "dunno"}])
    # def test_init__without_initial_population(self, upper_iteration, index, params):
    #     """
    #     Test '__init__' without passing 'initial_population' parameter.
    #
    #     :param upper_iteration: Example value of 'upper_iteration' parameter.
    #     :param index: Example value of 'index' attribute.
    #     :param params: Some additional parameters.
    #     """
    #     LowerAdaptiveEvolutionaryAlgorithm.__init__(self=self.mock_lower_evolutionary_algorithm_object,
    #                                                 upper_iteration=upper_iteration, index=index, **params)
    #     assert self.mock_lower_evolutionary_algorithm_object.upper_iteration == upper_iteration
    #     assert self.mock_lower_evolutionary_algorithm_object.index == index
    #     assert self.mock_lower_evolutionary_algorithm_object._population == []
    #     self.mock_evolutionary_algorithm_init.assert_called_once_with(**params)
    #
    # @pytest.mark.parametrize("upper_iteration", [5, 569])
    # @pytest.mark.parametrize("index", [0, 13])
    # @pytest.mark.parametrize("initial_population", ["some population", "some other population"])
    # @pytest.mark.parametrize("params", [{}, {"a": 1, "b": "abc", "something_other": "dunno"}])
    # def test_init__with_initial_population(self, upper_iteration, index, initial_population, params):
    #     """
    #     Test '__init__' without passing 'initial_population' parameter.
    #
    #     :param upper_iteration: Example value of 'upper_iteration' parameter.
    #     :param index: Example value of 'index' attribute.
    #     :param params: Some additional parameters.
    #     """
    #     LowerAdaptiveEvolutionaryAlgorithm.__init__(self=self.mock_lower_evolutionary_algorithm_object,
    #                                                 upper_iteration=upper_iteration, index=index,
    #                                                 initial_population=initial_population, **params)
    #     assert self.mock_lower_evolutionary_algorithm_object.upper_iteration == upper_iteration
    #     assert self.mock_lower_evolutionary_algorithm_object.index == index
    #     assert self.mock_lower_evolutionary_algorithm_object._population == initial_population
    #     self.mock_evolutionary_algorithm_init.assert_called_once_with(**params)

    # _log_iteration

    @pytest.mark.parametrize("iteration_index", [0, 1, 23])
    def test_log_iteration__without_logger(self, iteration_index):
        """
        Test for '_log_iteration' method without logger set.

        :param iteration_index: Example value of 'iteration_index'.
        """
        self.mock_lower_evolutionary_algorithm_object.logger = None
        LowerAdaptiveEvolutionaryAlgorithm._log_iteration(self=self.mock_lower_evolutionary_algorithm_object,
                                                          iteration_index=iteration_index)

    @pytest.mark.parametrize("iteration_index", [0, 1, 23])
    @pytest.mark.parametrize("upper_iteration", [2, 3])
    @pytest.mark.parametrize("lower_algorithm_index", [4, 5])
    @pytest.mark.parametrize("population", ["12345", "some population"])
    def test_log_iteration__with_logger(self, iteration_index, upper_iteration, lower_algorithm_index, population):
        """
        Test for '_log_iteration' method with logger set.

        :param iteration_index: Example value of 'iteration_index'.
        :param upper_iteration: Example value of 'upper_iteration' attribute.
        :param lower_algorithm_index: Example value of 'index' attribute.
        :param population: Example value of '_population' attribute.
        """
        mock_logger = Mock()
        self.mock_lower_evolutionary_algorithm_object.logger = mock_logger
        self.mock_lower_evolutionary_algorithm_object.upper_iteration = upper_iteration
        self.mock_lower_evolutionary_algorithm_object.index = lower_algorithm_index
        self.mock_lower_evolutionary_algorithm_object._population = population
        LowerAdaptiveEvolutionaryAlgorithm._log_iteration(self=self.mock_lower_evolutionary_algorithm_object,
                                                          iteration_index=iteration_index)
        mock_logger.log_lower_level_iteration.assert_called_once_with(upper_iteration=upper_iteration,
                                                                      lower_algorithm_index=lower_algorithm_index,
                                                                      lower_iteration=iteration_index,
                                                                      solutions=population)
