import pytest
from mock import Mock, patch

from optimization.algorithms.evolutionary_algorithm.adaptive_evolutionary_algorithm import \
    EvolutionaryAlgorithmAdaptationProblem, \
    OptimizationType, AdaptationType, SelectionType, CrossoverType, MutationType


class TestEvolutionaryAlgorithmAdaptationProblem:

    SCRIPT_LOCATION = "optimization.algorithms.evolutionary_algorithm.adaptive_evolutionary_algorithm"

    def setup(self):
        self.mock_validate_mandatory_parameters = Mock()
        self.mock_get_main_decision_variables = Mock()
        self.mock_get_objective_function = Mock()
        self.mock_get_additional_decision_variables = Mock()
        self.mock_get_constraints = Mock()
        self.mock_get_penalty_function = Mock()
        self.mock_adaptation_problem_object = Mock(spec=EvolutionaryAlgorithmAdaptationProblem,
                                                   _validate_mandatory_parameters=self.mock_validate_mandatory_parameters,
                                                   _get_main_decision_variables=self.mock_get_main_decision_variables,
                                                   _get_objective_function=self.mock_get_objective_function,
                                                   _get_additional_decision_variables=self.mock_get_additional_decision_variables,
                                                   _get_constraints=self.mock_get_constraints,
                                                   _get_penalty_function=self.mock_get_penalty_function)
        # patching
        self._patcher_optimization_problem_init = patch(f"{self.SCRIPT_LOCATION}.OptimizationProblem.__init__")
        self.mock_optimization_problem_init = self._patcher_optimization_problem_init.start()

    def teardown(self):
        self._patcher_optimization_problem_init.stop()

    # __init__

    @pytest.mark.parametrize("adaptation_params", [
        {},
        {"solutions_percentile": 0.1},
        {"solutions_number": 5},
        {"solutions_percentile": "something", "solutions_number": "else"}
    ])
    @pytest.mark.parametrize("additional_decision_variables_params", [
        {},
        {"value1": 1, "values2": 2},
        {"tournament_group_size": 5, "roulette_bias": 0.666, "ranking_bias": 1.39}
    ])
    def test_init(self, example_adaptation_type, example_population_size_boundaries, example_selection_types,
                  example_crossover_types, example_mutation_types, example_mutation_chance_boundaries,
                  example_apply_elitism_options, adaptation_params, additional_decision_variables_params):
        EvolutionaryAlgorithmAdaptationProblem.__init__(
            self=self.mock_adaptation_problem_object,
            adaptation_type=example_adaptation_type,
            population_size_boundaries=example_population_size_boundaries,
            selection_types=example_selection_types,
            crossover_types=example_crossover_types,
            mutation_types=example_mutation_types,
            mutation_chance_boundaries=example_mutation_chance_boundaries,
            apply_elitism_options=example_apply_elitism_options,
            **adaptation_params, **additional_decision_variables_params
        )
        self.mock_validate_mandatory_parameters.assert_called_once_with(
            adaptation_type=example_adaptation_type,
            population_size_boundaries=example_population_size_boundaries,
            selection_types=example_selection_types,
            crossover_types=example_crossover_types,
            mutation_types=example_mutation_types,
            mutation_chance_boundaries=example_mutation_chance_boundaries,
            apply_elitism_options=example_apply_elitism_options
        )
        self.mock_get_main_decision_variables.assert_called_once_with(
            population_size_boundaries=example_population_size_boundaries,
            selection_types=example_selection_types,
            crossover_types=example_crossover_types,
            mutation_types=example_mutation_types,
            mutation_chance_boundaries=example_mutation_chance_boundaries,
            apply_elitism_options=example_apply_elitism_options
        )
        self.mock_get_objective_function.assert_called_once_with(
            adaptation_type=example_adaptation_type,
            solutions_percentile=adaptation_params.get("solutions_percentile", None)
            if example_adaptation_type == AdaptationType.BestSolutionsPercentile else None,
            solutions_number=adaptation_params.get("solutions_number", None)
            if example_adaptation_type == AdaptationType.BestSolutions else None
        )
        self.mock_optimization_problem_init.assert_called_once_with(
            decision_variables=self.mock_get_main_decision_variables.return_value,
            constraints=self.mock_get_constraints.return_value,
            penalty_function=self.mock_get_penalty_function.return_value,
            objective_function=self.mock_get_objective_function.return_value,
            optimization_type=OptimizationType.Maximize
        )
        self.mock_get_additional_decision_variables.assert_called_once()
        assert self.mock_adaptation_problem_object.additional_decision_variable \
            == self.mock_get_additional_decision_variables.return_value

    # _validate_mandatory_parameters

    def test_validate_mandatory_parameters__valid(self, example_adaptation_type, example_population_size_boundaries,
                                                  example_selection_types, example_crossover_types,
                                                  example_mutation_types, example_mutation_chance_boundaries,
                                                  example_apply_elitism_options):
        assert EvolutionaryAlgorithmAdaptationProblem._validate_mandatory_parameters(
            self=self.mock_adaptation_problem_object,
            adaptation_type=example_adaptation_type,
            population_size_boundaries=example_population_size_boundaries,
            selection_types=example_selection_types,
            crossover_types=example_crossover_types,
            mutation_types=example_mutation_types,
            mutation_chance_boundaries=example_mutation_chance_boundaries,
            apply_elitism_options=example_apply_elitism_options) is None
        self.mock_adaptation_problem_object._validate_adaptation_type.assert_called_once_with(adaptation_type=example_adaptation_type)
        self.mock_adaptation_problem_object._validate_population_size_boundaries.assert_called_once_with(population_size_boundaries=example_population_size_boundaries)
        self.mock_adaptation_problem_object._validate_selection_types.assert_called_once_with(selection_types=example_selection_types)
        self.mock_adaptation_problem_object._validate_crossover_types.assert_called_once_with(crossover_types=example_crossover_types)
        self.mock_adaptation_problem_object._validate_mutation_types.assert_called_once_with(mutation_types=example_mutation_types)
        self.mock_adaptation_problem_object._validate_mutation_chance_boundaries.assert_called_once_with(mutation_chance_boundaries=example_mutation_chance_boundaries)
        self.mock_adaptation_problem_object._validate_apply_elitism_options.assert_called_once_with(apply_elitism_options=example_apply_elitism_options)

    # _validate_adaptation_type

    def test_validate_adaptation_type__valid(self, example_adaptation_type):
        assert EvolutionaryAlgorithmAdaptationProblem._validate_adaptation_type(
            adaptation_type=example_adaptation_type) is None

    @pytest.mark.parametrize("adaptation_type", [None, 1, "BestSolution"])
    def test_validate_adaptation_type__invalid_type(self, adaptation_type):
        with pytest.raises(TypeError):
            EvolutionaryAlgorithmAdaptationProblem._validate_adaptation_type(adaptation_type=adaptation_type)

    # _validate_population_size_boundaries

    @pytest.mark.parametrize("min_limit, min_population_size, max_population_size, max_limit", [
        (0, 6, 6, 6),
        (1, 2, 12, 13),
        (32, 32, 64, 64),
        (2, 54, 54, 100)
    ])
    def test_validate_adaptation_type__valid(self, min_limit, min_population_size, max_population_size, max_limit):
        self.mock_adaptation_problem_object.MIN_POPULATION_SIZE = min_limit
        self.mock_adaptation_problem_object.MAX_POPULATION_SIZE = max_limit
        assert EvolutionaryAlgorithmAdaptationProblem._validate_population_size_boundaries(
            self=self.mock_adaptation_problem_object,
            population_size_boundaries=(min_population_size, max_population_size)) is None

    @pytest.mark.parametrize("population_size_boundaries", [None, 1, "ab", [1, 2]])
    def test_validate_adaptation_type__invalid_type(self, population_size_boundaries):
        with pytest.raises(TypeError):
            EvolutionaryAlgorithmAdaptationProblem._validate_population_size_boundaries(
                self=self.mock_adaptation_problem_object,
                population_size_boundaries=population_size_boundaries)

    @pytest.mark.parametrize("min_limit, max_limit, population_size_boundaries", [
        (0, 6, (1, 4)),
        (1, 13, (8, 6)),
        (31, 64, (30, 64)),
        (2, 99, (54, 100)),
        (2, 100, ("abc", 50)),
        (2, 100, (50, None)),
        (2, 100, (4, 6, 8)),
        (2, 100, (4,)),
    ])
    def test_validate_adaptation_type__invalid_value(self, min_limit, max_limit, population_size_boundaries):
        self.mock_adaptation_problem_object.MIN_POPULATION_SIZE = min_limit
        self.mock_adaptation_problem_object.MAX_POPULATION_SIZE = max_limit
        with pytest.raises(ValueError):
            EvolutionaryAlgorithmAdaptationProblem._validate_population_size_boundaries(
                self=self.mock_adaptation_problem_object,
                population_size_boundaries=population_size_boundaries)

    # _validate_selection_types

    @pytest.mark.parametrize("selection_types", [
        {SelectionType.Ranking},
        (SelectionType.Roulette, SelectionType.DoubleTournament),
        list(SelectionType)
    ])
    def test_validate_selection_types__valid(self, selection_types):
        assert EvolutionaryAlgorithmAdaptationProblem._validate_selection_types(
            selection_types=selection_types) is None

    @pytest.mark.parametrize("selection_types", [None, 1, SelectionType.Ranking])
    def test_validate_selection_types__invalid_type(self, selection_types):
        with pytest.raises(TypeError):
            EvolutionaryAlgorithmAdaptationProblem._validate_selection_types(selection_types=selection_types)

    @pytest.mark.parametrize("selection_types", [
        {None, SelectionType.Ranking},
        "abcde"
    ])
    def test_validate_selection_types__invalid_value(self, selection_types):
        with pytest.raises(ValueError):
            EvolutionaryAlgorithmAdaptationProblem._validate_selection_types(selection_types=selection_types)

    # _validate_crossover_types
    
    @pytest.mark.parametrize("crossover_types", [
        {CrossoverType.SinglePoint},
        (CrossoverType.MultiPoint, CrossoverType.Adaptive),
        list(CrossoverType)
    ])
    def test_validate_crossover_types__valid(self, crossover_types):
        assert EvolutionaryAlgorithmAdaptationProblem._validate_crossover_types(
            crossover_types=crossover_types) is None

    @pytest.mark.parametrize("crossover_types", [None, 1, CrossoverType.SinglePoint])
    def test_validate_crossover_types__invalid_type(self, crossover_types):
        with pytest.raises(TypeError):
            EvolutionaryAlgorithmAdaptationProblem._validate_crossover_types(crossover_types=crossover_types)

    @pytest.mark.parametrize("crossover_types", [
        {None, CrossoverType.Uniform},
        "abcde"
    ])
    def test_validate_crossover_types__invalid_value(self, crossover_types):
        with pytest.raises(ValueError):
            EvolutionaryAlgorithmAdaptationProblem._validate_crossover_types(crossover_types=crossover_types)

    # _validate_mutation_types

    @pytest.mark.parametrize("mutation_types", [
        {MutationType.SinglePoint},
        (MutationType.MultiPoint, MutationType.Probabilistic),
        list(MutationType)
    ])
    def test_validate_mutation_types__valid(self, mutation_types):
        assert EvolutionaryAlgorithmAdaptationProblem._validate_mutation_types(
            mutation_types=mutation_types) is None

    @pytest.mark.parametrize("mutation_types", [None, 1, MutationType.SinglePoint])
    def test_validate_mutation_types__invalid_type(self, mutation_types):
        with pytest.raises(TypeError):
            EvolutionaryAlgorithmAdaptationProblem._validate_mutation_types(mutation_types=mutation_types)

    @pytest.mark.parametrize("mutation_types", [
        {None, MutationType.Probabilistic},
        "abcde"
    ])
    def test_validate_mutation_types__invalid_value(self, mutation_types):
        with pytest.raises(ValueError):
            EvolutionaryAlgorithmAdaptationProblem._validate_mutation_types(mutation_types=mutation_types)

    # _validate_mutation_chance_boundaries

    @pytest.mark.parametrize("min_limit, min_mutation_chance, max_mutation_chance, max_limit", [
        (0., 0., 1., 1.),
        (0.1, 0.15, 0.15, 0.2),
        (0.01, 0.1, 0.12, 0.2),
    ])
    def test_validate_mutation_chance_boundaries__valid(self, min_limit, min_mutation_chance, max_mutation_chance,
                                                        max_limit):
        self.mock_adaptation_problem_object.MIN_MUTATION_CHANCE = min_limit
        self.mock_adaptation_problem_object.MAX_MUTATION_CHANCE = max_limit
        EvolutionaryAlgorithmAdaptationProblem._validate_mutation_chance_boundaries(
            self=self.mock_adaptation_problem_object,
            mutation_chance_boundaries=(min_mutation_chance, max_mutation_chance)) is None

    @pytest.mark.parametrize("mutation_chance_boundaries", [None, 1, "ab", [0.1, 0.2]])
    def test_validate_mutation_chance_boundaries__invalid_type(self, mutation_chance_boundaries):
        with pytest.raises(TypeError):
            EvolutionaryAlgorithmAdaptationProblem._validate_mutation_chance_boundaries(
                self=self.mock_adaptation_problem_object,
                mutation_chance_boundaries=mutation_chance_boundaries)

    @pytest.mark.parametrize("min_limit, max_limit, mutation_chance_boundaries", [
        (0.1, 0.2, (0.09, 0.15)),
        (0.1, 0.2, (0.1, 0.21)),
        (0., 1., ("abc", 0.2)),
        (0., 1., (0.2, None)),
        (0., 1., (0.2, 0.6, 0.6)),
        (0., 1., (0.2,)),
    ])
    def test_validate_mutation_chance_boundaries__invalid_value(self, min_limit, max_limit, mutation_chance_boundaries):
        self.mock_adaptation_problem_object.MIN_MUTATION_CHANCE = min_limit
        self.mock_adaptation_problem_object.MAX_MUTATION_CHANCE = max_limit
        with pytest.raises(ValueError):
            EvolutionaryAlgorithmAdaptationProblem._validate_mutation_chance_boundaries(
                self=self.mock_adaptation_problem_object,
                mutation_chance_boundaries=mutation_chance_boundaries)

    # _validate_apply_elitism_options

    @pytest.mark.parametrize("apply_elitism_options", [
        {True},
        (True, False),
        [False]
    ])
    def test_validate_apply_elitism_options__valid(self, apply_elitism_options):
        assert EvolutionaryAlgorithmAdaptationProblem._validate_apply_elitism_options(
            apply_elitism_options=apply_elitism_options) is None

    @pytest.mark.parametrize("apply_elitism_options", [None, 1, True])
    def test_validate_apply_elitism_options__invalid_type(self, apply_elitism_options):
        with pytest.raises(TypeError):
            EvolutionaryAlgorithmAdaptationProblem._validate_apply_elitism_options(apply_elitism_options=apply_elitism_options)

    @pytest.mark.parametrize("apply_elitism_options", [
        {None, True},
        [],
        (False, "True")
    ])
    def test_validate_crossover_types__invalid_value(self, apply_elitism_options):
        with pytest.raises(ValueError):
            EvolutionaryAlgorithmAdaptationProblem._validate_apply_elitism_options(apply_elitism_options=apply_elitism_options)


# class TestLowerAdaptiveEvolutionaryAlgorithm:
#
#     SCRIPT_LOCATION = "optimization.algorithms.evolutionary_algorithm.adaptive_evolutionary_algorithm"
#
#     def setup(self):
#         self.mock_lower_evolutionary_algorithm_object = Mock(spec=LowerAdaptiveEvolutionaryAlgorithm)
#         # patching
#         self._patcher_evolutionary_algorithm_init = patch(f"{self.SCRIPT_LOCATION}.EvolutionaryAlgorithm.__init__")
#         self.mock_evolutionary_algorithm_init = self._patcher_evolutionary_algorithm_init.start()
#
#     def teardown(self):
#         self._patcher_evolutionary_algorithm_init.stop()
#
#     # __init__
#
#     # TODO: update these
#     # @pytest.mark.parametrize("upper_iteration", [5, 569])
#     # @pytest.mark.parametrize("index", [0, 13])
#     # @pytest.mark.parametrize("params", [{}, {"a": 1, "b": "abc", "something_other": "dunno"}])
#     # def test_init__without_initial_population(self, upper_iteration, index, params):
#     #     """
#     #     Test '__init__' without passing 'initial_population' parameter.
#     #
#     #     :param upper_iteration: Example value of 'upper_iteration' parameter.
#     #     :param index: Example value of 'index' attribute.
#     #     :param params: Some additional parameters.
#     #     """
#     #     LowerAdaptiveEvolutionaryAlgorithm.__init__(self=self.mock_lower_evolutionary_algorithm_object,
#     #                                                 upper_iteration=upper_iteration, index=index, **params)
#     #     assert self.mock_lower_evolutionary_algorithm_object.upper_iteration == upper_iteration
#     #     assert self.mock_lower_evolutionary_algorithm_object.index == index
#     #     assert self.mock_lower_evolutionary_algorithm_object._population == []
#     #     self.mock_evolutionary_algorithm_init.assert_called_once_with(**params)
#     #
#     # @pytest.mark.parametrize("upper_iteration", [5, 569])
#     # @pytest.mark.parametrize("index", [0, 13])
#     # @pytest.mark.parametrize("initial_population", ["some population", "some other population"])
#     # @pytest.mark.parametrize("params", [{}, {"a": 1, "b": "abc", "something_other": "dunno"}])
#     # def test_init__with_initial_population(self, upper_iteration, index, initial_population, params):
#     #     """
#     #     Test '__init__' without passing 'initial_population' parameter.
#     #
#     #     :param upper_iteration: Example value of 'upper_iteration' parameter.
#     #     :param index: Example value of 'index' attribute.
#     #     :param params: Some additional parameters.
#     #     """
#     #     LowerAdaptiveEvolutionaryAlgorithm.__init__(self=self.mock_lower_evolutionary_algorithm_object,
#     #                                                 upper_iteration=upper_iteration, index=index,
#     #                                                 initial_population=initial_population, **params)
#     #     assert self.mock_lower_evolutionary_algorithm_object.upper_iteration == upper_iteration
#     #     assert self.mock_lower_evolutionary_algorithm_object.index == index
#     #     assert self.mock_lower_evolutionary_algorithm_object._population == initial_population
#     #     self.mock_evolutionary_algorithm_init.assert_called_once_with(**params)
#
#     # _log_iteration
#
#     @pytest.mark.parametrize("iteration_index", [0, 1, 23])
#     def test_log_iteration__without_logger(self, iteration_index):
#         """
#         Test for '_log_iteration' method without logger set.
#
#         :param iteration_index: Example value of 'iteration_index'.
#         """
#         self.mock_lower_evolutionary_algorithm_object.logger = None
#         LowerAdaptiveEvolutionaryAlgorithm._log_iteration(self=self.mock_lower_evolutionary_algorithm_object,
#                                                           iteration_index=iteration_index)
#
#     @pytest.mark.parametrize("iteration_index", [0, 1, 23])
#     @pytest.mark.parametrize("upper_iteration", [2, 3])
#     @pytest.mark.parametrize("lower_algorithm_index", [4, 5])
#     @pytest.mark.parametrize("population", ["12345", "some population"])
#     def test_log_iteration__with_logger(self, iteration_index, upper_iteration, lower_algorithm_index, population):
#         """
#         Test for '_log_iteration' method with logger set.
#
#         :param iteration_index: Example value of 'iteration_index'.
#         :param upper_iteration: Example value of 'upper_iteration' attribute.
#         :param lower_algorithm_index: Example value of 'index' attribute.
#         :param population: Example value of '_population' attribute.
#         """
#         mock_logger = Mock()
#         self.mock_lower_evolutionary_algorithm_object.logger = mock_logger
#         self.mock_lower_evolutionary_algorithm_object.upper_iteration = upper_iteration
#         self.mock_lower_evolutionary_algorithm_object.index = lower_algorithm_index
#         self.mock_lower_evolutionary_algorithm_object._population = population
#         LowerAdaptiveEvolutionaryAlgorithm._log_iteration(self=self.mock_lower_evolutionary_algorithm_object,
#                                                           iteration_index=iteration_index)
#         mock_logger.log_lower_level_iteration.assert_called_once_with(upper_iteration=upper_iteration,
#                                                                       lower_algorithm_index=lower_algorithm_index,
#                                                                       lower_iteration=iteration_index,
#                                                                       solutions=population)
