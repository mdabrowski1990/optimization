import pytest
from mock import Mock, patch, call
from collections import OrderedDict

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
        self._patcher_discrete_variable_class = patch(f"{self.SCRIPT_LOCATION}.DiscreteVariable")
        self.mock_discrete_variable_class = self._patcher_discrete_variable_class.start()
        self._patcher_choice_variable_class = patch(f"{self.SCRIPT_LOCATION}.ChoiceVariable")
        self.mock_choice_variable_class = self._patcher_choice_variable_class.start()
        self._patcher_float_variable_class = patch(f"{self.SCRIPT_LOCATION}.FloatVariable")
        self.mock_float_variable_class = self._patcher_float_variable_class.start()
        self._patcher_integer_variable_class = patch(f"{self.SCRIPT_LOCATION}.IntegerVariable")
        self.mock_integer_variable_class = self._patcher_integer_variable_class.start()

    def teardown(self):
        self._patcher_optimization_problem_init.stop()
        self._patcher_discrete_variable_class.stop()
        self._patcher_choice_variable_class.stop()
        self._patcher_float_variable_class.stop()
        self._patcher_integer_variable_class.stop()

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
            if example_adaptation_type == AdaptationType.BestSolutions else None,
            population_size_boundaries=example_population_size_boundaries
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
        self.mock_adaptation_problem_object._validate_adaptation_type.assert_called_once_with(
            adaptation_type=example_adaptation_type)
        self.mock_adaptation_problem_object._validate_population_size_boundaries.assert_called_once_with(
            population_size_boundaries=example_population_size_boundaries)
        self.mock_adaptation_problem_object._validate_selection_types.assert_called_once_with(
            selection_types=example_selection_types)
        self.mock_adaptation_problem_object._validate_crossover_types.assert_called_once_with(
            crossover_types=example_crossover_types)
        self.mock_adaptation_problem_object._validate_mutation_types.assert_called_once_with(
            mutation_types=example_mutation_types)
        self.mock_adaptation_problem_object._validate_mutation_chance_boundaries.assert_called_once_with(
            mutation_chance_boundaries=example_mutation_chance_boundaries)
        self.mock_adaptation_problem_object._validate_apply_elitism_options.assert_called_once_with(
            apply_elitism_options=example_apply_elitism_options)

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
            EvolutionaryAlgorithmAdaptationProblem._validate_apply_elitism_options(
                apply_elitism_options=apply_elitism_options)

    @pytest.mark.parametrize("apply_elitism_options", [
        {None, True},
        [],
        (False, "True")
    ])
    def test_validate_crossover_types__invalid_value(self, apply_elitism_options):
        with pytest.raises(ValueError):
            EvolutionaryAlgorithmAdaptationProblem._validate_apply_elitism_options(
                apply_elitism_options=apply_elitism_options)

    # _create_objective_function

    @pytest.mark.parametrize("adaptation_type, solutions_percentile, solutions_number", [
        (AdaptationType.BestSolution, None, None),
        (AdaptationType.BestSolutions, None, 5),
        (AdaptationType.BestSolutionsPercentile, 0.1, None),
    ])
    def test_create_objective_function__valid(self, adaptation_type, solutions_percentile, solutions_number):
        objective = EvolutionaryAlgorithmAdaptationProblem._create_objective_function(
            adaptation_type=adaptation_type,
            solutions_percentile=solutions_percentile,
            solutions_number=solutions_number)
        assert callable(objective)

    @pytest.mark.parametrize("adaptation_type", ["Unknown value", None])
    def test_create_objective_function__invalid_type(self, adaptation_type):
        with pytest.raises(NotImplementedError):
            EvolutionaryAlgorithmAdaptationProblem._create_objective_function(adaptation_type=adaptation_type,
                                                                              solutions_percentile=None,
                                                                              solutions_number=None)

    # _get_objective_function

    @pytest.mark.parametrize("adaptation_type, solutions_percentile, solutions_number, population_size_boundaries", [
        (AdaptationType.BestSolution, None, None, (10, 100)),
        (AdaptationType.BestSolutions, None, 5, (10, 100)),
        (AdaptationType.BestSolutions, None, 10, (10, 100)),
        (AdaptationType.BestSolutions, None, 20, (20, 1000)),
        (AdaptationType.BestSolutionsPercentile, 0.1, None, (10, 100)),
        (AdaptationType.BestSolutionsPercentile, 0.00001, None, (10, 100)),
        (AdaptationType.BestSolutionsPercentile, 1., None, (10, 100)),
    ])
    def test_get_objective_function__valid(self, adaptation_type, solutions_percentile, solutions_number,
                                           population_size_boundaries):
        return_value = EvolutionaryAlgorithmAdaptationProblem._get_objective_function(
            self=self.mock_adaptation_problem_object,
            adaptation_type=adaptation_type,
            solutions_percentile=solutions_percentile,
            solutions_number=solutions_number,
            population_size_boundaries=population_size_boundaries)
        self.mock_adaptation_problem_object._create_objective_function.assert_called_once_with(
            adaptation_type=adaptation_type,
            solutions_percentile=solutions_percentile,
            solutions_number=solutions_number)
        assert return_value == self.mock_adaptation_problem_object._create_objective_function.return_value

    @pytest.mark.parametrize("adaptation_type, solutions_percentile, solutions_number", [
        (AdaptationType.BestSolutions, None, None),
        (AdaptationType.BestSolutions, None, 0.5),
        (AdaptationType.BestSolutions, None, "abc"),
        (AdaptationType.BestSolutionsPercentile, 1, None),
        (AdaptationType.BestSolutionsPercentile, False, None),
        (AdaptationType.BestSolutionsPercentile, None, None),
    ])
    def test_get_objective_function__invalid_type(self, adaptation_type, solutions_percentile, solutions_number,
                                                  example_population_size_boundaries):
        with pytest.raises(TypeError):
            EvolutionaryAlgorithmAdaptationProblem._get_objective_function(
                self=self.mock_adaptation_problem_object,
                adaptation_type=adaptation_type,
                solutions_percentile=solutions_percentile,
                solutions_number=solutions_number,
                population_size_boundaries=example_population_size_boundaries)

    @pytest.mark.parametrize("adaptation_type, solutions_percentile, solutions_number, population_size_boundaries", [
        (AdaptationType.BestSolutions, None, 1, (10, 100)),
        (AdaptationType.BestSolutions, None, 11, (10, 100)),
        (AdaptationType.BestSolutions, None, 21, (20, 1000)),
        (AdaptationType.BestSolutionsPercentile, 0., None, (10, 100)),
        (AdaptationType.BestSolutionsPercentile, 1.00001, None, (10, 100)),
    ])
    def test_get_objective_function__invalid_value(self, adaptation_type, solutions_percentile, solutions_number,
                                                   population_size_boundaries):
        with pytest.raises(ValueError):
            EvolutionaryAlgorithmAdaptationProblem._get_objective_function(
                self=self.mock_adaptation_problem_object,
                adaptation_type=adaptation_type,
                solutions_percentile=solutions_percentile,
                solutions_number=solutions_number,
                population_size_boundaries=population_size_boundaries)

    @pytest.mark.parametrize("adaptation_type", [None, "Something new"])
    def test_get_objective_function__ont_implemented(self, adaptation_type, example_population_size_boundaries):
        with pytest.raises(NotImplementedError):
            EvolutionaryAlgorithmAdaptationProblem._get_objective_function(
                self=self.mock_adaptation_problem_object,
                adaptation_type=adaptation_type,
                solutions_percentile=None,
                solutions_number=None,
                population_size_boundaries=example_population_size_boundaries)

    # _get_main_decision_variables

    @pytest.mark.parametrize("population_size_boundaries, mutation_chance_boundaries", [
        [(1, 2), (0.1, 0.2)],
        [(10, 100), (0, 1)],
    ])
    @pytest.mark.parametrize("selection_types, crossover_types, mutation_types", [
        ("abc", "def", "hij"),
        (range(3), range(3, 6), range(6, 9))
    ])
    @pytest.mark.parametrize("apply_elitism_options", [[True], {True, False}])
    def test_get_main_decision_variables(self, population_size_boundaries, selection_types, crossover_types,
                                         mutation_types, mutation_chance_boundaries, apply_elitism_options):
        return_value = EvolutionaryAlgorithmAdaptationProblem._get_main_decision_variables(
            population_size_boundaries=population_size_boundaries,
            selection_types=selection_types,
            crossover_types=crossover_types,
            mutation_types=mutation_types,
            mutation_chance_boundaries=mutation_chance_boundaries,
            apply_elitism_options=apply_elitism_options
        )
        self.mock_discrete_variable_class.assert_has_calls([call(min_value=population_size_boundaries[0],
                                                                 max_value=population_size_boundaries[1], step=2)])
        self.mock_choice_variable_class.assert_has_calls([call(possible_values=selection_types),
                                                          call(possible_values=crossover_types),
                                                          call(possible_values=mutation_types),
                                                          call(possible_values=apply_elitism_options)])
        self.mock_float_variable_class.assert_has_calls([call(min_value=mutation_chance_boundaries[0],
                                                              max_value=mutation_chance_boundaries[1])])
        assert isinstance(return_value, OrderedDict)
        assert return_value["population_size"] == self.mock_discrete_variable_class.return_value
        assert return_value["selection_type"] == self.mock_choice_variable_class.return_value
        assert return_value["crossover_type"] == self.mock_choice_variable_class.return_value
        assert return_value["mutation_type"] == self.mock_choice_variable_class.return_value
        assert return_value["mutation_chance"] == self.mock_float_variable_class.return_value
        assert return_value["apply_elitism"] == self.mock_choice_variable_class.return_value

    # _get_additional_decision_variables

    @pytest.mark.parametrize("additional_params", [
        {},
        {"min_tournament_group_size": 2},
        {"min_tournament_group_size": 3},
        {"max_tournament_group_size": 6},
        {"max_tournament_group_size": 5},
        {"min_roulette_bias": 1.},
        {"min_roulette_bias": 1.1},
        {"max_roulette_bias": 100.},
        {"max_roulette_bias": 99.9},
        {"min_ranking_bias": 1.},
        {"min_ranking_bias": 1.1},
        {"max_ranking_bias": 2.},
        {"max_ranking_bias": 1.9},
        {"min_tournament_group_size": 3, "max_tournament_group_size": 6,
         "min_roulette_bias": 10., "max_roulette_bias": 20.,
         "min_ranking_bias": 1.2, "max_ranking_bias": 1.8}
    ])
    def test_get_additional_decision_variables__valid(self, additional_params, default_min_tournament_group_size,
                                                      default_max_tournament_group_size,
                                                      default_min_ranking_bias, default_max_ranking_bias,
                                                      default_min_roulette_bias, default_max_roulette_bias):
        return_value = EvolutionaryAlgorithmAdaptationProblem._get_additional_decision_variables(**additional_params)
        self.mock_integer_variable_class.assert_has_calls([
            call(min_value=additional_params.get("min_tournament_group_size", default_min_tournament_group_size),
                 max_value=additional_params.get("max_tournament_group_size", default_max_tournament_group_size))
        ])
        self.mock_float_variable_class.assert_has_calls([
            call(min_value=additional_params.get("min_roulette_bias", default_min_roulette_bias),
                 max_value=additional_params.get("max_roulette_bias", default_max_roulette_bias)),
            call(min_value=additional_params.get("min_ranking_bias", default_min_ranking_bias),
                 max_value=additional_params.get("max_ranking_bias", default_max_ranking_bias)),
        ])
        assert isinstance(return_value, dict)
        assert return_value["tournament_group_size"] == self.mock_integer_variable_class.return_value
        assert return_value["roulette_bias"] == self.mock_float_variable_class.return_value
        assert return_value["ranking_bias"] == self.mock_float_variable_class.return_value
        assert return_value["crossover_points_number"] == self.mock_integer_variable_class.return_value
        assert return_value["crossover_patter"] == self.mock_integer_variable_class.return_value
        assert return_value["mutation_points_number"] == self.mock_integer_variable_class.return_value

    @pytest.mark.parametrize("additional_params", [
        {"something_strange": None},
        {"min_tournament_group_size": 1},
        {"max_tournament_group_size": 1000},
        {"min_roulette_bias": 0.},
        {"max_roulette_bias": 9999999999999999999.},
        {"min_ranking_bias": 0.},
        {"min_ranking_bias": 0.99},
        {"max_ranking_bias": 2.01},
        {"max_ranking_bias": 9999.99},
        {"min_tournament_group_size": 4, "max_tournament_group_size": 3},
        {"min_roulette_bias": 40., "max_roulette_bias": 39.9999},
        {"min_ranking_bias": 1.51, "max_ranking_bias": 1.5},
    ])
    def test_get_additional_decision_variables__valid(self, additional_params, default_min_tournament_group_size,
                                                      default_max_tournament_group_size,
                                                      default_min_ranking_bias, default_max_ranking_bias,
                                                      default_min_roulette_bias, default_max_roulette_bias):
        with pytest.raises(ValueError):
            EvolutionaryAlgorithmAdaptationProblem._get_additional_decision_variables(**additional_params)


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
