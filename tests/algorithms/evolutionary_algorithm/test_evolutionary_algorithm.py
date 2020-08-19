import pytest
from mock import Mock, patch, call

from collections import OrderedDict
from copy import deepcopy

from optimization.algorithms.evolutionary_algorithm.evolutionary_algorithm import EvolutionaryAlgorithm, \
    SelectionType, CrossoverType, MutationType, AbstractLogger, StopConditions, OptimizationProblem


class TestEvolutionaryAlgorithm:
    """Tests for 'EvolutionaryAlgorithm' class and their methods."""

    SCRIPT_LOCATION = "optimization.algorithms.evolutionary_algorithm.evolutionary_algorithm"

    def setup(self):
        self.mock_decision_variable_1_generate_random_value = Mock()
        self.mock_decision_variable_1 = Mock(generate_random_value=self.mock_decision_variable_1_generate_random_value)
        self.mock_decision_variable_2_generate_random_value = Mock()
        self.mock_decision_variable_2 = Mock(generate_random_value=self.mock_decision_variable_2_generate_random_value)
        self.mock_decision_variable_3_generate_random_value = Mock()
        self.mock_decision_variable_3 = Mock(generate_random_value=self.mock_decision_variable_3_generate_random_value)
        self.decision_variables = OrderedDict(var1=self.mock_decision_variable_1, var2=self.mock_decision_variable_2,
                                              var3=self.mock_decision_variable_3)
        self.mock_stop_conditions = Mock(spec=StopConditions)
        self.mock_problem = Mock(spec=OptimizationProblem, decision_variables=self.decision_variables, variables_number=3)
        self.mock_check_init_input = Mock()
        self.mock_check_additional_parameters = Mock()
        self.mock_perform_selection = Mock()
        self.mock_perform_crossover = Mock()
        self.mock_perform_mutation = Mock()
        self.mock_generate_random_population = Mock()
        self.mock_evolution_iteration = Mock()
        self.mock_evolutionary_algorithm_object = Mock(spec=EvolutionaryAlgorithm,
                                                       _check_init_input=self.mock_check_init_input,
                                                       _check_additional_parameters=self.mock_check_additional_parameters,
                                                       _perform_selection=self.mock_perform_selection,
                                                       _perform_crossover=self.mock_perform_crossover,
                                                       _perform_mutation=self.mock_perform_mutation,
                                                       _generate_random_population=self.mock_generate_random_population,
                                                       _evolution_iteration=self.mock_evolution_iteration)
        # patching
        self._patcher_abstract_algorithm_init = patch(f"{self.SCRIPT_LOCATION}.AbstractOptimizationAlgorithm.__init__")
        self.mock_abstract_algorithm_class_init = self._patcher_abstract_algorithm_init.start()
        self._patcher_check_selection_parameters = patch(f"{self.SCRIPT_LOCATION}.check_selection_parameters")
        self.mock_check_selection_parameters = self._patcher_check_selection_parameters.\
            start()
        self._patcher_check_crossover_parameters = patch(f"{self.SCRIPT_LOCATION}.check_crossover_parameters")
        self.mock_check_crossover_parameters = self._patcher_check_crossover_parameters.start()
        self._patcher_check_mutation_parameters = patch(f"{self.SCRIPT_LOCATION}.check_mutation_parameters")
        self.mock_check_mutation_parameters = self._patcher_check_mutation_parameters.start()

    def teardown(self):
        self._patcher_abstract_algorithm_init.stop()
        self._patcher_check_selection_parameters.stop()
        self._patcher_check_crossover_parameters.stop()
        self._patcher_check_mutation_parameters.stop()

    # __init__

    @pytest.mark.parametrize("population_size", [1, 9999])
    @pytest.mark.parametrize("apply_elitism", [True, False])
    @pytest.mark.parametrize("mutation_chance", [0.1, 0.5])
    @pytest.mark.parametrize("logger", [None, Mock(spec=AbstractLogger)])
    @pytest.mark.parametrize("selection_type, selection_args", [
        (SelectionType.Uniform, {}),
        (SelectionType.Ranking, {"ranking_bias": 1.5}),
    ])
    @pytest.mark.parametrize("crossover_type, crossover_args", [
        (CrossoverType.SinglePoint, {}),
        (CrossoverType.MultiPoint.value, {"crossover_points_number": 2}),
    ])
    @pytest.mark.parametrize("mutation_type, mutation_args", [
        (MutationType.SinglePoint, {}),
        (MutationType.MultiPoint.value, {"mutation_points_number": 2}),
    ])
    def test_init__valid(self, logger, population_size, apply_elitism, mutation_chance,
                         selection_type, selection_args, crossover_type, crossover_args, mutation_type, mutation_args):
        """
        Test valid initialization of 'EvolutionaryAlgorithm' class.

        :param logger: Example value of 'logger' parameter.
        :param population_size: Example value of 'population_size' parameter.
        :param apply_elitism: Example value of 'apply_elitism' parameter.
        :param mutation_chance: Example value of 'mutation_chance' parameter.
        :param selection_type: Example value of 'selection_type' parameter.
        :param selection_args: Example additional argument for given selection type.
        :param crossover_type: Example value of 'crossover_type' parameter.
        :param crossover_args: Example additional argument for given crossover type.
        :param mutation_type: Example value of 'mutation_type' parameter.
        :param mutation_args: Example additional argument for given mutation type.
        """
        EvolutionaryAlgorithm.__init__(self=self.mock_evolutionary_algorithm_object, logger=logger,
                                       problem=self.mock_problem, stop_conditions=self.mock_stop_conditions,
                                       population_size=population_size, apply_elitism=apply_elitism,
                                       mutation_chance=mutation_chance, selection_type=selection_type,
                                       crossover_type=crossover_type, mutation_type=mutation_type,
                                       **selection_args, **crossover_args, **mutation_args)
        self.mock_check_init_input.assert_called_once_with(population_size=population_size,
                                                           mutation_chance=mutation_chance, apply_elitism=apply_elitism)
        self.mock_abstract_algorithm_class_init.assert_called_once_with(problem=self.mock_problem,
                                                                        stop_conditions=self.mock_stop_conditions,
                                                                        logger=logger)
        self.mock_check_additional_parameters.assert_called_once_with()
        assert self.mock_evolutionary_algorithm_object.population_size == population_size
        assert self.mock_evolutionary_algorithm_object.mutation_chance == mutation_chance
        assert self.mock_evolutionary_algorithm_object.apply_elitism == apply_elitism
        assert self.mock_evolutionary_algorithm_object.selection_params == selection_args
        assert self.mock_evolutionary_algorithm_object.crossover_params == crossover_args
        assert self.mock_evolutionary_algorithm_object.mutation_params == mutation_args
        assert isinstance(self.mock_evolutionary_algorithm_object.selection_type, str)
        assert isinstance(self.mock_evolutionary_algorithm_object.crossover_type, str)
        assert isinstance(self.mock_evolutionary_algorithm_object.mutation_type, str)
        assert callable(self.mock_evolutionary_algorithm_object.selection_function)
        assert callable(self.mock_evolutionary_algorithm_object.crossover_function)
        assert callable(self.mock_evolutionary_algorithm_object.mutation_function)

    @pytest.mark.parametrize("selection_type, selection_args", [
        (SelectionType.Uniform, {}),
        (SelectionType.Ranking, {"ranking_bias": 1.5}),
    ])
    @pytest.mark.parametrize("crossover_type, crossover_args", [
        (CrossoverType.SinglePoint, {}),
        (CrossoverType.MultiPoint.value, {"crossover_points_number": 2}),
    ])
    @pytest.mark.parametrize("mutation_type, mutation_args", [
        (MutationType.SinglePoint, {}),
        (MutationType.MultiPoint.value, {"mutation_points_number": 2}),
    ])
    @pytest.mark.parametrize("incorrect_args", [
        {"something": None},
        {"select": True, "mutate": False}
    ])
    def test_init__additional_params(self, example_population_size, example_apply_elitism, example_mutation_chance,
                                     selection_type, selection_args, crossover_type, crossover_args, mutation_type,
                                     mutation_args, incorrect_args):
        """
        Test initialization of 'EvolutionaryAlgorithm' class with unexpected parameters.
        Check that ValueError is raised.

        :param example_population_size: Example value of 'population_size' parameter.
        :param example_apply_elitism: Example value of 'apply_elitism' parameter.
        :param example_mutation_chance: Example value of 'mutation_chance' parameter.
        :param selection_type: Example value of 'selection_type' parameter.
        :param selection_args: Example additional argument for given selection type.
        :param crossover_type: Example value of 'crossover_type' parameter.
        :param crossover_args: Example additional argument for given crossover type.
        :param mutation_type: Example value of 'mutation_type' parameter.
        :param mutation_args: Example additional argument for given mutation type.
        :param incorrect_args: Unexpected arguments.
        """
        with pytest.raises(ValueError):
            EvolutionaryAlgorithm.__init__(self=self.mock_evolutionary_algorithm_object,
                                           problem=self.mock_problem, stop_conditions=self.mock_stop_conditions,
                                           population_size=example_population_size, apply_elitism=example_apply_elitism,
                                           mutation_chance=example_mutation_chance, selection_type=selection_type,
                                           crossover_type=crossover_type, mutation_type=mutation_type,
                                           **selection_args, **crossover_args, **mutation_args, **incorrect_args)

    # _check_init_input

    @pytest.mark.parametrize("population_size_limits, population_size", [
        ([0, 100], 0),
        ([0, 100], 24),
        ([0, 100], 100),
        ([200, 202], 200),
        ([200, 202], 202),
    ])
    @pytest.mark.parametrize("mutation_chance_limits, mutation_chance", [
        ([0, 1], 0.),
        ([0, 1], 0.3345),
        ([0, 1], 1.),
        ([0.1, 0.2], 0.1),
        ([0.1, 0.2], 0.111111111),
        ([0.1, 0.2], 0.2),
    ])
    @pytest.mark.parametrize("apply_elitism", [True, False])
    def test_check_init_input__valid(self, population_size_limits, population_size,
                                     mutation_chance_limits, mutation_chance,
                                     apply_elitism):
        """
        Test that '_check_init_input' raises no exception when all values are in range.

        :param population_size_limits: Example min and max limits of 'population_size' parameter.
        :param population_size: Example value of 'population_size' parameter.
        :param mutation_chance_limits: Example min and max limits of 'mutation_chance' parameter.
        :param mutation_chance: Example value of 'mutation_chance' parameter.
        :param apply_elitism: Example value of 'apply_elitism' parameter.
        """
        self.mock_evolutionary_algorithm_object.MIN_POPULATION_SIZE = population_size_limits[0]
        self.mock_evolutionary_algorithm_object.MAX_POPULATION_SIZE = population_size_limits[1]
        self.mock_evolutionary_algorithm_object.MIN_MUTATION_CHANCE = mutation_chance_limits[0]
        self.mock_evolutionary_algorithm_object.MAX_MUTATION_CHANCE = mutation_chance_limits[1]
        assert EvolutionaryAlgorithm._check_init_input(self=self.mock_evolutionary_algorithm_object,
                                                       population_size=population_size,
                                                       mutation_chance=mutation_chance,
                                                       apply_elitism=apply_elitism) is None

    @pytest.mark.parametrize("population_size_limits, population_size", [
        ([0, 100], -2),
        ([0, 100], 23),
        ([0, 100], 102),
        ([200, 202], 201),
        ([200, 202], 199),
        ([200, 202], 204),
    ])
    @pytest.mark.parametrize("mutation_chance_limits, mutation_chance", [
        ([0, 1], 0.3345),
        ([0.1, 0.2], 0.111111111),
    ])
    @pytest.mark.parametrize("apply_elitism", [True, False])
    def test_check_init_input__invalid_value_population_size(self, population_size_limits, population_size,
                                                             mutation_chance_limits, mutation_chance,
                                                             apply_elitism):
        """
        Test that '_check_init_input' raises ValueError when 'population_size' is not in range or is odd integer.

        :param population_size_limits: Example min and max limits of 'population_size' parameter.
        :param population_size: Value out of range or odd number.
        :param mutation_chance_limits: Example min and max limits of 'mutation_chance' parameter.
        :param mutation_chance: Example value of 'mutation_chance' parameter.
        :param apply_elitism: Example value of 'apply_elitism' parameter.
        """
        self.mock_evolutionary_algorithm_object.MIN_POPULATION_SIZE = population_size_limits[0]
        self.mock_evolutionary_algorithm_object.MAX_POPULATION_SIZE = population_size_limits[1]
        self.mock_evolutionary_algorithm_object.MIN_MUTATION_CHANCE = mutation_chance_limits[0]
        self.mock_evolutionary_algorithm_object.MAX_MUTATION_CHANCE = mutation_chance_limits[1]
        with pytest.raises(ValueError):
            EvolutionaryAlgorithm._check_init_input(self=self.mock_evolutionary_algorithm_object,
                                                    population_size=population_size, mutation_chance=mutation_chance,
                                                    apply_elitism=apply_elitism)

    @pytest.mark.parametrize("population_size_limits, population_size", [
        ([0, 100], 24),
        ([200, 202], 200),
    ])
    @pytest.mark.parametrize("mutation_chance_limits, mutation_chance", [
        ([0, 1], -0.00001),
        ([0, 1], 1.000001),
        ([0.1, 0.2], 0.0999999999),
        ([0.1, 0.2], 0.2000000001),
    ])
    @pytest.mark.parametrize("apply_elitism", [True, False])
    def test_check_init_input__invalid_value_mutation_chance(self, population_size_limits, population_size,
                                                             mutation_chance_limits, mutation_chance,
                                                             apply_elitism):
        """
        Test that '_check_init_input' raises ValueError when 'mutation_chance' is not in range.

        :param population_size_limits: Example min and max limits of 'population_size' parameter.
        :param population_size: Example value of 'population_size' parameter.
        :param mutation_chance_limits: Example min and max limits of 'mutation_chance' parameter.
        :param mutation_chance: Value out of range.
        :param apply_elitism: Example value of 'apply_elitism' parameter.
        """
        self.mock_evolutionary_algorithm_object.MIN_POPULATION_SIZE = population_size_limits[0]
        self.mock_evolutionary_algorithm_object.MAX_POPULATION_SIZE = population_size_limits[1]
        self.mock_evolutionary_algorithm_object.MIN_MUTATION_CHANCE = mutation_chance_limits[0]
        self.mock_evolutionary_algorithm_object.MAX_MUTATION_CHANCE = mutation_chance_limits[1]
        with pytest.raises(ValueError):
            EvolutionaryAlgorithm._check_init_input(self=self.mock_evolutionary_algorithm_object,
                                                    population_size=population_size, mutation_chance=mutation_chance,
                                                    apply_elitism=apply_elitism)

    @pytest.mark.parametrize("population_size, mutation_chance, apply_elitism", [
        (None, 0.2, True),
        (122.5, 0.2, True),
        (100, None, True),
        (100, 0, True),
        (100, 0.2, None),
        (100, 0.2, "no"),
    ])
    def test_check_init_input__invalid_type(self, population_size, mutation_chance, apply_elitism):
        """
        Test that '_check_init_input' raises TypeError when any of parameter has incorrect type.

        :param population_size: Example value of 'population_size' parameter.
        :param mutation_chance: Example value of 'mutation_chance' parameter.
        :param apply_elitism: Example value of 'apply_elitism' parameter.
        """
        self.mock_evolutionary_algorithm_object.MIN_POPULATION_SIZE = 10
        self.mock_evolutionary_algorithm_object.MAX_POPULATION_SIZE = 1000
        self.mock_evolutionary_algorithm_object.MIN_MUTATION_CHANCE = 0.
        self.mock_evolutionary_algorithm_object.MAX_MUTATION_CHANCE = 1.
        with pytest.raises(TypeError):
            EvolutionaryAlgorithm._check_init_input(self=self.mock_evolutionary_algorithm_object,
                                                    population_size=population_size, mutation_chance=mutation_chance,
                                                    apply_elitism=apply_elitism)

    # _check_additional_parameters

    @pytest.mark.parametrize("variables_number", [5, 9])
    @pytest.mark.parametrize("selection_params", [
        {},
        {"abc": 1, "def": 2}
    ])
    @pytest.mark.parametrize("crossover_params", [
        {},
        {"points": 3}
    ])
    @pytest.mark.parametrize("mutation_params", [
        {},
        {"some_var1": "some value 1", "some_var2": "some value 2"}
    ])
    def test_check_additional_parameters(self, variables_number, selection_params, crossover_params, mutation_params):
        """
        Test '_check_additional_parameters' method.

        :param variables_number: Example number of decision variables.
        :param selection_params: Example selection parameters.
        :param crossover_params: Example crossover parameters.
        :param mutation_params: Example mutation parameters.
        """
        self.mock_evolutionary_algorithm_object.problem = Mock(variables_number=variables_number)
        self.mock_evolutionary_algorithm_object.selection_params = selection_params
        self.mock_evolutionary_algorithm_object.crossover_params = crossover_params
        self.mock_evolutionary_algorithm_object.mutation_params = mutation_params
        EvolutionaryAlgorithm._check_additional_parameters(self=self.mock_evolutionary_algorithm_object)
        self.mock_check_selection_parameters.assert_called_once_with(**selection_params)
        self.mock_check_crossover_parameters.assert_called_once_with(variables_number=variables_number, **crossover_params)
        self.mock_check_mutation_parameters.assert_called_once_with(variables_number=variables_number, **mutation_params)

    # _generate_random_population

    @pytest.mark.parametrize("population_size, solutions", [
        (2, [1, 2]),
        (10, ["a", 1.2, "b", 3, "c", "def", "xyz", None, True, False]),
    ])
    def test_generate_random_population(self, population_size, solutions):
        """
        Test '_generate_random_population'

        :param population_size:
        :param solutions:
        """
        self.mock_evolutionary_algorithm_object._population = []
        self.mock_evolutionary_algorithm_object.population_size = population_size
        self.mock_evolutionary_algorithm_object.SolutionClass = Mock(side_effect=solutions)
        EvolutionaryAlgorithm._generate_random_population(self=self.mock_evolutionary_algorithm_object)
        assert len(self.mock_evolutionary_algorithm_object._population) == population_size
        assert self.mock_evolutionary_algorithm_object._population == solutions
        self.mock_evolutionary_algorithm_object.SolutionClass.assert_has_calls([call() for _ in solutions])

    # perform_selection

    @pytest.mark.parametrize("population_size", [2, 20])
    @pytest.mark.parametrize("population", [[], range(10)])
    @pytest.mark.parametrize("selection_params", [{}, {"some_param1": "some value", "some_other_parama": 1}])
    @pytest.mark.parametrize("selection_output", [["abc", "def", "xyz"], "something other"])
    def test_perform_selection(self, population_size, population, selection_params, selection_output):
        """
        Test '_perform_selection' uses 'selection_function' and other params to determine new parents.

        :param population_size: Example value of 'population_size' attribute.
        :param population: Example value of '_population' attribute.
        :param selection_params: Example value of 'selection_params' attribute.
        :param selection_output: Example return of 'selection_function'.
        """
        mock_selection_function = Mock(return_value=selection_output)
        self.mock_evolutionary_algorithm_object.selection_function = mock_selection_function
        self.mock_evolutionary_algorithm_object.population_size = population_size
        self.mock_evolutionary_algorithm_object._population = population
        self.mock_evolutionary_algorithm_object.selection_params = selection_params
        assert EvolutionaryAlgorithm._perform_selection(self=self.mock_evolutionary_algorithm_object) == selection_output
        mock_selection_function.assert_called_once_with(population_size=population_size, population=population,
                                                        **selection_params)

    # _perform_crossover

    @pytest.mark.parametrize("parents", ["abc", (1, 2)])
    @pytest.mark.parametrize("variables_number", [4, 7])
    @pytest.mark.parametrize("crossover_params", [{}, {"p1": "v1", "p2": "v2"}])
    @pytest.mark.parametrize("crossover_output", ["some output", ("child1", "child2")])
    def test_perform_crossover(self, parents, variables_number, crossover_params, crossover_output):
        """
        Test '_perform_crossover' uses 'crossover_function' and other params to determine new children values.

        :param parents: Example value of 'parents' parameter.
        :param variables_number: Example value of 'variables_number' attribute.
        :param crossover_params: Example value of 'crossover_params' attribute.
        :param crossover_output: Example return of 'crossover_function'.
        """
        mock_crossover_function = Mock(return_value=crossover_output)
        self.mock_evolutionary_algorithm_object.crossover_function = mock_crossover_function
        self.mock_evolutionary_algorithm_object.crossover_params = crossover_params
        self.mock_evolutionary_algorithm_object.problem = Mock(variables_number=variables_number)
        assert EvolutionaryAlgorithm._perform_crossover(self=self.mock_evolutionary_algorithm_object, parents=parents) \
            == crossover_output
        mock_crossover_function.assert_called_once_with(parents=parents, variables_number=variables_number,
                                                        **crossover_params)

    # _perform_mutation

    @pytest.mark.parametrize("variables_number", [3, 5])
    @pytest.mark.parametrize("mutation_chance", [0.1, 0.2])
    @pytest.mark.parametrize("individual_values", [OrderedDict(var1=1, var2=2, var3=3),
                                                   OrderedDict(var1="a", var2="b", var3="c")])
    @pytest.mark.parametrize("mutation_points", [(), (0, 2), (1, )])
    @pytest.mark.parametrize("mutation_params", [{}, {"param1": "value1", "param2": "value2"}])
    def test_perform_mutation(self, individual_values, mutation_points, variables_number, mutation_chance,
                              mutation_params):
        """
        Test '_perform_mutation' uses 'mutation_function' and other params to determine mutation points.

        :param individual_values: Example value of 'individual_values' parameter.
        :param mutation_points: Example points of mutation to be returned by 'mutation_function'.
        :param variables_number: Example value of 'variables_number' attribute.
        :param mutation_chance: Example value of 'mutation_chance' attribute.
        :param mutation_params: Example value of 'mutation_params' attribute.
        """
        mock_mutation_function = Mock(return_value=mutation_points)
        individual_values_before_mutation = deepcopy(individual_values)
        self.mock_evolutionary_algorithm_object.mutation_function = mock_mutation_function
        self.mock_evolutionary_algorithm_object.mutation_chance = mutation_chance
        self.mock_evolutionary_algorithm_object.mutation_params = mutation_params
        self.mock_evolutionary_algorithm_object.problem = self.mock_problem
        self.mock_evolutionary_algorithm_object.problem.variables_number = variables_number
        EvolutionaryAlgorithm._perform_mutation(self=self.mock_evolutionary_algorithm_object,
                                                individual_values=individual_values)
        mock_mutation_function.assert_called_once_with(variables_number=variables_number,
                                                       mutation_chance=mutation_chance, **mutation_params)
        if 0 in mutation_points:
            assert individual_values["var1"] == self.mock_decision_variable_1_generate_random_value.return_value
        else:
            assert individual_values["var1"] == individual_values_before_mutation["var1"]
        if 1 in mutation_points:
            assert individual_values["var2"] == self.mock_decision_variable_2_generate_random_value.return_value
        else:
            assert individual_values["var2"] == individual_values_before_mutation["var2"]
        if 2 in mutation_points:
            assert individual_values["var3"] == self.mock_decision_variable_3_generate_random_value.return_value
        else:
            assert individual_values["var3"] == individual_values_before_mutation["var3"]
        # reset value as it mutate during test
        individual_values.update(individual_values_before_mutation)

    # _evolution_iteration

    @pytest.mark.parametrize("selected_parents, children_after_crossover, children", [
        [[("abc", "xyz"), (123, 987)], (({"a": 1, "b": 2}, {"a": 0, "b": 0}), ({"x": None}, {"y": 0.1})), ["child1", "child2", "child3", "child4"]],
        [[("parent1", "parent2")], [({"x": 1, "y": 2}, {"x": 0, "y": 0})], ["some child 1", "some child 2"]],
    ])
    def test_evolution_iteration__without_elitism(self, selected_parents, children_after_crossover, children):
        """
        Test '_evolution_iteration' function without elitism being set.

        :param selected_parents: Values to simulate selected parents.
        :param children_after_crossover: Values to simulate children values after crossover.
        :param children: Values to simulate children in new population.
        """
        mock_solution_class = Mock(side_effect=children)
        self.mock_perform_selection.return_value = selected_parents
        self.mock_perform_crossover.side_effect = children_after_crossover
        self.mock_evolutionary_algorithm_object.SolutionClass = mock_solution_class
        self.mock_evolutionary_algorithm_object.apply_elitism = False
        EvolutionaryAlgorithm._evolution_iteration(self=self.mock_evolutionary_algorithm_object)
        self.mock_perform_selection.assert_called_once_with()
        self.mock_perform_crossover.assert_has_calls([call(parents=(p1, p2)) for p1, p2 in selected_parents])
        self.mock_perform_mutation.assert_has_calls([call(child_values) for children_values in children_after_crossover
                                                     for child_values in children_values])
        mock_solution_class.assert_has_calls([call(**child_values) for children_values in children_after_crossover
                                              for child_values in children_values])
        assert self.mock_evolutionary_algorithm_object._population == children

    @pytest.mark.parametrize("selected_parents, children_after_crossover, children", [
        [[(1, 2), (3, 4)], (({"a": 1, "b": 2}, {"a": 0, "b": 0}), ({"x": None}, {"y": 0.1})), [11, 12, -1, -2]],
        [[(101, 102)], [({"x": 1, "y": 2}, {"x": 0, "y": 0})], [-1, 0]],
    ])
    def test_evolution_iteration__with_elitism(self, selected_parents, children_after_crossover, children):
        """
        Test '_evolution_iteration' function with elitism being set.

        :param selected_parents: Values to simulate selected parents.
        :param children_after_crossover: Values to simulate children values after crossover.
        :param children: Values to simulate children in new population.
        """
        mock_solution_class = Mock(side_effect=children)
        self.mock_perform_selection.return_value = selected_parents
        self.mock_perform_crossover.side_effect = children_after_crossover
        self.mock_evolutionary_algorithm_object.SolutionClass = mock_solution_class
        self.mock_evolutionary_algorithm_object.apply_elitism = True
        EvolutionaryAlgorithm._evolution_iteration(self=self.mock_evolutionary_algorithm_object)
        self.mock_perform_selection.assert_called_once_with()
        self.mock_perform_crossover.assert_has_calls([call(parents=(p1, p2)) for p1, p2 in selected_parents])
        self.mock_perform_mutation.assert_has_calls([call(child_values) for children_values in children_after_crossover
                                                     for child_values in children_values])
        mock_solution_class.assert_has_calls([call(**child_values) for children_values in children_after_crossover
                                              for child_values in children_values])
        expected_population = []
        i = 0
        for parents in selected_parents:
            expected_population.append(max(children[i], parents[0]))
            expected_population.append(max(children[i+1], parents[1]))
            i += 2
        assert self.mock_evolutionary_algorithm_object._population == expected_population

    # _perform_iteration

    @pytest.mark.parametrize("current_best", [None, - 5, 13, 2.53])
    @pytest.mark.parametrize("population", [range(-10, -6), range(-5, 10, 2)])
    def test_perform_iteration__iteration_zero_without_logger(self, current_best, population):
        """
        Test '_perform_iteration' method for iteration 0 and no logger.

        :param current_best: Currently best solution.
        :param population: Generated population.
        """
        self.mock_evolutionary_algorithm_object.logger = None
        self.mock_evolutionary_algorithm_object._best_solution = current_best
        self.mock_evolutionary_algorithm_object._population = population
        EvolutionaryAlgorithm._perform_iteration(self=self.mock_evolutionary_algorithm_object, iteration_index=0)
        self.mock_generate_random_population.assert_called_once_with()
        self.mock_evolution_iteration.assert_not_called()
        if current_best is None:
            assert self.mock_evolutionary_algorithm_object._best_solution == max(population)
        else:
            assert self.mock_evolutionary_algorithm_object._best_solution == max(*population, current_best)

    @pytest.mark.parametrize("current_best", [None, - 5, 13, 2.53])
    @pytest.mark.parametrize("population", [range(-10, -6), range(-5, 10, 2)])
    def test_perform_iteration__iteration_zero_with_logger(self, current_best, population):
        """
        Test '_perform_iteration' method for iteration 0 and logger.

        :param current_best: Currently best solution.
        :param population: Generated population.
        """
        logger = Mock()
        self.mock_evolutionary_algorithm_object.logger = logger
        self.mock_evolutionary_algorithm_object._best_solution = current_best
        self.mock_evolutionary_algorithm_object._population = population
        EvolutionaryAlgorithm._perform_iteration(self=self.mock_evolutionary_algorithm_object, iteration_index=0)
        self.mock_generate_random_population.assert_called_once_with()
        self.mock_evolution_iteration.assert_not_called()
        logger.log_iteration.assert_called_once_with(iteration=0, solutions=population)
        if current_best is None:
            assert self.mock_evolutionary_algorithm_object._best_solution == max(population)
        else:
            assert self.mock_evolutionary_algorithm_object._best_solution == max(*population, current_best)

    @pytest.mark.parametrize("iteration", [1, 323])
    @pytest.mark.parametrize("current_best", [None, - 5, 13, 2.53])
    @pytest.mark.parametrize("population", [range(-10, -6), range(-5, 10, 2)])
    def test_perform_iteration__following_iteration_without_logger(self, current_best, population, iteration):
        """
        Test '_perform_iteration' method for following iteration (non zero) and no logger.

        :param current_best: Currently best solution.
        :param population: Generated population.
        :param iteration: Example index of iteration.
        """
        self.mock_evolutionary_algorithm_object.logger = None
        self.mock_evolutionary_algorithm_object._best_solution = current_best
        self.mock_evolutionary_algorithm_object._population = population
        EvolutionaryAlgorithm._perform_iteration(self=self.mock_evolutionary_algorithm_object, iteration_index=iteration)
        self.mock_generate_random_population.assert_not_called()
        self.mock_evolution_iteration.assert_called_once_with()
        if current_best is None:
            assert self.mock_evolutionary_algorithm_object._best_solution == max(population)
        else:
            assert self.mock_evolutionary_algorithm_object._best_solution == max(*population, current_best)

    @pytest.mark.parametrize("iteration", [1, 323])
    @pytest.mark.parametrize("current_best", [None, - 5, 13, 2.53])
    @pytest.mark.parametrize("population", [range(-10, -6), range(-5, 10, 2)])
    def test_perform_iteration__following_iteration_with_logger(self, current_best, population, iteration):
        """
        Test '_perform_iteration' method for following iteration (non zero) and logger.

        :param current_best: Currently best solution.
        :param population: Generated population.
        :param iteration: Example index of iteration.
        """
        logger = Mock()
        self.mock_evolutionary_algorithm_object.logger = logger
        self.mock_evolutionary_algorithm_object._best_solution = current_best
        self.mock_evolutionary_algorithm_object._population = population
        EvolutionaryAlgorithm._perform_iteration(self=self.mock_evolutionary_algorithm_object, iteration_index=iteration)
        self.mock_generate_random_population.assert_not_called()
        self.mock_evolution_iteration.assert_called_once_with()
        logger.log_iteration.assert_called_once_with(iteration=iteration, solutions=population)
        if current_best is None:
            assert self.mock_evolutionary_algorithm_object._best_solution == max(population)
        else:
            assert self.mock_evolutionary_algorithm_object._best_solution == max(*population, current_best)

    # get_log_data

    @pytest.mark.parametrize("population_size", [2, 22])
    @pytest.mark.parametrize("apply_elitism", [True, False])
    @pytest.mark.parametrize("selection_type, selection_params", [
        ("Single", {}),
        ("Uniform", {"some_selection_param": "some value"}),
    ])
    @pytest.mark.parametrize("crossover_type, crossover_params", [
        ("Multi", {}),
        ("Triple", {"some_crossover_param": "some value XY"}),
    ])
    @pytest.mark.parametrize("mutation_type, mutation_params, mutation_chance", [
        ("Strange", {}, 0.1),
        ("Other", {"some_mutation_param": "some value ABC"}, 0.2),
    ])
    def test_get_log_data(self, population_size, apply_elitism, selection_type, selection_params,
                          crossover_type, crossover_params, mutation_type, mutation_params, mutation_chance):
        self.mock_evolutionary_algorithm_object.population_size = population_size
        self.mock_evolutionary_algorithm_object.apply_elitism = apply_elitism
        self.mock_evolutionary_algorithm_object.selection_type = selection_type
        self.mock_evolutionary_algorithm_object.selection_params = selection_params
        self.mock_evolutionary_algorithm_object.crossover_type = crossover_type
        self.mock_evolutionary_algorithm_object.crossover_params = crossover_params
        self.mock_evolutionary_algorithm_object.mutation_type = mutation_type
        self.mock_evolutionary_algorithm_object.mutation_params = mutation_params
        self.mock_evolutionary_algorithm_object.mutation_chance = mutation_chance
        log_data = EvolutionaryAlgorithm.get_log_data(self=self.mock_evolutionary_algorithm_object)
        assert isinstance(log_data, dict)
        assert log_data["type"] == "EvolutionaryAlgorithm"
        assert log_data["population_size"] == population_size
        assert log_data["apply_elitism"] == apply_elitism
        assert log_data["selection_type"] == selection_type
        assert log_data["selection_params"] == selection_params
        assert log_data["crossover_type"] == crossover_type
        assert log_data["crossover_params"] == crossover_params
        assert log_data["mutation_type"] == mutation_type
        assert log_data["mutation_params"] == mutation_params
        assert log_data["mutation_chance"] == mutation_chance
