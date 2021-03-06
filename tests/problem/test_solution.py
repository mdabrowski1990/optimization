import pytest
from mock import Mock, patch
from collections import OrderedDict

from optimization.problem.solution import AbstractSolution, OptimizationType


class TestSolution:
    """Tests for 'AbstractSolution' and their methods."""

    EXAMPLE_DECISION_VARIABLES_VALUES = [{"a": 1, "b": "black"}, {"x": "hi hello", "hgf32": -987.32}]

    def setup(self):
        self.mock_constraint = Mock()
        self.mock_decision_variable_is_proper_value = Mock(return_value=True)
        self.mock_decision_variable_generate_random_value = Mock()
        self.mock_decision_variable = Mock(is_proper_value=self.mock_decision_variable_is_proper_value,
                                           generate_random_value=self.mock_decision_variable_generate_random_value)
        self.mock_decision_variables_items = Mock(return_value=[])
        self.mock_decision_variables = Mock(items=self.mock_decision_variables_items)
        self.mock_optimization_problem_object = Mock(decision_variables=self.mock_decision_variables,
                                                     constraints=dict(c0=self.mock_constraint, c1=self.mock_constraint))
        self.mock_solution_object_calculate_constraints = Mock()
        self.mock_solution_object_calculate_objective = Mock()
        self.mock_solution_object_calculate_penalty = Mock()
        self.mock_solution_object_get_objective_value_with_penalty = Mock()
        self.mock_solution_object = Mock(spec=AbstractSolution,
                                         optimization_problem=self.mock_optimization_problem_object,
                                         get_objective_value_with_penalty=self.mock_solution_object_get_objective_value_with_penalty,
                                         _calculate_objective=self.mock_solution_object_calculate_objective,
                                         _calculate_penalty=self.mock_solution_object_calculate_penalty,
                                         _calculate_constraints=self.mock_solution_object_calculate_constraints)

    def teardown(self):
        ...

    # __init__

    @pytest.mark.parametrize("decision_variables_names", [("a", "b", "c"), ("x1", "ghi2", "bkAL")])
    def test_init__valid_input_without_variables(self, decision_variables_names):
        """
        Test initialization of 'AbstractSolution' class with valid input when no decision variables values are provided.

        :param decision_variables_names: Names of decision variables defined in optimization problem.
        """
        self.mock_decision_variables_items.return_value = [(name, self.mock_decision_variable)
                                                           for name in decision_variables_names]
        AbstractSolution.__init__(self_solution=self.mock_solution_object)
        assert isinstance(self.mock_solution_object.decision_variables_values, OrderedDict)
        assert set(self.mock_solution_object.decision_variables_values.keys()) == set(decision_variables_names)
        assert all([self.mock_solution_object.decision_variables_values[
                        var_name] == self.mock_decision_variable_generate_random_value.return_value
                    for var_name in decision_variables_names])
        assert self.mock_solution_object._objective_value_with_penalty is None

    @pytest.mark.parametrize("decision_variables_values", EXAMPLE_DECISION_VARIABLES_VALUES)
    def test_init__valid_input_with_all_variables(self, decision_variables_values):
        """
        Test initialization of 'AbstractSolution' class with valid input when all decision variables values are provided.

        :param decision_variables_values: Examples values of decision variables to be set in solution object.
        """
        self.mock_decision_variables_items.return_value = [(name, self.mock_decision_variable)
                                                           for name in decision_variables_values.keys()]
        AbstractSolution.__init__(self_solution=self.mock_solution_object, **decision_variables_values)
        assert isinstance(self.mock_solution_object.decision_variables_values, OrderedDict)
        assert set(self.mock_solution_object.decision_variables_values.keys()) == set(decision_variables_values.keys())
        assert all([self.mock_solution_object.decision_variables_values[var_name] == var_value
                    for var_name, var_value in decision_variables_values.items()])
        assert self.mock_solution_object._objective_value_with_penalty is None

    @pytest.mark.parametrize("decision_variables_values", EXAMPLE_DECISION_VARIABLES_VALUES)
    def test_init__invalid_variable_value(self, decision_variables_values):
        """
        Test initialization of 'AbstractSolution' class raises ValueError when invalid value of decision variable
        is provided.

        :param decision_variables_values: Examples values of decision variables to be set in solution object.
        """
        self.mock_decision_variable_is_proper_value.return_value = False
        self.mock_decision_variables_items.return_value = [(name, self.mock_decision_variable)
                                                           for name in decision_variables_values.keys()]
        with pytest.raises(ValueError):
            AbstractSolution.__init__(self_solution=self.mock_solution_object, **decision_variables_values)

    @pytest.mark.parametrize("decision_variables_values", EXAMPLE_DECISION_VARIABLES_VALUES)
    def test_init__unknown_variable(self, decision_variables_values):
        """
        Test initialization of 'AbstractSolution' class raises ValueError when values of unknown decision variables
        are provided.

        :param decision_variables_values: Examples values of decision variables to be set in solution object.
        """
        with pytest.raises(ValueError):
            AbstractSolution.__init__(self_solution=self.mock_solution_object, **decision_variables_values)

    # _calculate_objective

    @pytest.mark.parametrize("decision_variables_values", EXAMPLE_DECISION_VARIABLES_VALUES)
    def test_calculate_objective(self, decision_variables_values):
        """
        Test '_calculate_objective' method of AbstractSolution class calculates objective using
        optimization_problem model.

        :param decision_variables_values: Examples values of decision variables to be stored in solution object.
        """
        self.mock_solution_object.decision_variables_values = decision_variables_values
        assert AbstractSolution._calculate_objective(
            self.mock_solution_object) == self.mock_optimization_problem_object.objective_function.return_value
        self.mock_optimization_problem_object.objective_function.assert_called_once_with(
            **self.mock_solution_object.decision_variables_values)

    # _calculate_constraints

    @pytest.mark.parametrize("constraint_value", [-1.231, -2, 0, 2.5, 765])
    @pytest.mark.parametrize("decision_variables_values", EXAMPLE_DECISION_VARIABLES_VALUES)
    def test_calculate_constraints(self, decision_variables_values, constraint_value):
        """
        Test '_calculate_constraints' method of AbstractSolution class calculates constraints values using
        optimization_problem model.

        :param decision_variables_values: Examples values of decision variables to be stored in solution object.
        :param constraint_value: Value to be returned by constraint function.
        """
        self.mock_constraint.return_value = constraint_value
        self.mock_solution_object.decision_variables_values = decision_variables_values
        constraints_values = AbstractSolution._calculate_constraints(self.mock_solution_object)
        assert isinstance(constraints_values, dict)
        assert set(constraints_values.keys()) == set(self.mock_optimization_problem_object.constraints.keys())
        assert all([value == abs(self.mock_constraint.return_value) for value in constraints_values.values()])

    # _calculate_penalty

    @pytest.mark.parametrize("constraints_values", [{"c0": 0}, {"a": 1, "b": 2.34}])
    def test_calculate_penalty(self, constraints_values):
        """
        Test '_calculate_penalty' method of AbstractSolution class calculates penalty values using
        optimization_problem model.

        :param constraints_values: Values to be simulated as constraint values.
        """
        self.mock_solution_object_calculate_constraints.return_value = constraints_values
        assert AbstractSolution._calculate_penalty(
            self.mock_solution_object) == self.mock_optimization_problem_object.penalty_function.return_value
        self.mock_optimization_problem_object.penalty_function.assert_called_once_with(**constraints_values)

    # get_objective_value_with_penalty

    @pytest.mark.parametrize("objective_value_with_penalty", [12, 6554.62456])
    def test_get_objective_value_with_penalty__already_calculated(self, objective_value_with_penalty):
        """
        Test  'get_objective_value_with_penalty' method returns calculated value of objective (with penalty)
        if already calculated and stored.

        :param objective_value_with_penalty: Simulated value of objective with penalty.
        """
        self.mock_solution_object._objective_value_with_penalty = objective_value_with_penalty
        assert AbstractSolution.get_objective_value_with_penalty(
            self.mock_solution_object) == objective_value_with_penalty
        self.mock_solution_object_calculate_objective.assert_not_called()
        self.mock_solution_object_calculate_penalty.assert_not_called()

    @pytest.mark.parametrize("objective_value", [1, 2.34])
    @pytest.mark.parametrize("penalty_value", [954, 534.132])
    @pytest.mark.parametrize("optimization_type", [OptimizationType.Minimize, OptimizationType.Maximize])
    def test_get_objective_value_with_penalty__not_calculated(self, objective_value, penalty_value, optimization_type):
        """
        Test 'get_objective_value_with_penalty' method returns and stores calculated value  of objective (with penalty)
        if it has never been calculated before in this object.

        :param objective_value: Simulated value of objective.
        :param penalty_value: Simulated value of penalty.
        :param optimization_type: Simulated optimization type of optimization problem.
        """
        self.mock_optimization_problem_object.optimization_type = optimization_type
        self.mock_solution_object._objective_value_with_penalty = None
        self.mock_solution_object_calculate_penalty.return_value = penalty_value
        self.mock_solution_object_calculate_objective.return_value = objective_value
        if optimization_type == OptimizationType.Maximize:
            expected_objective_with_penalty = objective_value - penalty_value
        else:
            expected_objective_with_penalty = objective_value + penalty_value
        assert AbstractSolution.get_objective_value_with_penalty(self.mock_solution_object) \
               == expected_objective_with_penalty == self.mock_solution_object._objective_value_with_penalty

    # get_log_data

    @pytest.mark.parametrize("objective_value_with_penalty", [12, 6554.62456])
    def test_get_log_data(self, example_decision_variables, objective_value_with_penalty):
        self.mock_solution_object.decision_variables_values = example_decision_variables
        self.mock_solution_object_get_objective_value_with_penalty.return_value = objective_value_with_penalty
        log_data = AbstractSolution.get_log_data(self.mock_solution_object)
        assert isinstance(log_data, dict)
        assert log_data["decision_variables_values"] == example_decision_variables
        assert log_data["objective_value_with_penalty"] == objective_value_with_penalty
        self.mock_solution_object_get_objective_value_with_penalty.assert_called_once_with()


class TestSolutionComparison:
    """Tests for comparison methods of 'AbstractSolution' class."""

    def setup(self):
        self.mock_get_objective_value_with_penalty = Mock()
        self.mock_eq = Mock()
        self.mock_lt = Mock()
        self.mock_le = Mock()
        self.mock_optimization_problem = Mock()
        self.mock_solution_object = Mock(spec=AbstractSolution, __eq__=self.mock_eq, __le__=self.mock_le,
                                         __lt__=self.mock_lt, optimization_problem=self.mock_optimization_problem,
                                         get_objective_value_with_penalty=self.mock_get_objective_value_with_penalty)

    # __eq__

    @pytest.mark.parametrize("value", [0, -100, -0.453, 0., 232, 432.432])
    def test_eq__same(self, value):
        self.mock_get_objective_value_with_penalty.return_value = value
        assert AbstractSolution.__eq__(self.mock_solution_object, self.mock_solution_object) is True

    @pytest.mark.parametrize("values", [(0, 1), (-100, -99.9), (-0.453, -0.454), (2, 1)])
    def test_eq__other(self, values):
        self.mock_get_objective_value_with_penalty.side_effect = values
        assert AbstractSolution.__eq__(self.mock_solution_object, self.mock_solution_object) is False

    @pytest.mark.parametrize("other_type", [int, float, str, list, dict])
    def test_eq__invalid_type(self, other_type):
        with pytest.raises(TypeError):
            AbstractSolution.__eq__(self.mock_solution_object, Mock(spec=other_type))

    # __le__

    @pytest.mark.parametrize("optimization_type, values", [
        (OptimizationType.Maximize, (0, 0)),
        (OptimizationType.Maximize, (-123.456, -123.456)),
        (OptimizationType.Maximize, (-9.54, -9.53)),
        (OptimizationType.Maximize, (-12, 5.21)),
        (OptimizationType.Minimize, (0, 0)),
        (OptimizationType.Minimize, (-123.456, -123.456)),
        (OptimizationType.Minimize, (-9.54, -9.55)),
        (OptimizationType.Minimize, (12, -5.21)),
    ])
    def test_le__true(self, optimization_type, values):
        self.mock_solution_object.optimization_problem.optimization_type = optimization_type
        self.mock_get_objective_value_with_penalty.side_effect = values
        assert AbstractSolution.__le__(self.mock_solution_object, self.mock_solution_object) is True

    @pytest.mark.parametrize("optimization_type, values", [
        (OptimizationType.Maximize, (-123.456, -123.457)),
        (OptimizationType.Maximize, (5.21, -12)),
        (OptimizationType.Minimize, (-9.54, -9.53)),
        (OptimizationType.Minimize, (-5.21, 12)),
    ])
    def test_le__false(self, optimization_type, values):
        self.mock_solution_object.optimization_problem.optimization_type = optimization_type
        self.mock_get_objective_value_with_penalty.side_effect = values
        assert AbstractSolution.__le__(self.mock_solution_object, self.mock_solution_object) is False

    @pytest.mark.parametrize("other_type", [int, float, str, list, dict])
    def test_le__invalid_type(self, other_type):
        with pytest.raises(TypeError):
            AbstractSolution.__le__(self.mock_solution_object, Mock(spec=other_type))

    # __lt__

    @pytest.mark.parametrize("optimization_type, values", [
        (OptimizationType.Maximize, (-9.54, -9.53)),
        (OptimizationType.Maximize, (-12, 5.21)),
        (OptimizationType.Minimize, (-9.54, -9.55)),
        (OptimizationType.Minimize, (12, -5.21)),
    ])
    def test_lt__true(self, optimization_type, values):
        self.mock_solution_object.optimization_problem.optimization_type = optimization_type
        self.mock_get_objective_value_with_penalty.side_effect = values
        assert AbstractSolution.__lt__(self.mock_solution_object, self.mock_solution_object) is True

    @pytest.mark.parametrize("optimization_type, values", [
        (OptimizationType.Maximize, (0, 0)),
        (OptimizationType.Maximize, (-123.456, -123.456)),
        (OptimizationType.Maximize, (-123.456, -123.457)),
        (OptimizationType.Maximize, (5.21, -12)),
        (OptimizationType.Minimize, (0, 0)),
        (OptimizationType.Minimize, (-123.456, -123.456)),
        (OptimizationType.Minimize, (-9.54, -9.53)),
        (OptimizationType.Minimize, (-5.21, 12)),
    ])
    def test_lt__false(self, optimization_type, values):
        self.mock_solution_object.optimization_problem.optimization_type = optimization_type
        self.mock_get_objective_value_with_penalty.side_effect = values
        assert AbstractSolution.__lt__(self.mock_solution_object, self.mock_solution_object) is False

    @pytest.mark.parametrize("other_type", [int, float, str, list, dict])
    def test_lt__invalid_type(self, other_type):
        with pytest.raises(TypeError):
            AbstractSolution.__lt__(self.mock_solution_object, Mock(spec=other_type))

    # __ne__

    @pytest.mark.parametrize("result", [True, False])
    def test_ne(self, result):
        self.mock_eq.return_value = not result
        other = Mock()
        assert AbstractSolution.__ne__(self.mock_solution_object, other) is result
        self.mock_eq.assert_called_once_with(other)

    # __ge__

    @pytest.mark.parametrize("result", [True, False])
    def test_ge(self, result):
        self.mock_lt.return_value = not result
        other = Mock()
        assert AbstractSolution.__ge__(self.mock_solution_object, other) is result
        self.mock_lt.assert_called_once_with(other)

    # __gt__

    @pytest.mark.parametrize("result", [True, False])
    def test_gt(self, result):
        self.mock_le.return_value = not result
        other = Mock()
        assert AbstractSolution.__gt__(self.mock_solution_object, other) is result
        self.mock_le.assert_called_once_with(other)
