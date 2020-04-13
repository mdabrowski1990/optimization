import pytest
from mock import Mock, PropertyMock, patch

from optimization.problem_definition.decision_variables import choose_random_value
from optimization.problem_definition.problem import OptimizationProblem, OptimizationType, Solution
from .conftest import EXAMPLE_VALUE_TYPES, DECISION_VARIABLES_GROUPS, NUMBER_OF_CONSTRAINTS_EXAMPLES, \
    NUMBER_OF_PENALTY_FUNCTIONS_EXAMPLES, NUMBER_OF_OBJECTIVE_FUNCTIONS_EXAMPLES, \
    EXAMPLE_OBJECTIVE_FUNCTION_VALUES, EXAMPLE_PENALTY_FUNCTION_VALUES, EXAMPLE_CONSTRAINTS_VALUES, \
    EXAMPLE_DECISION_VARIABLES_VALUES


class TestOptimizationProblem:
    """Tests for 'OptimizationProblem' class."""

    @pytest.mark.parametrize("example_decision_variables", DECISION_VARIABLES_GROUPS, indirect=True)
    @pytest.mark.parametrize("example_constraints", range(NUMBER_OF_CONSTRAINTS_EXAMPLES), indirect=True)
    @pytest.mark.parametrize("example_penalty_function", range(NUMBER_OF_PENALTY_FUNCTIONS_EXAMPLES), indirect=True)
    @pytest.mark.parametrize("example_objective_function", range(NUMBER_OF_OBJECTIVE_FUNCTIONS_EXAMPLES), indirect=True)
    @pytest.mark.parametrize("example_optimization_type", OptimizationType)
    def test_init_with_valid_parameters(self, example_decision_variables, example_constraints, example_penalty_function,
                                        example_objective_function, example_optimization_type):
        """
        Check that 'OptimizationProblem' class can be initialized with correct parameters.
        Make sure that parameters are properly assigned.

        :param example_decision_variables: Example valid value of decision variables.
        :param example_constraints: Example valid value of constrains.
        :param example_penalty_function: Example valid value of penalty function.
        :param example_objective_function: Example valid value of objective function.
        :param example_optimization_type: Example valid value of optimization type.
        """
        problem = OptimizationProblem(decision_variables=example_decision_variables,
                                      constraints=example_constraints,
                                      penalty_function=example_penalty_function,
                                      objective_function=example_objective_function,
                                      optimization_type=example_optimization_type)
        assert problem.decision_variables == example_decision_variables
        assert problem.constraints == example_constraints
        assert problem.penalty_function == example_penalty_function
        assert problem.objective_function == example_objective_function
        assert problem.optimization_type == example_optimization_type

    # types checking

    @pytest.mark.parametrize("example_value", EXAMPLE_VALUE_TYPES.difference({dict}), indirect=True)
    @pytest.mark.parametrize("example_constraints", [0], indirect=True)
    @pytest.mark.parametrize("example_penalty_function", [0], indirect=True)
    @pytest.mark.parametrize("example_objective_function", [0], indirect=True)
    @pytest.mark.parametrize("example_optimization_type", OptimizationType)
    def test_init_with_invalid_decision_variables_type(self, example_value, example_constraints,
                                                       example_penalty_function, example_objective_function,
                                                       example_optimization_type):
        """
        Check that 'OptimizationProblem' class will not be initialized with incorrect parameter.

        :param example_value: Example value of decision variables that has invalid type.
        :param example_constraints: Example valid value of constrains.
        :param example_penalty_function: Example valid value of penalty function.
        :param example_objective_function: Example valid value of objective function.
        :param example_optimization_type: Example valid value of optimization type.
        """
        with pytest.raises(TypeError):
            OptimizationProblem(decision_variables=example_value,
                                constraints=example_constraints,
                                penalty_function=example_penalty_function,
                                objective_function=example_objective_function,
                                optimization_type=example_optimization_type)

    @pytest.mark.parametrize("example_decision_variables", DECISION_VARIABLES_GROUPS[:1], indirect=True)
    @pytest.mark.parametrize("example_value", EXAMPLE_VALUE_TYPES.difference({dict}), indirect=True)
    @pytest.mark.parametrize("example_penalty_function", [0], indirect=True)
    @pytest.mark.parametrize("example_objective_function", [0], indirect=True)
    @pytest.mark.parametrize("example_optimization_type", OptimizationType)
    def test_init_with_invalid_constraints_type(self, example_decision_variables, example_value,
                                                example_penalty_function, example_objective_function,
                                                example_optimization_type):
        """
        Check that 'OptimizationProblem' class will not be initialized with incorrect parameter.

        :param example_decision_variables: Example valid value of decision variables.
        :param example_value: Example value of constrains that has invalid type.
        :param example_penalty_function: Example valid value of penalty function.
        :param example_objective_function: Example valid value of objective function.
        :param example_optimization_type: Example valid value of optimization type.
        """
        with pytest.raises(TypeError):
            OptimizationProblem(decision_variables=example_decision_variables,
                                constraints=example_value,
                                penalty_function=example_penalty_function,
                                objective_function=example_objective_function,
                                optimization_type=example_optimization_type)

    @pytest.mark.parametrize("example_decision_variables", DECISION_VARIABLES_GROUPS[:1], indirect=True)
    @pytest.mark.parametrize("example_constraints", [0], indirect=True)
    @pytest.mark.parametrize("example_value", EXAMPLE_VALUE_TYPES.difference({"function"}), indirect=True)
    @pytest.mark.parametrize("example_objective_function", [0], indirect=True)
    @pytest.mark.parametrize("example_optimization_type", OptimizationType)
    def test_init_with_invalid_penalty_function_type(self, example_decision_variables, example_constraints,
                                                     example_value, example_objective_function,
                                                     example_optimization_type):
        """
        Check that 'OptimizationProblem' class will not be initialized with incorrect parameter.

        :param example_decision_variables: Example valid value of decision variables.
        :param example_constraints: Example valid value of constrains.
        :param example_value: Example value of penalty function variables that has invalid type.
        :param example_objective_function: Example valid value of objective function.
        :param example_optimization_type: Example valid value of optimization type.
        """
        with pytest.raises(TypeError):
            OptimizationProblem(decision_variables=example_decision_variables,
                                constraints=example_constraints,
                                penalty_function=example_value,
                                objective_function=example_objective_function,
                                optimization_type=example_optimization_type)

    @pytest.mark.parametrize("example_decision_variables", DECISION_VARIABLES_GROUPS[:1], indirect=True)
    @pytest.mark.parametrize("example_constraints", [0], indirect=True)
    @pytest.mark.parametrize("example_penalty_function", [0], indirect=True)
    @pytest.mark.parametrize("example_value", EXAMPLE_VALUE_TYPES.difference({"function"}), indirect=True)
    @pytest.mark.parametrize("example_optimization_type", OptimizationType)
    def test_init_with_invalid_objective_function_type(self, example_decision_variables, example_constraints,
                                                       example_penalty_function, example_value,
                                                       example_optimization_type):
        """
        Check that 'OptimizationProblem' class will not be initialized with incorrect parameter.

        :param example_decision_variables: Example valid value of decision variables.
        :param example_constraints: Example valid value of constrains.
        :param example_penalty_function: Example valid value of penalty function.
        :param example_value: Example value of objective function that has invalid type.
        :param example_optimization_type: Example valid value of optimization type.
        """
        with pytest.raises(TypeError):
            OptimizationProblem(decision_variables=example_decision_variables,
                                constraints=example_constraints,
                                penalty_function=example_penalty_function,
                                objective_function=example_value,
                                optimization_type=example_optimization_type)

    @pytest.mark.parametrize("example_decision_variables", DECISION_VARIABLES_GROUPS[:1], indirect=True)
    @pytest.mark.parametrize("example_constraints", [0], indirect=True)
    @pytest.mark.parametrize("example_penalty_function", [0], indirect=True)
    @pytest.mark.parametrize("example_objective_function", [0], indirect=True)
    @pytest.mark.parametrize("example_value", EXAMPLE_VALUE_TYPES, indirect=True)
    def test_init_with_invalid_objective_function_type(self, example_decision_variables, example_constraints,
                                                       example_penalty_function, example_objective_function,
                                                       example_value):
        """
        Check that 'OptimizationProblem' class will not be initialized with incorrect parameter.

        :param example_decision_variables: Example valid value of decision variables.
        :param example_constraints: Example valid value of constrains.
        :param example_penalty_function: Example valid value of penalty function.
        :param example_objective_function: Example valid value of objective function.
        :param example_value: Example value of optimization type that has invalid type.
        """
        with pytest.raises(TypeError):
            OptimizationProblem(decision_variables=example_decision_variables,
                                constraints=example_constraints,
                                penalty_function=example_penalty_function,
                                objective_function=example_objective_function,
                                optimization_type=example_value)

    # values checking

    @pytest.mark.parametrize("example_constraints", [0], indirect=True)
    @pytest.mark.parametrize("example_penalty_function", [0], indirect=True)
    @pytest.mark.parametrize("example_objective_function", [0], indirect=True)
    @pytest.mark.parametrize("example_optimization_type", OptimizationType)
    def test_init_with_empty_decision_variables(self, example_constraints, example_penalty_function,
                                                example_objective_function, example_optimization_type):
        """
        Check that 'OptimizationProblem' class will not be initialized with incorrect parameter.

        :param example_constraints: Example valid value of constrains.
        :param example_penalty_function: Example valid value of penalty function.
        :param example_objective_function: Example valid value of objective function.
        :param example_optimization_type: Example valid value of optimization type.
        """
        with pytest.raises(ValueError):
            OptimizationProblem(decision_variables={},
                                constraints=example_constraints,
                                penalty_function=example_penalty_function,
                                objective_function=example_objective_function,
                                optimization_type=example_optimization_type)

    @pytest.mark.parametrize("example_value", EXAMPLE_VALUE_TYPES.difference({str, dict, list, set}), indirect=True)
    @pytest.mark.parametrize("example_decision_variables", DECISION_VARIABLES_GROUPS, indirect=True)
    @pytest.mark.parametrize("example_constraints", [0], indirect=True)
    @pytest.mark.parametrize("example_penalty_function", [0], indirect=True)
    @pytest.mark.parametrize("example_objective_function", [0], indirect=True)
    @pytest.mark.parametrize("example_optimization_type", OptimizationType)
    def test_init_with_invalid_decision_variables_key_value(self, example_value, example_decision_variables,
                                                            example_constraints, example_penalty_function,
                                                            example_objective_function, example_optimization_type):
        """
        Check that 'OptimizationProblem' class will not be initialized with incorrect parameter.

        :param example_value: Example value of decision variables key that has invalid value.
        :param example_decision_variables: Example valid value of decision variables.
        :param example_constraints: Example valid value of constrains.
        :param example_penalty_function: Example valid value of penalty function.
        :param example_objective_function: Example valid value of objective function.
        :param example_optimization_type: Example valid value of optimization type.
        """
        example_decision_variables[example_value] = list(example_decision_variables.values())[0]
        with pytest.raises(ValueError):
            OptimizationProblem(decision_variables=example_decision_variables,
                                constraints=example_constraints,
                                penalty_function=example_penalty_function,
                                objective_function=example_objective_function,
                                optimization_type=example_optimization_type)

    @pytest.mark.parametrize("example_value", EXAMPLE_VALUE_TYPES, indirect=True)
    @pytest.mark.parametrize("example_decision_variables", DECISION_VARIABLES_GROUPS, indirect=True)
    @pytest.mark.parametrize("example_constraints", [0], indirect=True)
    @pytest.mark.parametrize("example_penalty_function", [0], indirect=True)
    @pytest.mark.parametrize("example_objective_function", [0], indirect=True)
    @pytest.mark.parametrize("example_optimization_type", OptimizationType)
    def test_init_with_invalid_decision_variables_value_value(self, example_value, example_decision_variables,
                                                              example_constraints, example_penalty_function,
                                                              example_objective_function, example_optimization_type):
        """
        Check that 'OptimizationProblem' class will not be initialized with incorrect parameter.

        :param example_value: Example value of decision variables value that has invalid value.
        :param example_decision_variables: Example valid value of decision variables.
        :param example_constraints: Example valid value of constrains.
        :param example_penalty_function: Example valid value of penalty function.
        :param example_objective_function: Example valid value of objective function.
        :param example_optimization_type: Example valid value of optimization type.
        """
        example_decision_variables[list(example_decision_variables.keys())[0]] = example_value
        with pytest.raises(ValueError):
            OptimizationProblem(decision_variables=example_decision_variables,
                                constraints=example_constraints,
                                penalty_function=example_penalty_function,
                                objective_function=example_objective_function,
                                optimization_type=example_optimization_type)

    @pytest.mark.parametrize("example_value", EXAMPLE_VALUE_TYPES.difference({str, dict, list, set}), indirect=True)
    @pytest.mark.parametrize("example_decision_variables", DECISION_VARIABLES_GROUPS, indirect=True)
    @pytest.mark.parametrize("example_constraints", [0], indirect=True)
    @pytest.mark.parametrize("example_penalty_function", [0], indirect=True)
    @pytest.mark.parametrize("example_objective_function", [0], indirect=True)
    @pytest.mark.parametrize("example_optimization_type", OptimizationType)
    def test_init_with_invalid_constraints_key_value(self, example_value, example_decision_variables,
                                                     example_constraints, example_penalty_function,
                                                     example_objective_function, example_optimization_type):
        """
        Check that 'OptimizationProblem' class will not be initialized with incorrect parameter.

        :param example_value: Example value of constraints key that has invalid value.
        :param example_decision_variables: Example valid value of decision variables.
        :param example_constraints: Example valid value of constrains.
        :param example_penalty_function: Example valid value of penalty function.
        :param example_objective_function: Example valid value of objective function.
        :param example_optimization_type: Example valid value of optimization type.
        """
        example_constraints[example_value] = list(example_constraints.values())[0]
        with pytest.raises(ValueError):
            OptimizationProblem(decision_variables=example_decision_variables,
                                constraints=example_constraints,
                                penalty_function=example_penalty_function,
                                objective_function=example_objective_function,
                                optimization_type=example_optimization_type)

    @pytest.mark.parametrize("example_value", EXAMPLE_VALUE_TYPES.difference({"function"}), indirect=True)
    @pytest.mark.parametrize("example_decision_variables", DECISION_VARIABLES_GROUPS, indirect=True)
    @pytest.mark.parametrize("example_constraints", [0], indirect=True)
    @pytest.mark.parametrize("example_penalty_function", [0], indirect=True)
    @pytest.mark.parametrize("example_objective_function", [0], indirect=True)
    @pytest.mark.parametrize("example_optimization_type", OptimizationType)
    def test_init_with_invalid_constraints_value_value(self, example_value, example_decision_variables,
                                                       example_constraints, example_penalty_function,
                                                       example_objective_function, example_optimization_type):
        """
        Check that 'OptimizationProblem' class will not be initialized with incorrect parameter.

        :param example_value: Example value of constraints value that has invalid value.
        :param example_decision_variables: Example valid value of decision variables.
        :param example_constraints: Example valid value of constrains.
        :param example_penalty_function: Example valid value of penalty function.
        :param example_objective_function: Example valid value of objective function.
        :param example_optimization_type: Example valid value of optimization type.
        """
        example_constraints[list(example_constraints.keys())[0]] = example_value
        with pytest.raises(ValueError):
            OptimizationProblem(decision_variables=example_decision_variables,
                                constraints=example_constraints,
                                penalty_function=example_penalty_function,
                                objective_function=example_objective_function,
                                optimization_type=example_optimization_type)


class TestSolution:
    def setup(self):
        mock_decision_variables = PropertyMock(return_value={})
        mock_constraints_items = Mock(return_value=[])
        mock_constraints = Mock(items=mock_constraints_items)
        mock_penalty_function = Mock(return_value=0)
        mock_objective_function = Mock(return_value=0)
        mock_optimization_type = PropertyMock(spec=OptimizationType, return_value="Minimize")
        mock_optimization_problem = Mock(spec=OptimizationProblem, decision_variables=mock_decision_variables,
                                         constraints=mock_constraints, penalty_function=mock_penalty_function,
                                         objective_function=mock_objective_function,
                                         optimization_type=mock_optimization_type)

        class MockedSolution(Solution):
            optimization_problem = mock_optimization_problem
        self.MockedSolution = MockedSolution
        self.mock_decision_variables = mock_decision_variables
        self.mock_constraints_items = mock_constraints_items
        self.mock_constraints = mock_constraints
        self.mock_penalty_function = mock_penalty_function
        self.mock_objective_function = mock_objective_function
        self.mock_optimization_problem = mock_optimization_problem
        self.mock_optimization_type = mock_optimization_type

    # __init__

    def test_abstract_class_init(self):
        """Check that abstract class 'Solution' cannot be directly initialized (exception is raised)."""
        with pytest.raises(Exception):
            Solution()

    @pytest.mark.parametrize("example_decision_variables", DECISION_VARIABLES_GROUPS, indirect=True)
    def test_subclass_init_without_values(self, example_decision_variables):
        """
        Check that 'Solution' subclass can be initialized without any parameters.

        :param example_decision_variables: Example valid value of decision variables definition.
        """
        self.MockedSolution.optimization_problem.decision_variables = example_decision_variables
        solution = self.MockedSolution()
        assert solution._objective_value is None
        assert isinstance(solution.decision_variables_values, dict)
        assert solution.decision_variables_values.keys() == example_decision_variables.keys()

    @pytest.mark.parametrize("example_decision_variables", DECISION_VARIABLES_GROUPS, indirect=True)
    def test_subclass_init_with_one_valid_value(self, example_decision_variables):
        """
        Check that 'Solution' subclass can be initialized with one valid value set.

        :param example_decision_variables: Example valid value of decision variables definition.
        """
        self.MockedSolution.optimization_problem.decision_variables = example_decision_variables
        var_name, var_definition = choose_random_value(example_decision_variables.items())
        var_value = var_definition.generate_random_value()
        solution = self.MockedSolution(**{var_name: var_value})
        assert solution._objective_value is None
        assert isinstance(solution.decision_variables_values, dict)
        assert solution.decision_variables_values.keys() == example_decision_variables.keys()

    @pytest.mark.parametrize("example_decision_variables", DECISION_VARIABLES_GROUPS, indirect=True)
    def test_subclass_init_with_valid_values(self, example_decision_variables):
        """
        Check that 'Solution' subclass can be initialized with all valid values set.

        :param example_decision_variables: Example valid value of decision variables definition.
        """
        self.MockedSolution.optimization_problem.decision_variables = example_decision_variables
        values_to_set = {
            var_name: var_definition.generate_random_value()
            for var_name, var_definition in example_decision_variables.items()
        }
        solution = self.MockedSolution(**values_to_set)
        assert solution._objective_value is None
        assert isinstance(solution.decision_variables_values, dict)
        assert solution.decision_variables_values == values_to_set

    def test_subclass_init_with_invalid_value_name(self):
        """
        Check that 'Solution' subclass cannot be initialized with variable with invalid name.
        """
        with pytest.raises(ValueError):
            self.MockedSolution(some_invalid_variable_name="some value")

    @pytest.mark.parametrize("example_decision_variables", DECISION_VARIABLES_GROUPS, indirect=True)
    def test_subclass_init_with_invalid_value_name(self, example_decision_variables):
        """
        Check that 'Solution' subclass cannot be initialized with variable with invalid value.

        :param example_decision_variables: Example valid value of decision variables definition.
        """
        self.MockedSolution.optimization_problem.decision_variables = example_decision_variables
        var_name = choose_random_value(example_decision_variables.keys())
        with pytest.raises(ValueError):
            self.MockedSolution(**{var_name: "some invalid value"})

    # get_objective_value_with_penalty

    @pytest.mark.parametrize("example_value", EXAMPLE_VALUE_TYPES.difference({None}), indirect=True)
    def test_get_objective_value_with_penalty_with_objective_value_set(self, example_value):
        """
        Check that 'get_objective_value_with_penalty' returns value of '_objective_value' attribute if set.

        :param example_value: Example value to set to _objective_value.
        """
        solution = self.MockedSolution()
        solution._objective_value = example_value
        assert solution.get_objective_value_with_penalty() == solution._objective_value == example_value

    @patch("optimization.problem_definition.problem.Solution._calculate_penalty")
    @patch("optimization.problem_definition.problem.Solution._calculate_objective")
    @pytest.mark.parametrize("optimization_type", OptimizationType)
    @pytest.mark.parametrize("objective_value", EXAMPLE_OBJECTIVE_FUNCTION_VALUES)
    @pytest.mark.parametrize("penalty_value", EXAMPLE_PENALTY_FUNCTION_VALUES)
    def test_get_objective_value_with_penalty_minimize(self, mock_calculate_objective, mock_calculate_penalty,
                                                       optimization_type, objective_value, penalty_value):
        """
        Check that 'get_objective_value_with_penalty' for minimization and maximization problems with
        '_objective_value' attribute value not set.

        :param mock_calculate_objective: Mock of '_calculate_objective' method.
        :param mock_calculate_penalty: Mock of '_calculate_penalty' method.
        :param optimization_type: Type of problem (Maximization or Minimization).
        :param objective_value: Value to return by 'mock_calculate_objective'.
        :param penalty_value: Value to return by 'mock_calculate_penalty'.
        """
        mock_calculate_objective.return_value = objective_value
        mock_calculate_penalty.return_value = penalty_value
        solution = self.MockedSolution()
        solution._objective_value = None
        solution.optimization_problem.optimization_type = optimization_type
        if optimization_type == OptimizationType.Minimize:
            expected_result = objective_value + penalty_value
        elif optimization_type == OptimizationType.Maximize:
            expected_result = objective_value - penalty_value
        else:
            raise ValueError
        assert solution.get_objective_value_with_penalty() == solution._objective_value
        assert solution._objective_value == expected_result

    # _calculate_objective

    @pytest.mark.parametrize("decision_variable_values", EXAMPLE_DECISION_VARIABLES_VALUES)
    @pytest.mark.parametrize("objective_value", EXAMPLE_OBJECTIVE_FUNCTION_VALUES)
    def test_calculate_objective(self, decision_variable_values, objective_value):
        """
        Check that '_calculate_objective' method uses 'objective_function' method.

        :param decision_variable_values: Values to set in 'decision_variables_values' mock.
        :param objective_value: Value to return by 'objective_function' mock.
        """
        self.mock_objective_function.return_value = objective_value
        self.MockedSolution.decision_variables_values = PropertyMock(return_value=decision_variable_values)
        solution = self.MockedSolution()
        assert solution._calculate_objective() == objective_value
        self.mock_objective_function.assert_called_once_with(**decision_variable_values)

    # _calculate_penalty

    @patch("optimization.problem_definition.problem.Solution._calculate_constraints")
    @pytest.mark.parametrize("constraints_values", EXAMPLE_CONSTRAINTS_VALUES)
    @pytest.mark.parametrize("penalty_value", EXAMPLE_PENALTY_FUNCTION_VALUES)
    def test_calculate_penalty(self, mock_calculate_constraints, constraints_values, penalty_value):
        """
        Check that '_calculate_penalty' method uses 'penalty_function' and '_calculate_constraints' methods.

        :param mock_calculate_constraints: Mock of '_calculate_constraints' method.
        :param constraints_values: Value to return by '_calculate_constraints' method.
        :param penalty_value: Value to return by 'penalty_function'.
        """
        self.mock_penalty_function.return_value = penalty_value
        mock_calculate_constraints.return_value = constraints_values
        solution = self.MockedSolution()
        assert solution._calculate_penalty() == abs(penalty_value)
        mock_calculate_constraints.assert_called_once_with()
        self.mock_penalty_function.assert_called_once_with(**constraints_values)

    # _calculate_constraints

    @pytest.mark.parametrize("decision_variable_values", EXAMPLE_DECISION_VARIABLES_VALUES)
    @pytest.mark.parametrize("constraints_names", [("c0", "c1"), ("abc", "def")])
    @pytest.mark.parametrize("constraints_values", [(0, 0), (1, 1), (-1, -1)])
    def test_calculate_constraints(self, decision_variable_values, constraints_names, constraints_values):
        """
        Check '_calculate_penalty' method uses 'penalty_function' and '_calculate_constraints' methods.

        """
        abs_constraints_values = [abs(val) for val in constraints_values]
        self.MockedSolution.decision_variables_values = PropertyMock(return_value=decision_variable_values)
        self.mock_constraints_items.return_value = zip(constraints_names,
                                                       [Mock(return_value=val) for val in constraints_values])
        solution = self.MockedSolution()
        assert solution._calculate_constraints() == dict(zip(constraints_names, abs_constraints_values))
        for _, mock_constraint_function in self.mock_constraints_items():
            mock_constraint_function.assert_called_once_with(**decision_variable_values)
