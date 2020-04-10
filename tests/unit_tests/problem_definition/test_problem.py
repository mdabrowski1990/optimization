import pytest

from optimization.problem_definition.problem import OptimizationProblem, OptimizationType
from .conftest import EXAMPLE_VALUE_TYPES, DECISION_VARIABLES_GROUPS, NUMBER_OF_CONSTRAINTS_EXAMPLES, \
    NUMBER_OF_PENALTY_FUNCTIONS_EXAMPLES, NUMBER_OF_OBJECTIVE_FUNCTIONS_EXAMPLES


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

    @pytest.mark.parametrize("example_value", EXAMPLE_VALUE_TYPES.difference({str, dict, list, set}), indirect=True)
    @pytest.mark.parametrize("example_decision_variables", DECISION_VARIABLES_GROUPS, indirect=True)
    @pytest.mark.parametrize("example_constraints", [0], indirect=True)
    @pytest.mark.parametrize("example_penalty_function", [0], indirect=True)
    @pytest.mark.parametrize("example_objective_function", [0], indirect=True)
    @pytest.mark.parametrize("example_optimization_type", OptimizationType)
    def test_init_with_invalid_decision_variables_key_value(self, example_value, example_decision_variables,
                                                            example_constraints, example_penalty_function,
                                                            example_objective_function,
                                                            example_optimization_type):
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
                                objective_function=example_value,
                                optimization_type=example_optimization_type)

    @pytest.mark.parametrize("example_value", EXAMPLE_VALUE_TYPES, indirect=True)
    @pytest.mark.parametrize("example_decision_variables", DECISION_VARIABLES_GROUPS, indirect=True)
    @pytest.mark.parametrize("example_constraints", [0], indirect=True)
    @pytest.mark.parametrize("example_penalty_function", [0], indirect=True)
    @pytest.mark.parametrize("example_objective_function", [0], indirect=True)
    @pytest.mark.parametrize("example_optimization_type", OptimizationType)
    def test_init_with_invalid_decision_variables_value_value(self, example_value, example_decision_variables,
                                                              example_constraints, example_penalty_function,
                                                              example_objective_function,
                                                              example_optimization_type):
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
                                objective_function=example_value,
                                optimization_type=example_optimization_type)

    @pytest.mark.parametrize("example_value", EXAMPLE_VALUE_TYPES.difference({str, dict, list, set}), indirect=True)
    @pytest.mark.parametrize("example_decision_variables", DECISION_VARIABLES_GROUPS, indirect=True)
    @pytest.mark.parametrize("example_constraints", [0], indirect=True)
    @pytest.mark.parametrize("example_penalty_function", [0], indirect=True)
    @pytest.mark.parametrize("example_objective_function", [0], indirect=True)
    @pytest.mark.parametrize("example_optimization_type", OptimizationType)
    def test_init_with_invalid_constraints_key_value(self, example_value, example_decision_variables,
                                                     example_constraints, example_penalty_function,
                                                     example_objective_function,
                                                     example_optimization_type):
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
                                objective_function=example_value,
                                optimization_type=example_optimization_type)

    @pytest.mark.parametrize("example_value", EXAMPLE_VALUE_TYPES.difference({"function"}), indirect=True)
    @pytest.mark.parametrize("example_decision_variables", DECISION_VARIABLES_GROUPS, indirect=True)
    @pytest.mark.parametrize("example_constraints", [0], indirect=True)
    @pytest.mark.parametrize("example_penalty_function", [0], indirect=True)
    @pytest.mark.parametrize("example_objective_function", [0], indirect=True)
    @pytest.mark.parametrize("example_optimization_type", OptimizationType)
    def test_init_with_invalid_decision_variables_value_value(self, example_value, example_decision_variables,
                                                              example_constraints, example_penalty_function,
                                                              example_objective_function,
                                                              example_optimization_type):
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
                                objective_function=example_value,
                                optimization_type=example_optimization_type)
