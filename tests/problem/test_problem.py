import pytest
from mock import Mock, patch

from optimization.problem.problem import OptimizationType, OptimizationProblem


class TestOptimizationProblem:
    """Tests for 'OptimizationProblem' class and their methods."""

    SCRIPT_LOCATION = "optimization.problem.problem"

    @pytest.fixture(autouse=True)
    def setup(self, example_decision_variables, example_constraints, example_penalty_function,
              example_objective_function):
        self.mock_optimization_problem_object = Mock(spec=OptimizationProblem,
                                                     decision_variables=example_decision_variables,
                                                     constraints=example_constraints,
                                                     penalty_function=example_penalty_function,
                                                     objective_function=example_objective_function,
                                                     optimization_type=OptimizationType.Maximize)
        # patching
        self._patcher_log_function_code = patch(f"{self.SCRIPT_LOCATION}.log_function_code")
        self.mock_log_function_code = self._patcher_log_function_code.start()

    def teardown(self):
        self._patcher_log_function_code.stop()

    # __init__

    @pytest.mark.parametrize("optimization_type", [OptimizationType.Maximize, OptimizationType.Minimize.value])
    def test_init__valid_input(self, example_decision_variables, example_constraints, example_penalty_function,
                               example_objective_function, optimization_type):
        """
        Test for initialization of 'OptimizationProblem' with valid parameters values.

        :param example_decision_variables: Example value of 'decision_variables' param.
        :param example_constraints: Example value of 'constraints' param.
        :param example_penalty_function: Example value of 'penalty_function' param.
        :param example_objective_function: Example value of 'objective_function' param.
        :param optimization_type: Example value of 'optimization_type' param.
        """
        OptimizationProblem.__init__(self=self.mock_optimization_problem_object,
                                     decision_variables=example_decision_variables, constraints=example_constraints,
                                     penalty_function=example_penalty_function,
                                     objective_function=example_objective_function, optimization_type=optimization_type)
        assert self.mock_optimization_problem_object.decision_variables == example_decision_variables
        assert self.mock_optimization_problem_object.constraints == example_constraints
        assert self.mock_optimization_problem_object.penalty_function == example_penalty_function
        assert self.mock_optimization_problem_object.objective_function == example_objective_function
        assert isinstance(self.mock_optimization_problem_object.optimization_type, OptimizationType)

    @pytest.mark.parametrize("invalid_decision_variables", [None, 1, True])
    @pytest.mark.parametrize("optimization_type", [OptimizationType.Maximize, OptimizationType.Minimize.value])
    def test_init__invalid_decision_variables_type(self, invalid_decision_variables, example_constraints,
                                                   example_penalty_function, example_objective_function,
                                                   optimization_type):
        """
        Test that during initialization of 'OptimizationProblem' will be raised TypeError if one of parameters
        has invalid type.

        :param invalid_decision_variables: Value of 'decision_variables' param of invalid type.
        :param example_constraints: Example value of 'constraints' param.
        :param example_penalty_function: Example value of 'penalty_function' param.
        :param example_objective_function: Example value of 'objective_function' param.
        :param optimization_type: Example value of 'optimization_type' param.
        """
        with pytest.raises(TypeError):
            OptimizationProblem.__init__(self=self.mock_optimization_problem_object,
                                         decision_variables=invalid_decision_variables,
                                         constraints=example_constraints, penalty_function=example_penalty_function,
                                         objective_function=example_objective_function,
                                         optimization_type=optimization_type)

    @pytest.mark.parametrize("invalid_constraints", [None, 1, True])
    @pytest.mark.parametrize("optimization_type", [OptimizationType.Maximize, OptimizationType.Minimize.value])
    def test_init__invalid_constraints_type(self, example_decision_variables, invalid_constraints,
                                            example_penalty_function, example_objective_function,
                                            optimization_type):
        """
        Test that during initialization of 'OptimizationProblem' will be raised TypeError if one of parameters
        has invalid type.

        :param example_decision_variables: Example value of 'decision_variables' param.
        :param invalid_constraints: Value of 'constraints' param of invalid type.
        :param example_penalty_function: Example value of 'penalty_function' param.
        :param example_objective_function: Example value of 'objective_function' param.
        :param optimization_type: Example value of 'optimization_type' param.
        """
        with pytest.raises(TypeError):
            OptimizationProblem.__init__(self=self.mock_optimization_problem_object,
                                         decision_variables=example_decision_variables,
                                         constraints=invalid_constraints, penalty_function=example_penalty_function,
                                         objective_function=example_objective_function,
                                         optimization_type=optimization_type)

    @pytest.mark.parametrize("invalid_penalty_function", [None, 1, True])
    @pytest.mark.parametrize("optimization_type", [OptimizationType.Maximize, OptimizationType.Minimize.value])
    def test_init__invalid_penalty_function_type(self, example_decision_variables, example_constraints,
                                                 invalid_penalty_function, example_objective_function,
                                                 optimization_type):
        """
        Test that during initialization of 'OptimizationProblem' will be raised TypeError if one of parameters
        has invalid type.

        :param example_decision_variables: Example value of 'decision_variables' param.
        :param example_constraints: Example value of 'constraints' param.
        :param invalid_penalty_function: Value of 'penalty_function' param of invalid type.
        :param example_objective_function: Example value of 'objective_function' param.
        :param optimization_type: Example value of 'optimization_type' param.
        """
        with pytest.raises(TypeError):
            OptimizationProblem.__init__(self=self.mock_optimization_problem_object,
                                         decision_variables=example_decision_variables,
                                         constraints=example_constraints, penalty_function=invalid_penalty_function,
                                         objective_function=example_objective_function,
                                         optimization_type=optimization_type)

    @pytest.mark.parametrize("invalid_objective_function", [None, 1, True])
    @pytest.mark.parametrize("optimization_type", [OptimizationType.Maximize, OptimizationType.Minimize.value])
    def test_init__invalid_objective_function_type(self, example_decision_variables, example_constraints,
                                                   example_penalty_function, invalid_objective_function,
                                                   optimization_type):
        """
        Test that during initialization of 'OptimizationProblem' will be raised TypeError if one of parameters
        has invalid type.

        :param example_decision_variables: Example value of 'decision_variables' param.
        :param example_constraints: Example value of 'constraints' param.
        :param example_penalty_function: Example value of 'penalty_function' param.
        :param invalid_objective_function: Value of 'objective_function' param of invalid type.
        :param optimization_type: Example value of 'optimization_type' param.
        """
        with pytest.raises(TypeError):
            OptimizationProblem.__init__(self=self.mock_optimization_problem_object,
                                         decision_variables=example_decision_variables,
                                         constraints=example_constraints, penalty_function=example_penalty_function,
                                         objective_function=invalid_objective_function,
                                         optimization_type=optimization_type)

    @pytest.mark.parametrize("invalid_optimization_type", [None, 1, True])
    def test_init__invalid_optimization_type_type(self, example_decision_variables, example_constraints,
                                                  example_penalty_function, example_objective_function,
                                                  invalid_optimization_type):
        """
        Test that during initialization of 'OptimizationProblem' will be raised TypeError if one of parameters
        has invalid type.

        :param example_decision_variables: Example value of 'decision_variables' param.
        :param example_constraints: Example value of 'constraints' param.
        :param example_penalty_function: Example value of 'penalty_function' param.
        :param example_objective_function: Example value of 'objective_function' param.
        :param invalid_optimization_type: Value of 'optimization_type' param of invalid type.
        """
        with pytest.raises(TypeError):
            OptimizationProblem.__init__(self=self.mock_optimization_problem_object,
                                         decision_variables=example_decision_variables,
                                         constraints=example_constraints, penalty_function=example_penalty_function,
                                         objective_function=example_objective_function,
                                         optimization_type=invalid_optimization_type)

    @pytest.mark.parametrize("optimization_type", [OptimizationType.Maximize, OptimizationType.Minimize.value])
    def test_init__invalid_decision_variables_data_1(self, invalid_decision_variables__keys_not_str,
                                                     example_constraints, example_penalty_function,
                                                     example_objective_function, optimization_type):
        """
        Test that during initialization of 'OptimizationProblem' will be raised ValueError if one of parameters
        contains valid data.

        :param invalid_decision_variables__keys_not_str: Invalid value of 'decision_variables' param (keys are not str type).
        :param example_constraints: Example value of 'constraints' param.
        :param example_penalty_function: Example value of 'penalty_function' param.
        :param example_objective_function: Example value of 'objective_function' param.
        :param optimization_type: Example value of 'optimization_type' param.
        """
        with pytest.raises(ValueError):
            OptimizationProblem.__init__(self=self.mock_optimization_problem_object,
                                         decision_variables=invalid_decision_variables__keys_not_str,
                                         constraints=example_constraints, penalty_function=example_penalty_function,
                                         objective_function=example_objective_function,
                                         optimization_type=optimization_type)

    @pytest.mark.parametrize("optimization_type", [OptimizationType.Maximize, OptimizationType.Minimize.value])
    def test_init__invalid_decision_variables_data_2(self, invalid_decision_variables__values_not_decision_variable,
                                                     example_constraints, example_penalty_function,
                                                     example_objective_function, optimization_type):
        """
        Test that during initialization of 'OptimizationProblem' will be raised ValueError if one of parameters
        contains valid data.

        :param invalid_decision_variables__values_not_decision_variable: Invalid value of 'decision_variables' param
            (values are not DecisionVaraible type).
        :param example_constraints: Example value of 'constraints' param.
        :param example_penalty_function: Example value of 'penalty_function' param.
        :param example_objective_function: Example value of 'objective_function' param.
        :param optimization_type: Example value of 'optimization_type' param.
        """
        with pytest.raises(ValueError):
            OptimizationProblem.__init__(self=self.mock_optimization_problem_object,
                                         decision_variables=invalid_decision_variables__values_not_decision_variable,
                                         constraints=example_constraints, penalty_function=example_penalty_function,
                                         objective_function=example_objective_function,
                                         optimization_type=optimization_type)

    @pytest.mark.parametrize("optimization_type", [OptimizationType.Maximize, OptimizationType.Minimize.value])
    def test_init__invalid_constraints_data_1(self, example_decision_variables, invalid_constraints__keys_not_str,
                                              example_penalty_function, example_objective_function,
                                              optimization_type):
        """
        Test that during initialization of 'OptimizationProblem' will be raised ValueError if one of parameters
        contains valid data.

        :param example_decision_variables: Example value of 'decision_variables' param.
        :param invalid_constraints__keys_not_str: Invalid value of 'constraints' param (keys are not str type).
        :param example_penalty_function: Example value of 'penalty_function' param.
        :param example_objective_function: Example value of 'objective_function' param.
        :param optimization_type: Example value of 'optimization_type' param.
        """
        with pytest.raises(ValueError):
            OptimizationProblem.__init__(self=self.mock_optimization_problem_object,
                                         decision_variables=example_decision_variables,
                                         constraints=invalid_constraints__keys_not_str,
                                         penalty_function=example_penalty_function,
                                         objective_function=example_objective_function,
                                         optimization_type=optimization_type)

    @pytest.mark.parametrize("optimization_type", [OptimizationType.Maximize, OptimizationType.Minimize.value])
    def test_init__invalid_constraints_data_2(self, example_decision_variables,
                                              invalid_constraints__values_not_callable,
                                              example_penalty_function, example_objective_function,
                                              optimization_type):
        """
        Test that during initialization of 'OptimizationProblem' will be raised ValueError if one of parameters
        contains valid data.

        :param example_decision_variables: Example value of 'decision_variables' param.
        :param invalid_constraints__values_not_callable: Invalid value of 'constraints' param (values are not callable).
        :param example_penalty_function: Example value of 'penalty_function' param.
        :param example_objective_function: Example value of 'objective_function' param.
        :param optimization_type: Example value of 'optimization_type' param.
        """
        with pytest.raises(ValueError):
            OptimizationProblem.__init__(self=self.mock_optimization_problem_object,
                                         decision_variables=example_decision_variables,
                                         constraints=invalid_constraints__values_not_callable,
                                         penalty_function=example_penalty_function,
                                         objective_function=example_objective_function,
                                         optimization_type=optimization_type)

    # get_log_data

    def test_get_log_data(self):
        """
        Tests that 'get_log_data' method calls 'log_function_code' function and returns dictionary with certain keys.
        """
        log_data = OptimizationProblem.get_log_data(self=self.mock_optimization_problem_object)
        assert isinstance(log_data, dict)
        assert "optimization_type" in log_data
        assert "decision_variables" in log_data
        assert "constraints" in log_data
        assert "penalty_function" in log_data
        assert "objective_function" in log_data
        self.mock_log_function_code.assert_called()
