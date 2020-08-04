import pytest
from mock import patch

from optimization_old.optimization_problem.decision_variables import DecisionVariable,\
    IntegerVariable, FloatVariable, ChoiceVariable, \
    generate_random_int, generate_random_float, choose_random_value
from .conftest import EXAMPLE_VALUE_TYPES, NUMBER_OF_DECISION_VARIABLE_EXAMPLES, \
    EXAMPLE_INT_VARIABLE_LIMITS, NUMBER_OF_INT_VARIABLE_EXAMPLES, \
    EXAMPLE_FLOAT_VARIABLE_LIMITS, NUMBER_OF_FLOAT_VARIABLE_EXAMPLES, \
    EXAMPLE_CHOICE_VARIABLE_OPTIONS, NUMBER_OF_CHOICE_VARIABLE_EXAMPLES


class TestIntegerVariableClass:
    """Tests for methods of 'IntegerVariable' class"""
    @pytest.mark.parametrize("min_value, max_value", EXAMPLE_INT_VARIABLE_LIMITS)
    def test_init_valid_params(self, min_value, max_value):
        """
        Check that 'IntegerVariable' can be initialized with proper int values.

        :param min_value: Some int value.
        :param max_value: Int value that is larger than 'min_value'.
        """
        IntegerVariable(min_value=min_value, max_value=max_value)

    @pytest.mark.parametrize("max_value, min_value", EXAMPLE_INT_VARIABLE_LIMITS)
    def test_init_min_larger_than_max(self, max_value, min_value):
        """
        Check that 'IntegerVariable' raises ValueError during initialization when max_value < min_value.

        :param max_value: Some int value.
        :param min_value: Int value that is larger than 'max_value'.
        """
        with pytest.raises(ValueError):
            IntegerVariable(min_value=min_value, max_value=max_value)

    @pytest.mark.parametrize("example_value", EXAMPLE_VALUE_TYPES.difference({int}), indirect=True)
    def test_init_invalid_min_type(self, example_value):
        """
        Check that 'IntegerVariable' raises TypeError during initialization when min_value is not int type.

        :param example_value: Example value that is not int type.
        """
        with pytest.raises(TypeError):
            IntegerVariable(min_value=example_value, max_value=0)

    @pytest.mark.parametrize("example_value", EXAMPLE_VALUE_TYPES.difference({int}), indirect=True)
    def test_init_invalid_max_type(self, example_value):
        """
        Check that 'IntegerVariable' raises TypeError during initialization when max_value is not int type.

        :param example_value: Example value that is not int type.
        """
        with pytest.raises(TypeError):
            IntegerVariable(min_value=0, max_value=example_value)

    @patch("optimization_old.optimization_problem.decision_variables.generate_random_int")
    @pytest.mark.parametrize("example_integer_decision_variable", range(NUMBER_OF_INT_VARIABLE_EXAMPLES), indirect=True)
    def test_generate_random_value(self, mock_generate_random_int, example_integer_decision_variable):
        """
        Check that 'IntegerVariable' uses 'generate_random_int' function to generate random value.
        It is also checked that min_value and max_value are passed to the function.

        :param mock_generate_random_int: Mock of 'generate_random_int' function.
        :param example_integer_decision_variable: Example instance of 'IntegerVariable' class.
        """
        example_integer_decision_variable.generate_random_value()
        mock_generate_random_int.assert_called_once_with(example_integer_decision_variable.min_value,
                                                         example_integer_decision_variable.max_value)

    @pytest.mark.parametrize("example_value", EXAMPLE_VALUE_TYPES.difference({int}), indirect=True)
    @pytest.mark.parametrize("example_integer_decision_variable", range(NUMBER_OF_INT_VARIABLE_EXAMPLES), indirect=True)
    def test_is_value_correct_wrong_type(self, example_value, example_integer_decision_variable):
        """
        Check that 'is_value_correct' method of 'IntegerVariable' will return False when checked value is not int type.

        :param example_value: Example value that is not int type.
        :param example_integer_decision_variable: Example instance of 'IntegerVariable' class.
        """
        assert not example_integer_decision_variable.is_value_correct(example_value)

    @pytest.mark.parametrize("check_type", ["min", "max", "middle"])
    @pytest.mark.parametrize("example_integer_decision_variable", range(NUMBER_OF_INT_VARIABLE_EXAMPLES), indirect=True)
    def test_is_value_correct_in_range(self, check_type, example_integer_decision_variable):
        """
        Check that 'is_value_correct' method of 'IntegerVariable' will return True when checked value is in range.

        :param check_type: Describes value location in the valid range.
        :param example_integer_decision_variable: Example instance of 'IntegerVariable' class.
        """
        if check_type == "min":
            value = example_integer_decision_variable.min_value
        elif check_type == "max":
            value = example_integer_decision_variable.max_value
        elif check_type == "middle":
            value = (example_integer_decision_variable.min_value + example_integer_decision_variable.max_value) // 2
        else:
            raise ValueError
        assert example_integer_decision_variable.is_value_correct(value)

    @pytest.mark.parametrize("check_type", ["below_min", "above_max"])
    @pytest.mark.parametrize("example_integer_decision_variable", range(NUMBER_OF_INT_VARIABLE_EXAMPLES), indirect=True)
    def test_is_value_correct_out_of_range(self, check_type, example_integer_decision_variable):
        """
        Check that 'is_value_correct' method of 'IntegerVariable' will return False when checked value is out of range.

        :param check_type: Describes value location in the invalid range.
        :param example_integer_decision_variable: Example instance of 'IntegerVariable' class.
        """
        if check_type == "below_min":
            value = example_integer_decision_variable.min_value - 1
        elif check_type == "above_max":
            value = example_integer_decision_variable.max_value + 1
        else:
            raise ValueError
        assert not example_integer_decision_variable.is_value_correct(value)


class TestFloatVariableClass:
    """Tests for methods of 'FloatVariable' class"""
    @pytest.mark.parametrize("min_value, max_value", EXAMPLE_FLOAT_VARIABLE_LIMITS)
    def test_init_valid_params(self, min_value, max_value):
        """
        Check that 'FloatVariable' can be initialized with proper int values.

        :param min_value: Some float value.
        :param max_value: Float value that is larger than 'min_value'.
        """
        FloatVariable(min_value=min_value, max_value=max_value)

    @pytest.mark.parametrize("max_value, min_value", EXAMPLE_FLOAT_VARIABLE_LIMITS)
    def test_init_min_larger_than_max(self, max_value, min_value):
        """
        Check that 'FloatVariable' raises ValueError during initialization when max_value < min_value.

        :param max_value: Some float value.
        :param min_value: Float value that is larger than 'max_value'.
        """
        with pytest.raises(ValueError):
            FloatVariable(min_value=min_value, max_value=max_value)

    @pytest.mark.parametrize("example_value", EXAMPLE_VALUE_TYPES.difference({float}), indirect=True)
    def test_init_invalid_min_type(self, example_value):
        """
        Check that 'FloatVariable' raises TypeError during initialization when min_value is not float type.

        :param example_value: Example value that is not float type.
        """
        with pytest.raises(TypeError):
            FloatVariable(min_value=example_value, max_value=0.)

    @pytest.mark.parametrize("example_value", EXAMPLE_VALUE_TYPES.difference({float}), indirect=True)
    def test_init_invalid_max_type(self, example_value):
        """
        Check that 'FloatVariable' raises TypeError during initialization when max_value is not float type.

        :param example_value: Example value that is not float type.
        """
        with pytest.raises(TypeError):
            FloatVariable(min_value=0., max_value=example_value)

    @patch("optimization_old.optimization_problem.decision_variables.generate_random_float")
    @pytest.mark.parametrize("example_float_decision_variable", range(NUMBER_OF_FLOAT_VARIABLE_EXAMPLES), indirect=True)
    def test_generate_random_value(self, mock_generate_random_float, example_float_decision_variable):
        """
        Check that 'IntegerVariable' uses 'generate_random_int' function to generate random value.
        It is also checked that min_value and max_value are passed to the function.

        :param mock_generate_random_float: Mock of 'generate_random_float' function.
        :param example_float_decision_variable: Example instance of 'FloatVariable' class.
        """
        example_float_decision_variable.generate_random_value()
        mock_generate_random_float.assert_called_once_with(example_float_decision_variable.min_value,
                                                           example_float_decision_variable.max_value)

    @pytest.mark.parametrize("example_value", EXAMPLE_VALUE_TYPES.difference({float}), indirect=True)
    @pytest.mark.parametrize("example_float_decision_variable", range(NUMBER_OF_FLOAT_VARIABLE_EXAMPLES), indirect=True)
    def test_is_value_correct_wrong_type(self, example_value, example_float_decision_variable):
        """
        Check that 'is_value_correct' method of 'FloatVariable' will return False when checked value is not float type.

        :param example_value: Example value that is not float type.
        :param example_float_decision_variable: Example instance of 'FloatVariable' class.
        """
        assert not example_float_decision_variable.is_value_correct(example_value)

    @pytest.mark.parametrize("check_type", ["min", "max", "middle"])
    @pytest.mark.parametrize("example_float_decision_variable", range(NUMBER_OF_FLOAT_VARIABLE_EXAMPLES), indirect=True)
    def test_is_value_correct_in_range(self, check_type, example_float_decision_variable):
        """
        Check that 'is_value_correct' method of 'FloatVariable' will return True when checked value is in range.

        :param check_type: Describes value location in the valid range.
        :param example_float_decision_variable: Example instance of 'FloatVariable' class.
        """
        if check_type == "min":
            value = example_float_decision_variable.min_value
        elif check_type == "max":
            value = example_float_decision_variable.max_value
        elif check_type == "middle":
            value = (example_float_decision_variable.min_value + example_float_decision_variable.max_value) / 2.
        else:
            raise ValueError
        assert example_float_decision_variable.is_value_correct(value)

    @pytest.mark.parametrize("check_type", ["below_min", "above_max"])
    @pytest.mark.parametrize("example_float_decision_variable", range(NUMBER_OF_FLOAT_VARIABLE_EXAMPLES), indirect=True)
    def test_is_value_correct_out_of_range(self, check_type, example_float_decision_variable):
        """
        Check that 'is_value_correct' method of 'FloatVariable' will return False when checked value is out of range.

        :param check_type: Describes value location in the invalid range.
        :param example_float_decision_variable: Example instance of 'FloatVariable' class.
        """
        if check_type == "below_min":
            value = example_float_decision_variable.min_value - 0.0000000001
        elif check_type == "above_max":
            value = example_float_decision_variable.max_value + 0.0000000001
        else:
            raise ValueError
        assert not example_float_decision_variable.is_value_correct(value)


class TestChoiceVariableClass:
    """Tests for methods of 'ChoiceVariable' class"""
    @pytest.mark.parametrize("values_pool", EXAMPLE_CHOICE_VARIABLE_OPTIONS)
    def test_init_valid_params(self, values_pool):
        """
        Check that 'ChoiceVariable' can be initialized with proper values.

        :param values_pool: Set with some values.
        """
        ChoiceVariable(possible_values=values_pool)

    @pytest.mark.parametrize("example_value", EXAMPLE_VALUE_TYPES.difference({set}), indirect=True)
    def test_init_invalid_possible_values_type(self, example_value):
        """
        Check that 'ChoiceVariable' raises TypeError during initialization when possible_values is not set type.

        :param example_value: Example value that is not set type.
        """
        with pytest.raises(TypeError):
            ChoiceVariable(possible_values=example_value)

    @patch("optimization_old.optimization_problem.decision_variables.choose_random_value")
    @pytest.mark.parametrize("example_choice_decision_variable", range(NUMBER_OF_CHOICE_VARIABLE_EXAMPLES),
                             indirect=True)
    def test_generate_random_value(self, mock_choose_random_value, example_choice_decision_variable):
        """
        Check that 'ChoiceVariable' uses 'choose_random_value' function to choose random value.
        It is also checked that possible values are passed to the function.

        :param mock_choose_random_value: Mock of 'choose_random_value' function.
        :param example_choice_decision_variable: Example instance of 'ChoiceVariable' class.
        """
        example_choice_decision_variable.generate_random_value()
        mock_choose_random_value.assert_called_once_with(example_choice_decision_variable.possible_values)

    @pytest.mark.parametrize("values_pool", EXAMPLE_CHOICE_VARIABLE_OPTIONS)
    def test_is_value_correct_in_range(self, values_pool):
        """
        Check that 'is_value_correct' method of 'ChoiceVariable' will return True when checked value is in
        possible values pool.

        :param values_pool: Set with some values.
        """
        choice_var = ChoiceVariable(possible_values=values_pool)
        assert all(choice_var.is_value_correct(value) for value in values_pool)

    @pytest.mark.parametrize("example_value", EXAMPLE_VALUE_TYPES.difference({list, dict}), indirect=True)
    @pytest.mark.parametrize("example_choice_decision_variable", range(NUMBER_OF_CHOICE_VARIABLE_EXAMPLES),
                             indirect=True)
    def test_is_value_correct_with_random_values(self, example_value, example_choice_decision_variable):
        """
        Check that 'is_value_correct' method of 'ChoiceVariable' will return True when checked value is in range and
        False when it is not.

        :param example_value: Some random value.
        :param example_choice_decision_variable: Example instance of 'ChoiceVariable' class.
        """
        assert example_choice_decision_variable.is_value_correct(example_value) == \
               (example_value in example_choice_decision_variable.possible_values)


class TestDecisionVariableClass:
    """Tests for methods of 'DecisionVariable' class."""

    @pytest.mark.parametrize("example_decision_variable", range(NUMBER_OF_DECISION_VARIABLE_EXAMPLES),
                             indirect=True)
    def test_generate_random_value_raises_error(self, example_decision_variable):
        """
        Check that abstractmethod 'generate_random_value' of 'DecisionVariable' class always raises NotImplementedError.

        :param example_decision_variable: Instance of 'DecisionVariable' sub-class.
        """
        with pytest.raises(NotImplementedError):
            DecisionVariable.generate_random_value(example_decision_variable)

    @pytest.mark.parametrize("example_value", EXAMPLE_VALUE_TYPES, indirect=True)
    @pytest.mark.parametrize("example_decision_variable", range(NUMBER_OF_DECISION_VARIABLE_EXAMPLES),
                             indirect=True)
    def test_is_value_correct_raises_error(self, example_value, example_decision_variable):
        """
        Check that abstractmethod 'is_value_correct' of 'DecisionVariable' class always raises NotImplementedError.

        :param example_value: Some random value.
        :param example_decision_variable: Instance of 'DecisionVariable' sub-class.
        """
        with pytest.raises(NotImplementedError):
            DecisionVariable.is_value_correct(example_decision_variable, example_value)


class TestRandomValues:
    def setup(self):
        self.base_repetitions = 2500

    @pytest.mark.parametrize("tested_function, output_type", [(generate_random_int, int),
                                                              (generate_random_float, float)])
    def test_valid_input_output(self, tested_function, output_type):
        """
        Check that 'generate_random_int' and 'generate_random_float' functions returns proper type of the output value.

        :param tested_function: Tested function - either 'generate_random_int' or 'generate_random_float'.
        :param output_type: Expected type of returned variable.
        """
        random_value = tested_function(0, 1)
        assert isinstance(random_value, output_type)
        assert 0 <= random_value <= 1

    @pytest.mark.parametrize("min_value, max_value", [(0, 10), (-100, 100)])
    def test_generate_random_int_distribution(self, min_value, max_value):
        """
        Simple test to check that 'generate_random_int' generates all integer values relatively similar probability.

        :param min_value: Minimal value that can be drawn.
        :param max_value: Maximal value that can be drawn.
        """
        repetitions = (max_value - min_value + 1) * self.base_repetitions
        random_values_distribution = {}
        for _ in range(repetitions):
            rand_val = generate_random_int(min_value, max_value)
            random_values_distribution[rand_val] = random_values_distribution.setdefault(rand_val, 0) + 1
        assert set(random_values_distribution.keys()) == set(range(min_value, max_value+1))
        assert all(0.9*self.base_repetitions <= drawn_times <= 1.1*self.base_repetitions
                   for drawn_times in random_values_distribution.values())

    @pytest.mark.parametrize("min_value, max_value", [(0, 10), (-100, 100)])
    def test_generate_random_float_distribution(self, min_value, max_value):
        """
        Simple test to check that 'generate_random_float' generates float values in all regions
        with relatively similar probability.

        :param min_value: Minimal value that can be drawn.
        :param max_value: Maximal value that can be drawn.
        """
        repetitions = (max_value - min_value) * self.base_repetitions
        random_values = set()
        integer_part_distribution = {}
        for _ in range(repetitions):
            rand_val = generate_random_float(min_value, max_value)
            random_values.add(rand_val)
            int_part = divmod(rand_val, 1)[0]
            integer_part_distribution[int_part] = integer_part_distribution.setdefault(int_part, 0) + 1
        assert len(random_values) >= repetitions-1
        assert all(0.9*self.base_repetitions <= drawn_times <= 1.1*self.base_repetitions
                   for drawn_times in integer_part_distribution.values())

    @pytest.mark.parametrize("values_pool", [set(range(10)), {"black", "white", "red", "blue", "yellow"}])
    def test_choose_random_value_simple(self, values_pool):
        """
        Check that 'choose_random_value' function returns value from values_pool.

        :param values_pool: Values available to be drawn.
        """
        assert all(choose_random_value(values_pool) in values_pool for _ in range(10))

    @pytest.mark.parametrize("values_pool", [set(range(-10, 11)), set(range(-1000, 1000, 13))])
    def test_choose_random_value_distribution(self, values_pool):
        """
        Simple test to check that 'choose_random_value' function chooses random values from possible pool with
        relatively similar probability.

        :param values_pool: Set with possible values to choose.
        """
        repetitions = len(values_pool) * self.base_repetitions
        random_values_distribution = {}
        for _ in range(repetitions):
            rand_val = choose_random_value(values_pool)
            random_values_distribution[rand_val] = random_values_distribution.setdefault(rand_val, 0) + 1
        assert set(random_values_distribution.keys()) == values_pool
        assert all(0.9*self.base_repetitions <= drawn_times <= 1.1*self.base_repetitions
                   for drawn_times in random_values_distribution.values())
