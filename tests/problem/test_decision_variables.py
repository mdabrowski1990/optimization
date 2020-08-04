import pytest

from mock import Mock, patch
from optimization.problem.decision_variables import IntegerVariable, DiscreteVariable, FloatVariable, ChoiceVariable


class TestIntegerVariable:
    """
    Tests for 'IntegerVariable' class and their methods.
    """
    SCRIPT_LOCATION = "optimization.problem.decision_variables"
    EXAMPLE_INT_VARIABLE_LIMITS = [(0, 1), (-100, -1), (1, 100), (-98765432109876543210, 98765432109876543210)]
    EXAMPLE_NOT_INT_VALUES = [0., None, "123", [0]]

    def setup(self):
        self.mock_decision_variable_object = Mock(spec=IntegerVariable)
        # patching
        self._patcher_generate_random_int = patch(f"{self.SCRIPT_LOCATION}.generate_random_int")
        self.mock_generate_random_int = self._patcher_generate_random_int.start()

    def teardown(self):
        self._patcher_generate_random_int.stop()

    @pytest.mark.parametrize("min_value, max_value", EXAMPLE_INT_VARIABLE_LIMITS)
    def test_init__valid_params(self, min_value, max_value):
        """
        Check that 'IntegerVariable' can be initialized with proper int values.

        :param min_value: Some int value.
        :param max_value: Int value that is greater than 'min_value'.
        """
        IntegerVariable.__init__(self=self.mock_decision_variable_object, min_value=min_value, max_value=max_value)
        assert self.mock_decision_variable_object.min_value == min_value
        assert self.mock_decision_variable_object.max_value == max_value

    @pytest.mark.parametrize("max_value, min_value", EXAMPLE_INT_VARIABLE_LIMITS)
    def test_init__min_greater_than_max(self, min_value, max_value):
        """
        Check that 'IntegerVariable' raises ValueError during init if min_value > max_value.

        :param min_value: Some int value.
        :param max_value: Int value that is less than 'min_value'.
        """
        with pytest.raises(ValueError):
            IntegerVariable.__init__(self=self.mock_decision_variable_object, min_value=min_value, max_value=max_value)

    @pytest.mark.parametrize("not_int", EXAMPLE_NOT_INT_VALUES)
    def test_init__min_value_invalid_type(self, not_int):
        """
        Check that 'IntegerVariable' raises TypeError during init if min_value is not int type.

        :param not_int: Value that is not int type.
        """
        with pytest.raises(TypeError):
            IntegerVariable.__init__(self=self.mock_decision_variable_object, min_value=not_int, max_value=0)

    @pytest.mark.parametrize("not_int", EXAMPLE_NOT_INT_VALUES)
    def test_init__max_value_invalid_type(self, not_int):
        """
        Check that 'IntegerVariable' raises TypeError during init if max_value is not int type.

        :param not_int: Value that is not int type.
        """
        with pytest.raises(TypeError):
            IntegerVariable.__init__(self=self.mock_decision_variable_object, min_value=0, max_value=not_int)

    @pytest.mark.parametrize("min_value, max_value", EXAMPLE_INT_VARIABLE_LIMITS)
    def test_generate_random_value(self, min_value, max_value):
        """
        Check that 'generate_random_value' returns 'generate_random_int' function output.

        :param min_value: Some int value.
        :param max_value: Int value that is greater than 'min_value'.
        """
        self.mock_decision_variable_object.min_value = min_value
        self.mock_decision_variable_object.max_value = max_value
        assert IntegerVariable.generate_random_value(self=self.mock_decision_variable_object) \
            is self.mock_generate_random_int.return_value
        self.mock_generate_random_int.assert_called_once_with(min_value, max_value)

    @pytest.mark.parametrize("min_value, max_value", EXAMPLE_INT_VARIABLE_LIMITS)
    def test_is_proper_value__low_boundary(self, min_value, max_value):
        self.mock_decision_variable_object.min_value = min_value
        self.mock_decision_variable_object.max_value = max_value
        assert IntegerVariable.is_proper_value(self.mock_decision_variable_object, value=min_value) is True

    @pytest.mark.parametrize("min_value, max_value", EXAMPLE_INT_VARIABLE_LIMITS)
    def test_is_proper_value__high_boundary(self, min_value, max_value):
        self.mock_decision_variable_object.min_value = min_value
        self.mock_decision_variable_object.max_value = max_value
        assert IntegerVariable.is_proper_value(self.mock_decision_variable_object, value=max_value) is True

    @pytest.mark.parametrize("min_value, max_value", EXAMPLE_INT_VARIABLE_LIMITS)
    def test_is_proper_value__outer_low_boundary(self, min_value, max_value):
        self.mock_decision_variable_object.min_value = min_value
        self.mock_decision_variable_object.max_value = max_value
        assert IntegerVariable.is_proper_value(self.mock_decision_variable_object, value=min_value-1) is False

    @pytest.mark.parametrize("min_value, max_value", EXAMPLE_INT_VARIABLE_LIMITS)
    def test_is_proper_value__outer_high_boundary(self, min_value, max_value):
        self.mock_decision_variable_object.min_value = min_value
        self.mock_decision_variable_object.max_value = max_value
        assert IntegerVariable.is_proper_value(self.mock_decision_variable_object, value=max_value+1) is False

    @pytest.mark.parametrize("min_value, max_value", EXAMPLE_INT_VARIABLE_LIMITS)
    @pytest.mark.parametrize("not_int", EXAMPLE_NOT_INT_VALUES)
    def test_is_proper_value__invalid_type(self, min_value, max_value, not_int):
        self.mock_decision_variable_object.min_value = min_value
        self.mock_decision_variable_object.max_value = max_value
        assert IntegerVariable.is_proper_value(self.mock_decision_variable_object, value=not_int) is False

    @pytest.mark.parametrize("min_value, max_value", EXAMPLE_INT_VARIABLE_LIMITS)
    def test_get_log_data(self, min_value, max_value):
        self.mock_decision_variable_object.min_value = min_value
        self.mock_decision_variable_object.max_value = max_value
        log_data = IntegerVariable.get_log_data(self.mock_decision_variable_object)
        assert isinstance(log_data, dict)
        assert "min_value" in log_data
        assert "max_value" in log_data
        assert "type" in log_data


class TestDiscreteVariable:
    """
    Tests for 'DiscreteVariable' class and their methods.
    """
    SCRIPT_LOCATION = TestIntegerVariable.SCRIPT_LOCATION
    EXAMPLE_LIMITS_AND_STEP = [(0, 1, 0.25), (-100, 0.1, 11), (0.1, 100, 11), (-0.1, 0.1, 0.005)]
    EXAMPLE_NOT_INT_FLOAT_VALUES = [None, "123", [0]]

    def setup(self):
        self.mock_decision_variable_object = Mock(spec=DiscreteVariable)
        # patching
        self._patcher_generate_random_int = patch(f"{self.SCRIPT_LOCATION}.generate_random_int")
        self.mock_generate_random_int = self._patcher_generate_random_int.start()

    def teardown(self):
        self._patcher_generate_random_int.stop()

    @pytest.mark.parametrize("min_value, max_value, step", EXAMPLE_LIMITS_AND_STEP)
    def test_init__valid_params(self, min_value, max_value, step):
        """
        Check that 'DiscreteVariable' can be initialized with proper values.

        :param min_value: Some int or float value.
        :param max_value: Int or float value that is greater than 'min_value'.
        :param step: Int or float value that is greater than zero.
        """
        DiscreteVariable.__init__(self=self.mock_decision_variable_object,
                                  min_value=min_value, max_value=max_value, step=step)
        assert self.mock_decision_variable_object.min_value == min_value
        assert self.mock_decision_variable_object.max_value == max_value
        assert self.mock_decision_variable_object.step == step

    @pytest.mark.parametrize("max_value, min_value, step", EXAMPLE_LIMITS_AND_STEP)
    def test_init__min_greater_than_max(self, min_value, max_value, step):
        """
        Check that 'DiscreteVariable' raises ValueError during init if min_value > max_value.

        :param min_value: Some int or float value.
        :param max_value: Int or float that is less than 'min_value'.
        :param step: Int or float value that is greater than zero.
        """
        with pytest.raises(ValueError):
            DiscreteVariable.__init__(self=self.mock_decision_variable_object,
                                      min_value=min_value, max_value=max_value, step=step)

    @pytest.mark.parametrize("not_int_float", EXAMPLE_NOT_INT_FLOAT_VALUES)
    def test_init__min_value_invalid_type(self, not_int_float):
        """
        Check that 'DiscreteVariable' raises TypeError during init if min_value is not int or float type.

        :param not_int_float: Value that is not int or float type.
        """
        with pytest.raises(TypeError):
            DiscreteVariable.__init__(self=self.mock_decision_variable_object,
                                      min_value=not_int_float, max_value=0, step=1)

    @pytest.mark.parametrize("not_int_float", EXAMPLE_NOT_INT_FLOAT_VALUES)
    def test_init__max_value_invalid_type(self, not_int_float):
        """
        Check that 'DiscreteVariable' raises TypeError during init if max_value is not int or float type.

        :param not_int_float: Value that is not int or float type.
        """
        with pytest.raises(TypeError):
            DiscreteVariable.__init__(self=self.mock_decision_variable_object,
                                      min_value=0, max_value=not_int_float, step=1)

    @pytest.mark.parametrize("not_int_float", EXAMPLE_NOT_INT_FLOAT_VALUES)
    def test_init__step_invalid_type(self, not_int_float):
        """
        Check that 'DiscreteVariable' raises TypeError during init if step is not int or float type.

        :param not_int_float: Value that is not int or float type.
        """
        with pytest.raises(TypeError):
            DiscreteVariable.__init__(self=self.mock_decision_variable_object,
                                      min_value=0, max_value=1, step=not_int_float)

    @pytest.mark.parametrize("min_value, max_value, step", EXAMPLE_LIMITS_AND_STEP)
    def test_generate_random_value(self, min_value, max_value, step):
        """
        Check that 'generate_random_value' uses 'generate_random_int' function output.

        :param min_value: Some int or float value.
        :param max_value: Int or float that is less than 'min_value'.
        :param step: Int or float value that is greater than zero.
        """
        self.mock_generate_random_int.return_value = 0
        self.mock_decision_variable_object.min_value = min_value
        self.mock_decision_variable_object.max_value = max_value
        self.mock_decision_variable_object.step = step
        self.mock_decision_variable_object._max_rand = (max_value - min_value) // step
        ret_value = DiscreteVariable.generate_random_value(self=self.mock_decision_variable_object)
        assert isinstance(ret_value, (int, float)) and min_value <= ret_value <= max_value
        self.mock_generate_random_int.assert_called_once_with(0, self.mock_decision_variable_object._max_rand)

    @pytest.mark.parametrize("min_value, max_value, step", EXAMPLE_LIMITS_AND_STEP)
    def test_is_proper_value__low_boundary(self, min_value, max_value, step):
        self.mock_decision_variable_object.min_value = min_value
        self.mock_decision_variable_object.max_value = max_value
        self.mock_decision_variable_object.step = step
        assert DiscreteVariable.is_proper_value(self.mock_decision_variable_object, value=min_value) is True

    @pytest.mark.parametrize("min_value, max_value, step", EXAMPLE_LIMITS_AND_STEP)
    def test_is_proper_value__high_boundary(self, min_value, max_value, step):
        self.mock_decision_variable_object.min_value = min_value
        self.mock_decision_variable_object.max_value = max_value
        self.mock_decision_variable_object.step = step
        max_valid_value = max_value - ((max_value - min_value) % step)
        assert DiscreteVariable.is_proper_value(self.mock_decision_variable_object, value=max_valid_value) is True

    @pytest.mark.parametrize("min_value, max_value, step", EXAMPLE_LIMITS_AND_STEP)
    def test_is_proper_value__outer_low_boundary(self, min_value, max_value, step):
        self.mock_decision_variable_object.min_value = min_value
        self.mock_decision_variable_object.max_value = max_value
        self.mock_decision_variable_object.step = step
        assert DiscreteVariable.is_proper_value(self.mock_decision_variable_object, value=min_value-step) is False

    @pytest.mark.parametrize("min_value, max_value, step", EXAMPLE_LIMITS_AND_STEP)
    def test_is_proper_value__outer_high_boundary(self, min_value, max_value, step):
        self.mock_decision_variable_object.min_value = min_value
        self.mock_decision_variable_object.max_value = max_value
        self.mock_decision_variable_object.step = step
        max_valid_value = max_value - ((max_value - min_value) % step)
        assert DiscreteVariable.is_proper_value(self.mock_decision_variable_object, value=max_valid_value+step) is False

    @pytest.mark.parametrize("min_value, max_value, step", EXAMPLE_LIMITS_AND_STEP)
    def test_is_proper_value__invalid_value_in_range(self, min_value, max_value, step):
        self.mock_decision_variable_object.min_value = min_value
        self.mock_decision_variable_object.max_value = max_value
        self.mock_decision_variable_object.step = step
        invalid_value_in_range = min_value + 0.5*step
        assert DiscreteVariable.is_proper_value(self.mock_decision_variable_object, value=invalid_value_in_range) is False

    @pytest.mark.parametrize("min_value, max_value, step", EXAMPLE_LIMITS_AND_STEP)
    @pytest.mark.parametrize("not_int_float", EXAMPLE_NOT_INT_FLOAT_VALUES)
    def test_is_proper_value__invalid_type(self, min_value, max_value, step, not_int_float):
        self.mock_decision_variable_object.min_value = min_value
        self.mock_decision_variable_object.max_value = max_value
        self.mock_decision_variable_object.step = step
        assert DiscreteVariable.is_proper_value(self.mock_decision_variable_object, value=not_int_float) is False

    @pytest.mark.parametrize("min_value, max_value, step", EXAMPLE_LIMITS_AND_STEP)
    def test_get_log_data(self, min_value, max_value, step):
        self.mock_decision_variable_object.min_value = min_value
        self.mock_decision_variable_object.max_value = max_value
        self.mock_decision_variable_object.step = step
        log_data = DiscreteVariable.get_log_data(self.mock_decision_variable_object)
        assert isinstance(log_data, dict)
        assert "min_value" in log_data
        assert "max_value" in log_data
        assert "step" in log_data
        assert "type" in log_data


class TestFloatVariable:
    """
    Tests for 'FloatVariable' class and their methods.
    """
    SCRIPT_LOCATION = TestIntegerVariable.SCRIPT_LOCATION
    EXAMPLE_FLOAT_VARIABLE_LIMITS = [(0., 1.), (-1.555, 1.7653423), (0.0001, 0.0002), (-100000., 100000.)]
    EXAMPLE_NOT_FLOAT_VALUES = [0, None, "123", [0.]]

    def setup(self):
        self.mock_decision_variable_object = Mock(spec=FloatVariable)
        # patching
        self._patcher_generate_random_float = patch(f"{self.SCRIPT_LOCATION}.generate_random_float")
        self.mock_generate_random_float = self._patcher_generate_random_float.start()

    def teardown(self):
        self._patcher_generate_random_float.stop()

    @pytest.mark.parametrize("min_value, max_value", EXAMPLE_FLOAT_VARIABLE_LIMITS)
    def test_init__valid_params(self, min_value, max_value):
        """
        Check that 'FloatVariable' can be initialized with proper float values.

        :param min_value: Some float value.
        :param max_value: Float value that is greater than 'min_value'.
        """
        FloatVariable.__init__(self=self.mock_decision_variable_object, min_value=min_value, max_value=max_value)
        assert self.mock_decision_variable_object.min_value == min_value
        assert self.mock_decision_variable_object.max_value == max_value

    @pytest.mark.parametrize("max_value, min_value", EXAMPLE_FLOAT_VARIABLE_LIMITS)
    def test_init__min_greater_than_max(self, min_value, max_value):
        """
        Check that 'FloatVariable' raises ValueError during init if min_value > max_value.

        :param min_value: Some float value.
        :param max_value: Float value that is less than 'min_value'.
        """
        with pytest.raises(ValueError):
            FloatVariable.__init__(self=self.mock_decision_variable_object, min_value=min_value, max_value=max_value)

    @pytest.mark.parametrize("not_float", EXAMPLE_NOT_FLOAT_VALUES)
    def test_init__min_value_invalid_type(self, not_float):
        """
        Check that 'FloatVariable' raises TypeError during init if min_value is not float type.

        :param not_float: Value that is not float type.
        """
        with pytest.raises(TypeError):
            FloatVariable.__init__(self=self.mock_decision_variable_object, min_value=not_float, max_value=0.)

    @pytest.mark.parametrize("not_float", EXAMPLE_NOT_FLOAT_VALUES)
    def test_init__max_value_invalid_type(self, not_float):
        """
        Check that 'FloatVariable' raises TypeError during init if max_value is not float type.

        :param not_float: Value that is not float type.
        """
        with pytest.raises(TypeError):
            FloatVariable.__init__(self=self.mock_decision_variable_object, min_value=0., max_value=not_float)

    @pytest.mark.parametrize("min_value, max_value", EXAMPLE_FLOAT_VARIABLE_LIMITS)
    def test_generate_random_value(self, min_value, max_value):
        """
        Check that 'generate_random_value' returns 'generate_random_int' function output.

        :param min_value: Some float value.
        :param max_value: Float value that is less than 'min_value'.
        """
        self.mock_decision_variable_object.min_value = min_value
        self.mock_decision_variable_object.max_value = max_value
        assert FloatVariable.generate_random_value(self=self.mock_decision_variable_object) \
            is self.mock_generate_random_float.return_value
        self.mock_generate_random_float.assert_called_once_with(min_value, max_value)

    @pytest.mark.parametrize("min_value, max_value", EXAMPLE_FLOAT_VARIABLE_LIMITS)
    def test_is_proper_value__low_boundary(self, min_value, max_value):
        self.mock_decision_variable_object.min_value = min_value
        self.mock_decision_variable_object.max_value = max_value
        assert FloatVariable.is_proper_value(self.mock_decision_variable_object, value=min_value) is True

    @pytest.mark.parametrize("min_value, max_value", EXAMPLE_FLOAT_VARIABLE_LIMITS)
    def test_is_proper_value__high_boundary(self, min_value, max_value):
        self.mock_decision_variable_object.min_value = min_value
        self.mock_decision_variable_object.max_value = max_value
        assert FloatVariable.is_proper_value(self.mock_decision_variable_object, value=max_value) is True

    @pytest.mark.parametrize("min_value, max_value", EXAMPLE_FLOAT_VARIABLE_LIMITS)
    def test_is_proper_value__outer_low_boundary(self, min_value, max_value):
        self.mock_decision_variable_object.min_value = min_value
        self.mock_decision_variable_object.max_value = max_value
        assert FloatVariable.is_proper_value(self.mock_decision_variable_object, value=min_value-0.000001) is False

    @pytest.mark.parametrize("min_value, max_value", EXAMPLE_FLOAT_VARIABLE_LIMITS)
    def test_is_proper_value__outer_high_boundary(self, min_value, max_value):
        self.mock_decision_variable_object.min_value = min_value
        self.mock_decision_variable_object.max_value = max_value
        assert FloatVariable.is_proper_value(self.mock_decision_variable_object, value=max_value+0.000001) is False

    @pytest.mark.parametrize("min_value, max_value", EXAMPLE_FLOAT_VARIABLE_LIMITS)
    @pytest.mark.parametrize("not_float", EXAMPLE_NOT_FLOAT_VALUES)
    def test_is_proper_value__invalid_type(self, min_value, max_value, not_float):
        self.mock_decision_variable_object.min_value = min_value
        self.mock_decision_variable_object.max_value = max_value
        assert FloatVariable.is_proper_value(self.mock_decision_variable_object, value=not_float) is False

    @pytest.mark.parametrize("min_value, max_value", EXAMPLE_FLOAT_VARIABLE_LIMITS)
    def test_get_log_data(self, min_value, max_value):
        self.mock_decision_variable_object.min_value = min_value
        self.mock_decision_variable_object.max_value = max_value
        log_data = FloatVariable.get_log_data(self.mock_decision_variable_object)
        assert isinstance(log_data, dict)
        assert "min_value" in log_data
        assert "max_value" in log_data
        assert "type" in log_data


class TestChoiceVariable:
    """
    Tests for 'ChoiceVariable' class and their methods.
    """
    SCRIPT_LOCATION = TestIntegerVariable.SCRIPT_LOCATION
    EXAMPLE_NOT_ITERABLE_VALUES = [1, None, 0.]
    EXAMPLE_VALUES_POOLS = [range(10), {"black", "red", "white", "blue", "green"}, "abcdef"]
    EXAMPLE_VALUES_NOT_IN_POOL = [-1, None, 0.12345]

    def setup(self):
        self.mock_decision_variable_object = Mock(spec=ChoiceVariable)
        # patching
        self._patcher_choose_random_value = patch(f"{self.SCRIPT_LOCATION}.choose_random_value")
        self.mock_choose_random_value = self._patcher_choose_random_value.start()

    def teardown(self):
        self._patcher_choose_random_value.stop()

    @pytest.mark.parametrize("values_pool", EXAMPLE_VALUES_POOLS)
    def test_init__valid_params(self, values_pool):
        """
        Check that 'ChoiceVariable' can be initialized with proper container value.

        :param values_pool: Some iterable with possible values to pick.
        """
        ChoiceVariable.__init__(self=self.mock_decision_variable_object, possible_values=values_pool)
        assert isinstance(self.mock_decision_variable_object.possible_values, set)
        assert all([value in self.mock_decision_variable_object.possible_values for value in values_pool])

    @pytest.mark.parametrize("not_iterable", EXAMPLE_NOT_ITERABLE_VALUES)
    def test_init__invalid_param_type(self, not_iterable):
        """
        Check that 'ChoiceVariable' cannot be initialized with value that cannot be converted to set.

        :param not_iterable: Value that is not iterable and cannot be converted to set.
        """
        with pytest.raises(TypeError):
            ChoiceVariable.__init__(self=self.mock_decision_variable_object, possible_values=not_iterable)

    @pytest.mark.parametrize("values_pool", EXAMPLE_VALUES_POOLS)
    def test_generate_random_value(self, values_pool):
        """
        Check that 'generate_random_value' return value provided by 'choose_random_value'.

        :param values_pool: Some iterable with possible values to pick.
        """
        self.mock_decision_variable_object.possible_values = set(values_pool)
        assert ChoiceVariable.generate_random_value(self.mock_decision_variable_object) \
            == self.mock_choose_random_value.return_value
        self.mock_choose_random_value.assert_called_once_with(set(values_pool))

    @pytest.mark.parametrize("values_pool", EXAMPLE_VALUES_POOLS)
    def test_is_proper_value__valid(self, values_pool):
        self.mock_decision_variable_object.possible_values = set(values_pool)
        assert all([ChoiceVariable.is_proper_value(self=self.mock_decision_variable_object, value=value) is True
                    for value in values_pool])

    @pytest.mark.parametrize("values_pool", EXAMPLE_VALUES_POOLS)
    @pytest.mark.parametrize("not_in_pool", EXAMPLE_VALUES_NOT_IN_POOL)
    def test_is_proper_value__invalid(self, values_pool, not_in_pool):
        self.mock_decision_variable_object.possible_values = set(values_pool)
        assert ChoiceVariable.is_proper_value(self=self.mock_decision_variable_object, value=not_in_pool) is False

    @pytest.mark.parametrize("values_pool", EXAMPLE_VALUES_POOLS)
    def test_get_log_data(self, values_pool):
        self.mock_decision_variable_object.possible_values = set(values_pool)
        log_data = ChoiceVariable.get_log_data(self.mock_decision_variable_object)
        assert isinstance(log_data, dict)
        assert "possible_values" in log_data
        assert "type" in log_data
