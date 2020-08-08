import pytest

from .conftest import EXAMPLE_FUNCTION_TYPES

from optimization.logging.utilities import log_function_code


class TestFunctions:
    """Tests of functions located in logging.utilities."""

    @pytest.mark.parametrize("not_a_function", [None, 1, 43.3])
    def test_log_function_code__wrong_type(self, not_a_function):
        """Test 'log_function_code' raises TypeError when input is not a callable type."""
        with pytest.raises(TypeError):
            log_function_code(not_a_function)

    @pytest.mark.parametrize("example_function", EXAMPLE_FUNCTION_TYPES, indirect=True)
    def test_log_function_code__valid(self, example_function):
        """
        Test 'log_function_code' return str when proper function is provided.

        :param example_function: Example function object.
        """
        function_code = log_function_code(example_function)
        assert isinstance(function_code, str)
        # todo: how to check that 'function_code' makes sense? exec(function_code) and then what?
