import pytest

from optimization.utilities.other import binary_search


class TestFunctions:

    @pytest.mark.parametrize("sorted_list, value, expected_output", [
        (list(range(10)), 0, 0),
        (list(range(10)), 9, 9),
        (list(range(10)), 0.5, 1),
        (list(range(10)), 8.9, 9),
        ([0.1, 0.2, 0.5, 2.5, 6.5, 180, 180.1], 0.5, 2),
        ([0.1, 0.2, 0.5, 2.5, 6.5, 180, 180.1], 0.4999, 2),
        ([0.1, 0.2, 0.5, 2.5, 6.5, 180, 180.1], 0.2001, 2),
        ([0.1, 0.2, 0.5, 2.5, 6.5, 180, 180.1], 0.2, 1),
        ([0.1, 0.2, 0.5, 2.5, 6.5, 180, 180.1], 180.05, 6),
        ([0.1, 0.2, 0.5, 2.5, 6.5, 180, 180.1], 180.1, 6),
        ([0.1, 0.2, 0.5, 2.5, 6.5, 180, 180.1], 90, 5),
        ([0.1, 0.2, 0.5, 2.5, 6.5, 180, 180.1], 0.1, 0),
    ])
    def test_binary_search__without_list_size(self, sorted_list, value, expected_output):
        """
        Test 'binary_search' function output when 'list_size' is not given.

        :param sorted_list: Example value of 'sorted_list' parameter.
        :param value: Example value of 'value' parameter.
        :param expected_output: Expected output from 'binary_search' function.
        """
        assert binary_search(sorted_list=sorted_list, value=value) == expected_output

    @pytest.mark.parametrize("sorted_list, value, expected_output", [
        (list(range(10)), 0, 0),
        (list(range(10)), 9, 9),
        (list(range(10)), 0.5, 1),
        (list(range(10)), 8.9, 9),
        ([0.1, 0.2, 0.5, 2.5, 6.5, 180, 180.1], 0.5, 2),
        ([0.1, 0.2, 0.5, 2.5, 6.5, 180, 180.1], 0.4999, 2),
        ([0.1, 0.2, 0.5, 2.5, 6.5, 180, 180.1], 0.2001, 2),
        ([0.1, 0.2, 0.5, 2.5, 6.5, 180, 180.1], 0.2, 1),
        ([0.1, 0.2, 0.5, 2.5, 6.5, 180, 180.1], 180.05, 6),
        ([0.1, 0.2, 0.5, 2.5, 6.5, 180, 180.1], 180.1, 6),
        ([0.1, 0.2, 0.5, 2.5, 6.5, 180, 180.1], 90, 5),
        ([0.1, 0.2, 0.5, 2.5, 6.5, 180, 180.1], 0.1, 0),
    ])
    def test_binary_search__with_list_size(self, sorted_list, value, expected_output):
        """
        Test 'binary_search' function output when 'list_size' is given.

        :param sorted_list: Example value of 'sorted_list' parameter.
        :param value: Example value of 'value' parameter.
        :param expected_output: Expected output from 'binary_search' function.
        """
        assert binary_search(sorted_list=sorted_list, value=value, list_size=len(sorted_list)) == expected_output

    @pytest.mark.parametrize("sorted_list, value", [
        (list(range(10)), -0.0000001),
        (list(range(10)), -1),
        (list(range(10)), 9.00000001),
        (list(range(10)), 10),
        ([0.1, 0.2, 0.5, 2.5, 6.5, 180, 180.1], 0.09999999),
        ([0.1, 0.2, 0.5, 2.5, 6.5, 180, 180.1], 0),
        ([0.1, 0.2, 0.5, 2.5, 6.5, 180, 180.1], 180.10000001),
        ([0.1, 0.2, 0.5, 2.5, 6.5, 180, 180.1], 181),
    ])
    def test_binary_search__value_out_of_range(self, sorted_list, value):
        """
        Tests that 'binary_search' raises ValueError when 'value' is out of sorted_list range.

        :param sorted_list: Example value of 'sorted_list' parameter.
        :param value: Invalid (out of range) value of 'value' parameter.
        """
        with pytest.raises(ValueError):
            binary_search(sorted_list=sorted_list, value=value)
