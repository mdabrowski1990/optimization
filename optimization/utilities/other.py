"""Other useful functions."""

__all__ = ["binary_search"]


from typing import Any, List, Optional


def binary_search(sorted_list: List[Any], value: Any, list_size: Optional[int] = None) -> int:  # todo: remove?
    """
    Performs binary search.

    :param sorted_list: List with values in ascending order.
    :param value: Value for which position in the list is searched.
    :param list_size: Size of the sorted_list.

    :return: Index i of element such as:
        sorted_list[i-1] < value <= sorted_list[i]
    """
    if value < sorted_list[0] or value > sorted_list[-1]:
        raise ValueError(f"Parameter 'value' is out of range. Expected: {sorted_list[0]} <= value <= {sorted_list[-1]}."
                         f" Actual value: {value}.")
    i_low = 0
    i_high = len(sorted_list) - 1 if list_size is None else list_size - 1
    while i_low < i_high:
        i_mid = (i_low + i_high) // 2
        if sorted_list[i_mid] < value:
            i_low = i_mid + 1
        else:
            i_high = i_mid
    return i_high
