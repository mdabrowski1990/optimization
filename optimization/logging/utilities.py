from typing import Callable
import inspect


__all__ = ["log_function_code"]


def log_function_code(func_to_log: Callable) -> str:
    """
    Extracts function code to 'str' so it can be logged to external file.

    :param func_to_log: Function which code to be extracted.

    :return: Code of the function.
    """
    # todo: test
    function_definition = inspect.getsource(func_to_log)
    # todo: remove prefix '... lambda **decision_variables_values:'
    # todo: remove suffix ', ...'
    return function_definition
