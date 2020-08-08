"""Helper functions for logging purposes."""

__all__ = ["log_function_code"]


from typing import Callable
import inspect


def log_function_code(func_to_log: Callable) -> str:
    """
    Extracts function code into str.

    It is used for preparing functions code to be logged into external files.

    :param func_to_log: Function object for which code to be extracted.

    :return: Code of the function.
    """
    if not callable(func_to_log):
        TypeError(f"Parameter 'func_to_log' is not function. Actual value: {func_to_log}.")
    function_definition = inspect.getsource(func_to_log)
    if function_definition.startswith("return "):
        function_definition = function_definition[7:]
    return repr(function_definition.strip())
