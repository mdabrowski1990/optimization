from typing import Callable
import inspect


__all__ = ["log_function_code"]


def log_function_code(func_to_log: Callable) -> str:
    """
    Extracts function code to 'str' so it can be logged to external file.

    :param func_to_log: Function which code to be extracted.

    :return: Code of the function.
    """
    # todo: test and make output easier to read
    function_definition = inspect.getsource(func_to_log)
    return function_definition.strip().replace("\n", " ").replace("\r", " ").replace("  ", " ").rstrip(",")
