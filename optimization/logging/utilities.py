from typing import Callable
import inspect


def log_function_code(func_to_log: Callable) -> str:
    # todo: description
    # todo: check
    function_definition = inspect.getsource(func_to_log)
    # todo: remove prefix '... lambda **decision_variables_values:'
    # todo: remove suffix ', ...'
    return function_definition
