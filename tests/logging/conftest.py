import pytest


EXAMPLE_FUNCTION_TYPES = ["one line function", "indented function", "indented function with multi-line arguments",
                          "lambda", "named lambda", "indented multi-line lambda"]


def one_line_function(a: int, b: float, c: list) -> bool:
    return a in c or b in c


named_lambda = lambda x1, x2, x3: min([x1, x2, x3])


@pytest.fixture
def example_function(request):
    if request.param == "one line function":
        return one_line_function
    if request.param == "indented function":
        def indented_function(p1, p2, p3):
            return p1 + p2 + p3
        return indented_function
    if request.param == "indented function with multi-line arguments":
        def multi_line_function(some_long_argument1,
                                some_long_argument2,
                                some_long_argument3):
            """
            Some docstring

            :param some_long_argument1: wondering
            :param some_long_argument2: no idea
            :param some_long_argument3: whatever is here it does not matter

            :return: Who knows
            """
            some_long_argument1.update(some_long_argument2)
            some_long_argument3.set(some_long_argument3)
            return some_long_argument1 != some_long_argument3 != some_long_argument2
        return multi_line_function
    if request.param == "lambda":
        return lambda x, y: x + y
    if request.param == "named lambda":
        return named_lambda
    if request.param == "indented multi-line lambda":
        return lambda some_long_argument1, some_long_argument2, some_long_argument3, some_long_argument4: \
            some_long_argument1 + some_long_argument2 - some_long_argument3*some_long_argument4
    raise ValueError("Dunno what you want from me :(")
