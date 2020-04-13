import pytest
from random import choices, uniform, randint
from functools import reduce
from string import printable

from optimization.optimization_problem.decision_variables import IntegerVariable, FloatVariable, ChoiceVariable


# ------------------------------------------------------ General ----------------------------------------------------- #


EXAMPLE_VALUE_TYPES = {int, float, str, bytes, list, tuple, set, dict, "function", None}


@pytest.fixture
def random_text():
    return "".join(choices(population=printable, k=randint(4, 20)))


@pytest.fixture
def example_value(request, random_text):
    if request.param == int:
        return randint(-1000, 1000)
    elif request.param == float:
        return uniform(-1000, 1000)
    elif request.param == str:
        return random_text
    elif request.param == bytes:
        return b"something"
    elif request.param == list:
        return []
    elif request.param == tuple:
        return ()
    elif request.param == set:
        return set()
    elif request.param == dict:
        return {}
    elif request.param == "function":
        return lambda a, b: a+b
    else:
        return None


# ------------------------------------------------ Decision Variables ------------------------------------------------ #


EXAMPLE_INT_VARIABLE_LIMITS = [(0, 10), (-100, 100), sorted(choices(population=range(-1000, 1001), k=2))]
NUMBER_OF_INT_VARIABLE_EXAMPLES = len(EXAMPLE_INT_VARIABLE_LIMITS)


@pytest.fixture
def example_integer_decision_variable(request):
    return IntegerVariable(*EXAMPLE_INT_VARIABLE_LIMITS[request.param])


EXAMPLE_FLOAT_VARIABLE_LIMITS = [(0., 1.), (-100., 100.), sorted([uniform(-1000, 1000), uniform(-1000, 1000)])]
NUMBER_OF_FLOAT_VARIABLE_EXAMPLES = len(EXAMPLE_FLOAT_VARIABLE_LIMITS)


@pytest.fixture
def example_float_decision_variable(request):
    return FloatVariable(*EXAMPLE_FLOAT_VARIABLE_LIMITS[request.param])


EXAMPLE_CHOICE_VARIABLE_OPTIONS = [{"white", "black", "red", "yellow", "blue"}, {1, 2.3, 4, 5.67, 8, 9.101112, 13},
                                   set(range(10, 230, 13))]
NUMBER_OF_CHOICE_VARIABLE_EXAMPLES = len(EXAMPLE_CHOICE_VARIABLE_OPTIONS)


@pytest.fixture
def example_choice_decision_variable(request):
    return ChoiceVariable(EXAMPLE_CHOICE_VARIABLE_OPTIONS[request.param])


NUMBER_OF_DECISION_VARIABLE_EXAMPLES = NUMBER_OF_INT_VARIABLE_EXAMPLES + NUMBER_OF_FLOAT_VARIABLE_EXAMPLES \
                                       + NUMBER_OF_CHOICE_VARIABLE_EXAMPLES


@pytest.fixture
def example_decision_variable(request):
    # int var
    if request.param < NUMBER_OF_INT_VARIABLE_EXAMPLES:
        return IntegerVariable(*EXAMPLE_INT_VARIABLE_LIMITS[request.param])
    else:
        request.param -= NUMBER_OF_INT_VARIABLE_EXAMPLES
    # float var
    if request.param < NUMBER_OF_FLOAT_VARIABLE_EXAMPLES:
        return FloatVariable(*EXAMPLE_FLOAT_VARIABLE_LIMITS[request.param])
    else:
        request.param -= NUMBER_OF_FLOAT_VARIABLE_EXAMPLES
    # choice var
    return ChoiceVariable(EXAMPLE_CHOICE_VARIABLE_OPTIONS[request.param])


DECISION_VARIABLES_GROUPS = ["integers", "floats", "choices", "all"]
EXAMPLE_DECISION_VARIABLES_VALUES = [{"x0": 0}, {"x": 123.456, "y": 6.231, "z": 9.65}, {"abc": -1, "def": -2.3}]


@pytest.fixture
def example_decision_variables(request):
    if request.param == "integers":
        return {f"x{i}": IntegerVariable(*limits) for i, limits in enumerate(EXAMPLE_INT_VARIABLE_LIMITS)}
    elif request.param == "floats":
        return {f"x{i}": FloatVariable(*limits) for i, limits in enumerate(EXAMPLE_FLOAT_VARIABLE_LIMITS)}
    elif request.param == "choices":
        return {f"x{i}": ChoiceVariable(values_pool)
                for i, values_pool in enumerate(EXAMPLE_CHOICE_VARIABLE_OPTIONS, 1)}
    elif request.param == "all":
        return {
            "x0": IntegerVariable(*EXAMPLE_INT_VARIABLE_LIMITS[0]),
            "x1": FloatVariable(*EXAMPLE_FLOAT_VARIABLE_LIMITS[0]),
            "x2": ChoiceVariable(EXAMPLE_CHOICE_VARIABLE_OPTIONS[0]),
        }
    else:
        raise ValueError


# ---------------------------------------------------- Constraints --------------------------------------------------- #


EXAMPLE_CONSTRAINTS = [
    # constraints # 1
    {
        "c0": lambda **decision_variables_values:
        0 if decision_variables_values["x0"] <= decision_variables_values["x1"]
        else decision_variables_values["x0"] - decision_variables_values["x1"],
        "c1": lambda **decision_variables_values:
        abs(decision_variables_values["x2"] // 10 - int(decision_variables_values["x1"])),
    },
    # constraints # 2
    {
        "c0": lambda **decision_variables_values:
        0 if decision_variables_values["x0"] + 1.5 <= decision_variables_values["x2"]
        else decision_variables_values["x0"] - decision_variables_values["x2"] + 1.5,
        "c1": lambda **decision_variables_values:
        int(decision_variables_values["x0"] == decision_variables_values["x1"] == decision_variables_values["x2"])
    }
]
NUMBER_OF_CONSTRAINTS_EXAMPLES = len(EXAMPLE_CONSTRAINTS)
EXAMPLE_CONSTRAINTS_VALUES = [{}, {"c0": 0, "c1": 1, "c2": 0.1234, "c3": -987.5}, {"abc": -3, "def": 3}]


@pytest.fixture
def example_constraints(request):
    return EXAMPLE_CONSTRAINTS[request.param]


# ------------------------------------------------------ Penalty ----------------------------------------------------- #


EXAMPLE_PENALTY_FUNCTIONS = [
    lambda **constraints_values: sum(constraints_values.values())*10000.,
    lambda **constraints_values: float("inf") if any(constraints_values.values()) else 0,
]
NUMBER_OF_PENALTY_FUNCTIONS_EXAMPLES = len(EXAMPLE_PENALTY_FUNCTIONS)
EXAMPLE_PENALTY_FUNCTION_VALUES = [-0.00123, 0., 12345.6789]


@pytest.fixture
def example_penalty_function(request):
    return EXAMPLE_PENALTY_FUNCTIONS[request.param]


# ----------------------------------------------------- Objective ---------------------------------------------------- #


EXAMPLE_OBJECTIVE_FUNCTIONS = [
    lambda **decision_variables_values: sum(decision_variables_values.values()),
    lambda **decision_variables_values: reduce(lambda a, b: a*b, decision_variables_values.values(), 1)
]
NUMBER_OF_OBJECTIVE_FUNCTIONS_EXAMPLES = len(EXAMPLE_OBJECTIVE_FUNCTIONS)
EXAMPLE_OBJECTIVE_FUNCTION_VALUES = [-0.00321, 0., 98765.4321]


@pytest.fixture
def example_objective_function(request):
    return EXAMPLE_OBJECTIVE_FUNCTIONS[request.param]

