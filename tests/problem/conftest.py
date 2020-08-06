import pytest
from collections import OrderedDict
from random import choice, randint
from functools import reduce

from optimization.problem.decision_variables import IntegerVariable, DiscreteVariable, FloatVariable, ChoiceVariable


# ------------------------------------------------ Decision Variables ------------------------------------------------ #


EXAMPLE_INTEGER_DECISION_VARIABLES_PARAMS = [(-10, 10), (0, 100), (-987, -876), (-1, 7), (-612, 729)]
EXAMPLE_DISCRETE_DECISION_VARIABLES_PARAMS = [(-10, 10, 0.5), (0, 100, 5), (-1.25, 2.5, 0.075), (-1, -0.1, 0.00001)]
EXAMPLE_FLOAT_DECISION_VARIABLES_PARAMS = [(-10., 10.), (0., 100.), (0.01, 0.02), (-0.00002, -0.00001)]
EXAMPLE_CHOICE_DECISION_VARIABLES_PARAMS = [
    {"white", "yellow", "pink", "red", "green", "blue", "brown", "gray", "black"},
    {"car", "bike", "bus", "plane", "ship", "space ship", "other"},
    {0.432, 1, 2.5, 6.25, 7.654, 85.213, 987654321, "infinity"},
]


def example_integer_decision_variable():
    return IntegerVariable(*choice(EXAMPLE_INTEGER_DECISION_VARIABLES_PARAMS))


def example_discrete_decision_variable():
    return DiscreteVariable(*choice(EXAMPLE_DISCRETE_DECISION_VARIABLES_PARAMS))


def example_float_decision_variable():
    return FloatVariable(*choice(EXAMPLE_FLOAT_DECISION_VARIABLES_PARAMS))


def example_choice_decision_variable():
    return ChoiceVariable(choice(EXAMPLE_CHOICE_DECISION_VARIABLES_PARAMS))


def example_decision_variable():
    return choice([example_integer_decision_variable, example_discrete_decision_variable,
                   example_float_decision_variable, example_choice_decision_variable])()


@pytest.fixture
def example_decision_variables():
    """:return: Example value of decision_variables OrderedDict."""
    variables_number = randint(3, 5)
    return OrderedDict([(f"x{i}", example_decision_variable()) for i in range(variables_number)])


@pytest.fixture()
def invalid_decision_variables__keys_not_str():
    """
    :return: OrderedDict in similar format to decision_variables but containing invalid data (keys are not str type).
    """
    return OrderedDict([(1, example_integer_decision_variable()), (None, example_float_decision_variable())])


@pytest.fixture()
def invalid_decision_variables__values_not_decision_variable():
    """
    :return: OrderedDict in similar format to decision_variables but containing invalid data
    (values are not DecisionVariable type).
    """
    return OrderedDict([("a", 1), ("b", 2.)])


# ---------------------------------------------------- Constraints --------------------------------------------------- #


def example_constraint():
    return choice([
        lambda **vars_values: int(vars_values["x0"] <= vars_values["x1"]),
        lambda **vars_values: int(isinstance(vars_values["x0"], int)),
        lambda **vars_values: int(isinstance(vars_values["x1"], str)),
        lambda **vars_values: int(isinstance(vars_values["x2"], float)),
        lambda **vars_values: abs(vars_values["x1"] - vars_values["x2"])
        if isinstance(vars_values["x1"], (float, int)) and isinstance(vars_values["x2"], (float, int)) else 0,
    ])


@pytest.fixture
def example_constraints():
    """:return: Example value of constrains dict."""
    constraints_number = randint(2, 5)
    return {f"c{i}": example_constraint() for i in range(constraints_number)}


@pytest.fixture
def invalid_constraints__keys_not_str():
    """:return: Dict in similar format to constrains but containing invalid data (keys are not str type)."""
    return {1: lambda **x: 0, None: lambda **x: 1.}


@pytest.fixture
def invalid_constraints__values_not_callable():
    """:return: Dict in similar format to constrains but containing invalid data (values are not callable)."""
    return {"c1": 0, "c2": 1.}


# ------------------------------------------------------ Penalty ----------------------------------------------------- #


@pytest.fixture
def example_penalty_function():
    return choice([
        lambda **constraints_values: 10. * sum(constraints_values.values()),
        lambda **constraints_values: float("inf") if any(constraints_values.values()) else 0,
        lambda **constraints_values: 0
    ])


# ----------------------------------------------------- Objective ---------------------------------------------------- #


@pytest.fixture
def example_objective_function():
    return choice([
        lambda **decision_variables_values: sum(decision_variables_values.values()),
        lambda **decision_variables_values: reduce(lambda a, b: a*b, decision_variables_values.values(), 1)
    ])


