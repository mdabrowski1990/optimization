"""Module with definition of mathematically described problem to be optimized."""

__all__ = ["OptimizationType", "OptimizationProblem"]


from typing import Union, Callable, Dict
from typing import OrderedDict as OrderedDictTyping
from collections import OrderedDict
from enum import Enum

from .decision_variables import DecisionVariable
from optimization.logging import log_function_code


class OptimizationType(Enum):
    """Enum with types of optimization problems."""

    Maximize = "Maximize"
    Minimize = "Minimize"


class OptimizationProblem:
    """Class for defining the problem for which optimal solution to be searched."""

    def __init__(self,
                 decision_variables: OrderedDictTyping[str, DecisionVariable],  # type: ignore
                 constraints: Dict[str, Callable],
                 penalty_function: Callable,
                 objective_function: Callable,
                 optimization_type: Union[OptimizationType, str]) -> None:
        """
        Definition of optimization problem.

        :param decision_variables: OrderedDict (for some algorithms order of variables is relevant) with decision
            variables definitions.
        :param constraints: Dictionary with constrains definitions.
        :param penalty_function: Function that calculates penalty value of the solution
            (used if constrains are not meet).
        :param objective_function: Function that calculates objective value of the solution (does not include penalty).
        :param optimization_type: Type of optimization problem (either searching for minimal or maximal value).

        :raise TypeError: For some parameter a value has incorrect type.
        :raise ValueError: For some parameter a value is incorrect.
        """
        # check: decision_variables
        if not isinstance(decision_variables, OrderedDict):
            raise TypeError(f"Parameter 'decision_variables' is not OrderDict type. "
                            f"Actual value: {decision_variables}.")
        if any([not isinstance(key, str) for key in decision_variables.keys()]):  # type: ignore
            raise ValueError(f"Some keys of 'decision_variables' are not str type. "  # type: ignore
                             f"Keys: {list(decision_variables.keys())}.")
        if any([not isinstance(value, DecisionVariable) for value in decision_variables.values()]):  # type: ignore
            raise ValueError(f"Some values of 'decision_variables' are not DecisionVariable type. "  # type: ignore
                             f"Values: {list(decision_variables.values())}.")
        # check: constraints
        if not isinstance(constraints, dict):
            raise TypeError(f"Parameter 'constraints' is not dict type. Actual value: {constraints}.")
        if any([not isinstance(key, str) for key in constraints.keys()]):
            raise ValueError(f"Some keys of 'constraints' are not str type. Keys: {list(constraints.keys())}.")
        if any([not callable(value) for value in constraints.values()]):
            raise ValueError(f"Some values of 'constraints' are not callable. Values: {list(constraints.values())}.")
        # check: penalty_function
        if not callable(penalty_function):
            raise TypeError(f"Parameter 'penalty_function' is not callable. Actual value: {penalty_function}.")
        # check: objective_function
        if not callable(objective_function):
            raise TypeError(f"Parameter 'objective_function' is not callable. Actual value: {objective_function}.")
        # check and set value: optimization_type
        if isinstance(optimization_type, OptimizationType):
            self.optimization_type = optimization_type
        elif isinstance(optimization_type, str):
            self.optimization_type = OptimizationType[optimization_type]
        else:
            raise TypeError(f"Parameter 'optimization_type' is not str or OptimizationType type. "
                            f"Actual value: {OptimizationType}.")
        # set other values
        self.decision_variables = decision_variables
        self.constraints = constraints
        self.penalty_function = penalty_function
        self.objective_function = objective_function

    def get_log_data(self) -> Dict[str, Union[str, dict, list]]:
        """
        Gets data for logging purposes.

        :return: Dictionary with this Optimization Problem crucial data.
        """
        return {
            "optimization_type": self.optimization_type.value,
            "decision_variables": [
                {"name": name, "definition": decision_var.get_log_data()}
                for name, decision_var in self.decision_variables.items()  # type: ignore
            ],
            "constraints": {name: log_function_code(constraint) for name, constraint in self.constraints.items()},
            "penalty_function": log_function_code(self.penalty_function),
            "objective_function": log_function_code(self.objective_function),
        }
