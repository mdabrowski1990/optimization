"""Optimization problem solution implementation."""

__all__ = ["AbstractSolution"]


from typing import Any, Union, Dict
from abc import ABC, abstractmethod
from collections import OrderedDict

from .problem import OptimizationProblem, OptimizationType


class AbstractSolution(ABC):
    """Abstract definition of optimization problem solution."""

    @property
    @abstractmethod
    def optimization_problem(self) -> OptimizationProblem:
        """Optimization problem for which this class is able to create solutions (as objects)."""
        ...

    def __init__(self, **decision_variables_values: Any) -> None:
        """
        Creates object that carries all information about single solution of 'optimization_problem'.

        :param decision_variables_values: Values of decision variables defined for the 'optimization_problem'.
            Keys: Names of decision variables.
            Values: Values assigned to the decision variables.

        :raise ValueError: Unknown decision variable (name of variable is not defined in the optimization problem) or
            incorrect value of decision variable was provided.
        """
        # find values for all variables
        values_to_set = OrderedDict()
        for variable_name, variable_definition in self.optimization_problem.decision_variables.items():  # type: ignore
            if variable_name in decision_variables_values:
                value = decision_variables_values.pop(variable_name)
                if not variable_definition.is_proper_value(value):
                    raise ValueError(f"Received value of '{variable_name}' decision variables is not compatible "
                                     f"with variable definition. Actual value: {value}.")
            else:
                value = variable_definition.generate_random_value()
            values_to_set[variable_name] = value
        # check if assignment of decision variables values was executed successfully
        if decision_variables_values:
            raise ValueError(f"Values for unknown decision variables were provided: "
                             f"{list(decision_variables_values.keys())}.")
        # set attributes
        self.decision_variables_values = values_to_set
        self._objective_value_with_penalty = None

    # todo: implement comparison methods

    def _calculate_objective(self) -> Union[float, int]:
        """:return: Value of solution objective without penalty."""
        return self.optimization_problem.objective_function(**self.decision_variables_values)

    def _calculate_constraints(self) -> Dict[str, Union[float, int]]:
        """
        Calculates constraints values.

        Each constraint function should return 0 if constraint is fulfilled (solution meets restrictions).
        If constraint is not fulfilled, then function should return value other than 0.

        :return: Dictionary with constraints values.
            Keys: Names of constraint functions.
            Values: Calculated of the corresponding constraint.
        """
        constraints_values = {
            constraint_name: abs(constraint_function(**self.decision_variables_values))
            for constraint_name, constraint_function in self.optimization_problem.constraints.items()
        }
        return constraints_values

    def _calculate_penalty(self) -> Union[float, int]:
        """:return: Value of solution penalty."""
        return self.optimization_problem.penalty_function(**self._calculate_constraints())  # TODO: consider abs here

    def get_objective_value_with_penalty(self):
        """:return: Value of solution objective with penalty."""
        if self._objective_value_with_penalty is None:
            if self.optimization_problem.optimization_type == OptimizationType.Minimize:
                self._objective_value_with_penalty = self._calculate_objective() + self._calculate_penalty()
            else:  # only OptimizationType.Maximize value is possible here
                self._objective_value_with_penalty = self._calculate_objective() - self._calculate_penalty()
        return self._objective_value_with_penalty

    def get_log_data(self) -> Dict[str, Union[dict, int, float]]:
        """
        Gets data for logging purposes.

        :return: Dictionary with this Integer Variable crucial data.
        """
        return {
            "decision_variables_values": self.decision_variables_values,
            "objective_value_with_penalty": self.get_objective_value_with_penalty(),
        }
