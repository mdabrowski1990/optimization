from typing import Callable, Dict, Any
from enum import Enum
from abc import ABC, abstractmethod

from optimization.optimization_problem.decision_variables import DecisionVariable


class OptimizationType(Enum):
    """Enum with types of optimization problems."""
    Maximize = "Maximize"
    Minimize = "Minimize"


class OptimizationProblem:
    """Class for defining the problem for which optimal solution to be searched."""

    def __init__(self, decision_variables: Dict[str, DecisionVariable], constraints: Dict[str, Callable],
                 penalty_function: Callable, objective_function: Callable, optimization_type: OptimizationType) -> None:
        """
        Definition of optimization problem.

        :param decision_variables: Dictionary with decision variables definitions.
            Keys: Names of decision variables.
            Values: Instances of DecisionVariable sub-classes with decision variable definitions.
        :param constraints: Dictionary with constrains definitions.
            Keys: Names of constrains.
            Values: Constraints functions.
        :param penalty_function: Function for calculating penalty value of the solution
            (in case if constrains are not meet).
        :param objective_function: Function for calculating objective value of the solution (does not include penalty).
        :param optimization_type: Type optimization problem (either looking for minimal or maximal value).
        :raise TypeError: Some parameter has incorrect type.
        :raise ValueError: Parameter 'decision_variables' or 'constraints has invalid value.
        """
        if not isinstance(decision_variables, dict):
            raise TypeError(f"Parameter 'decision_variables' must be dict type. "
                            f"Received: {decision_variables} ({type(decision_variables)}).")
        if not decision_variables:
            raise ValueError("Parameter 'decision_variables' cannot be empty.")
        if not all(isinstance(key, str) and isinstance(value, DecisionVariable)
                   for key, value in decision_variables.items()):
            raise ValueError(f"Keys of 'decision_variables' parameter must be str type and values must be "
                             f"DecisionVariable type. Received: {decision_variables}.")
        if not isinstance(constraints, dict):
            raise TypeError(f"Parameter 'constraints' must be dict type. "
                            f"Received: {constraints} ({type(constraints)}).")
        if not all(isinstance(key, str) and callable(value) for key, value in constraints.items()):
            raise ValueError(f"Keys of 'constraints' parameter must be str type and values must be callable. "
                             f"Received: {constraints}.")
        if not callable(penalty_function):
            raise TypeError(f"Parameter 'penalty_function' must be callable. Received: {penalty_function}")
        if not callable(objective_function):
            raise TypeError(f"Parameter 'objective_function' must be callable. Received: {objective_function}")
        if not isinstance(optimization_type, OptimizationType):
            raise TypeError(f"Parameter 'optimization_type' must be instance of 'OptimizationType' enum. "
                            f"Received: {optimization_type}.")
        self.decision_variables = decision_variables
        self.constraints = constraints
        self.penalty_function = penalty_function
        self.objective_function = objective_function
        self.optimization_type = optimization_type

    def get_data_for_logging(self) -> dict:
        # todo
        pass


class Solution(ABC):
    """Abstract definition of optimization problem solution."""

    @property
    @abstractmethod
    def optimization_problem(self) -> OptimizationProblem:
        """
        Abstract definition of a property that stores reference to optimization problem.

        :raises NotImplementedError: Abstract method was called.
        """
        raise NotImplementedError("You have called abstract property 'optimization_problem' of 'Solution' "
                                  "abstract class.")

    def __init__(self, **decision_variables_values: Any) -> None:
        """
        Creates solution object that carries solution information.

        :param decision_variables_values:
            Keys: Names of decision variables.
            Values: Values assigned to the decision variables.
        :raise ValueError: Incorrect value of decision variable war provided or
            value for unknown decision variable was received.
        """
        # find values for all variables
        values_to_set = {}
        for variable_name, variable_definition in self.optimization_problem.decision_variables.items():
            value = decision_variables_values.pop(variable_name, None)
            if value is None:
                value = variable_definition.generate_random_value()
            elif not variable_definition.is_value_correct(value):
                raise ValueError(f"Received value of '{variable_name}' decision variable does not match the definition."
                                 f" Received value: {value}.")
            values_to_set[variable_name] = value
        # check if assignment of decision variables values was executed successfully
        if decision_variables_values:
            raise ValueError(f"Values for unknown decision variables were received: {decision_variables_values}."
                             f"Known decision variables names: "
                             f"{list(self.optimization_problem.decision_variables.keys())}.")
        # set attributes
        self.decision_variables_values = values_to_set
        self._objective_value = None

    def _calculate_objective(self) -> float:
        """
        Calculates value of the solution objective.

        :return: Value of objective function without penalty.
        """
        return self.optimization_problem.objective_function(**self.decision_variables_values)

    def _calculate_penalty(self) -> float:
        """
        Calculates value of the solution penalty.

        :return: Value of penalty function.
        """
        return abs(self.optimization_problem.penalty_function(**self._calculate_constraints()))

    def _calculate_constraints(self) -> dict:
        """
        Calculates constraints function values.
        Each constraint function should be return 0 if constraint is fulfilled (solution meets restrictions).
        If constraint is not fulfilled, then function should return value greater than 0.

        :return: Dictionary with constraints values.
            Keys: Names of constraint functions.
            Values: Values calculated for the corresponding constraint functions.
        """
        constraints_values = {
            constraint_name: abs(constraint_function(**self.decision_variables_values))
            for constraint_name, constraint_function in self.optimization_problem.constraints.items()
        }
        return constraints_values

    def get_objective_value_with_penalty(self) -> float:
        """
        Method for determining value of objective function with penalty.

        :return: Value calculated according to definition of optimization problem.
        """
        if self._objective_value is None:
            if self.optimization_problem.optimization_type == OptimizationType.Minimize:
                self._objective_value = self._calculate_objective() + self._calculate_penalty()
            else:
                self._objective_value = self._calculate_objective() - self._calculate_penalty()
        return self._objective_value

    def get_data_for_logging(self):
        # todo
        pass
