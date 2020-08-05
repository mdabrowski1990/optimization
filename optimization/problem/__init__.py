"""
Module with optimization problem and solution definition.

Provides:
    - OptimizationProblem - class for defining optimization problem using mathematical model
    - OptimizationType - enum storing available optimization types
    - AbstractSolution - Abstract class (used internally) for defining types (child classes) that creates certain
        optimization problem solutions (objects of child classes).
    - DiscreteVariable - Abstract class (used internally) for typing and definition of children classes:
        - IntegerVariable - definition of Decision Variable that stores integer value (any int in range)
        - DiscreteVariable - definition of Decision Variable that stores discrete value (with certain step)
        - FloatVariable - definition of Decision Variable that stores float value (any float in range)
        - ChoiceVariable - definition of Decision Variable that stores value of any type but from predefined possible
            values pool
"""

from .problem import OptimizationType, OptimizationProblem
from .decision_variables import IntegerVariable, DiscreteVariable, FloatVariable, ChoiceVariable
from .solution import AbstractSolution
