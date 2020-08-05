"""
Package for performing optimization with any of (currently implemented) optimization algorithms.

Current supported list of optimization algorithms:
    - Evolutionary Algorithms (https://en.wikipedia.org/wiki/Evolutionary_algorithm)
    - Self-adaptive Evolutionary Algorithm (similar to Evolutionary Algorithms, but during optimization process,
        evolutionary algorithm searching method is also optimized)
    - Random Algorithm

Full package documentation: https://github.com/mdabrowski1990/optimization
"""

__author__ = "Maciej Dąbrowski (maciek_dabrowski@o2.pl)"

from .problem import OptimizationProblem, OptimizationType, IntegerVariable, DiscreteVariable, FloatVariable, \
    ChoiceVariable
