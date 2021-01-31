"""
Package for performing optimization with heuristic algorithms.

Current supported list of optimization algorithms:
 - Random Algorithm
 - Evolutionary Algorithms

Full package documentation: https://github.com/mdabrowski1990/optimization
"""

__author__ = "Maciej Dąbrowski (maciek_dabrowski@o2.pl)"

from .problem import OptimizationProblem, OptimizationType, IntegerVariable, DiscreteVariable, FloatVariable, \
    ChoiceVariable
from .stop_conditions import StopConditions
from .logging import AbstractLogger, Logger, LoggingFormat, LoggingVerbosity
from .algorithms import RandomAlgorithm, EvolutionaryAlgorithm, SelectionType, CrossoverType, MutationType, \
    AdaptationType, AdaptiveEvolutionaryAlgorithm, EvolutionaryAlgorithmAdaptationProblem
