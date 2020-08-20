"""
Optimization algorithms.

Available algorithms:
 - RandomAlgorithm - algorithm that creates totally random solutions
 - EvolutionaryAlgorithm - algorithm that uses biological evolution mechanisms such as reproduction, mutation,
    recombination and selection
"""

from .random_algorithm import RandomAlgorithm
from .evolutionary_algorithm import EvolutionaryAlgorithm, SelectionType, CrossoverType, MutationType
