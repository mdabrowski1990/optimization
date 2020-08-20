"""
Evolutionary algorithms.

In this package, you can find following algorithms:
- EvolutionaryAlgorithm - classic Evolutionary algorithm

Additionally, there enums with implemented and possible to choose selection, crossover and mutation functions:
- SelectionType - enum with all implemented selection types supported by EvolutionaryAlgorithm
- CrossoverType - enum with all implemented crossover types supported by EvolutionaryAlgorithm
- MutationType - enum with all implemented mutation types supported by EvolutionaryAlgorithm
"""

from .evolutionary_algorithm import EvolutionaryAlgorithm
from .selection import SelectionType
from .crossover import CrossoverType
from .mutation import MutationType
