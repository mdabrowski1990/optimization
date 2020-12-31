"""Evolutionary algorithms boundary values definitions."""

# general Evolutionary Algorithm params
MIN_EA_POPULATION_SIZE: int = 10  # must be even
MAX_EA_POPULATION_SIZE: int = 1000  # must be even
MIN_EA_MUTATION_CHANCE: float = 0.001
MAX_EA_MUTATION_CHANCE: float = 0.2

# selection variables
MIN_TOURNAMENT_GROUP_SIZE: int = 2
MAX_TOURNAMENT_GROUP_SIZE: int = 6
MIN_ROULETTE_BIAS: float = 1.1
MAX_ROULETTE_BIAS: float = 100.
MIN_RANKING_BIAS: float = 1.
MAX_RANKING_BIAS: float = 2.
