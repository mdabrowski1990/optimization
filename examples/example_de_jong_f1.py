"""
Example use of optimization package.

Problem to be optimized: De Jong F1 function
More info:
https://www.researchgate.net/publication/279561942_A_simple_and_global_optimization_algorithm_for_engineering_problems_Differential_evolution_algorithm
http://www2.denizyuret.com/pub/aitr1569/node19.html

Stop conditions (when algorithm must stop the optimization):
- maximal time (10s) is reached

Algorithms used:
- Random Algorithm
- Evolutionary Algorithm (ea1):
    population size: 100 individuals
    selection: roulette with bias 90
    crossover: single point
    mutation: two-point (1% chance)
    applies elitism (children replace their parents in a new population only if they are better adjusted)
- Evolutionary Algorithm (ea2):
    population size:  20 individuals
    selection: uniform
    crossover: two-point
    mutation: probabilistic (5% chance)
    does not apply elitism (children always replace their parents)
TODO: update description
"""

from collections import OrderedDict
from datetime import timedelta
from os import getcwd, path

from optimization import OptimizationProblem, FloatVariable, OptimizationType, StopConditions, \
    Logger, LoggingVerbosity, \
    RandomAlgorithm, EvolutionaryAlgorithm, AdaptiveEvolutionaryAlgorithm, \
    SelectionType, CrossoverType, MutationType

# De Jong F1 function problem definition
x = FloatVariable(min_value=-5.12, max_value=5.12)
problem_de_jong_f1 = OptimizationProblem(
    decision_variables=OrderedDict(x1=x, x2=x, x3=x),
    constraints={},
    penalty_function=lambda **values: 0,
    objective_function=lambda **values: values["x1"] ** 2 + values["x2"] ** 2 + values["x3"] ** 2,
    optimization_type=OptimizationType.Minimize
)

# Define when Algorithm to be stopped
stop_conditions_f1 = StopConditions(time_limit=timedelta(seconds=10))  # stop if optimal solution (objective == 0) is found

# Define loggers for optimization process recording
example_logs_path = path.join(getcwd(), "Example De Jong F1 logs")
ra_logs_path = path.join(example_logs_path, "Random Algorithm")
ea1_logs_path = path.join(example_logs_path, "Evolutionary Algorithm 1")
ea2_logs_path = path.join(example_logs_path, "Evolutionary Algorithm 2")
aea1_logs_path = path.join(example_logs_path, "Adaptive Evolutionary Algorithm 1")
aea2_logs_path = path.join(example_logs_path, "Adaptive Evolutionary Algorithm 2")
logger_f1_ra = Logger(logs_dir=ra_logs_path, verbosity=LoggingVerbosity.AllSolutions)
logger_f1_ea1 = Logger(logs_dir=ea1_logs_path, verbosity=LoggingVerbosity.AllSolutions)
logger_f1_ea2 = Logger(logs_dir=ea2_logs_path, verbosity=LoggingVerbosity.AllSolutions)
logger_f1_aea1 = Logger(logs_dir=aea1_logs_path, verbosity=LoggingVerbosity.AllSolutions)
logger_f1_aea2 = Logger(logs_dir=aea2_logs_path, verbosity=LoggingVerbosity.AllSolutions)

# define algorithms to be used
ra = RandomAlgorithm(problem=problem_de_jong_f1, stop_conditions=stop_conditions_f1, logger=logger_f1_ra)
ea1 = EvolutionaryAlgorithm(problem=problem_de_jong_f1,
                            stop_conditions=stop_conditions_f1,
                            logger=logger_f1_ea1,
                            population_size=100,
                            selection_type=SelectionType.Roulette,
                            crossover_type=CrossoverType.SinglePoint,
                            mutation_type=MutationType.MultiPoint,
                            mutation_chance=0.01,
                            apply_elitism=True,
                            roulette_bias=90.,
                            mutation_points_number=2)
ea2 = EvolutionaryAlgorithm(problem=problem_de_jong_f1,
                            stop_conditions=stop_conditions_f1,
                            logger=logger_f1_ea2,
                            population_size=20,
                            selection_type=SelectionType.Uniform,
                            crossover_type=CrossoverType.MultiPoint,
                            mutation_type=MutationType.Probabilistic,
                            mutation_chance=0.05,
                            apply_elitism=False,
                            crossover_points_number=2)
# TODO: aea1 and eaa2 to be defined

# Perform optimization

ra.perform_optimization()
ea1.perform_optimization()
ea2.perform_optimization()
