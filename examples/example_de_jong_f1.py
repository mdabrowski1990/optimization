"""
Example use of optimization package.

Problem to be optimized: De Jong F1 function
More info:
https://www.researchgate.net/publication/279561942_A_simple_and_global_optimization_algorithm_for_engineering_problems_Differential_evolution_algorithm
http://www2.denizyuret.com/pub/aitr1569/node19.html

Stop conditions (when algorithm must stop the optimization):
- maximal time (10s) is reached

Algorithms used:
- Random Algorithm (ra):
    population size: default value

- Evolutionary Algorithm (ea1):
    population size: 100 individuals
    selection: roulette with bias equal 90
    crossover: single-point
    mutation: two-point (10% chance)
    applies elitism (children replace their parents in a new population only if they are better adjusted)
- Evolutionary Algorithm (ea2):
    population size: 20 individuals
    selection: uniform
    crossover: two-point
    mutation: probabilistic (15% chance)
    does not apply elitism (children always replace their parents)
- Evolutionary Algorithm (ea3):
    population size: 250 individuals
    selection: ranking with bias equal 1.9
    crossover: uniform
    mutation: single-point
    applies elitism

- Adaptive Evolutionary Algorithm (aea1):
    population size: 10 individuals
    selection: roulette with bias equal 90
    crossover: single-point
    mutation: two-point (10% chance)
- Adaptive Evolutionary Algorithm (aea2):
    population size: 10 individuals
    selection: uniform
    crossover: two-point
    mutation: probabilistic (15% chance)
- Adaptive Evolutionary Algorithm (aea3):
    population size: 10 individuals
    selection: ranking with bias equal 1.9
    crossover: uniform
    mutation: single-point
"""

from collections import OrderedDict
from datetime import timedelta
from os import getcwd, path

from optimization import OptimizationProblem, FloatVariable, OptimizationType, StopConditions, \
    Logger, LoggingVerbosity, \
    RandomAlgorithm, EvolutionaryAlgorithm, AdaptiveEvolutionaryAlgorithm, \
    SelectionType, CrossoverType, MutationType, \
    AdaptationType, EvolutionaryAlgorithmAdaptationProblem


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
stop_conditions_f1 = StopConditions(time_limit=timedelta(seconds=10))  # stop after 10 s

# Define logger for optimization process recording
example_logs_path = path.join(getcwd(), "De Jong F1 - logs")
logger = Logger(logs_dir=example_logs_path, verbosity=LoggingVerbosity.AllSolutions)

# DEFINE ALGORITHMS
# random algorithm
ra = RandomAlgorithm(problem=problem_de_jong_f1, stop_conditions=stop_conditions_f1, logger=logger)
# evolutionary algorithms
ea1 = EvolutionaryAlgorithm(problem=problem_de_jong_f1,
                            stop_conditions=stop_conditions_f1,
                            logger=logger,
                            population_size=100,
                            selection_type=SelectionType.Roulette,
                            roulette_bias=90.,
                            crossover_type=CrossoverType.SinglePoint,
                            mutation_type=MutationType.MultiPoint,
                            mutation_points_number=2,
                            mutation_chance=0.1,
                            apply_elitism=True)
ea2 = EvolutionaryAlgorithm(problem=problem_de_jong_f1,
                            stop_conditions=stop_conditions_f1,
                            logger=logger,
                            population_size=20,
                            selection_type=SelectionType.Uniform,
                            crossover_type=CrossoverType.MultiPoint,
                            crossover_points_number=2,
                            mutation_type=MutationType.Probabilistic,
                            mutation_chance=0.15,
                            apply_elitism=False)
ea3 = EvolutionaryAlgorithm(problem=problem_de_jong_f1,
                            stop_conditions=stop_conditions_f1,
                            logger=logger,
                            population_size=250,
                            selection_type=SelectionType.Ranking,
                            ranking_bias=1.9,
                            crossover_type=CrossoverType.Uniform,
                            mutation_type=MutationType.SinglePoint,
                            mutation_chance=0.2,
                            apply_elitism=True)
# adaptive evolutionary algorithms
example_adaptation_problem = EvolutionaryAlgorithmAdaptationProblem(
    adaptation_type=AdaptationType.BestSolution,
    population_size_boundaries=(10, 50))
aea1 = AdaptiveEvolutionaryAlgorithm(
    problem=problem_de_jong_f1,
    adaptation_problem=example_adaptation_problem,
    stop_conditions=stop_conditions_f1,
    logger=logger,
    population_size=10,
    selection_type=SelectionType.Roulette,
    roulette_bias=90.,
    crossover_type=CrossoverType.SinglePoint,
    mutation_type=MutationType.MultiPoint,
    mutation_points_number=2,
    mutation_chance=0.1)
aea2 = AdaptiveEvolutionaryAlgorithm(
    problem=problem_de_jong_f1,
    adaptation_problem=example_adaptation_problem,
    stop_conditions=stop_conditions_f1,
    logger=logger,
    population_size=10,
    selection_type=SelectionType.Uniform,
    crossover_type=CrossoverType.MultiPoint,
    crossover_points_number=2,
    mutation_type=MutationType.Probabilistic,
    mutation_chance=0.15)
aea3 = AdaptiveEvolutionaryAlgorithm(
    problem=problem_de_jong_f1,
    adaptation_problem=example_adaptation_problem,
    stop_conditions=stop_conditions_f1,
    logger=logger,
    population_size=10,
    selection_type=SelectionType.Ranking,
    ranking_bias=1.9,
    crossover_type=CrossoverType.Uniform,
    mutation_type=MutationType.SinglePoint,
    mutation_chance=0.2)

# Perform optimization
print("Optimization started for: ra")
ra.perform_optimization()
print("Optimization completed for: ra")

print("Optimization started for: ea1")
ea1.perform_optimization()
print("Optimization completed for: ea1")

print("Optimization started for: ea2")
ea2.perform_optimization()
print("Optimization completed for: ea2")

print("Optimization started for: ea3")
ea3.perform_optimization()
print("Optimization completed for: ea3")

print("Optimization started for: aea1")
aea1.perform_optimization()
print("Optimization completed for: aea1")

print("Optimization started for: aea2")
aea2.perform_optimization()
print("Optimization completed for: aea2")

print("Optimization started for: aea3")
aea3.perform_optimization()
print("Optimization completed for: aea3")

print("SUCCESS - Experiment is completed")
