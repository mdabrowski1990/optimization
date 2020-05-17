from collections import OrderedDict
from datetime import timedelta

from optimization.logging import Logger
from optimization.optimization_algorithms import StopCondition, EvolutionaryAlgorithm, SelectionType, \
    CrossoverType, MutationType
from optimization.optimization_problem import OptimizationProblem, OptimizationType, IntegerVariable, FloatVariable, \
    ChoiceVariable

logger = Logger(logs_location="E:\\MGR")

stop_condition = StopCondition(time_limit=timedelta(seconds=1))
problem = OptimizationProblem(decision_variables=OrderedDict(a=IntegerVariable(min_value=0, max_value=100),
                                                             b=FloatVariable(min_value=0., max_value=100.),
                                                             c=ChoiceVariable(possible_values=set(range(10)))),
                              constraints={},
                              penalty_function=lambda **_: 0,
                              objective_function=lambda **vars_values: sum(vars_values.values()),
                              optimization_type=OptimizationType.Maximize)

ea = EvolutionaryAlgorithm(optimization_problem=problem, stop_condition=stop_condition, logger=logger,
                           population_size=100, selection_type=SelectionType.STOCHASTIC,
                           crossover_type=CrossoverType.UNIFORM, mutation_type=MutationType.SINGLE_POINT,
                           mutation_chance=0.01, apply_elitism=False)

ea.perform_optimization()
