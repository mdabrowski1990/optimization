from collections import OrderedDict
from datetime import timedelta

from optimization_old.logging import Logger
from optimization_old.optimization_algorithms import StopCondition, EvolutionaryAlgorithm, SelectionType, \
    CrossoverType, MutationType
from optimization_old.optimization_problem import OptimizationProblem, OptimizationType, IntegerVariable, FloatVariable, \
    ChoiceVariable

logger = Logger(logs_location="E:\\MGR")

stop_condition = StopCondition(time_limit=timedelta(seconds=10))
problem = OptimizationProblem(decision_variables=OrderedDict(a=IntegerVariable(min_value=0, max_value=100),
                                                             b=FloatVariable(min_value=0., max_value=100.),
                                                             c=ChoiceVariable(possible_values=set(range(10)))),
                              constraints={},
                              penalty_function=lambda **_: 0,
                              objective_function=lambda **vars_values: sum(vars_values.values()),
                              optimization_type=OptimizationType.Maximize)

ea = EvolutionaryAlgorithm(optimization_problem=problem, stop_condition=stop_condition, logger=logger,
                           population_size=100, selection_type=SelectionType.RANKING,
                           crossover_type=CrossoverType.MULTI_POINT, mutation_type=MutationType.MULTI_POINT,
                           mutation_chance=0.01, apply_elitism=False, ranking_bias=1.5, crossover_points_number=1,
                           mutation_points_number=2)

ea.perform_optimization()
