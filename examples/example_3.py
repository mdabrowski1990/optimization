from collections import OrderedDict
from datetime import timedelta

from optimization_old.logging import SEALogger
from optimization_old.optimization_algorithms import SEA, SEAOptimizationProblem, SEAOptimizationType, StopCondition, \
    SelectionType, CrossoverType, MutationType
from optimization_old.optimization_problem import OptimizationProblem, OptimizationType, IntegerVariable, FloatVariable, \
    ChoiceVariable

logger = SEALogger(logs_location="E:\\MGR")

stop_condition = StopCondition(time_limit=timedelta(seconds=60))
problem = OptimizationProblem(decision_variables=OrderedDict(a=IntegerVariable(min_value=0, max_value=100),
                                                             b=FloatVariable(min_value=0., max_value=100.),
                                                             c=ChoiceVariable(possible_values=set(range(10)))),
                              constraints={},
                              penalty_function=lambda **_: 0,
                              objective_function=lambda **vars_values: sum(vars_values.values()),
                              optimization_type=OptimizationType.Maximize)

sea_problem = SEAOptimizationProblem(population_size=(10, 100), selection_types=list(SelectionType),
                                     crossover_types=list(CrossoverType), mutation_types=list(MutationType),
                                     sea_optimization_type=SEAOptimizationType.BEST_SOLUTIONS,
                                     optimization_problem=problem)

sea = SEA(sae_optimization_problem=sea_problem, stop_condition=stop_condition, population_size=10,
          iterations_number=2, selection_type=SelectionType.STOCHASTIC, crossover_type=CrossoverType.UNIFORM,
          mutation_type=MutationType.SINGLE_POINT, mutation_chance=0.05, logger=logger)
sea.perform_optimization()
