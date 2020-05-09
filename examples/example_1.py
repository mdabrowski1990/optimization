from collections import OrderedDict
from datetime import timedelta

from optimization.logging import Logger
from optimization.optimization_algorithms import RandomAlgorithm, StopCondition
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
ra = RandomAlgorithm(optimization_problem=problem, population_size=100, stop_condition=stop_condition, logger=logger)

x = ra.perform_optimization()
print(x.__dict__)
