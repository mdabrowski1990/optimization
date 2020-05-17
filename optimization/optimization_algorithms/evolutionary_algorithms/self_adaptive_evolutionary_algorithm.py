from typing import Set, Union, Optional, Any
from enum import Enum
from collections import OrderedDict
from random import shuffle

from optimization.optimization_algorithms.evolutionary_algorithms.evolutionary_algorithm import EvolutionaryAlgorithm
from optimization.optimization_algorithms.algorithm_definition import OptimizationAlgorithm
from optimization.optimization_algorithms.stop_conditions import StopCondition
from optimization.optimization_algorithms.evolutionary_algorithms.selection import SelectionType, \
    SELECTION_FUNCTIONS, ADDITIONAL_SELECTION_PARAMETERS
from optimization.optimization_algorithms.evolutionary_algorithms.crossover import CrossoverType, \
    CROSSOVER_FUNCTIONS, ADDITIONAL_CROSSOVER_PARAMETERS
from optimization.optimization_algorithms.evolutionary_algorithms.mutation import MutationType, \
    MUTATION_FUNCTIONS, ADDITIONAL_MUTATION_PARAMETERS
from optimization.optimization_problem import OptimizationProblem, AbstractSolution, OptimizationType, \
    ChoiceVariable, FloatVariable, IntegerVariable


__all__ = ["SAEOptimizationType", "SAEOptimizationProblem"]


class SAEOptimizationType(Enum):
    BEST_SOLUTIONS = "Best solutions"


class SAEOptimizationProblem(OptimizationProblem):
    def __init__(self, available_selection_types: Set[SelectionType], available_crossover_types: Set[CrossoverType],
                 available_mutation_types: Set[MutationType], sae_optimization_type: SAEOptimizationType,
                 tracked_solution_number: int, optimization_problem: OptimizationProblem):
        _variables_number = len(optimization_problem.decision_variables)
        decision_vars = OrderedDict(selection_type=ChoiceVariable(possible_values=available_selection_types),
                                    crossover_type=ChoiceVariable(possible_values=available_crossover_types),
                                    mutation_type=ChoiceVariable(possible_values=available_mutation_types),
                                    apply_elitism=ChoiceVariable(possible_values={True, False}),
                                    mutation_chance=FloatVariable(min_value=0.01, max_value=0.2))
        # additional selection parameters
        additional_decision_vars = OrderedDict()
        if SelectionType.TOURNAMENT in available_selection_types:
            additional_decision_vars.update(tournament_group_size=IntegerVariable(min_value=3, max_value=6))
        if SelectionType.ROULETTE in available_selection_types:
            additional_decision_vars.update(roulette_bias=FloatVariable(min_value=1, max_value=100))
        if SelectionType.RANKING in available_selection_types:
            additional_decision_vars.update(ranking_bias=FloatVariable(min_value=1, max_value=2))
        # additional crossover parameters
        if CrossoverType.MULTI_POINT in available_crossover_types:
            additional_decision_vars.update(crossover_points_number=IntegerVariable(min_value=2,
                                                                                    max_value=_variables_number))
        if CrossoverType.ADAPTIVE in available_crossover_types:
            max_value = (1 << _variables_number) - 1
            additional_decision_vars.update(crossover_pattern=IntegerVariable(min_value=0, max_value=max_value))
        # additional mutation parameters
        if MutationType.MULTI_POINT in available_mutation_types:
            additional_decision_vars.update(crossover_pattern=IntegerVariable(min_value=2, max_value=_variables_number))
        # determine objective value
        if sae_optimization_type == SAEOptimizationType.BEST_SOLUTIONS:
            def objective_function(population):
                return sum(solution.get_objective_value_with_penalty()
                           for solution in population[:tracked_solution_number])
        else:
            raise NotImplementedError("This type of SAE Optimization is currently not supported.")
        # create object
        super().__init__(decision_variables=decision_vars, constraints={}, penalty_function=lambda **_: 0,
                         objective_function=objective_function,
                         optimization_type=optimization_problem.optimization_type)
        self.additional_decision_variables = additional_decision_vars

    def spawn_solution_definition(self) -> type:
        # todo: return SAE Lower Lever AE
        pass


class SAE(EvolutionaryAlgorithm):
    def __init__(self, sae_optimization_problem: SAEOptimizationProblem, stop_condition: StopCondition,
                 population_size: int, selection_type: Union[str, SelectionType], crossover_type: [str, CrossoverType],
                 mutation_type: Union[str, MutationType], apply_elitism: bool, mutation_chance: float,
                 logger: Optional[object] = None, **other_params: Any) -> None:
        super().__init__(optimization_problem=sae_optimization_problem, stop_condition=stop_condition,
                         population_size=population_size, selection_type=selection_type, crossover_type=crossover_type,
                         mutation_type=mutation_type, apply_elitism=apply_elitism, mutation_chance=mutation_chance,
                         logger=logger, **other_params)


class SAELower(EvolutionaryAlgorithm, AbstractSolution):
    # todo: mix these parent classes into one
    pass
