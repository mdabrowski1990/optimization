"""Classic evolutionary algorithm."""

__all__ = ["EvolutionaryAlgorithm"]


from typing import Optional, Union, Any, Dict

from ..abstract_algorithm import AbstractOptimizationAlgorithm
from ...problem import OptimizationProblem
from ...stop_conditions import StopConditions
from ...logging import AbstractLogger
from .selection import SelectionType, SELECTION_FUNCTIONS, SELECTION_ADDITIONAL_PARAMS, check_selection_parameters
from .crossover import CrossoverType, CROSSOVER_FUNCTIONS, CROSSOVER_ADDITIONAL_PARAMS, check_crossover_parameters
from .mutation import MutationType


class EvolutionaryAlgorithm(AbstractOptimizationAlgorithm):
    """
    Evolutionary optimization algorithm implementation.

    This algorithm uses mechanisms inspired by biological evolution in searches of optimal solution
    for given optimization problem.
    """

    def __init__(self, problem: OptimizationProblem,  # pylint: disable=too-many-arguments
                 stop_conditions: StopConditions,
                 population_size: int,
                 selection_type: Union[SelectionType, str],
                 crossover_type: Union[CrossoverType, str],
                 mutation_type: Union[CrossoverType, str],
                 mutation_chance: float,
                 apply_elitism: bool,
                 logger: Optional[AbstractLogger] = None,
                 **other_params: Any) -> None:
        """
        Configuration of Evolutionary Algorithm.

        :param problem: Optimization problem to be solved by the algorithm.
        :param stop_conditions: Conditions when optimization algorithm shall be stopped.
        :param population_size: Size of the algorithm's solution population.
        :param selection_type: Type of selection function to use.
        :param crossover_type: Type of crossover function to use.
        :param mutation_type: Type of mutation function to use.
        :param mutation_chance: Probability of a single decision variable (gene) mutation.
        :param apply_elitism: Information whether elitism should be applied.
            When True, then only better adopted children will replace their parents.
            When False, then children will always replace their parents.
        :param logger: Logger used for optimization process recording.
        :param other_params: Parameter related to selected selection, crossover and mutation type.
        """
        if not isinstance(population_size, int):
            raise TypeError(f"Parameter 'population_size' value is not int type. Actual value: {population_size}.")
        if not 10 <= population_size <= 1000 or population_size & 1:
            raise ValueError(f"Parameter 'population_size' value is not even value in range. "
                             f"Expected value: 10 <= population_size <= 1000. Actual value: {population_size}.")
        if not isinstance(mutation_chance, float):
            raise TypeError(f"Parameter 'mutation_chance' value is not float type. Actual value: {mutation_chance}.")
        if not isinstance(apply_elitism, bool):
            raise TypeError(f"Parameter 'apply_elitism' value is not bool type. Actual value: {apply_elitism}.")
        if isinstance(selection_type, SelectionType):
            self.selection_type = selection_type.value
        elif isinstance(selection_type, str):
            self.selection_type = getattr(SelectionType, selection_type)
        else:
            raise TypeError(f"Parameter 'selection_type' value is not str or SelectionType type. "
                            f"Actual value: {selection_type}.")
        if isinstance(crossover_type, CrossoverType):
            self.crossover_type = crossover_type.value
        elif isinstance(crossover_type, str):
            self.crossover_type = getattr(CrossoverType, crossover_type)
        else:
            raise TypeError(f"Parameter 'crossover_type' value is not str or CrossoverType type. "
                            f"Actual value: {crossover_type}.")
        if isinstance(mutation_type, MutationType):
            self.mutation_type = mutation_type.value
        elif isinstance(mutation_type, str):
            self.mutation_type = getattr(MutationType, mutation_type)
        else:
            raise TypeError(f"Parameter 'mutation_type' value is not str or MutationType type. "
                            f"Actual value: {mutation_type}.")
        super().__init__(problem=problem, stop_conditions=stop_conditions, logger=logger)
        self.population_size = population_size
        self.mutation_chance = mutation_chance
        self.apply_elitism = apply_elitism
        self.selection_function = SELECTION_FUNCTIONS[self.selection_type]
        self.selection_function = CROSSOVER_FUNCTIONS[self.crossover_type]
        # self.selection_function = MUTATION_FUNCTIONS[self.mutation_type]
        self.selection_params: Dict[str, Any] = {}
        self.crossover_params: Dict[str, Any] = {}
        self.mutation_params: Dict[str, Any] = {}
        for selection_param in SELECTION_ADDITIONAL_PARAMS[self.selection_type]:
            self.selection_params[selection_param] = other_params.pop(selection_param)
        check_selection_parameters(**self.selection_params)
        for crossover_param in CROSSOVER_ADDITIONAL_PARAMS[self.crossover_type]:
            self.crossover_params[crossover_param] = other_params.pop(crossover_param)
        check_crossover_parameters(variables_number=len(self.problem.decision_variables), **self.crossover_params)
        # todo: extract selection_params, crossover_params, mutation_params and raise meaningful exception if not there
