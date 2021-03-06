"""Classic evolutionary algorithm."""

__all__ = ["EvolutionaryAlgorithm"]


from typing import Optional, Union, Any, Dict, Tuple, OrderedDict

from ..abstract_algorithm import AbstractOptimizationAlgorithm
from ...problem import OptimizationProblem, AbstractSolution
from ...stop_conditions import StopConditions
from ...logging import AbstractLogger
from .selection import SelectionType, SELECTION_FUNCTIONS, SELECTION_ADDITIONAL_PARAMS, check_selection_parameters, \
    SelectionOutput
from .crossover import CrossoverType, CROSSOVER_FUNCTIONS, CROSSOVER_ADDITIONAL_PARAMS, check_crossover_parameters, \
    ChildrenValuesTyping
from .mutation import MutationType, MUTATION_FUNCTIONS, MUTATION_ADDITIONAL_PARAMS, check_mutation_parameters
from .limits import MIN_EA_POPULATION_SIZE, MAX_EA_POPULATION_SIZE, MIN_EA_MUTATION_CHANCE, MAX_EA_MUTATION_CHANCE


class EvolutionaryAlgorithm(AbstractOptimizationAlgorithm):
    """
    Evolutionary optimization algorithm implementation.

    This algorithm uses mechanisms inspired by biological evolution in searches of optimal solution
    for given optimization problem.
    """

    MIN_POPULATION_SIZE: int = MIN_EA_POPULATION_SIZE
    MAX_POPULATION_SIZE: int = MAX_EA_POPULATION_SIZE
    MIN_MUTATION_CHANCE: float = MIN_EA_MUTATION_CHANCE
    MAX_MUTATION_CHANCE: float = MAX_EA_MUTATION_CHANCE

    def __init__(self_ea,  # noqa
                 problem: OptimizationProblem,
                 stop_conditions: StopConditions,
                 population_size: int,
                 selection_type: Union[SelectionType, str],
                 crossover_type: Union[CrossoverType, str],
                 mutation_type: Union[MutationType, str],
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
        :param other_params: Parameter related to selected selection, crossover and mutation type such as:
            - :param tournament_group_size: int - determines group size used in tournament
                and double tournament selections
            - :param roulette_bias: float - bias towards promoting better adopted individuals in roulette selection
            - :param ranking_bias: float - represents selection pressure (the higher the value the higher the pressure)
                in ranking selection
            - :param crossover_points_number: int - number of crossover points to use in multipoint crossover
            - :param crossover_pattern: int - pattern of crossover to be used in adaptive crossover
            - :param mutation_points_number: int - number of mutation points to be used in multipoint mutation.

        :raise ValueError: Unexpected value of 'other_params'.
        """
        self_ea._check_init_input(population_size=population_size, mutation_chance=mutation_chance,
                                  apply_elitism=apply_elitism)
        super().__init__(problem=problem, stop_conditions=stop_conditions, logger=logger)
        self_ea.population_size = population_size
        self_ea._population: list = []
        self_ea.mutation_chance = mutation_chance
        self_ea.apply_elitism = apply_elitism
        self_ea.selection_type = selection_type.value if isinstance(selection_type, SelectionType) \
            else getattr(SelectionType, selection_type).value
        self_ea.crossover_type = crossover_type.value if isinstance(crossover_type, CrossoverType) \
            else getattr(CrossoverType, crossover_type).value
        self_ea.mutation_type = mutation_type.value if isinstance(mutation_type, MutationType) \
            else getattr(MutationType, mutation_type).value
        self_ea.selection_function = SELECTION_FUNCTIONS[self_ea.selection_type]
        self_ea.crossover_function = CROSSOVER_FUNCTIONS[self_ea.crossover_type]
        self_ea.mutation_function = MUTATION_FUNCTIONS[self_ea.mutation_type]
        self_ea.selection_params: Dict[str, Any] = {}
        self_ea.crossover_params: Dict[str, Any] = {}
        self_ea.mutation_params: Dict[str, Any] = {}
        for selection_param in SELECTION_ADDITIONAL_PARAMS[self_ea.selection_type]:
            self_ea.selection_params[selection_param] = other_params.pop(selection_param)
        for crossover_param in CROSSOVER_ADDITIONAL_PARAMS[self_ea.crossover_type]:
            self_ea.crossover_params[crossover_param] = other_params.pop(crossover_param)
        for mutation_param in MUTATION_ADDITIONAL_PARAMS[self_ea.mutation_type]:
            self_ea.mutation_params[mutation_param] = other_params.pop(mutation_param)
        if other_params:
            raise ValueError(f"Unexpected 'other_params' received: {other_params}.")
        self_ea._check_additional_parameters()

    def _check_init_input(self, population_size: int, mutation_chance: float, apply_elitism: bool) -> None:
        """
        Checks if input parameter provided to __init__ method have proper values.

        :param population_size: Size of the algorithm's solution population.
        :param mutation_chance: Probability of a single decision variable (gene) mutation.
        :param apply_elitism: Information whether elitism should be applied.

        :raise TypeError: One of parameters stores value of incorrect type.
        :raise ValueError: One of parameters stores incorrect value.

        :return: None
        """
        if not isinstance(population_size, int):
            raise TypeError(f"Parameter 'population_size' value is not int type. Actual value: {population_size}.")
        if not self.MIN_POPULATION_SIZE <= population_size <= self.MAX_POPULATION_SIZE or population_size & 1:
            raise ValueError(f"Parameter 'population_size' value is not even value in range. "
                             f"Expected value: 10 <= population_size <= 1000. Actual value: {population_size}.")
        if not isinstance(mutation_chance, float):
            raise TypeError(f"Parameter 'mutation_chance' value is not float type. Actual value: {mutation_chance}.")
        if not self.MIN_MUTATION_CHANCE <= mutation_chance <= self.MAX_MUTATION_CHANCE:
            raise ValueError(f"Parameter 'mutation_chance' value is not in expected range. Expected value: "
                             f"{self.MIN_MUTATION_CHANCE} <= mutation_chance <= {self.MAX_MUTATION_CHANCE}. "
                             f"Actual value: {mutation_chance}.")
        if not isinstance(apply_elitism, bool):
            raise TypeError(f"Parameter 'apply_elitism' value is not bool type. Actual value: {apply_elitism}.")

    def _check_additional_parameters(self) -> None:
        """
        Checks if additional (function specific) selection, crossover and mutation parameters have proper values.

        :return: None
        """
        check_selection_parameters(**self.selection_params)
        check_crossover_parameters(variables_number=self.problem.variables_number, **self.crossover_params)
        check_mutation_parameters(variables_number=self.problem.variables_number, **self.mutation_params)

    def _log_iteration(self, iteration_index: int) -> None:
        """
        Logs population data in given algorithm's iteration.

        :param iteration_index: Index number (counted from 0) of optimization algorithm iteration.

        :return: None
        """
        if self.logger is not None:
            self.logger.log_iteration(iteration=iteration_index, solutions=self._population)

    def _generate_random_population(self) -> None:
        """
        Creates initial random population of solutions. To be called as initial iteration.

        :return: None
        """
        while len(self._population) < self.population_size:
            self._population.append(self.SolutionClass())

    def _perform_selection(self) -> SelectionOutput:
        """
        Selects pairs of individuals that will become parents for new population.

        :return: Parents pair generator.
        """
        return self.selection_function(population_size=self.population_size,
                                       population=self._population,
                                       **self.selection_params)

    def _perform_crossover(self, parents: Tuple[AbstractSolution, AbstractSolution]) -> ChildrenValuesTyping:
        """
        Performs crossover of two parents.

        :param parents: Solution objects with values of selected parents.

        :return: Values of children decision variables (genes).
        """
        return self.crossover_function(parents=parents,
                                       variables_number=self.problem.variables_number,
                                       **self.crossover_params)

    def _perform_mutation(self, individual_values: OrderedDict[str, Any]) -> None:  # type: ignore
        """
        Performs mutation on individual decision variables values.

        :param individual_values: Individual values to be mutated.

        :return: None
        """
        decision_variables_list = list(self.problem.decision_variables.items())  # type: ignore
        for mutation_point in self.mutation_function(variables_number=self.problem.variables_number,
                                                     mutation_chance=self.mutation_chance,
                                                     **self.mutation_params):
            name, var = decision_variables_list[mutation_point]
            individual_values[name] = var.generate_random_value()  # type: ignore

    def _evolution_iteration(self, **_: Any) -> None:
        """
        Perform iteration according to evolutionary algorithm. To be called as following iteration.

        :return: None
        """
        new_population = []
        for parent1, parent2 in self._perform_selection():
            child1_values, child2_values = self._perform_crossover(parents=(parent1, parent2))
            self._perform_mutation(child1_values)
            self._perform_mutation(child2_values)
            child1 = self.SolutionClass(**child1_values)
            child2 = self.SolutionClass(**child2_values)
            if self.apply_elitism:
                new_population.append(child1 if child1 >= parent1 else parent1)
                new_population.append(child2 if child2 >= parent2 else parent2)
            else:
                new_population.append(child1)
                new_population.append(child2)
        self._population = new_population

    def _perform_iteration(self, iteration_index: int) -> None:
        """
        Executes following iteration of optimization algorithm.

        :param iteration_index: Index number (counted from 0) of optimization algorithm iteration.

        :return: None
        """
        if iteration_index == 0:
            self._generate_random_population()
        else:
            self._evolution_iteration()
        self._best_solution = max(*self._population) if self._best_solution is None \
            else max(*self._population, self._best_solution)
        self._log_iteration(iteration_index=iteration_index)

    def get_log_data(self) -> Dict[str, Any]:
        """
        Gets data for logging purposes.

        :return: Dictionary with this Evolutionary Algorithm crucial data.
        """
        log_data = super().get_log_data()
        log_data.update(population_size=self.population_size, apply_elitism=self.apply_elitism,
                        selection_type=self.selection_type, selection_params=self.selection_params,
                        crossover_type=self.crossover_type, crossover_params=self.crossover_params,
                        mutation_type=self.mutation_type, mutation_params=self.mutation_params,
                        mutation_chance=self.mutation_chance)
        return log_data
