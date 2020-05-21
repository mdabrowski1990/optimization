from typing import List, Tuple, Any, Iterator, Optional, Union
from collections import OrderedDict

from optimization.logging import Logger
from optimization.optimization_algorithms.algorithm_definition import OptimizationAlgorithm
from optimization.optimization_algorithms.stop_conditions import StopCondition
from optimization.optimization_algorithms.evolutionary_algorithms.selection import SelectionType, \
    SELECTION_FUNCTIONS, ADDITIONAL_SELECTION_PARAMETERS
from optimization.optimization_algorithms.evolutionary_algorithms.crossover import CrossoverType, \
    CROSSOVER_FUNCTIONS, ADDITIONAL_CROSSOVER_PARAMETERS
from optimization.optimization_algorithms.evolutionary_algorithms.mutation import MutationType, \
    MUTATION_FUNCTIONS, ADDITIONAL_MUTATION_PARAMETERS
from optimization.optimization_problem import OptimizationProblem, AbstractSolution


__all__ = ["EvolutionaryAlgorithm"]


class EvolutionaryAlgorithm(OptimizationAlgorithm):
    """Optimization algorithm that uses mechanism inspired by biological evolution."""

    def __init__(self, optimization_problem: OptimizationProblem, stop_condition: StopCondition,
                 population_size: int, selection_type: Union[str, SelectionType], crossover_type: [str, CrossoverType],
                 mutation_type: Union[str, MutationType], apply_elitism: bool, mutation_chance: float,
                 logger: Optional[Logger] = None, **other_params: Any) -> None:
        """
        Initialization of evolutionary algorithm.

        :param optimization_problem: Definition of optimization problem to solve.
        :param stop_condition: Definition of condition when optimization should be stopped.
        :param population_size: Number of solutions generated in one iteration.
        :param selection_type: Type of selection function to be used by the algorithm.
            It can be either passed as name of selection function or selection function itself.
        :param crossover_type: Type of crossover function to be used by the algorithm.
            It can be either passed as name of crossover function or crossover function itself.
        :param mutation_type: ype of mutation function to be used by the algorithm.
            It can be either passed as name of mutation function or mutation function itself.
        :param apply_elitism: Flag whether elitism should be applied, that means:
            Children would only replace parents if they are better in terms of objective function.
        :param mutation_chance: Probability of mutation (~chance of mutating a single gen for a single individual).
        :param logger: Configured logger that would report optimization process.
        :param other_params: Parameters configuring selection, crossover and mutation functions.

        :raise ValueError: Parameter 'other_params' contains unused values.
        """
        super().__init__(optimization_problem=optimization_problem, stop_condition=stop_condition, logger=logger)
        self.population_size = population_size
        # selection configuration
        self.selection_type = selection_type if isinstance(selection_type, str) else selection_type.value
        self.selection_function = SELECTION_FUNCTIONS[self.selection_type]
        self.selection_params = {selection_param: other_params.pop(selection_param)
                                 for selection_param in ADDITIONAL_SELECTION_PARAMETERS[self.selection_type]}
        # crossover configuration
        self.crossover_type = crossover_type if isinstance(crossover_type, str) else crossover_type.value
        self.crossover_function = CROSSOVER_FUNCTIONS[self.crossover_type]
        self.crossover_params = {crossover_param: other_params.pop(crossover_param)
                                 for crossover_param in ADDITIONAL_CROSSOVER_PARAMETERS[self.crossover_type]}
        # mutation configuration
        self.mutation_type = mutation_type if isinstance(mutation_type, str) else mutation_type.value
        self.mutation_function = MUTATION_FUNCTIONS[self.mutation_type]
        self.mutation_params = {mutation_param: other_params.pop(mutation_param)
                                for mutation_param in ADDITIONAL_MUTATION_PARAMETERS[self.mutation_type]}
        self.mutation_chance = mutation_chance
        # other params
        self.apply_elitism = apply_elitism
        self.population = None
        self.optimized_variables_number = len(self.optimization_problem.decision_variables)
        if other_params:
            raise ValueError(f"Unexpected parameters received: {other_params}")

    def _selection(self, population: List[AbstractSolution]) -> Iterator[Tuple[AbstractSolution, AbstractSolution]]:
        return self.selection_function(population_size=self.population_size, population=population,
                                       **self.selection_params)

    def _crossover(self, parents: Tuple[AbstractSolution, AbstractSolution]) -> Tuple[OrderedDict, OrderedDict]:
        return self.crossover_function(parents=parents, variables_number=self.optimized_variables_number,
                                       **self.crossover_params)

    def _mutation(self) -> List[int]:
        return self.mutation_function(variables_number=self.optimized_variables_number,
                                      mutation_chance=self.mutation_chance, **self.mutation_params)

    def _initial_iteration(self) -> List[AbstractSolution]:
        """
        Searches for solutions in first iterations of evolutionary algorithm.

        :return: List of solution sorted by objective value (calculation includes penalty).
        """
        solutions = [self.solution_type() for _ in range(self.population_size)]
        self.sort_solutions(solutions=solutions)
        self.population = solutions
        return solutions

    def _following_iteration(self) -> List[AbstractSolution]:
        new_population = []
        decision_variables_list = list(self.optimization_problem.decision_variables.items())
        for parents_pair in self._selection(population=self.population):
            # crossover
            child_1_values, child_2_values = self._crossover(parents=parents_pair)
            # mutation child 1
            for mutation_point in self._mutation():
                name, var = decision_variables_list[mutation_point]
                child_1_values[name] = var.generate_random_value()
            # mutation child 2
            for mutation_point in self._mutation():
                name, var = decision_variables_list[mutation_point]
                child_2_values[name] = var.generate_random_value()
            # create children object
            child1 = self.solution_type(**child_1_values)
            child2 = self.solution_type(**child_2_values)
            # update new population
            if self.apply_elitism:
                parent1, parent2 = parents_pair
                new_population.append(child1 if child1 >= parent1 else parent1)
                new_population.append(child2 if child2 >= parent2 else parent2)
            else:
                new_population.extend([child1, child2])
        self.sort_solutions(new_population)
        self.population = new_population
        return new_population

    def get_data_for_logging(self):
        data = super().get_data_for_logging()
        data.update({
            # selection
            "selection_params": self.selection_params, "selection_type": self.selection_type,
            # crossover
            "crossover_params": self.crossover_params, "crossover_type": self.crossover_type,
            # mutation
            "mutation_params": self.mutation_params, "mutation_type": self.mutation_type,
            "mutation_chance": self.mutation_chance,
            # other
            "apply_elitism": self.apply_elitism,
        })
        return data
