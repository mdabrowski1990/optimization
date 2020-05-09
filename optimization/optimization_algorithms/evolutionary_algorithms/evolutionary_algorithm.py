from typing import List, Tuple, Any, Callable, Iterator, Optional

from optimization.optimization_algorithms.algorithm_definition import OptimizationAlgorithm
from optimization.optimization_algorithms.stop_conditions import StopCondition
from optimization.optimization_algorithms.evolutionary_algorithms.selection import SELECTION_PARAMETERS
from optimization.optimization_algorithms.evolutionary_algorithms.crossover import CROSSOVER_PARAMETERS
from optimization.optimization_algorithms.evolutionary_algorithms.mutation import MUTATION_PARAMETERS
from optimization.optimization_problem import OptimizationProblem, AbstractSolution


__all__ = ["EvolutionaryAlgorithm"]


class EvolutionaryAlgorithm(OptimizationAlgorithm):
    """Optimization algorithm that uses mechanism inspired by biological evolution."""

    def __init__(self, optimization_problem: OptimizationProblem, stop_condition: StopCondition,
                 population_size: int, selection_type: Callable, crossover_type: Callable,
                 mutation_type: Callable, apply_elitism: bool, logger: Optional[object] = None,
                 **other_params: Any) -> None:
        # todo: description
        super().__init__(optimization_problem=optimization_problem, stop_condition=stop_condition, logger=logger)
        self.population_size = population_size
        self.selection_type = selection_type
        self.selection_params = {selection_param: other_params.pop(selection_param)
                                 for selection_param in SELECTION_PARAMETERS.get(selection_type, [])}
        self.crossover_type = crossover_type
        self.crossover_params = {crossover_param: other_params.pop(crossover_param)
                                 for crossover_param in CROSSOVER_PARAMETERS.get(crossover_type, [])}
        self.crossover_params.update(variables_number=len(optimization_problem.decision_variables),
                                     solution_class=self.solution_type)
        self.mutation_type = mutation_type
        self.mutation_params = {mutation_param: other_params.pop(mutation_param)
                                for mutation_param in MUTATION_PARAMETERS.get(mutation_type, [])}
        self.mutation_params["mutation_chance"] = other_params.pop("mutation_chance")
        self.apply_elitism = apply_elitism
        self.population = None

    def _selection(self, population: List[AbstractSolution]) -> Iterator[Tuple[AbstractSolution, AbstractSolution]]:
        return self.selection_type(population_size=self.population_size, population=population, **self.selection_params)

    def _crossover(self, parents):
        return self.crossover_type(parents=parents, **self.crossover_params)

    def _mutation(self, individual):
        self.mutation_type(individual=individual, **self.mutation_params)

    def initial_iteration(self) -> List[AbstractSolution]:
        """
        Searches for solutions in first iterations of evolutionary algorithm.

        :return: List of solution sorted by objective value (calculation includes penalty).
        """
        solutions = [self.solution_type() for _ in range(self.population_size)]
        self.sort_solution_by_objective_value(solutions=solutions)
        self.population = solutions
        return solutions

    def following_iteration(self) -> List[AbstractSolution]:
        new_population = []
        for parents_pair in self._selection(population=self.population):
            children1, children2 = self._crossover(parents=parents_pair)
            self._mutation(children1)
            self._mutation(children2)
            if self.apply_elitism:
                parent1, parent2 = parents_pair
                new_population.append(children1 if children1 >= parent1 else parent1)
                new_population.append(children2 if children2 >= parent2 else children2)
            else:
                new_population.extend([children1, children2])
        self.sort_solution_by_objective_value(new_population)
        self.population = new_population
        return new_population

    def get_data_for_logging(self):
        data = super().get_data_for_logging()
        data.update({
            "apply_elitism": self.apply_elitism,
            "selection_params": self.selection_params, "selection_type": self.selection_type.__name__,
            "crossover_params": self.crossover_params, "crossover_type": self.crossover_type.__name__,
            "mutation_params": self.mutation_params, "mutation_type": self.mutation_type.__name__,
        })
        return data
