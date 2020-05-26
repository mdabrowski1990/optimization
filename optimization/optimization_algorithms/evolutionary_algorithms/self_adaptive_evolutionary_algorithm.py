from typing import Union, Optional, Any, Tuple, Iterable, List
from datetime import datetime
from enum import Enum
from collections import OrderedDict
from abc import abstractmethod

from optimization.logging import SEALogger
from optimization.utilities import shuffled
from optimization.optimization_algorithms.evolutionary_algorithms.evolutionary_algorithm import EvolutionaryAlgorithm
from optimization.optimization_algorithms.stop_conditions import StopCondition
from optimization.optimization_algorithms.evolutionary_algorithms.selection import SelectionType, SELECTION_FUNCTIONS, \
    ADDITIONAL_SELECTION_PARAMETERS
from optimization.optimization_algorithms.evolutionary_algorithms.crossover import CrossoverType, CROSSOVER_FUNCTIONS, \
    ADDITIONAL_CROSSOVER_PARAMETERS
from optimization.optimization_algorithms.evolutionary_algorithms.mutation import MutationType, MUTATION_FUNCTIONS, \
    ADDITIONAL_MUTATION_PARAMETERS
from optimization.optimization_problem import OptimizationProblem, AbstractSolution, \
    ChoiceVariable, FloatVariable, IntegerVariable


__all__ = ["SEAOptimizationType", "SEAOptimizationProblem", "SEA"]


class SEAOptimizationType(Enum):
    BEST_SOLUTIONS = "Best solutions"
    BEST_SOLUTIONS_PROGRESS = "Best solutions progress"
    BEST_SOLUTIONS_PERCENTILE = "Best solutions percentile"
    BEST_SOLUTIONS_PERCENTILE_PROGRESS = "Best solutions percentile progress"


class SEAOptimizationProblem(OptimizationProblem):
    def __init__(self, population_size: Tuple[int, int], selection_types: Iterable[SelectionType],
                 crossover_types: Iterable[CrossoverType], mutation_types: Iterable[MutationType],
                 sea_optimization_type: SEAOptimizationType, optimization_problem: OptimizationProblem,
                 apply_elitism: Iterable[bool] = (True, False), mutation_chance: Tuple[float, float] = (0.01, 0.2),
                 **optional_params):
        # main decision variables definition
        main_decision_variables = OrderedDict(
            population_size=ChoiceVariable(possible_values=set(range(population_size[0], population_size[1]+1, 2))),
            selection_type=ChoiceVariable(possible_values=set(selection_types)),
            crossover_type=ChoiceVariable(possible_values=set(crossover_types)),
            mutation_type=ChoiceVariable(possible_values=set(mutation_types)),
            apply_elitism=ChoiceVariable(possible_values=set(apply_elitism)),
            mutation_chance=FloatVariable(min_value=mutation_chance[0], max_value=mutation_chance[1])
        )
        # side decision variables definition (they are only used for certain selection/crossover/mutation type)
        _variables_number = len(optimization_problem.decision_variables)
        side_decision_variables = {}
        if SelectionType.TOURNAMENT in selection_types:
            _min, _max = optional_params.pop("tournament_group_size", (2, 6))
            side_decision_variables.update(tournament_group_size=IntegerVariable(min_value=_min, max_value=_max))
        if SelectionType.ROULETTE in selection_types:
            _min, _max = optional_params.pop("roulette_bias", (1., 100.))
            side_decision_variables.update(roulette_bias=FloatVariable(min_value=_min, max_value=_max))
        if SelectionType.RANKING in selection_types:
            _min, _max = optional_params.pop("ranking_bias", (1., 2.))
            side_decision_variables.update(ranking_bias=FloatVariable(min_value=_min, max_value=_max))
        if CrossoverType.MULTI_POINT in crossover_types:
            _min, _max = optional_params.pop("crossover_points_number", (1, _variables_number-1))
            side_decision_variables.update(crossover_points_number=IntegerVariable(min_value=_min, max_value=_max))
        if CrossoverType.ADAPTIVE in crossover_types:
            _min, _max = (0, (1 << _variables_number) - 1)
            side_decision_variables.update(crossover_pattern=IntegerVariable(min_value=_min, max_value=_max))
        if MutationType.MULTI_POINT in mutation_types:
            _min, _max = optional_params.pop("mutation_points_number", (1, _variables_number - 1))
            side_decision_variables.update(mutation_points_number=IntegerVariable(min_value=_min, max_value=_max))
        # determine objective value
        if sea_optimization_type in {SEAOptimizationType.BEST_SOLUTIONS, SEAOptimizationType.BEST_SOLUTIONS_PROGRESS}:
            self.tracked_solutions = optional_params.pop("tracked_solutions_number", 3)
        elif sea_optimization_type in {SEAOptimizationType.BEST_SOLUTIONS_PERCENTILE,
                                       SEAOptimizationType.BEST_SOLUTIONS_PERCENTILE_PROGRESS}:
            self.tracked_solutions_percentile = optional_params.pop("tracked_solutions_percentile", 10)
        else:
            raise NotImplementedError("This type of SAE Optimization is currently not supported.")
        # set values of the object
        super().__init__(decision_variables=main_decision_variables, constraints={}, penalty_function=lambda **_: 0,
                         objective_function=lambda: None,  # unused by SAELower
                         optimization_type=optimization_problem.optimization_type)
        self.additional_decision_variables = side_decision_variables
        self.optimization_problem = optimization_problem
        self.sea_optimization_type = sea_optimization_type

    def spawn_solution_definition(self) -> type:
        class ThisSEALower(SEALower):
            optimization_problem = self
        return ThisSEALower

    def get_data_for_logging(self) -> dict:
        data = {
            "decision_variables": {
                var_name: var_object.get_data_for_logging() for var_name, var_object in self.decision_variables.items()
            },
            "additional_decision_variables": {
                var_name: var_object.get_data_for_logging()
                for var_name, var_object in self.additional_decision_variables.items()
            },
            "optimization_type": self.optimization_type.value,
            "sea_optimization_type": self.sea_optimization_type.value,
        }
        if "tracked_solutions" in self.__dict__:
            data.update(tracked_solutions=self.tracked_solutions)
        if "tracked_solutions_percentile" in self.__dict__:
            data.update(tracked_solutions_percentile=self.tracked_solutions_percentile)
        return data


class SEALower(AbstractSolution, EvolutionaryAlgorithm):
    logger: SEALogger = None
    stop_condition: StopCondition = None

    @property
    @abstractmethod
    def optimization_problem(self) -> SEAOptimizationProblem:
        """
        Abstract definition of a property that stores reference to optimization problem.

        :raise NotImplementedError: Abstract method was called.
        """
        raise NotImplementedError("You have called abstract property 'optimization_problem' of 'AbstractSolution' "
                                  "abstract class.")

    def __init__(self, sea_iteration, order_index, population, **decision_variables_values: Any) -> None:
        main_decision_variables_values = {}
        side_decision_variables_values = {}
        for name, value in decision_variables_values.items():
            if name in self.optimization_problem.decision_variables:
                main_decision_variables_values[name] = value
            elif name in self.optimization_problem.additional_decision_variables:
                side_decision_variables_values[name] = value
            else:
                raise ValueError(f"Unexpected variable received: {name}={value}")
        AbstractSolution.__init__(self, **main_decision_variables_values)
        for main_function, additional_params in (("selection_type", ADDITIONAL_SELECTION_PARAMETERS),
                                                 ("crossover_type", ADDITIONAL_CROSSOVER_PARAMETERS),
                                                 ("mutation_type", ADDITIONAL_MUTATION_PARAMETERS)):
            for param_name in additional_params[self.__getattribute__(main_function)]:
                if param_name not in side_decision_variables_values:
                    side_decision_variables_values[param_name] = \
                        self.optimization_problem.additional_decision_variables[param_name].generate_random_value()
        self.additional_decision_variables_values = side_decision_variables_values
        # AE params
        self.population = population[:self.population_size]
        self.sea_iteration = sea_iteration
        self.order_index = order_index
        self.optimized_variables_number = len(self.optimization_problem.optimization_problem.decision_variables)
        self.best_solution = None
        self.start_time = None
        self.end_time = None
        self.solution_type = self.optimization_problem.optimization_problem.spawn_solution_definition()

    def _mutation(self, individual_values: OrderedDict) -> None:
        decision_variables_list = list(self.optimization_problem.optimization_problem.decision_variables.items())
        for mutation_point in self.mutation_function(variables_number=self.optimized_variables_number,
                                                     mutation_chance=self.mutation_chance, **self.mutation_params):
            name, var = decision_variables_list[mutation_point]
            individual_values[name] = var.generate_random_value()

    @property
    def population_size(self):
        return self.decision_variables_values["population_size"]

    @property
    def selection_type(self):
        return self.decision_variables_values["selection_type"].value

    @property
    def selection_function(self):
        return SELECTION_FUNCTIONS[self.selection_type]

    @property
    def selection_params(self):
        return {selection_param: self.additional_decision_variables_values[selection_param]
                for selection_param in ADDITIONAL_SELECTION_PARAMETERS[self.selection_type]}

    @property
    def crossover_type(self):
        return self.decision_variables_values["crossover_type"].value

    @property
    def crossover_function(self):
        return CROSSOVER_FUNCTIONS[self.crossover_type]

    @property
    def crossover_params(self):
        return {crossover_param: self.additional_decision_variables_values[crossover_param]
                for crossover_param in ADDITIONAL_CROSSOVER_PARAMETERS[self.crossover_type]}

    @property
    def mutation_type(self):
        return self.decision_variables_values["mutation_type"].value

    @property
    def mutation_function(self):
        return MUTATION_FUNCTIONS[self.mutation_type]

    @property
    def mutation_params(self):
        return {mutation_param: self.additional_decision_variables_values[mutation_param]
                for mutation_param in ADDITIONAL_MUTATION_PARAMETERS[self.mutation_type]}

    @property
    def mutation_chance(self):
        return self.decision_variables_values["mutation_chance"]

    @property
    def apply_elitism(self):
        return self.decision_variables_values["apply_elitism"]

    def _calculate_objective(self) -> float:
        self.perform_optimization()
        return self._objective_value

    def get_data_for_logging(self):
        return EvolutionaryAlgorithm.get_data_for_logging(self)

    def _initial_iteration(self) -> List[AbstractSolution]:
        solutions = self.population
        for _ in range(self.population_size - len(solutions)):
            solutions.append(self.solution_type())
        self.sort_solutions(solutions=solutions)
        self.population = solutions
        self.best_solution = solutions[0]
        return solutions

    def perform_optimization(self) -> AbstractSolution:
        sea_opt_type = self.optimization_problem.sea_optimization_type
        if sea_opt_type in {SEAOptimizationType.BEST_SOLUTIONS, SEAOptimizationType.BEST_SOLUTIONS_PROGRESS}:
            tracked_solutions = self.optimization_problem.tracked_solutions
        elif sea_opt_type in {SEAOptimizationType.BEST_SOLUTIONS_PERCENTILE,
                              SEAOptimizationType.BEST_SOLUTIONS_PERCENTILE_PROGRESS}:
            result, rest = divmod(self.population_size * self.optimization_problem.tracked_solutions_percentile, 100)
            tracked_solutions = result + 1 if rest else result
        else:
            raise ValueError
        # initial iteration
        i = 0
        self.start_time = datetime.now()
        solutions = self._initial_iteration()
        if self.logger is not None:
            self.logger.log_sea_lower_solutions(sea_upper_iteration=self.sea_iteration, sea_lower_iteration=i,
                                                sea_lower_index=self.order_index, solutions=solutions)
        if sea_opt_type in {SEAOptimizationType.BEST_SOLUTIONS_PROGRESS,
                            SEAOptimizationType.BEST_SOLUTIONS_PERCENTILE_PROGRESS}:
            initial_objective_state = sum(solution.get_objective_value_with_penalty()
                                          for solution in solutions[:tracked_solutions])
        else:
            best_solutions = solutions[:tracked_solutions]
        # following iterations
        while not self._is_stop_condition_achieved(solutions=solutions):
            i += 1
            solutions = self._following_iteration()
            _tmp = [self.best_solution, solutions[0]]
            self.sort_solutions(_tmp)
            self.best_solution = _tmp[0]
            if self.logger is not None:
                self.logger.log_sea_lower_solutions(sea_upper_iteration=self.sea_iteration, sea_lower_iteration=i,
                                                    sea_lower_index=self.order_index, solutions=solutions)
            if self.optimization_problem.optimization_type in {SEAOptimizationType.BEST_SOLUTIONS,
                                                               SEAOptimizationType.BEST_SOLUTIONS_PERCENTILE}:
                best_solutions.extend(solutions[:tracked_solutions])
                self.sort_solutions(best_solutions)
                best_solutions = best_solutions[:tracked_solutions]
        # end
        if sea_opt_type in {SEAOptimizationType.BEST_SOLUTIONS_PROGRESS,
                            SEAOptimizationType.BEST_SOLUTIONS_PERCENTILE_PROGRESS}:
            end_objective_state = sum(solution.get_objective_value_with_penalty()
                                      for solution in solutions[:tracked_solutions])
            self._objective_value = end_objective_state / initial_objective_state
        else:
            self._objective_value = sum(solution.get_objective_value_with_penalty() for solution in best_solutions) \
                                    / tracked_solutions
        self.end_time = datetime.now()
        return self.best_solution


class SEA(EvolutionaryAlgorithm):
    def __init__(self, sae_optimization_problem: SEAOptimizationProblem, stop_condition: StopCondition,
                 population_size: int, iterations_number: int, selection_type: Union[str, SelectionType],
                 crossover_type: [str, CrossoverType], mutation_type: Union[str, MutationType],
                 mutation_chance: float, logger: Optional[SEALogger] = None, **other_params: Any) -> None:
        super().__init__(optimization_problem=sae_optimization_problem, stop_condition=stop_condition,
                         population_size=population_size, selection_type=selection_type, crossover_type=crossover_type,
                         mutation_type=mutation_type, apply_elitism=False, mutation_chance=mutation_chance,
                         logger=logger, **other_params)
        # update stop condition for SEALower
        sea_lower_time_limit = stop_condition.time_limit / (5*population_size*iterations_number)
        self.solution_type.stop_condition = StopCondition(time_limit=sea_lower_time_limit)
        self.solution_type.logger = logger

    def _crossover(self, parents: Tuple[SEALower, SEALower]) -> Tuple[OrderedDict, OrderedDict]:
        child_1_main_values, child_2_main_values = super()._crossover(parents=parents)
        child_1_side_values, child_2_side_values = {}, {}
        for main_function, additional_params in (("selection_type", ADDITIONAL_SELECTION_PARAMETERS),
                                                 ("crossover_type", ADDITIONAL_CROSSOVER_PARAMETERS),
                                                 ("mutation_type", ADDITIONAL_MUTATION_PARAMETERS)):
            val1, val2 = child_1_main_values[main_function], child_2_main_values[main_function]
            child_1_params, child_2_params = additional_params[val1.value], additional_params[val2.value]
            if not child_1_params and not child_2_params:
                continue
            elif child_1_params == child_2_params:
                for param in child_1_params:
                    child_1_side_values[param], child_2_side_values[param] = \
                        shuffled([parents[0].additional_decision_variables_values[param],
                                  parents[1].additional_decision_variables_values[param]])
                continue
            for param in child_1_params:
                if param in parents[0].additional_decision_variables_values:
                    child_1_side_values[param] = parents[0].additional_decision_variables_values[param]
                elif param in parents[1].additional_decision_variables_values:
                    child_1_side_values[param] = parents[1].additional_decision_variables_values[param]
                else:
                    raise ValueError
            for param in child_2_params:
                if param in parents[0].additional_decision_variables_values:
                    child_2_side_values[param] = parents[0].additional_decision_variables_values[param]
                elif param in parents[1].additional_decision_variables_values:
                    child_2_side_values[param] = parents[1].additional_decision_variables_values[param]
                else:
                    raise ValueError
        return OrderedDict(**child_1_main_values, **child_1_side_values), \
            OrderedDict(**child_2_main_values, **child_2_side_values)

    def _mutation(self, individual_values: OrderedDict) -> None:
        variables_to_mutate = set()
        individual_variables_names = list(individual_values.keys())
        for mutation_point in self.mutation_function(variables_number=len(individual_values),
                                                     mutation_chance=self.mutation_chance, **self.mutation_params):
            variables_to_mutate.add(individual_variables_names[mutation_point])
        # perform mutation
        for main_function, additional_params in (("selection_type", ADDITIONAL_SELECTION_PARAMETERS),
                                                 ("crossover_type", ADDITIONAL_CROSSOVER_PARAMETERS),
                                                 ("mutation_type", ADDITIONAL_MUTATION_PARAMETERS)):
            if main_function in variables_to_mutate:
                variables_to_mutate.discard(main_function)
                new_value = self.optimization_problem.decision_variables[main_function].generate_random_value()
                old_value = individual_values[main_function]
                if new_value != old_value:
                    individual_values[main_function] = new_value
                    variables_to_mutate = variables_to_mutate.difference(additional_params[old_value.value])
                    variables_to_mutate = variables_to_mutate.union(additional_params[new_value.value])
                    for _old_side_param in additional_params[old_value.value]:
                        individual_values.pop(_old_side_param)
        for var_name in variables_to_mutate:
            if var_name in self.optimization_problem.decision_variables:
                value = self.optimization_problem.decision_variables[var_name].generate_random_value()
            else:
                value = self.optimization_problem.additional_decision_variables[var_name].generate_random_value()
            individual_values[var_name] = value

    def _initial_iteration(self) -> List[SEALower]:
        sea_lower_population = []
        for i in range(self.population_size):
            new_sea_lower = self.solution_type(sea_iteration=0, order_index=i, population=[])
            sea_lower_population.append(new_sea_lower)
        self.sort_solutions(sea_lower_population)
        self.population = sea_lower_population
        return sea_lower_population

    def _following_iteration(self, iteration_index: int) -> List[AbstractSolution]:
        new_population = []
        for pair_i, (parent1, parent2) in enumerate(self._selection(population=self.population)):
            # crossover
            child_1_values, child_2_values = self._crossover(parents=(parent1, parent2))
            # mutation
            self._mutation(child_1_values)
            self._mutation(child_2_values)
            # create children object
            lower_population = parent1.population + parent2.population
            lower_population = shuffled(lower_population)
            p_1_i = child_1_values["population_size"]
            if len(lower_population) >= child_2_values["population_size"]:
                p_2_i = len(lower_population) - child_2_values["population_size"]
            else:
                p_2_i = 0
            child1 = self.solution_type(population=lower_population[:p_1_i],
                                        sea_iteration=iteration_index, order_index=2*pair_i, **child_1_values)
            child2 = self.solution_type(population=lower_population[p_2_i:],
                                        sea_iteration=iteration_index, order_index=2*pair_i + 1, **child_2_values)
            # update new population
            new_population.extend([child1, child2])
        self.sort_solutions(new_population)
        self.population = new_population
        return new_population

    def perform_optimization(self) -> AbstractSolution:
        # pre start
        if self.logger is not None:
            self.logger.log_at_start(sea=self)
        # initial iteration
        i = 0
        self.start_time = datetime.now()
        sae_lower_list = self._initial_iteration()
        best_solutions = [sae_lower.best_solution for sae_lower in sae_lower_list]
        self.sort_solutions(best_solutions)
        best_solution = best_solutions[0]
        if self.logger is not None:
            self.logger.log_solutions(iteration=i, solutions=sae_lower_list)
        # following iterations
        while not self._is_stop_condition_achieved(solutions=sae_lower_list):
            i += 1
            sae_lower_list = self._following_iteration(iteration_index=i)
            best_solutions = [best_solution] + [sae_lower.best_solution for sae_lower in sae_lower_list]
            self.sort_solutions(best_solutions)
            best_solution = best_solutions[0]
            if self.logger is not None:
                self.logger.log_solutions(iteration=i, solutions=sae_lower_list)
        # end
        self.end_time = datetime.now()
        if self.logger is not None:
            self.logger.log_at_end(best_solution=best_solution)
        return best_solution

