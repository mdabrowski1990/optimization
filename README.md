# optimization
Optimization is a process of searching a better solution in order to find the best possible solution.

This package provides tools for defining optimization problems and effective search using some of the most popular 
(mostly heuristic) algorithms.


### Optimization problem definition
Before we can start an optimization process, it is necessary to define mathematical description of a problem.
In order to fully describe the problem, we must define following aspects of it:
- **Decision variables** - are variables for which optimal values we are looking for (e.g. dimension of some object).
- **Constraints** - are also called **soft constrains**. They are helping to penalize solution if any of restriction 
is not met. For example if we prefer solutions where dimension a should be greater than dimension b, we can create 
constraint a > b to promote such solutions over all others.
- **Penalty function** - is a function that calculates penalty value basing on constraints that are not meet. 
It is equal 0 if all constraints are satisfied.
- **Objective function** - is a function that determines quality of a solution. It can be considered in single or multi 
criteria:
    - Single Criteria - there is only one criteria that determines quality of a solution (e.g. costs, lines of code, 
    time needed)
    - Multi Criteria - quality of a solution is determined as mix of a few factors (e.g. both costs and durability 
    of some product) 
     
    Example criteria that might be calculates as part of objective function:
    - costs
    - time needed
    - durability, hardiness, toughness
    - materials needed
    - number of wastes or pollutions produced
- **Optimization type** - determines whether we look for solution with the lowest or the highest objective value.


#### Decision Variables
As a part of optimization problem definition, we must have a common way of defining proper Decision Variables. 
In this package you can find following types of decision variables:

- **Integer Decision Variable** - is a variable that can take any integer value within given range. Examples:
  - numbers in range from 0 to 100:  
  0, 1, 2, ..., 99, 100  
  ```python
  import optimization
      
  int_var = optimization.IntegerVariable(min_value=0, max_value=100)
  ```
  - numbers in range from -10 to 10:  
  -10, -9, -8, ..., 9, 10
  ```python
  import optimization
      
  int_var = optimization.IntegerVariable(min_value=-10, max_value=10)
  ```
- **Discrete Decision Variable** - is a variable that can take integer and/or float value within given range, 
but also having in mind defined step. Examples:
  - numbers in range from 0 to 100 with step 2:  
  0, 2, 4, ..., 98, 100
  ```python
  import optimization
      
  discrete_var = optimization.DiscreteVariable(min_value=0, max_value=100, step=2)
  ```
  - numbers in range from 0.1 to 10 with step 0.1:  
  0.1, 0.2, 0.3, ... 9.9, 10
  ```python
  import optimization
      
  discrete_var = optimization.DiscreteVariable(min_value=0.1, max_value=10, step=0.1)
  ```
- **Float Decision Variable** - is a variable that can take any float value within given range. Examples:
  - numbers in range from -1 to 1
  ```python
  import optimization
      
  float_var = optimization.FloatVariable(min_value=-1., max_value=1.)
  ```
  - numbers in range from 0.1 to 0.11
  ```python
  import optimization
      
  float_var = optimization.FloatVariable(min_value=0.1, max_value=0.11)
  ```
- **Choice Decision Variable** - is a variable that can take any value from given list of possible values.
  - one of colors from following list: "yellow", "orange", "green", "blue", "pink", "white"
  ```python
  import optimization
      
  choice_var = optimization.ChoiceVariable(possible_values={"yellow", "orange", "green", "blue", "pink", "white"})
  ```
  - one of values from following list: 0, 1.25, 6.5, 987
  ```python
  import optimization
      
  choice_var = optimization.ChoiceVariable(possible_values={0, 1.25, 6.5, 987})
  ```

### Stop conditions
Before we can start an optimization process, it is necessary to determine when to stop it.
Using this package you can define stop conditions object that will help you to stop further optimization in one of 
following cases that you can configure:
- **Time limit expired** - you have to (to avoid infinite process) define maximal time that optimization process may last.
- **Satisfying solution found** - you can define objective value (including penalty) that is boundary value. 
    If a solution with better value (lower in case of minimization, higher in case of maximization), 
    then optimization process will be stopped in this iteration.
- **No progress** - this option gives you possibility to force optimization process stop either after:
    - ```n``` optimization algorithm iteration during which no better solution was found
    - ```t``` time during which no better solution was found by optimization algorithm  

Examples:
1) Stop condition for optimization process that is supposed to last 1 hour:
```python
import datetime

import optimization

stop_condition_1_hour = optimization.StopCondition(time_limit=datetime.timedelta(hours=1))
```
2) Stop conditions for optimization process to last:
- maximal 1 day
- until solution with objective value of ```100``` or better is found
- stop optimization process if optimization algorithm could not found better solution for 1000 iterations
- stop optimization process if optimization algorithm could not found better solution for 2 hours
```python
import datetime

import optimization

stop_conditions = optimization.StopCondition(time_limit=datetime.timedelta(days=1),
                                             satisfying_objective_value=100,
                                             max_iter_without_progress=1000,
                                             max_time_without_progress=datetime.timedelta(hours=2))
```


### Logging
If you are interested in logging optimization process, you can:
- use ```Logger``` (build-in functionality) 
```python
import optimization

optimization.Logger(logs_dir="path\\to\\directory\\where\\you\\want\\to\\have\\logs",
                    verbosity=optimization.LoggingVerbosity.AllSolutions,  # level of logs verbosity you want
                    log_format=optimization.LoggingFormat.YAML)  # format in which logs to be created
```
- create your own logged basing on ```AbstractLogger```
```python
import optimization

class CustomLogger(optimization.AbstractLogger):
    ...  # your custom implementation here with all abstract methods implemented
```


### Optimization algorithms
In order to execute optimization process, you must select and optimization algorithm, configure it and execute 
optimization process.

#### Random algorithm
Random algorithm searches for optimal solution by simply checking totally random solutions.
It is advised to use it only for research, comparison or education purposes.  
Example use:
```python
import optimization

stop_conditions = optimization.StopConditions(...)  # look above to learn how to configure it properly
problem = optimization.OptimizationProblem(...)  # look above to learn how to configure it properly

random_algorithm = optimization.RandomAlgorithm(
  stop_conditions=stop_conditions, 
  problem=problem
)
random_algorithm.perform_optimization()
```

#### Evolutionary Algorithm
Evolutionary algorithm simulates mechanisms that takes place in nature (biological evolution, natural selection, reproduction, mutation etc.). 
Each solution is considered an individual and a group of solutions is considered a population.
In following iterations of this algorithm a population is evolving in order to produce the best adjusted individuals.
Evolution process consists of 3 stages:
- selection - pairs of individuals are chosen to be parents
- crossover - parents genes (decision variables values) are mixed and passed for two new children (two new solutions are created)
- mutation - random gene(s) (decision variables values) of each child might be changed  

Example use:
```python
import optimization

stop_conditions = optimization.StopConditions(...)  # look above to learn how to configure it properly
problem = optimization.OptimizationProblem(...)  # look above to learn how to configure it properly

evolutionary_algorithm = optimization.EvolutionaryAlgorithm(
    stop_conditions=stop_conditions, 
    problem=problem,
    population_size=100, 
    selection_type=optimization.SelectionType.Uniform, 
    crossover_type=optimization.CrossoverType.SinglePoint,
    mutation_type=optimization.MutationType.Probabilistic, 
    mutation_chance=0.1, 
    apply_elitism=False
)
evolutionary_algorithm.perform_optimization()
```

#### Adaptive Evolutionary Algorithm
Adaptive evolutionary algorithm acts like evolutionary algorithm, but it performs two level optimization (instead of just one) and solves two problems at the same time.
These two problems are:
- optimization problem - the main problem that the algorithm faces (look [Optimization problem definition](#optimization-problem-definition)) 
- adaptation problem - problem related to algorithm configuration 
  (searching perfect algorithm configuration to be possibly the most efficient in searching of optimization problem solution described by EvolutionaryAlgorithmAdaptationProblem)
  
Example use:
```python
import optimization

stop_conditions = optimization.StopConditions(...)  # look above to learn how to configure it properly
problem = optimization.OptimizationProblem(...)  # look above to learn how to configure it properly
adaptation_problem = optimization.EvolutionaryAlgorithmAdaptationProblem(
    adaptation_type=optimization.AdaptationType.BestSolution,
)

adaptive_evolutionary_algorithm = optimization.AdaptiveEvolutionaryAlgorithm(
    adaptation_problem=adaptation_problem,
    problem=problem,
    stop_conditions=stop_conditions,
    selection_type=optimization.SelectionType.Uniform, 
    crossover_type=optimization.CrossoverType.SinglePoint,
    mutation_type=optimization.MutationType.Probabilistic, 
    mutation_chance=0.05, 
)
adaptive_evolutionary_algorithm.perform_optimization()
```

### More Examples
Examples can be found in [examples directory][myexample].

[myexample]: /examples "optimization examples"

### Optimization knowledge base
https://www.extremeoptimization.com/Documentation/Mathematics/Optimization/Default.aspx
https://en.wikipedia.org/wiki/Constrained_optimization
https://en.wikipedia.org/wiki/Penalty_method
https://en.wikipedia.org/wiki/Multi-objective_optimization
