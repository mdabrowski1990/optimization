# optimization - WORK IN PROGRESS, soon to be officially released!
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
    - numbers in range from -10 to 10:  
    -10, -9, -8, ..., 9, 10
- **Discrete Decision Variable** - is a variable that can take integer and/or float value within given range, 
but also having in mind defined step. Examples:
    - numbers in range from 0 to 100 with step 2:  
    0, 2, 4, ..., 98, 100
    - numbers in range from 0.1 to 10 with step 0.1:  
    0.1, 0.2, 0.3, ... 9.9, 10
- **Float Decision Variable** - is a variable that can take any float value within given range. Examples:
    - numbers in range from -1 to 1
    - numbers in range from 0.1 to 0.11
- **Choice Decision Variable** - is a variable that can take any value from given list of possible values.


### Optimization knowledge base
https://www.extremeoptimization.com/Documentation/Mathematics/Optimization/Default.aspx
https://en.wikipedia.org/wiki/Constrained_optimization
https://en.wikipedia.org/wiki/Penalty_method
https://en.wikipedia.org/wiki/Multi-objective_optimization


### Development
This part is restricted for development information

##### Static code analysis:
prospector --profile prospector_profile.yaml optimization
