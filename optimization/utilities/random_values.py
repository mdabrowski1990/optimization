"""
Module with (pseudo) random values generation and random shuffling.

Note: It uses built-in random package (it could have been used directly), but this module is meant to be
abstraction layer that enables to easily change this implementation.
"""

__all__ = ["generate_random_int", "generate_random_float", "choose_random_value", "choose_random_values",
           "shuffle", "shuffled"]


from typing import Any, Set, List, Iterable
from random import randint as generate_random_int
from random import uniform as generate_random_float
from random import sample, shuffle
from copy import deepcopy


def choose_random_value(values_pool: Set[Any]) -> Any:
    """
    Picks randomly chosen value from 'values_pool'.

    :param values_pool: Container with possible values to pick.

    :return: Randomly chosen value.
    """
    return sample(population=values_pool, k=1)[0]


def choose_random_values(values_pool: Set[Any], values_number: int) -> List[Any]:
    """
    Picks randomly chosen values from 'values_pool'.

    :param values_pool: Container with possible values to pick.
    :param values_number: Number of values to be picked.

    :return: List with two randomly picked values.
    """
    return sample(population=values_pool, k=values_number)


def shuffled(values: Iterable[Any]) -> List[Any]:
    """
    Randomly shuffles values.

    :param values: Iterable with values to shuffle.

    :return: List with randomly ordered values.
    """
    values = list(deepcopy(values))
    shuffle(values)
    return values
