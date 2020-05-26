from typing import Any, List, Container, Iterable
from random import randint as generate_random_int
from random import uniform as generate_random_float
from random import sample, shuffle
from copy import deepcopy


__all__ = ["generate_random_int", "generate_random_float", "choose_random_value", "choose_random_values", "shuffled"]


def choose_random_value(values_pool: Container[Any]) -> Any:
    """
    Picks randomly chosen value from 'values_pool'.

    :param values_pool: Container with possible values to pick.

    :return: Randomly chosen value.
    """
    return sample(population=values_pool, k=1)[0]


def choose_random_values(values_pool: Container[Any], values_number: int) -> List[Any]:
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
