from typing import Any, List, Container, Iterable
from random import randint as generate_random_int
from random import uniform as generate_random_float
from random import sample, shuffle


def choose_random_value(values_pool: Container[Any]) -> Any:
    """
    Picks randomly chosen value from 'values_pool'.

    :param values_pool: Container with possible values to pick.

    :return: Randomly chosen value.
    """
    return sample(population=values_pool, k=1)[0]
