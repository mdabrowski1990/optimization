"""
Helper function package.

It contains:
    - abstraction layer for random values generation
    - binary search algorithm
"""

from .random_values import generate_random_int, generate_random_float, \
    choose_random_value, choose_random_values, choose_random_value_with_weights, \
    shuffle, shuffled
from .other import binary_search
