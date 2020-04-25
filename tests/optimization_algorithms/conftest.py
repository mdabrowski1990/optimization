import pytest
from datetime import timedelta

from ..common_conftest import example_value, random_text, random_int, random_float, EXAMPLE_VALUE_TYPES, \
    random_positive_timedelta, random_negative_timedelta, random_positive_int, random_negative_int


EXAMPLE_VALID_SATISFYING_OBJECTIVE_VALUES = [None, 0., -1., 2154355.23453452]
EXAMPLE_VALID_MAX_ITERATIONS_WITHOUT_PROGRESS_VALUES = [None, 0, 1, 1000]
EXAMPLE_INVALID_MAX_ITERATIONS_WITHOUT_PROGRESS_VALUES = [-1, -100]
EXAMPLE_VALID_MAX_TIME_WITHOUT_PROGRESS_VALUES = [None, timedelta(microseconds=1), timedelta(seconds=9)]
