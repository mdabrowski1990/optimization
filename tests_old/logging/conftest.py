import pytest
from operator import lt, le, eq, ne, ge, gt

from ..common_conftest import *


COMPARISON_FUNCTIONS = [lt, le, eq, ne, ge, gt]

VALID_EXAMPLE_VALUES_DIFF_LESS = [1, 100]
INVALID_EXAMPLE_VALUES_DIFF_LESS = [0, -1, -100]

VALID_EXAMPLE_VALUES_DIFF_LESSEQUAL = [0, 1, 100]
INVALID_EXAMPLE_VALUES_DIFF_LESSEQUAL = [-1, -100]

VALID_EXAMPLE_VALUES_DIFF_EQUAL = [0]
INVALID_EXAMPLE_VALUES_DIFF_EQUAL = [-1, -100, 1, 100]

VALID_EXAMPLE_VALUES_DIFF_NOTEQUAL = INVALID_EXAMPLE_VALUES_DIFF_EQUAL
INVALID_EXAMPLE_VALUES_DIFF_NOTEQUAL = VALID_EXAMPLE_VALUES_DIFF_EQUAL

VALID_EXAMPLE_VALUES_DIFF_GREATER = INVALID_EXAMPLE_VALUES_DIFF_LESSEQUAL
INVALID_EXAMPLE_VALUES_DIFF_GREATER = VALID_EXAMPLE_VALUES_DIFF_LESSEQUAL

VALID_EXAMPLE_VALUES_DIFF_GREATEREQUAL = INVALID_EXAMPLE_VALUES_DIFF_LESS
INVALID_EXAMPLE_VALUES_DIFF_GREATEREQUAL = VALID_EXAMPLE_VALUES_DIFF_LESS


VALID_COMPARISON_DATA_SETS = [
    (lt, VALID_EXAMPLE_VALUES_DIFF_LESS),
    (le, VALID_EXAMPLE_VALUES_DIFF_LESSEQUAL),
    (eq, VALID_EXAMPLE_VALUES_DIFF_EQUAL),
    (ne, VALID_EXAMPLE_VALUES_DIFF_NOTEQUAL),
    (gt, VALID_EXAMPLE_VALUES_DIFF_GREATER),
    (ge, VALID_EXAMPLE_VALUES_DIFF_GREATEREQUAL),
]

INVALID_COMPARISON_DATA_SETS = [
    (lt, INVALID_EXAMPLE_VALUES_DIFF_LESS),
    (le, INVALID_EXAMPLE_VALUES_DIFF_LESSEQUAL),
    (eq, INVALID_EXAMPLE_VALUES_DIFF_EQUAL),
    (ne, INVALID_EXAMPLE_VALUES_DIFF_NOTEQUAL),
    (gt, INVALID_EXAMPLE_VALUES_DIFF_GREATER),
    (ge, INVALID_EXAMPLE_VALUES_DIFF_GREATEREQUAL),
]
