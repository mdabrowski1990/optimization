import pytest
from random import choices, uniform, randint
from string import printable
from datetime import datetime, timedelta


EXAMPLE_VALUE_TYPES = {int, float, str, bytes, list, tuple, set, dict, "function", None, datetime, timedelta}


@pytest.fixture
def random_text():
    return "".join(choices(population=printable, k=randint(4, 20)))


@pytest.fixture
def random_int():
    return randint(-1000000, 1000000)


@pytest.fixture
def random_positive_int():
    return randint(1, 1000000)


@pytest.fixture
def random_negative_int():
    return randint(-1000000, -1)


@pytest.fixture
def random_float():
    return uniform(-1000000, 1000000)


@pytest.fixture
def random_positive_timedelta():
    return timedelta(seconds=uniform(0.000001, 1000000))


@pytest.fixture
def random_negative_timedelta():
    return timedelta(seconds=-uniform(0.000001, 1000000))


@pytest.fixture
def example_value(request, random_text, random_int, random_float):
    if request.param == int:
        return random_int
    elif request.param == float:
        return random_float
    elif request.param == str:
        return random_text
    elif request.param == bytes:
        return b"something"
    elif request.param == list:
        return []
    elif request.param == tuple:
        return ()
    elif request.param == set:
        return set()
    elif request.param == dict:
        return {}
    elif request.param == "function":
        return lambda a, b: a+b
    elif request.param == datetime:
        return datetime.now()
    elif request.param == timedelta:
        return timedelta(seconds=random_float)
    else:
        return None
