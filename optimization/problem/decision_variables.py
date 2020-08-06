"""
Decision variables definition.

Decision variables are variables for which optimal values (in certain criteria) are searched.
Decision variables are main part of optimization problem model.
"""

__all__ = ["DecisionVariable", "IntegerVariable", "DiscreteVariable", "FloatVariable", "ChoiceVariable"]


from typing import Any, Union, Dict, Iterable
from abc import ABC, abstractmethod

from optimization.utilities.random_values import generate_random_int, generate_random_float, choose_random_value


class DecisionVariable(ABC):
    """Abstract definition of decision variable."""

    @abstractmethod
    def generate_random_value(self) -> Any:
        """:return: Random value according to this Decision Variable definition."""
        ...

    @abstractmethod
    def is_proper_value(self, value: Any) -> bool:
        """:return: True if value is compatible with this Decision Variable definition, False otherwise."""
        ...

    @abstractmethod
    def get_log_data(self) -> Dict[str, str]:
        """
        Gets data for logging purposes.

        :return: Dictionary with this Decision Variable crucial data.
        """
        ...


class IntegerVariable(DecisionVariable):
    """
    Integer Decision Variable definition.

    This class is Decision Variable type of variable that can only take integer value within given range with step 1.
    """

    def __init__(self, min_value: int, max_value: int) -> None:
        """
        Creates definition of Integer Decision Variable.

        :param min_value: Minimal value that this variable can store.
        :param max_value: Maximal value that this variable can store.

        :raise TypeError: Parameter 'min_value' or 'max_value' is not int type.
        :raise ValueError: Value of parameter 'min_value' is greater or equal value of 'max_value'.
        """
        if not isinstance(min_value, int):
            raise TypeError(f"Value of 'min_value' parameter is not int type. Actual value: '{min_value}'.")
        if not isinstance(max_value, int):
            raise TypeError(f"Value of 'max_value' parameter is not int type. Actual value: '{max_value}'.")
        if min_value >= max_value:
            raise ValueError(f"Value of 'min_value' parameter is not less than value of 'max_value' parameter. "
                             f"Actual values: min_value={min_value}, max_value={max_value}.")
        self.min_value = min_value
        self.max_value = max_value

    def generate_random_value(self) -> int:
        """:return: Random value according to this Integer Variable definition."""
        return generate_random_int(self.min_value, self.max_value)

    def is_proper_value(self, value: Any) -> bool:
        """:return: True if value is compatible with this Integer Variable definition, False otherwise."""
        return isinstance(value, int) and self.min_value <= value <= self.max_value

    def get_log_data(self) -> Dict[str, Union[str, int]]:
        """
        Gets data for logging purposes.

        :return: Dictionary with this Integer Variable crucial data.
        """
        return {
            "type": self.__class__.__name__,
            "min_value": self.min_value,
            "max_value": self.max_value,
        }


class DiscreteVariable(DecisionVariable):
    """
    Discrete Decision Variable definition.

    This class is Decision Variable type of variable that can take integer and/or float value within given range with
    given step. Examples:
        - odd integer in inclusive range 1-99 (1, 3, 5, ..., 97, 99):
            min_value=1, max_value=99, step=2
        - even integer in inclusive range 2-10 (2, 4, 6, 8, 10):
            min_value=2, max_value=10, step=2
        - one of values from arithmetic sequence 0, 0.1, 0.2, ..., 9.9, 10:
            min_value=0, max_value=10, step=0.1

    Note: If step==1 and min_value is int type, then 'IntegerVariable' can be used instead.
    !WARNING! This variable has precision issues, you can use 'IntegerVariable' and update objective function to
        properly recalculate the result.
    """

    def __init__(self, min_value: Union[int, float], max_value: Union[int, float], step: Union[int, float]) -> None:
        """
        Creates definition of Discrete Decision Variable.

        Possible value are equal: [min_value] + [i]*[step]
        where [i] such that: [min_value] + [i]*[step] <= [max_value]

        :param min_value: Minimal value that this variable can store.
        :param max_value: Maximal value that this variable can store.
        :param step: Difference between following possible values.

        :raise TypeError: Parameter 'min_value' or 'max_value' is not int or float type.
        :raise ValueError: Value of parameter 'min_value' is greater or equal value of 'max_value'
            or 'step' is lower equal 0.
        """
        if not isinstance(min_value, (int, float)):
            raise TypeError(f"Value of 'min_value' parameter is not int nor float type. Actual value: '{min_value}'.")
        if not isinstance(max_value, (int, float)):
            raise TypeError(f"Value of 'max_value' parameter is not int nor float type. Actual value: '{max_value}'.")
        if not isinstance(step, (int, float)):
            raise TypeError(f"Value of 'step' parameter is not int nor float type. Actual value: '{step}'.")
        if min_value >= max_value:
            raise ValueError(f"Value of 'min_value' parameter is not less than value of 'max_value' parameter. "
                             f"Actual values: min_value={min_value}, max_value={max_value}.")
        if step <= 0:
            raise ValueError(f"Value of 'step' parameter less or equal 0. Actual value: {step}.")
        self.min_value = min_value
        self.max_value = max_value
        self.step = step
        self._max_rand = int((self.max_value - self.min_value) // self.step)

    def generate_random_value(self) -> Union[int, float]:
        """:return: Random value according to this Discrete Variable definition."""
        return self.min_value + generate_random_int(0, self._max_rand)*self.step

    def is_proper_value(self, value: Any) -> bool:
        """:return: True if value is compatible with this Discrete Variable definition, False otherwise."""
        if isinstance(value, (int, float)) and self.min_value <= value <= self.max_value:
            _rest = (value - self.min_value) % self.step
            return round(_rest, 15) in {self.step, 0.}
        return False

    def get_log_data(self) -> Dict[str, Union[str, float, int]]:
        """
        Gets data for logging purposes.

        :return: Dictionary with this Discrete Variable crucial data.
        """
        return {
            "type": self.__class__.__name__,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "step": self.step,
        }


class FloatVariable(DecisionVariable):
    """
    Float Decision Variable definition.

    This class is Decision Variable type of variable that can only take float value within given range.
    """

    def __init__(self, min_value: float, max_value: float) -> None:
        """
        Creates definition of Float Decision Variable.

        :param min_value: Minimal value that this variable can store.
        :param max_value: Maximal value that this variable can store.

        :raise TypeError: Parameter 'min_value' or 'max_value' is not float type.
        :raise ValueError: Value of parameter 'min_value' is greater  equal value of 'max_value'.
        """
        if not isinstance(min_value, float):
            raise TypeError(f"Value of 'min_value' parameter is not float type. Actual value: '{min_value}'.")
        if not isinstance(max_value, float):
            raise TypeError(f"Value of 'max_value' parameter is not float type. Actual value: '{max_value}'.")
        if min_value >= max_value:
            raise ValueError(f"Value of 'min_value' parameter is not less than value of 'max_value' parameter. "
                             f"Actual values: min_value={min_value}, max_value={max_value}.")
        self.min_value = min_value
        self.max_value = max_value

    def generate_random_value(self) -> float:
        """:return: Random value according to this Float Variable definition."""
        return generate_random_float(self.min_value, self.max_value)

    def is_proper_value(self, value: Any) -> bool:
        """:return: True if value is compatible with this Float Variable definition, False otherwise."""
        return isinstance(value, float) and self.min_value <= value <= self.max_value

    def get_log_data(self) -> Dict[str, Union[str, float]]:
        """
        Gets data for logging purposes.

        :return: Dictionary with this Integer Variable crucial data.
        """
        return {
            "type": self.__class__.__name__,
            "min_value": self.min_value,
            "max_value": self.max_value,
        }


class ChoiceVariable(DecisionVariable):
    """
    Choice Decision Variable definition.

    This class is Decision Variable type of variable that can take any (hashable) value from given iterable.
    """

    def __init__(self, possible_values: Iterable[Any]) -> None:
        """
        Creates definition of Choice Decision Variable.

        :param possible_values: Iterable with possible values to set for this Decision Variable.
        """
        self.possible_values = set(possible_values)

    def generate_random_value(self) -> Any:
        """:return: Random value according to this Choice Variable definition."""
        return choose_random_value(self.possible_values)

    def is_proper_value(self, value: Any) -> bool:
        """:return: True if value is compatible with this Choice Variable definition, False otherwise."""
        return value in self.possible_values

    def get_log_data(self) -> Dict[str, str]:
        """
        Method which prepares data of the instance of this class for logging.

        :return: Crucial data of this object.
        """
        return {
            "type": self.__class__.__name__,
            "possible_values": ", ".join([str(value) for value in self.possible_values]),
        }
