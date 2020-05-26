from typing import Any, Set
from abc import ABC, abstractmethod

from optimization.utilities import generate_random_int, generate_random_float, choose_random_value


__all__ = ["DecisionVariable", "IntegerVariable", "FloatVariable", "ChoiceVariable"]


class DecisionVariable(ABC):
    """Abstract definition of decision variable that optimal value is searched during optimization process."""

    @abstractmethod
    def generate_random_value(self) -> Any:
        """
        Abstract definition of a method that generates random value according to Decision Variable definition.

        :raise NotImplementedError: Abstract method was called.
        """
        raise NotImplementedError("You have called abstract method 'generate_random_value' of 'DecisionVariable' "
                                  "abstract class.")

    @abstractmethod
    def is_value_correct(self, value: Any) -> bool:
        """
        Abstract definition of a method that checks if received value is compatible with Decision Variable definition.

        :raise NotImplementedError: Abstract method was called.
        """
        raise NotImplementedError("You have called abstract method 'generate_random_value' of 'DecisionVariable' "
                                  "abstract class.")

    @abstractmethod
    def get_data_for_logging(self) -> dict:
        """
        Abstract definition of a method which prepares data of the instance of this class for logging.

        :raise NotImplementedError: Abstract method was called.
        """
        raise NotImplementedError("You have called abstract method 'get_data_for_logging' of 'DecisionVariable' "
                                  "abstract class.")


class IntegerVariable(DecisionVariable):
    """Class for defining decision variable that carries integer value."""

    def __init__(self, min_value: int, max_value: int) -> None:
        """
        Creates definition of Integer Decision Variable.

        :param min_value: Minimal value that can be set.
        :param max_value: Maximal value that can be set.
        :raise TypeError: Parameter 'min_value' or 'max_value' is not int type.
        :raise ValueError: Value of parameter 'min_value' is not less than value of 'max_value'.
        """
        if not isinstance(min_value, int):
            raise TypeError(f"Parameter 'min_value' must be int type. Received: {min_value} ({type(min_value)}).")
        if not isinstance(max_value, int):
            raise TypeError(f"Parameter 'max_value' must be int type. Received: {max_value} ({type(max_value)}).")
        if min_value >= max_value:
            raise ValueError(f"Value of 'min_value' must be less than value of 'max_value'. "
                             f"Received: min_value={min_value}, max_value={max_value}.")
        self.min_value = min_value
        self.max_value = max_value

    def generate_random_value(self) -> int:
        """
        Generates random integer value according to the class definition.

        :return: Valid random value of the Integer Variable.
        """
        return generate_random_int(self.min_value, self.max_value)

    def is_value_correct(self, value: Any) -> bool:
        """
        Checks if received value is compatible with the Integer Variable definition.

        :param value: Value to be checked.

        :return: True if value is compatible with the definition, False if it is not.
        """
        return isinstance(value, int) and self.min_value <= value <= self.max_value

    def get_data_for_logging(self) -> dict:
        """
        Method which prepares data of the instance of this class for logging.

        :return: Crucial data of this object.
        """
        return {
            "type": "IntegerVariable",
            "min_value": self.min_value,
            "max_value": self.max_value
        }


class FloatVariable(DecisionVariable):
    """Class for defining decision variable that carries float value."""

    def __init__(self, min_value: float, max_value: float) -> None:
        """
        Creates definition of Float Decision Variable.

        :param min_value: Minimal value that can be set.
        :param max_value: Maximal value that can be set.
        :raise TypeError: Parameter 'min_value' or 'max_value' is not float or int type.
        :raise ValueError: Value of parameter 'min_value' is not less than value of 'max_value'.
        """
        if not isinstance(min_value, (int, float)):
            raise TypeError(f"Parameter 'min_value' must be float type. Received: {min_value} ({type(min_value)}).")
        if not isinstance(max_value, (int, float)):
            raise TypeError(f"Parameter 'max_value' must be float type. Received: {max_value} ({type(max_value)}).")
        if min_value >= max_value:
            raise ValueError(f"Value of 'min_value' must be less than value of 'max_value'. "
                             f"Received: min_value={min_value}, max_value={max_value}.")
        self.min_value = float(min_value)
        self.max_value = float(max_value)

    def generate_random_value(self) -> float:
        """
        Generates random integer value according to the class definition.

        :return: Valid random value of the Float Variable.
        """
        return generate_random_float(self.min_value, self.max_value)

    def is_value_correct(self, value: Any) -> bool:
        """
        Checks if received value is compatible with the Float Variable definition.

        :param value: Value to be checked.

        :return: True if value is compatible with the definition, False if it is not.
        """
        return isinstance(value, float) and self.min_value <= value <= self.max_value

    def get_data_for_logging(self) -> dict:
        """
        Method which prepares data of the instance of this class for logging.

        :return: Crucial data of this object.
        """
        return {
            "type": "FloatVariable",
            "min_value": self.min_value,
            "max_value": self.max_value
        }


class ChoiceVariable(DecisionVariable):
    """Class for defining decision variable that carries value from possible values pool."""

    def __init__(self, possible_values: Set) -> None:
        """
        Creates definition of Choice Decision Variable.

        :param possible_values:
        :raise TypeError: Parameter 'possible_values' is not set type.
        """
        if not isinstance(possible_values, set):
            raise TypeError(f"Parameter 'possible_values' must be set type. "
                            f"Received: {possible_values} ({type(possible_values)}).")
        self.possible_values = possible_values

    def generate_random_value(self) -> Any:
        """
        Generates random integer value according to the class definition.

        :return: Valid random value of the Choice Variable.
        """
        return choose_random_value(self.possible_values)

    def is_value_correct(self, value: Any) -> bool:
        """
        Checks if received value is compatible with the Choice Variable definition.

        :param value: Value to be checked.

        :return: True if value is compatible with the definition, False if it is not.
        """
        return value in self.possible_values

    def get_data_for_logging(self) -> dict:
        """
        Method which prepares data of the instance of this class for logging.

        :return: Crucial data of this object.
        """
        return {
            "type": "ChoiceVariable",
            "possible_values": [val if isinstance(val, bool) else repr(val) for val in self.possible_values],
        }
