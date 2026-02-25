from typing import Any, Callable, Dict, List, Optional

import json_repair


class OutputStringParser:
    @staticmethod
    def parse_dict(
        content: str,
        expected_keys: List[str] = None,
        value_converter: Callable[[Any], Any] | dict[str, Callable] = None,
        expected_value_types: Dict[str, type] = None,
        validator: Callable[[Dict[str, Any]], None] = None,
        fallback: Any = {},
    ) -> tuple[Dict[str, Any], Any]:
        """
        Parses a string content into a dictionary with optional validation and conversion.

        check expected keys -> convert values -> check expected types -> custom validation

        Args:
            content (str): The string content to parse as JSON.
            expected_keys (List[str], optional): List of keys that must be present in the parsed dictionary.
            value_converter (Callable[[Any], Any] | dict[str, Callable], optional): Function or mapping to apply to all values in the dictionary.
            expected_value_types (Dict[str, type], optional): Dictionary mapping keys to their expected types.
            validator (Callable[[Dict[str, Any]], None], optional): Custom validation function for the parsed dictionary.
            fallback (Any, optional): Value to return if parsing or validation fails. Defaults to {}.

        Returns:
            tuple: (parsed dictionary or fallback value, fail_reason or None)
        """
        fail_reason = None
        try:
            solution = json_repair.loads(content)
            assert isinstance(solution, dict), (
                f"Expected parsed content to be a dictionary, got {type(solution)}"
            )

            # Check for required keys
            if expected_keys:
                for key in expected_keys:
                    assert key in solution, (
                        f"Required key '{key}' not found in parsed dictionary"
                    )

            # Apply value conversion to all values
            if value_converter:
                if callable(value_converter):
                    solution = {k: value_converter(v) for k, v in solution.items()}
                elif isinstance(value_converter, dict):
                    for k, converter in value_converter.items():
                        if k in solution:
                            solution[k] = converter(solution[k])

            # Check expected types
            if expected_value_types:
                for key, expected_type in expected_value_types.items():
                    assert isinstance(solution[key], expected_type), (
                        f"Expected key '{key}' to be of type {expected_type}, got {type(solution[key])}"
                    )

            # Apply custom validation
            if validator:
                validator(solution)

            return solution, fail_reason
        except Exception as e:
            fail_reason = str(e)
            return fallback, fail_reason

    @staticmethod
    def parse_list(
        content: str,
        expected_length: Optional[int] = None,
        item_converter: Optional[Callable[[Any], Any]] = None,
        expected_item_type: Optional[type] = None,
        validator: Optional[Callable[[List[Any]], None]] = None,
        fallback: Any = [],
    ) -> tuple[List[Any], Any]:
        """
        Parses a string content into a list with optional validation and conversion.

        Args:
            content (str): The string content to parse as JSON.
            expected_length (int, optional): Expected length of the parsed list.
            item_converter (Callable[[Any], Any], optional): Function to apply to all items in the list.
            expected_item_type (type, optional): Expected type for all items in the list.
            validator (Callable[[List[Any]], None], optional): Custom validation function for the parsed list.
            fallback (Any, optional): Value to return if parsing or validation fails. Defaults to [].

        Returns:
            tuple: (parsed list or fallback value, fail_reason or None)
        """
        fail_reason = None
        try:
            solution = json_repair.loads(content)
            assert isinstance(solution, list), (
                f"Expected parsed content to be a list, got {type(solution)}"
            )

            # Check expected length
            if expected_length is not None:
                assert len(solution) == expected_length, (
                    f"Expected list length to be {expected_length}, got {len(solution)}"
                )

            # Apply item conversion to all items
            if item_converter:
                solution = [item_converter(item) for item in solution]

            # Check expected item types
            if expected_item_type:
                for item in solution:
                    assert isinstance(item, expected_item_type), (
                        f"Expected list item to be of type {expected_item_type}, got {type(item)}"
                    )

            # Apply custom validation
            if validator:
                validator(solution)

            return solution, fail_reason
        except Exception as e:
            fail_reason = str(e)
            return fallback, fail_reason


# Test cases for OutputStringParser
def test_parse_dict_success():
    content = '{"a": 1, "b": 2}'
    result, fail_reason = OutputStringParser.parse_dict(content)
    assert result == {"a": 1, "b": 2}
    assert fail_reason is None


def test_parse_dict_expected_keys():
    content = '{"a": 1, "b": 2}'
    result, fail_reason = OutputStringParser.parse_dict(content, expected_keys=["a"])
    assert result == {"a": 1, "b": 2}
    assert fail_reason is None
    # Test failure
    result, fail_reason = OutputStringParser.parse_dict(
        content, expected_keys=["c"], fallback={"error": True}
    )
    assert result == {"error": True}
    assert fail_reason is not None


def test_parse_dict_value_converter():
    content = '{"a": "1", "b": "2"}'
    result, fail_reason = OutputStringParser.parse_dict(content, value_converter=int)
    assert result == {"a": 1, "b": 2}
    assert fail_reason is None


def test_parse_dict_expected_types():
    content = '{"a": 1, "b": "2"}'
    result, fail_reason = OutputStringParser.parse_dict(
        content, expected_value_types={"a": int, "b": str}
    )
    assert result == {"a": 1, "b": "2"}
    assert fail_reason is None
    # Failure
    result, fail_reason = OutputStringParser.parse_dict(
        content, expected_value_types={"a": str}, fallback={"error": True}
    )
    assert result == {"error": True}
    assert fail_reason is not None


def test_parse_dict_validator():
    def validator(d):
        if d.get("a") != 1:
            raise ValueError("Invalid")

    content = '{"a": 1}'
    result, fail_reason = OutputStringParser.parse_dict(content, validator=validator)
    assert result == {"a": 1}
    assert fail_reason is None
    content2 = '{"a": 2}'
    result, fail_reason = OutputStringParser.parse_dict(
        content2, validator=validator, fallback={"error": True}
    )
    assert result == {"error": True}
    assert fail_reason is not None


def test_parse_dict_invalid_json():
    content = "invalid"
    result, fail_reason = OutputStringParser.parse_dict(
        content, fallback={"error": True}
    )
    assert result == {"error": True}
    assert fail_reason is not None


def test_parse_list_success():
    content = "[1, 2, 3]"
    result, fail_reason = OutputStringParser.parse_list(content)
    assert result == [1, 2, 3]
    assert fail_reason is None


def test_parse_list_expected_length():
    content = "[1, 2]"
    result, fail_reason = OutputStringParser.parse_list(content, expected_length=2)
    assert result == [1, 2]
    assert fail_reason is None
    result, fail_reason = OutputStringParser.parse_list(
        content, expected_length=3, fallback=[-1]
    )
    assert result == [-1]
    assert fail_reason is not None


def test_parse_list_item_converter():
    content = '["1", "2"]'
    result, fail_reason = OutputStringParser.parse_list(content, item_converter=int)
    assert result == [1, 2]
    assert fail_reason is None


def test_parse_list_expected_item_type():
    content = "[1, 2]"
    result, fail_reason = OutputStringParser.parse_list(content, expected_item_type=int)
    assert result == [1, 2]
    assert fail_reason is None
    result, fail_reason = OutputStringParser.parse_list(
        content, expected_item_type=str, fallback=[-1]
    )
    assert result == [-1]
    assert fail_reason is not None


def test_parse_list_validator():
    def validator(lst):
        if len(lst) > 2:
            raise ValueError("Too long")

    content = "[1, 2]"
    result, fail_reason = OutputStringParser.parse_list(content, validator=validator)
    assert result == [1, 2]
    assert fail_reason is None
    content2 = "[1, 2, 3]"
    result, fail_reason = OutputStringParser.parse_list(
        content2, validator=validator, fallback=[-1]
    )
    assert result == [-1]
    assert fail_reason is not None


def test_parse_list_invalid_json():
    content = "invalid"
    result, fail_reason = OutputStringParser.parse_list(content, fallback=[-1])
    assert result == [-1]
    assert fail_reason is not None


if __name__ == "__main__":
    test_parse_dict_success()
    test_parse_dict_expected_keys()
    test_parse_dict_value_converter()
    test_parse_dict_expected_types()
    test_parse_dict_validator()
    test_parse_dict_invalid_json()
    test_parse_list_success()
    test_parse_list_expected_length()
    test_parse_list_item_converter()
    test_parse_list_expected_item_type()
    test_parse_list_validator()
    test_parse_list_invalid_json()
    print("All tests passed!")
