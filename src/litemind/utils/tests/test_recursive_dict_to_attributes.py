from types import SimpleNamespace

import pytest

from litemind.utils.recursive_dict_to_attributes import recursive_dict_to_attributes


def test_recursive_dict_to_attributes():
    # Test with a simple dictionary
    input_dict = {"a": 1, "b": 2}
    result = recursive_dict_to_attributes(input_dict)
    assert isinstance(result, SimpleNamespace)
    assert result.a == 1
    assert result.b == 2

    # Test with a nested dictionary
    input_dict = {"a": 1, "b": {"c": 2, "d": 3}}
    result = recursive_dict_to_attributes(input_dict)
    assert isinstance(result, SimpleNamespace)
    assert result.a == 1
    assert isinstance(result.b, SimpleNamespace)
    assert result.b.c == 2
    assert result.b.d == 3

    # Test with a list of dictionaries
    input_dict = {"a": [{"b": 1}, {"c": 2}]}
    result = recursive_dict_to_attributes(input_dict)
    assert isinstance(result, SimpleNamespace)
    assert isinstance(result.a, list)
    assert isinstance(result.a[0], SimpleNamespace)
    assert result.a[0].b == 1
    assert isinstance(result.a[1], SimpleNamespace)
    assert result.a[1].c == 2

    # Test with a mixed dictionary
    input_dict = {"a": 1, "b": [{"c": 2}, {"d": {"e": 3}}]}
    result = recursive_dict_to_attributes(input_dict)
    assert isinstance(result, SimpleNamespace)
    assert result.a == 1
    assert isinstance(result.b, list)
    assert isinstance(result.b[0], SimpleNamespace)
    assert result.b[0].c == 2
    assert isinstance(result.b[1], SimpleNamespace)
    assert isinstance(result.b[1].d, SimpleNamespace)
    assert result.b[1].d.e == 3


if __name__ == "__main__":
    pytest.main()
