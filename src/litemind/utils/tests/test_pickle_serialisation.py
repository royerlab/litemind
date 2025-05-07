import pickle

import pytest

from litemind.utils.pickle_serialisation import PickleSerializable


class PickleObject(PickleSerializable):
    """Test class that inherits from PickleSerializable."""

    def __init__(self, data=None):
        self.data = data

    def __eq__(self, other):
        if not isinstance(other, PickleObject):
            return False
        return self.data == other.data


class CustomType:
    def __init__(self, value):
        self.value = value

    def __eq__(self, other):
        return isinstance(other, CustomType) and self.value == other.value


def test_to_pickle_returns_bytes():
    obj = PickleObject("test data")
    pickled = obj.to_pickle()
    assert isinstance(pickled, bytes)


def test_from_pickle_restores_object():
    original = PickleObject("test data")
    pickled = original.to_pickle()
    restored = PickleObject.from_pickle(pickled)
    assert isinstance(restored, PickleObject)
    assert restored.data == original.data


def test_to_base64_returns_string():
    obj = PickleObject("test data")
    base64_str = obj.to_base64()
    assert isinstance(base64_str, str)


def test_from_base64_restores_object():
    original = PickleObject("test data")
    base64_str = original.to_base64()
    restored = PickleObject.from_base64(base64_str)
    assert isinstance(restored, PickleObject)
    assert restored.data == original.data


def test_pickle_roundtrip():
    original = PickleObject("test data")
    pickled = original.to_pickle()
    restored = PickleObject.from_pickle(pickled)
    assert restored.data == original.data


def test_base64_roundtrip():
    original = PickleObject("test data")
    base64_str = original.to_base64()
    restored = PickleObject.from_base64(base64_str)
    assert restored.data == original.data


def test_complex_data_roundtrip():
    complex_data = {
        "string": "test string",
        "number": 123,
        "float": 3.14,
        "list": [1, 2, 3],
        "dict": {"key": "value"},
        "bool": True,
        "none": None,
    }
    original = PickleObject(complex_data)
    pickled = original.to_pickle()
    restored = PickleObject.from_pickle(pickled)
    assert restored.data == original.data


def test_from_pickle_invalid_data():
    with pytest.raises(ValueError):
        PickleObject.from_pickle(b"invalid pickle data")


def test_from_base64_invalid_data():
    with pytest.raises(ValueError):
        PickleObject.from_base64("invalid base64 data")


def test_nested_objects():
    nested = PickleObject(PickleObject("nested value"))
    pickled = nested.to_pickle()
    restored = PickleObject.from_pickle(pickled)
    assert isinstance(restored.data, PickleObject)
    assert restored.data.data == "nested value"


def test_empty_object():
    empty = PickleObject()
    pickled = empty.to_pickle()
    restored = PickleObject.from_pickle(pickled)
    assert restored.data is None


def test_large_data():
    large_string = "x" * 100000  # 100KB string
    large_obj = PickleObject(large_string)
    base64_str = large_obj.to_base64()
    restored = PickleObject.from_base64(base64_str)
    assert restored.data == large_string


def test_pickle_protocol_compatibility():
    obj = PickleObject("test data")
    # Manually pickle with lowest protocol
    pickled_low = pickle.dumps(obj, protocol=0)
    # Manually pickle with highest protocol
    pickled_high = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)

    # Both should be deserializable
    restored_low = PickleObject.from_pickle(pickled_low)
    restored_high = PickleObject.from_pickle(pickled_high)

    assert restored_low.data == "test data"
    assert restored_high.data == "test data"


def test_custom_types():
    obj = PickleObject(CustomType(42))
    restored = PickleObject.from_pickle(obj.to_pickle())
    assert restored.data == obj.data
