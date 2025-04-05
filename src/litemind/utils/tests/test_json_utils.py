import pytest

from litemind.utils.json_utils import JSONSerializable


def test_jsonserializable_basic_round_trip():
    class Example(JSONSerializable):
        def __init__(self, value=None):
            self.value = value

    instance = Example(value="test")
    json_str = instance.to_json()
    assert "value" in json_str

    reloaded = Example.from_json(json_str)
    assert reloaded.value == "test"


def test_jsonserializable_with_list():
    class ListHolder(JSONSerializable):
        def __init__(self, items=None):
            self.items = items if items is not None else []

    instance = ListHolder(items=[1, 2, 3])
    json_str = instance.to_json()
    assert '"items": [1, 2, 3]' in json_str

    reloaded = ListHolder.from_json(json_str)
    assert reloaded.items == [1, 2, 3]


def test_jsonserializable_with_dict():
    class DictHolder(JSONSerializable):
        def __init__(self, mapping=None):
            self.mapping = mapping if mapping is not None else {}

    instance = DictHolder(mapping={"key": "value"})
    json_str = instance.to_json()
    assert '"key": "value"' in json_str

    reloaded = DictHolder.from_json(json_str)
    assert reloaded.mapping["key"] == "value"


def test_jsonserializable_nested():
    class Nested(JSONSerializable):
        def __init__(self, name, child=None):
            self.name = name
            self.child = child

    child_obj = Nested(name="child")
    parent_obj = Nested(name="parent", child=child_obj)
    json_str = parent_obj.to_json()
    assert '"name": "child"' in json_str

    reloaded = Nested.from_json(json_str)
    assert reloaded.name == "parent"
    assert reloaded.child.name == "child"


def test_jsonserializable_malformed_json():
    class Example(JSONSerializable):
        def __init__(self, value=None):
            self.value = value

    with pytest.raises(ValueError):
        Example.from_json("{invalid_json}")
