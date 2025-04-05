from pydantic import BaseModel

from litemind.utils.parse_json_output import parse_json


class ExampleModel(BaseModel):
    name: str
    age: int


def test_parse_json():
    json_str = '{"name": "John", "age": 30}'
    result = parse_json(json_str, ExampleModel)
    assert result.name == "John"
    assert result.age == 30
