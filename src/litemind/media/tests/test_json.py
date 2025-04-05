import pytest

from litemind.media.types.media_json import Json


class TestJson:
    def test_init(self):
        """Test initializing Json with valid parameters."""
        json = {"name": "test", "value": 42}
        json_media = Json(json)
        assert json_media.get_content() == json

    def test_init_with_string(self):
        """Test initializing Json with a JSON string."""
        json_str = '{"name": "test", "value": 42}'
        json_media = Json(json_str)
        assert json_media.get_content() == {"name": "test", "value": 42}

    def test_str(self):
        """Test string representation of Json."""
        json = {"name": "test", "value": 42}
        json_media = Json(json)
        assert (
            "name" in str(json_media)
            and "test" in str(json_media)
            and "value" in str(json_media)
            and "42" in str(json_media)
        )

    def test_len(self):
        """Test length calculation for Json."""
        json = {"name": "test", "value": 42}
        json_media = Json(json)
        assert len(json_media) == 29

    def test_invalid_json(self):
        """Test initialization with invalid JSON."""
        with pytest.raises(ValueError):
            Json("not valid json")

    def test_init_with_valid_json_str(self):
        valid_str = '{"name": "Test", "value": 123}'
        json_media = Json(valid_str)
        assert json_media.json["name"] == "Test"
        assert json_media.json["value"] == 123

    def test_init_with_invalid_json_str(self):
        invalid_str = '{"name": "Test", "value": }'
        with pytest.raises(ValueError) as excinfo:
            Json(invalid_str)
        assert "Invalid JSON string" in str(excinfo.value)

    def test_init_with_valid_json_data(self):
        data = {"name": "Test", "value": 123}
        json_media = Json(data)
        assert json_media.json["name"] == "Test"
        assert json_media.json["value"] == 123

    def test_str_representation(self):
        json_media = Json('{"a": 1}')
        assert str(json_media) == '{"a": 1}'
        assert repr(json_media) == '{"a": 1}'

    def test_markdown_output(self):
        json_media = Json({"b": 2})
        md_string = json_media.to_markdown_string()
        assert md_string.startswith("```json\n")
        assert md_string.endswith("\n```")

        md_text = json_media.to_markdown_text_media()
        assert "```json" in str(md_text)
