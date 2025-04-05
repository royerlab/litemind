import pytest

from litemind.utils.extract_thinking import extract_thinking_content


def test_extract_thinking_content():
    # Test case with <thinking> tags
    text_with_thinking = "This is a test. <thinking>Thinking content goes here.</thinking> This is the end."
    expected_result = (
        "This is a test.  This is the end.",
        "Thinking content goes here.",
    )
    assert extract_thinking_content(text_with_thinking) == expected_result

    # Test case without <thinking> tags
    text_without_thinking = "This is a test. This is the end."
    expected_result = (text_without_thinking, "")
    assert extract_thinking_content(text_without_thinking) == expected_result

    # Test case with multiple <thinking> tags
    text_with_multiple_thinking = "Start. <thinking>First thinking content.</thinking> Middle. <thinking>Second thinking content.</thinking> End."
    expected_result = ("Start.  Middle.  End.", "First thinking content.")
    assert extract_thinking_content(text_with_multiple_thinking) == expected_result

    # Test case with empty <thinking> tags
    text_with_empty_thinking = "Start. <thinking></thinking> End."
    expected_result = ("Start.  End.", "")
    assert extract_thinking_content(text_with_empty_thinking) == expected_result


if __name__ == "__main__":
    pytest.main()
