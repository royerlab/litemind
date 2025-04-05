from litemind.utils.markdown import extract_outermost_markdown_block


def test_extract_outermost_markdown_block():
    # Test basic extraction with ```markdown
    text1 = (
        "Some text before\n```markdown\nThis is markdown content\n```\nSome text after"
    )
    assert extract_outermost_markdown_block(text1) == "This is markdown content"

    # Test with ```md marker
    text2 = "Some text before\n```md\nThis is markdown content\n```\nSome text after"
    assert extract_outermost_markdown_block(text2) == "This is markdown content"

    # Test with no markdown block
    text3 = "Just regular text without markdown blocks"
    assert extract_outermost_markdown_block(text3) == text3

    # Test with unclosed block
    text4 = "Some text\n```markdown\nUnclosed block"
    assert extract_outermost_markdown_block(text4) == text4

    # Test with nested code blocks
    text5 = (
        "Some text before\n"
        "```markdown\n"
        "Outer markdown block\n"
        "```python\n"
        "def hello_world():\n"
        "    print('Hello, world!')\n"
        "```\n"
        "More markdown content\n"
        "```\n"
        "Some text after"
    )
    expected5 = (
        "Outer markdown block\n"
        "```python\n"
        "def hello_world():\n"
        "    print('Hello, world!')\n"
        "```\n"
        "More markdown content"
    )
    assert extract_outermost_markdown_block(text5) == expected5

    # Test with multiple markdown blocks (should extract from first marker to last end marker)
    text6 = (
        "```markdown\nFirst block\n```\n" "Some text\n" "```markdown\nSecond block\n```"
    )
    expected6 = "First block\n```\nSome text\n```markdown\nSecond block"
    assert extract_outermost_markdown_block(text6) == expected6
