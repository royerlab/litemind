import re
from typing import List, Optional, Tuple


def extract_thinking_content(
    text: str, thinking_tags: Optional[List[str]] = None
) -> Tuple[str, str]:
    """
    Extracts content within <thinking>...</thinking> tags and returns the string without the tags and their contents,
    and the contents of the thinking block. If no tags are present, returns the original string and an empty string.

    Parameters
    ----------
    text : str
        The input string containing the <thinking>...</thinking> tags.

    Returns
    -------
    Tuple[str, str]
        A tuple containing the string without the tags and their contents, and the contents of the thinking block.
    """

    # Default thinking tags:
    if thinking_tags is None:
        thinking_tags = ["think", "thinking", "thought", "reasoning"]

    # First we try to detect the thinking block using the possible tags:
    for tag in thinking_tags:
        pattern = re.compile(rf"<{tag}>(.*?)</{tag}>", re.DOTALL)
        match = pattern.search(text)
        if match:
            thinking_content = match.group(1).strip()
            text_without_thinking = pattern.sub("", text).strip()
            return text_without_thinking, thinking_content
        else:
            continue

    return text, ""
