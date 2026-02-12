"""Utilities for extracting thinking/reasoning content from XML-tagged text."""

import re
from typing import List, Optional, Tuple


def extract_thinking_content(
    text: str, thinking_tags: Optional[List[str]] = None
) -> Tuple[str, str]:
    """
    Extract content within thinking-style XML tags from text.

    Searches for tags like ``<thinking>``, ``<thought>``, ``<reasoning>``,
    etc. and separates the thinking content from the rest of the text.

    Parameters
    ----------
    text : str
        The input string potentially containing thinking tags.
    thinking_tags : list of str, optional
        Tag names to search for. Defaults to
        ``["think", "thinking", "thought", "reasoning"]``.

    Returns
    -------
    Tuple[str, str]
        A tuple of ``(text_without_thinking, thinking_content)``.
        If no thinking tags are found, returns ``(text, "")``.
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
