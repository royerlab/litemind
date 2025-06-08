import re

# Compile once on import
_GEMINI_REASONING_RE = re.compile(
    r"""
    ^gemini-                # prefix
    (?:1\.5|2\.0|2\.5)-      # supported generations
    (?:                     # variations with reasoning
        pro |               # pro
        flash(?:-lite)? |   # flash or flash-lite
        flash-thinking      # flash-thinking experimental
    )
    (?:-.+)?$               # optional suffix: -preview-05-06, -exp-01-21, etc.
    """,
    re.VERBOSE,
)


def is_gemini_reasoning_model(model_name: str) -> bool:
    """
    True ⇔ `model_name` supports Gemini’s built-in thinking/reasoning trace.

    >>> is_gemini_reasoning_model("gemini-2.5-pro-preview-05-06")
    True
    >>> is_gemini_reasoning_model("gemini-nano")
    False
    """
    return bool(_GEMINI_REASONING_RE.fullmatch(model_name))
