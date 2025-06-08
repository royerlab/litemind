from litemind.utils.text_compressor import TextCompressor

# -----------------------------------------------------------------------------
# Sample text used across tests
# -----------------------------------------------------------------------------
SAMPLE_TEXT = """
# This is a python comment
# Second comment line

def foo(bar):
    pass  # TODO: implement later

// This is a C-style comment
// Another C comment

-- SQL comment
-- Another SQL comment

This line has multiple newlines after it.


This line follows multiple newlines.

This has repeated characters: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa and zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz

This line has trailing spaces       
    This line has 4 spaces at the beginning
        This line has 8 spaces (2 tabs worth)
"""

# Separate snippet with contiguous import lines for the “imports” scheme test
IMPORT_SAMPLE = """
from arbol import aprint
from pydantic import BaseModel
from litemind import API_IMPLEMENTATIONS
from litemind.agent.agent import Agent
from litemind.agent.messages.message import Message
from litemind.apis.model_features import ModelFeatures

def bar():
    return 42
"""


# -----------------------------------------------------------------------------
# Helper
# -----------------------------------------------------------------------------


def _compress_with(schemes, **kwargs):
    """Utility to run compression with specific schemes enabled."""
    cmp = TextCompressor(schemes=schemes, **kwargs)
    return cmp.compress(SAMPLE_TEXT)


# -----------------------------------------------------------------------------
# Individual scheme tests
# -----------------------------------------------------------------------------


def test_newlines():
    out = _compress_with(("newlines",))
    # No more than one consecutive newline
    assert "\n\n" not in out
    # Content is preserved
    assert "def foo(bar):" in out


def test_comments():
    out = _compress_with(("comments",))
    # Python comments are removed
    assert "# This is a python comment" not in out
    assert "# Second comment line" not in out

    # C-style comments are removed
    assert "// This is a C-style comment" not in out
    assert "// Another C comment" not in out

    # SQL comments are removed
    assert "-- SQL comment" not in out
    assert "-- Another SQL comment" not in out

    # Ensure code content itself survives
    assert "def foo" in out

    # Inline comments should remain (not implemented in the simplified version)
    assert "TODO: implement later" in out


def test_repeats():
    # Test with default max_repeats=10
    out = _compress_with(("repeats",))

    # 'a' repeated many times should be compressed to few 'a's
    assert "aaaaaaaaaaaaaaaa" in out
    assert "aaaaaaaaaaaaaaaaaaaaaaaa" not in out

    # 'z' repeated many times should be compressed to few 'z's
    assert "zzzzzzzzzzzzzzzz" in out
    assert "zzzzzzzzzzzzzzzzzzzzzzzz" not in out

    # Test with custom max_repeats
    cmp = TextCompressor(schemes=("repeats",), max_repeats=5)
    out = cmp.compress("aaaaaaaaaaaaaaaa")
    assert out == "aaaaa"


def test_custom_max_repeats():
    cmp = TextCompressor(schemes=("repeats",), max_repeats=3)
    out = cmp.compress("aaaaaaaaaa")
    assert out == "aaa"


def test_spaces_to_tabs():
    out = _compress_with(("spaces",))
    # Check that 4 spaces are converted to tabs
    assert "\t" in out
    assert "    " not in out


def test_trailing_spaces():
    out = _compress_with(("trailing",))
    # Check that trailing spaces are removed
    assert "This line has trailing spaces       \n" not in out
    assert "This line has trailing spaces\n" in out


def test_import_blocks():
    """Ensure contiguous import blocks are collapsed to a single placeholder."""
    cmp = TextCompressor(schemes=("imports",))
    out = cmp.compress(IMPORT_SAMPLE)

    # Placeholder comment inserted exactly once
    assert out.count("# [shortened] removed imports") == 1

    # All original import lines are gone
    assert "from arbol import aprint" not in out
    assert "from pydantic import BaseModel" not in out

    # Code that follows the imports is preserved
    assert "def bar():" in out


# -----------------------------------------------------------------------------
# Pipeline & behavior tests
# -----------------------------------------------------------------------------


def test_combined_pipeline_reduces_size():
    schemes = ("newlines", "comments", "repeats", "spaces", "trailing", "imports")
    compressed = _compress_with(schemes)
    assert len(compressed) < len(SAMPLE_TEXT)


def test_enable_disable_runtime():
    cmp = TextCompressor(schemes=("newlines",))  # comments initially off

    # With comments disabled
    out1 = cmp.compress("# Comment\ncode")
    assert "# Comment" in out1

    # Enable comments
    cmp.enable("comments")
    out2 = cmp.compress("# Comment\ncode")
    assert "# Comment" not in out2

    # Disable comments
    cmp.disable("comments")
    out3 = cmp.compress("# Comment\ncode")
    assert "# Comment" in out3


def test_unknown_scheme():
    try:
        TextCompressor(schemes=("invalid_scheme",))
        assert False, "Should raise ValueError for unknown scheme"
    except ValueError:
        pass


def test_empty_text():
    cmp = TextCompressor(
        schemes=("newlines", "comments", "repeats", "spaces", "trailing", "imports")
    )
    assert cmp.compress("") == ""


def test_preserve_code_content():
    code = "def function():\n    x = 1\n    return x"
    cmp = TextCompressor(schemes=("newlines", "comments", "repeats"))
    assert cmp.compress(code) == code
