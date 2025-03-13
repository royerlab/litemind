from typing import List, Tuple
from functools import lru_cache

@lru_cache
def is_semanticText_available() -> bool:
    # Check if package semantic-text-splitter s available:
    try:
        import importlib.util

        return importlib.util.find_spec("semantic-text-splitter") is not None

    except ImportError:
        return False

@lru_cache
def is_treeSitter_available() -> bool:
    # Check if package tree-sitter-languages s available:
    try:
        import importlib.util

        return importlib.util.find_spec("tree-sitter-languages") is not None

    except ImportError:
        return False


def semantic_splitting(
    text: str, size: int | Tuple[int, int], overlap: int, trim: bool
) -> List[str]:
    """
    Convert a text string into a list of semantic strings

    Parameters
    ----------
    text: str
        The text string to split.
    size: int | Tuple[int,int]
        The size of the chunk to get
    overlap: int
        The size of the overlap with the other chunks
    trim: bool
        If True, each chunks will be trimmed

    Returns
    -------
    List[str]
        The resulting list of chunks string.

    """
    if not is_semanticText_available():
        raise ImportError(
            "semantic-text-splitter is not available. Please install it to use this function."
        )

    # Import the module
    import semantic_text_splitter

    # create the Textsplitter
    splitter = semantic_text_splitter.TextSplitter(
        capacity=size, overlap=overlap, trim=trim
    )

    splitted_text = splitter.chunks(text)

    return splitted_text


def markdown_splitting(
    text: str, size: int | Tuple[int, int], overlap: int, trim: bool
) -> List[str]:
    """
    Convert a markdown string into a list of semantic strings

    Parameters
    ----------
    text: str
        The markdown string to split.
    size: int | Tuple[int,int]
        The size of the chunk to get
    overlap: int
        The size of the overlap with the other chunks
    trim: bool
        If True, each chunks will be trimmed

    Returns
    -------
    List[str]
        The resulting list of chunks string.

    """
    if not is_semanticText_available():
        raise ImportError(
            "semantic-text-splitter is not available. Please install it to use this function."
        )

    # Import the module
    import semantic_text_splitter

    # create the Textsplitter
    splitter = semantic_text_splitter.MarkdownSplitter(
        capacity=size, overlap=overlap, trim=trim
    )

    splitted_text = splitter.chunks(text)

    return splitted_text


def code_splitting(
    text: str, size: int | Tuple[int, int], overlap: int, trim: bool
) -> List[str]:
    """
    Convert a markdown string into a list of semantic strings

    Parameters
    ----------
    text: str
        The markdown string to split.
    size: int | Tuple[int,int]
        The size of the chunk to get
    overlap: int
        The size of the overlap with the other chunks
    trim: bool
        If True, each chunks will be trimmed

    Returns
    -------
    List[str]
        The resulting list of chunks string.

    """
    if not is_treeSitter_available():
        raise ImportError(
            "tree-sitter-languages is not available. Please install it to use this function."
        )

    # Import the module
    import semantic_text_splitter
    import tree_sitter_languages

    # create the Textsplitter
    splitter = semantic_text_splitter.CodeSplitter(
        language=tree_sitter_languages.language(),
        capacity=size,
        overlap=overlap,
        trim=trim,
    )

    splitted_text = splitter.chunks(text)

    return splitted_text
