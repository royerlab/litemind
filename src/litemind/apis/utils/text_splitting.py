from typing import List, Tuple
from functools import lru_cache

@lru_cache
def is_semanticText_available() -> bool:
    # Check if package semantic-text-splitter s available:
    try:
        import importlib.util

        return importlib.util.find_spec("semantic_text_splitter") is not None

    except ImportError:
        return False

@lru_cache
def is_treeSitterPython_available() -> bool:
    # Check if package tree-sitter-languages s available:
    try:
        import importlib.util

        return importlib.util.find_spec("tree_sitter_python") is not None

    except ImportError:
        return False

@lru_cache
def is_treeSitterJava_available() -> bool:
    # Check if package tree-sitter-languages s available:
    try:
        import importlib.util

        return importlib.util.find_spec("tree_sitter_java") is not None

    except ImportError:
        return False
    
@lru_cache
def is_treeSitterC_available() -> bool:
    # Check if package tree-sitter-languages s available:
    try:
        import importlib.util

        return importlib.util.find_spec("tree_sitter_c") is not None

    except ImportError:
        return False
@lru_cache
def is_treeSitterCPP_available() -> bool:
    # Check if package tree-sitter-languages s available:
    try:
        import importlib.util

        return importlib.util.find_spec("tree_sitter_cpp") is not None

    except ImportError:
        return False
@lru_cache
def is_treeSitterCSharp_available() -> bool:
    # Check if package tree-sitter-languages s available:
    try:
        import importlib.util

        return importlib.util.find_spec("tree_sitter_c_sharp") is not None

    except ImportError:
        return False
@lru_cache
def is_treeSitterHtml_available() -> bool:
    # Check if package tree-sitter-languages s available:
    try:
        import importlib.util

        return importlib.util.find_spec("tree_sitter_html") is not None

    except ImportError:
        return False
@lru_cache
def is_treeSitterJson_available() -> bool:
    # Check if package tree-sitter-languages s available:
    try:
        import importlib.util

        return importlib.util.find_spec("tree_sitter_json") is not None

    except ImportError:
        return False
@lru_cache
def is_treeSitterBash_available() -> bool:
    # Check if package tree-sitter-languages s available:
    try:
        import importlib.util

        return importlib.util.find_spec("tree_sitter_bash") is not None

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
    language: str, text: str, size: int | Tuple[int, int], overlap: int, trim: bool
) -> List[str]:
    """
    Convert a code string into a list of semantic strings

    Parameters
    ----------
    language: str
        The coding language to split. Use the extension name (i.e. 'py', 'java', ect)
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
    # define available languages
    languages = {
        "py": "python",
        "java": "java",
        "html": "html",
        "c": "c",
        "cpp": "c++",
        "cs": "c#",
        "json": "json",
        "sh": "bash"
    }
    if language not in languages.keys():
        raise TypeError(
            "This coding language is not supported."
        )
    
    python_extension = languages[language]

    # Wanted to use match case but this python version <3.10 are accepted
    if python_extension == "python":
        # check module availability
        if not is_treeSitterPython_available():
                raise ImportError(
                    "tree-sitter-python is not available. Please install it to use this function."
                )
        
        import tree_sitter_python
        language_parser = tree_sitter_python.language()

    elif python_extension == "java":
        # check module availability
        if not is_treeSitterJava_available():
                raise ImportError(
                    "tree-sitter-java is not available. Please install it to use this function."
                )
        
        import tree_sitter_java
        language_parser = tree_sitter_java.language()

    elif python_extension == "html":
        # check module availability
        if not is_treeSitterHtml_available():
                raise ImportError(
                    "tree-sitter-html is not available. Please install it to use this function."
                )
        
        import tree_sitter_html
        language_parser = tree_sitter_html.language()

    elif python_extension == "c":
        # check module availability
        if not is_treeSitterC_available():
                raise ImportError(
                    "tree-sitter-c is not available. Please install it to use this function."
                )
        
        import tree_sitter_c
        language_parser = tree_sitter_c.language()

    elif python_extension == "c++":
        # check module availability
        if not is_treeSitterCPP_available():
                raise ImportError(
                    "tree-sitter-cpp is not available. Please install it to use this function."
                )
        
        import tree_sitter_cpp
        language_parser = tree_sitter_cpp.language()

    elif python_extension == "c#":
        # check module availability
        if not is_treeSitterCSharp_available():
                raise ImportError(
                    "tree-sitter-c-sharp is not available. Please install it to use this function."
                )
        
        import tree_sitter_c_sharp
        language_parser = tree_sitter_c_sharp.language()

    elif python_extension == "json":
        # check module availability
        if not is_treeSitterJson_available():
                raise ImportError(
                    "tree-sitter-json is not available. Please install it to use this function."
                )
        
        import tree_sitter_json
        language_parser = tree_sitter_json.language()

    else:
         # check module availability bash
        if not is_treeSitterBash_available():
                raise ImportError(
                    "tree-sitter-bash is not available. Please install it to use this function."
                )
        
        import tree_sitter_bash
        language_parser = tree_sitter_bash.language()        
    

    # Import the module
    import semantic_text_splitter

    # create the Textsplitter
    splitter = semantic_text_splitter.CodeSplitter(
        language=language_parser,
        capacity=size,
        overlap=overlap,
        trim=trim,
    )

    splitted_text = splitter.chunks(text)

    return splitted_text
