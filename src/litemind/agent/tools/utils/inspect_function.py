"""Utilities for inspecting Python functions and extracting metadata.

Provides helper functions to retrieve a function's docstring and signature,
used primarily by ``FunctionTool`` to auto-generate tool descriptions and
argument schemas from wrapped callables.
"""

import inspect


def extract_docstring(func):
    """
    Extract the docstring of a function.

    Parameters
    ----------
    func : callable
        A Python function or callable.

    Returns
    -------
    str or None
        The cleaned docstring, or None if the function has no docstring.
    """
    return inspect.getdoc(func)


def extract_function_info(func):
    """
    Extract both the signature and docstring of a function.

    Parameters
    ----------
    func : callable
        A Python function or callable.

    Returns
    -------
    tuple of (str, str or None)
        A tuple of ``(signature_string, docstring_string)``. The
        docstring is None if the function has no docstring.
    """
    sig = str(inspect.signature(func))
    doc = inspect.getdoc(func)
    return sig, doc
