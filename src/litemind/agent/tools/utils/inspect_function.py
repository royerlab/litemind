import inspect


def extract_docstring(func):
    """
    Returns the docstring of the given function object as a string.

    Args:
        func: A Python function or callable.

    Returns:
        A string containing the docstring, or None if there is no docstring.
    """
    return inspect.getdoc(func)


def extract_function_info(func):
    """
    Returns both the docstring and signature of the given function object.

    Args:
        func: A Python function or callable.

    Returns:
        A tuple of (signature_string, docstring_string).
        If docstring is missing, docstring_string is None.
    """
    sig = str(inspect.signature(func))
    doc = inspect.getdoc(func)
    return sig, doc
