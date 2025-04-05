from litemind.agent.tools.utils.inspect_function import (
    extract_docstring,
    extract_function_info,
)


def test_extract_docstring():
    def sample_function():
        """This is a sample docstring."""
        pass

    assert extract_docstring(sample_function) == "This is a sample docstring."

    def no_docstring_function():
        pass

    assert extract_docstring(no_docstring_function) is None


def test_extract_function_info():
    def sample_function(arg1, arg2):
        """This is a sample docstring."""
        pass

    sig, doc = extract_function_info(sample_function)
    assert sig == "(arg1, arg2)"
    assert doc == "This is a sample docstring."

    def no_docstring_function(arg1, arg2):
        pass

    sig, doc = extract_function_info(no_docstring_function)
    assert sig == "(arg1, arg2)"
    assert doc is None
