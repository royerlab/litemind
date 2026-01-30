"""
Standardized error handling utilities for API operations.

This module provides decorators and utilities for consistent exception
handling across all API providers.
"""

import functools
import logging
from typing import Callable, TypeVar

from litemind.apis.exceptions import APIError

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable)


def api_error_handler(operation_name: str) -> Callable[[F], F]:
    """
    Decorator for consistent API error handling.

    Wraps a function to catch exceptions and re-raise them as APIError
    with consistent logging.

    Parameters
    ----------
    operation_name : str
        A descriptive name for the operation (e.g., "OpenAI generate text").
        This will be included in the error message.

    Returns
    -------
    Callable
        A decorator that wraps the function with error handling.

    Examples
    --------
    >>> @api_error_handler("OpenAI generate text")
    ... def generate_text(self, messages, **kwargs):
    ...     # implementation
    ...     pass
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except APIError:
                # Re-raise our own errors without modification
                raise
            except Exception as e:
                logger.exception(f"{operation_name} failed")
                raise APIError(f"{operation_name} error: {e}") from e

        return wrapper  # type: ignore

    return decorator
