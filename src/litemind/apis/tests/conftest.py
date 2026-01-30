"""
Pytest configuration and fixtures for API tests.
"""

import pytest


@pytest.fixture(autouse=True)
def clear_api_caches():
    """Clear API availability caches before each test to prevent cross-test contamination."""
    # Import and call all clear functions
    from litemind.apis.providers.anthropic.utils.check_availability import (
        clear_availability_cache as clear_anthropic_cache,
    )
    from litemind.apis.providers.google.utils.check_availability import (
        clear_availability_cache as clear_gemini_cache,
    )
    from litemind.apis.providers.ollama.utils.check_availability import (
        clear_availability_cache as clear_ollama_cache,
    )
    from litemind.apis.providers.openai.utils.check_availability import (
        clear_availability_cache as clear_openai_cache,
    )

    # Clear all caches before each test
    clear_openai_cache()
    clear_gemini_cache()
    clear_ollama_cache()
    clear_anthropic_cache()

    yield

    # Optionally clear caches after each test as well
    clear_openai_cache()
    clear_gemini_cache()
    clear_ollama_cache()
    clear_anthropic_cache()
