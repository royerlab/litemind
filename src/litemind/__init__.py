"""
Litemind: A unified API wrapper and agentic AI framework.

Litemind provides a consistent interface to multiple LLM providers (OpenAI,
Anthropic, Google Gemini, Ollama) along with an agentic framework for building
tool-augmented, multimodal AI applications.

This top-level package exposes the main user-facing classes and lazy
availability checks for API backends and vector databases.
"""

__version__ = "2026.2.7"

from arbol import aprint, asection

# Core exports for user-facing API:
from litemind.agent.agent import Agent
from litemind.agent.augmentations.information.information import Information

# API class imports:
from litemind.agent.augmentations.vector_db.in_memory_vector_db import (
    InMemoryVectorDatabase,
)
from litemind.agent.augmentations.vector_db.qdrant_vector_db import QdrantVectorDatabase
from litemind.agent.messages.conversation import Conversation
from litemind.agent.messages.message import Message
from litemind.agent.tools.toolset import ToolSet
from litemind.apis.combined_api import CombinedApi
from litemind.apis.default_api import DefaultApi
from litemind.apis.model_features import ModelFeatures
from litemind.apis.providers.anthropic.anthropic_api import AnthropicApi
from litemind.apis.providers.google.google_api import GeminiApi
from litemind.apis.providers.ollama.ollama_api import OllamaApi
from litemind.apis.providers.openai.openai_api import OpenAIApi

# All API classes (static list for lazy initialization):
_ALL_API_CLASSES = [
    DefaultApi,
    OpenAIApi,
    OllamaApi,
    AnthropicApi,
    GeminiApi,
    CombinedApi,
]

# All vector database classes (static list for lazy initialization):
_ALL_VECDB_CLASSES = [QdrantVectorDatabase, InMemoryVectorDatabase]

# Cached results (None means not yet computed):
_api_implementations = None
_vecdb_implementations = None


def get_available_apis():
    """
    Get a list of available API implementations.

    Lazily checks availability and credentials of each registered API
    provider on the first call, then caches the result for subsequent
    calls. ``CombinedApi`` is included only when at least one concrete
    provider (other than ``DefaultApi``) is available.

    Returns
    -------
    list[type[BaseApi]]
        API classes whose providers are reachable and have valid
        credentials.
    """
    global _api_implementations

    if _api_implementations is not None:
        return _api_implementations

    # Build new list (don't modify _ALL_API_CLASSES)
    available = []

    for api_class in _ALL_API_CLASSES:
        # Skip CombinedApi initially - we'll handle it at the end
        if api_class == CombinedApi:
            continue

        try:
            with asection(f"Checking availability of: {api_class.__name__}"):
                # Try to instantiate the API class:
                api_instance = api_class()

                # Check if the API is available and credentials are correct:
                if (
                    not api_instance.check_availability_and_credentials()
                    or len(api_instance.list_models()) == 0
                ):
                    aprint(f"API {api_class.__name__} is not available. Skipping.")
                else:
                    aprint(
                        f"API {api_class.__name__} is available and has {len(api_instance.list_models())} models."
                    )
                    available.append(api_class)

        except Exception as e:
            aprint(
                f"API {api_class.__name__} could not be instantiated: {e}. Skipping."
            )

    # Add CombinedApi only if there are other APIs available
    # (need more than just DefaultApi)
    non_default_apis = [a for a in available if a != DefaultApi]
    if len(non_default_apis) > 0:
        available.append(CombinedApi)
    else:
        aprint("No individual APIs available. Skipping CombinedApi.")

    _api_implementations = available
    return _api_implementations


def get_available_vecdbs():
    """
    Get a list of available vector database implementations.

    Lazily checks that the required dependencies for each vector database
    backend are installed on the first call, then caches the result.
    ``QdrantVectorDatabase`` is excluded when no API provider is available
    because it relies on external embeddings.

    Returns
    -------
    list[type[BaseVectorDB]]
        Vector database classes whose dependencies are satisfied.
    """
    global _vecdb_implementations

    if _vecdb_implementations is not None:
        return _vecdb_implementations

    # Build new list (don't modify _ALL_VECDB_CLASSES)
    available = []

    for vecdb_class in _ALL_VECDB_CLASSES:
        try:
            # Check specific dependencies based on the vector database class
            if vecdb_class == QdrantVectorDatabase:
                import qdrant_client  # noqa: F401

            # InMemoryVectorDatabase doesn't require external dependencies
            available.append(vecdb_class)

        except ImportError as e:
            aprint(
                f"Vector database {vecdb_class.__name__} dependencies are not installed: {e}. Skipping."
            )
        except Exception as e:
            aprint(
                f"Error checking vector database {vecdb_class.__name__}: {e}. Skipping."
            )

    # If no APIs are available, QdrantVectorDatabase cannot be used
    # (it needs embeddings). Check lazily only when needed.
    if QdrantVectorDatabase in available:
        apis = get_available_apis()
        if len(apis) == 0:
            aprint("No API implementations available. Removing QdrantVectorDatabase.")
            available = [v for v in available if v != QdrantVectorDatabase]

    _vecdb_implementations = available
    return _vecdb_implementations


# For backwards compatibility, provide API_IMPLEMENTATIONS and VECDB_IMPLEMENTATIONS
# as static lists of all API/VecDB classes. These lists no longer filter by availability
# at import time (to avoid network calls during import).
# To get only available implementations, use get_available_apis() or get_available_vecdbs().
API_IMPLEMENTATIONS = _ALL_API_CLASSES.copy()
VECDB_IMPLEMENTATIONS = _ALL_VECDB_CLASSES.copy()

# Initialize and silence the Abseil logging system:
import absl.logging

absl.logging.set_verbosity(absl.logging.FATAL)
absl.logging.use_absl_handler()

__all__ = [
    # Version
    "__version__",
    # API classes
    "DefaultApi",
    "OpenAIApi",
    "OllamaApi",
    "AnthropicApi",
    "GeminiApi",
    "CombinedApi",
    # Vector databases
    "InMemoryVectorDatabase",
    "QdrantVectorDatabase",
    # Core agent API
    "Agent",
    "Message",
    "Conversation",
    "ToolSet",
    "ModelFeatures",
    "Information",
    # Availability functions
    "get_available_apis",
    "get_available_vecdbs",
    # Legacy (static lists)
    "API_IMPLEMENTATIONS",
    "VECDB_IMPLEMENTATIONS",
]
