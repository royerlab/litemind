__version__ = "2025.07.01.1"

from arbol import aprint, asection

from litemind.agent.augmentations.vector_db.in_memory_vector_db import (
    InMemoryVectorDatabase,
)
from litemind.agent.augmentations.vector_db.qdrant_vector_db import QdrantVectorDatabase
from litemind.apis.combined_api import CombinedApi
from litemind.apis.default_api import DefaultApi
from litemind.apis.providers.anthropic.anthropic_api import AnthropicApi
from litemind.apis.providers.google.google_api import GeminiApi
from litemind.apis.providers.ollama.ollama_api import OllamaApi
from litemind.apis.providers.openai.openai_api import OpenAIApi

# LLM API implementations:
API_IMPLEMENTATIONS = [
    DefaultApi,
    OpenAIApi,
    OllamaApi,
    AnthropicApi,
    GeminiApi,
    CombinedApi,
]

# Check availability of each API and remove it from the list if it is not available:
for api_class in API_IMPLEMENTATIONS:
    try:
        with asection(f"Checking availability of: {api_class.__name__}"):
            # Trying to instantiate the API class:
            api_instance = api_class()

            # Checking if the API is available and credentials are correct:
            if (
                not api_instance.check_availability_and_credentials()
                or len(api_instance.list_models()) == 0
            ):
                aprint(
                    f"API {api_class.__name__} is not available. Removing it from the list."
                )
                API_IMPLEMENTATIONS.remove(api_class)
            else:
                aprint(
                    f"API {api_class.__name__} is available and has {len(api_instance.list_models())} models."
                )

    except Exception as e:
        # If an exception is raised, remove the API from the list:
        aprint(
            f"API {api_class.__name__} could not be instantiated: {e}. Removing it from the list."
        )
        API_IMPLEMENTATIONS.remove(api_class)

# Remove the default and combined API from the list of implementations if they are the only ones in the list:
if len(API_IMPLEMENTATIONS) <= 2 and CombinedApi in API_IMPLEMENTATIONS:
    API_IMPLEMENTATIONS.remove(CombinedApi)
    aprint(
        "Removing CombinedApi from the list of implementations as it is the only one available."
    )

# Vector database implementations:
VECDB_IMPLEMENTATIONS = [QdrantVectorDatabase, InMemoryVectorDatabase]

# Check availability of each vector database implementation:
for vecdb_class in list(VECDB_IMPLEMENTATIONS):
    try:
        # Check specific dependencies based on the vector database class
        if vecdb_class == QdrantVectorDatabase:
            import qdrant_client
        # InMemoryVectorDatabase doesn't require external dependencies

    except ImportError as e:
        # If dependencies are not installed, remove the vector database from the list
        aprint(
            f"Vector database {vecdb_class.__name__} dependencies are not installed: {e}. Removing it from the list."
        )
        VECDB_IMPLEMENTATIONS.remove(vecdb_class)
    except Exception as e:
        # Handle any other exceptions
        aprint(
            f"Error checking vector database {vecdb_class.__name__}: {e}. Removing it from the list."
        )
        VECDB_IMPLEMENTATIONS.remove(vecdb_class)

# If no API_IMPLEMENTATIONS are available then the QdrantVectorDatabase cannot be used:
if len(API_IMPLEMENTATIONS) == 0 and QdrantVectorDatabase in VECDB_IMPLEMENTATIONS:
    aprint(
        "No API implementations available. Removing QdrantVectorDatabase from the list of implementations."
    )
    VECDB_IMPLEMENTATIONS.remove(QdrantVectorDatabase)

# Initialize and silence the Abseil logging system:
import absl.logging

absl.logging.set_verbosity(absl.logging.FATAL)
absl.logging.use_absl_handler()
