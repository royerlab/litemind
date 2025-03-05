__version__ = "2025.3.4"

from arbol import aprint

from litemind.apis.combined_api import CombinedApi
from litemind.apis.default_api import DefaultApi
from litemind.apis.providers.anthropic.anthropic_api import AnthropicApi
from litemind.apis.providers.google.google_api import GeminiApi
from litemind.apis.providers.ollama.ollama_api import OllamaApi
from litemind.apis.providers.openai.openai_api import OpenAIApi

# Put all your implementations in this list:
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
        # Trying to instantiate the API class:
        api_instance = api_class()

        # Checking if the API is available and credentials are correct:
        if not api_instance.check_availability_and_credentials():
            aprint(
                f"API {api_class.__name__} is not available. Removing it from the list."
            )
            API_IMPLEMENTATIONS.remove(api_class)
    except Exception as e:
        # If an exception is raised, remove the API from the list:
        aprint(
            f"API {api_class.__name__} could not be instantiated: {e}. Removing it from the list."
        )
        API_IMPLEMENTATIONS.remove(api_class)

# Initialize and silence the Abseil logging system:
import absl.logging

absl.logging.set_verbosity(absl.logging.FATAL)
absl.logging.use_absl_handler()
