__version__ = "2025.3.1"

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
    api_instance = api_class()
    if not api_instance.check_availability_and_credentials():
        aprint(f"API {api_class.__name__} is not available. Removing it from the list.")
        API_IMPLEMENTATIONS.remove(api_class)
