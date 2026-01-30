from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from google.genai import Client


def _get_gemini_models_list(client: "Client"):
    """
    Get the list of available Gemini models from the API.

    Includes:
    - Gemini models (text generation, multimodal)
    - Gemma models (open-weight text generation)
    - Imagen models (image generation)
    - Veo models (video generation)
    - Embedding models

    Excludes:
    - Deprecated or discontinued models
    - Legacy models (aqa, embedding-gecko-001)
    - Models that only support Interactions API (native-audio, computer-use, live, deep-research)

    Parameters
    ----------
    client : Client
        The google-genai Client instance.
    """
    model_list = []

    # Models to explicitly exclude (legacy or special purpose)
    excluded_models = {
        "models/aqa",  # Legacy question answering model
        "models/embedding-gecko-001",  # Legacy embedding model
    }

    # Patterns to exclude (models that only support Interactions API or are not
    # compatible with standard generateContent method)
    excluded_patterns = (
        "native-audio",  # Audio-only models that require Interactions API
        "computer-use",  # Computer Use models require special tool configuration
        "live-",  # Live/realtime models require special API handling
        "deep-research",  # Deep research models only support Interactions API
        "robotics",  # Robotics-focused models not suitable for general text generation
    )

    for model_obj in client.models.list():
        name = model_obj.name.lower()
        desc = (model_obj.description or "").lower()

        # Skip explicitly excluded models
        if model_obj.name in excluded_models:
            continue

        # Skip models matching excluded patterns (Interactions API only models)
        if any(pattern in name for pattern in excluded_patterns):
            continue

        # Skip deprecated or discontinued models
        if "will be discontinued" in desc or "deprecated" in desc:
            continue

        # Include all other models from supported families
        supported_prefixes = (
            "models/gemini-",  # Gemini models (includes gemini-embedding-001)
            "models/gemma-",  # Gemma open-weight models
            "models/imagen-",  # Imagen image generation
            "models/veo-",  # Veo video generation
            "models/text-embedding-",  # Legacy text embedding models
            "models/nano-",  # Experimental small models
        )

        if any(name.startswith(prefix) for prefix in supported_prefixes):
            model_list.append(model_obj.name)

    # Sort models to prioritize text generation models over specialized models
    # Priority order: gemini > gemma > nano > text-embedding > imagen > veo
    def model_priority(model_name: str) -> int:
        name_lower = model_name.lower()
        if name_lower.startswith("models/gemini-"):
            return 0  # Highest priority - main text generation
        elif name_lower.startswith("models/gemma-"):
            return 1  # Open-weight text generation
        elif name_lower.startswith("models/nano-"):
            return 2  # Experimental text models
        elif name_lower.startswith("models/text-embedding-"):
            return 3  # Embedding models
        elif name_lower.startswith("models/imagen-"):
            return 4  # Image generation
        elif name_lower.startswith("models/veo-"):
            return 5  # Video generation (lowest priority)
        return 6  # Unknown models

    # Sort by priority, then reverse within each group (newer models first)
    model_list.sort(key=lambda m: (model_priority(m), m), reverse=False)

    # Within each priority group, we want newer models first
    # The API typically returns models in chronological order, so reverse
    # each priority group separately
    from itertools import groupby

    sorted_list = []
    for _, group in groupby(model_list, key=model_priority):
        group_list = list(group)
        group_list.reverse()  # Newer models first within group
        sorted_list.extend(group_list)

    return sorted_list
