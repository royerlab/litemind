"""Utility for fetching and sorting the Ollama model list.

Retrieves locally available models from the Ollama server, sorts them
by size (largest/most capable first), and creates ``-thinking`` suffixed
variants for each model to enable thinking/reasoning mode.
"""

from typing import List


def _get_ollama_models_list(client) -> List[str]:
    """Get available Ollama models, sorted by size, with thinking variants.

    For each model, a ``-thinking`` suffixed variant is also included
    to enable thinking/reasoning mode.

    Parameters
    ----------
    client : Any
        The Ollama client instance.

    Returns
    -------
    List[str]
        Model names sorted by size (largest first), with thinking variants.
    """
    # Get the list of models:
    model_list = list(client.list().models)

    # Get the list of models:
    model_list_names = [model.model for model in model_list]

    # Get the list of model sizes:
    model_list_sizes = [model.size for model in model_list]

    # Sort by decreasing ollama model size:
    model_list = [
        str(x) for _, x in sorted(zip(model_list_sizes, model_list_names), reverse=True)
    ]

    # Ensure it is a list:
    model_list = list(model_list)

    # For each model, insert a thinking version with a '-thinking' suffix to the same list:
    model_list_with_thinking = []
    for model in model_list:
        model_list_with_thinking.append(model)
        model_list_with_thinking.append(f"{model}-thinking")
    model_list = model_list_with_thinking

    # Return the list of models:
    return model_list
