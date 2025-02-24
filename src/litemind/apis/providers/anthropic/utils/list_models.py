from typing import List


def _get_anthropic_models_list(client, max_num_models: int = 100) -> List[str]:
    # List Anthropic models:
    from anthropic.types import ModelInfo

    # Get the first n models, to be safe:
    models_info: List[ModelInfo] = client.models.list(limit=max_num_models).data

    # Extract model IDs
    model_list: List[str] = list([str(info.id) for info in models_info])

    # Return model list:
    return model_list
