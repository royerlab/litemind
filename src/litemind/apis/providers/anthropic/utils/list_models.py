from typing import List


def _get_anthropic_models_list(client, max_num_models: int = 100) -> List[str]:
    # List Anthropic models:
    from anthropic.types import ModelInfo

    # Get the first n models, to be safe:
    models_info: List[ModelInfo] = client.models.list(limit=max_num_models).data

    # Extract model IDs
    model_list: List[str] = list([str(info.id) for info in models_info])

    # remove deprecated models: claude-3-sonnet-20240229
    if "claude-3-sonnet-20240229" in model_list:
        model_list.remove("claude-3-sonnet-20240229")

    # IF claude 3.7 is available, then add reasoning variant:
    if any("claude-3-7" in m for m in model_list):
        # get the model with claude-3-7 in its name from the list:
        claude_3_7_model = [m for m in model_list if "claude-3-7" in m][0]

        # Insert claude-3.7-thinking in place of claude-3.7, pushing the rest of the list down:
        model_list.insert(
            model_list.index(claude_3_7_model), claude_3_7_model + "-thinking-high"
        )
        model_list.insert(
            model_list.index(claude_3_7_model), claude_3_7_model + "-thinking-mid"
        )
        model_list.insert(
            model_list.index(claude_3_7_model), claude_3_7_model + "-thinking-low"
        )

    # If claude-opus-4 is available, then add reasoning variants:
    if any("claude-opus-4" in m for m in model_list):
        # get the model with claude-4 in its name from the list:
        claude_opus_4_model = [m for m in model_list if "claude-opus-4" in m][0]

        # Insert claude-opus-4-thinking in place of claude-4, pushing the rest of the list down:
        model_list.insert(
            model_list.index(claude_opus_4_model),
            claude_opus_4_model + "-thinking-high",
        )
        model_list.insert(
            model_list.index(claude_opus_4_model), claude_opus_4_model + "-thinking-mid"
        )
        model_list.insert(
            model_list.index(claude_opus_4_model), claude_opus_4_model + "-thinking-low"
        )

    # If claude-sonnet-4 is available, then add reasoning variants:
    if any("claude-sonnet-4" in m for m in model_list):
        # get the model with claude-sonnet-4 in its name from the list:
        claude_sonnet_4_model = [m for m in model_list if "claude-sonnet-4" in m][0]

        # Insert claude-sonnet-4-thinking in place of claude-sonnet-4, pushing the rest of the list down:
        model_list.insert(
            model_list.index(claude_sonnet_4_model),
            claude_sonnet_4_model + "-thinking-high",
        )
        model_list.insert(
            model_list.index(claude_sonnet_4_model),
            claude_sonnet_4_model + "-thinking-mid",
        )
        model_list.insert(
            model_list.index(claude_sonnet_4_model),
            claude_sonnet_4_model + "-thinking-low",
        )

    # Return model list:
    return model_list
