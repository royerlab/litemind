from typing import List


def _get_ollama_models_list(client) -> List[str]:
    # Get the ls of models:
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
