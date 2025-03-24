import traceback
from typing import Any, List, Optional

from arbol import aprint


def _get_raw_openai_model_list(client: "OpenAI"):
    from openai import OpenAI

    # Explicit typing:
    client: OpenAI = client

    # Raw model list:
    models = client.models.list().data

    # Return the list of models:
    return models


def get_openai_model_list(
    raw_model_list: List[Any],
    included: Optional[List[str]] = None,
    excluded: Optional[List[str]] = None,
    exclude_dated_models: bool = True,
    verbose: bool = False,
) -> List[str]:
    """
    Get the list of all OpenAI ChatGPT models.

    Parameters
    ----------
    raw_model_list : list
        Raw list of models.
    included : str
        Filter to apply to the list of models. If None, all models are returned.
        Models must contain at least one of the filters to be included in the list.
    excluded : str
        Excluded models. If None, no models are excluded.
        Models must not contain any of the excluded models to be included in the list.
    exclude_dated_models: bool
        If True, remove models with a date in the model id. This is useful to keep only the most recent version of each model.
    verbose : bool
        Verbosity flag.

    Returns
    -------
    list[str]
        List of models.

    """

    try:
        if included is None:
            # Base filtering of models:
            included = [
                "dall-e",
                "audio",
                "gpt",
                "o1",
                "o3",
                "text-embedding",
                "whisper",
                "ada-002",
            ]

        if excluded is None:
            # Exclude models that are not supported by the API:
            excluded = ["ada-002"]

        # Convert to list of model ids:
        models = [model.id for model in raw_model_list]

        if exclude_dated_models:
            models = _remove_dated_models(models)

        # Model list to populate:
        model_list = []

        # Goes through models and populate the list:
        for model in models:

            # only keep models that match the filter:
            if not included or any(f in model for f in included):
                model_list.append(model)
                if verbose:
                    aprint(f"Included: {model}")

            # Only keep models that do not match the excluded:
            if excluded and any(e in model for e in excluded):
                model_list.remove(model)
                if verbose:
                    aprint(f"Excluded: {model}")

        # Remove duplicates:
        model_list = list(set(model_list))

        # Actual sorting:
        sorted_model_list = sorted(model_list, key=model_key, reverse=True)

        # List of reasoning models:
        reasoning_models = ["o1", "o1-mini", "o3-mini"]

        # Replace each reasoning model 'X' with its three variants:  X-low, X-mid, X-high:
        for reasoning_model in reasoning_models:
            if reasoning_model in sorted_model_list:
                # Find index of reasoning model:
                index = sorted_model_list.index(reasoning_model)
                # Insert three variants:
                sorted_model_list.insert(index + 1, f"{reasoning_model}-high")
                sorted_model_list.insert(index + 2, f"{reasoning_model}-medium")
                sorted_model_list.insert(index + 3, f"{reasoning_model}-low")
                # Remove the original reasoning model:
                sorted_model_list.remove(reasoning_model)

        return sorted_model_list

    except Exception as e:
        # Error message:
        aprint(
            f"Error: {type(e).__name__} with message: '{str(e)}' occured while trying to get the list of OpenAI models. "
        )
        # print stacktrace:
        traceback.print_exc()

        return []


def _remove_dated_models(models):
    nodate_models = []
    for model in models:

        # Remove '-preview' from name:
        model = model.replace("-preview", "")

        if "-" in model:
            parts = model.split("-")

            if len(parts) >= 4:

                year = parts[-3]
                month = parts[-2]
                day = parts[-1]

                # Check if the date is valid:
                if len(year) == 4 and len(month) == 2 and len(day) == 2:
                    continue

            if len(parts) >= 2:
                if parts[-1].isdigit() and len(parts[-1]) == 4:
                    continue

        nodate_models.append(model)
    return nodate_models


# Next we sort models so the best ones are at the beginning of the list:
def model_key(model):
    score = 0

    if "o3" in model:
        score += 120
    elif "o1" in model:
        score += 110
    elif "gpt-4o" in model:
        score += 100
    elif "gpt-4" in model:
        score += 90
    elif "gpt-3.5" in model:
        score += 80
    elif "gpt-3" in model:
        score += 70

    if "mini" in model:
        score -= 5

    if "preview" in model:
        score -= 2

    return score


def postprocess_openai_model_list(model_list: list) -> list:
    """
    Postprocess the list of OpenAI models. This is useful to remove problematic models from the list and sort models in decreasing order of quality.

    Parameters
    ----------
    model_list : list
        List of models.

    Returns
    -------
    list
        Post-processed list of models.

    """

    try:
        # First, sort the list of models:
        model_list = sorted(model_list)

        # get list of bad models for main LLM:
        bad_models_filters = {
            "0613",
            "vision",
            "turbo-instruct",
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-0613",
            "gpt-3.5-turbo-0301",
            "gpt-3.5-turbo-1106",
            "gpt-3.5-turbo-0125",
            "gpt-3.5-turbo-16k",
            "chatgpt-4o-latest",
        }

        # get list of best models for main LLM:
        best_models_filters = {"0314", "0301", "1106", "gpt-4", "gpt-4o"}

        # Ensure that some 'bad' or unsupported models are excluded:
        bad_models = [
            m for m in model_list if any(bm in m for bm in bad_models_filters)
        ]
        for bad_model in bad_models:
            if bad_model in model_list:
                model_list.remove(bad_model)
                # model_list.append(bad_model)

        # Ensure that the best models are at the top of the list:
        best_models = [
            m for m in model_list if any(bm in m for bm in best_models_filters)
        ]
        model_list = best_models + [m for m in model_list if m not in best_models]

        # Ensure that the very best models are at the top of the list:
        very_best_models = [
            m for m in model_list if ("gpt-4o" in m and "mini" not in m)
        ]
        model_list = very_best_models + [
            m for m in model_list if m not in very_best_models
        ]

    except Exception as exc:
        aprint(f"Error occurred: {exc}")

        # print stacktrace:
        traceback.print_exc()

    return model_list
