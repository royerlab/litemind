import traceback
from functools import lru_cache
from typing import List, Optional

from arbol import aprint


def get_openai_model_list(included: Optional[List[str]] = None,
                          excluded: Optional[List[str]] = None,
                          verbose: bool = False) -> list[str]:
    """
    Get the list of all OpenAI ChatGPT models.

    Parameters
    ----------
    included : str
        Filter to apply to the list of models. If None, all models are returned.
        Models must contain at least one of the filters to be included in the list.
    excluded : str
        Excluded models. If None, no models are excluded.
        Models must not contain any of the excluded models to be included in the list.
    verbose : bool
        Verbosity flag.

    Returns
    -------
    list[str]
        List of models.

    """

    # Local imports to avoid issues:

    try:
        if included is None:
            included = ['gpt']

        models = _get_raw_openai_model_list()

        # Model list to populate:
        model_list = []

        # Goes through models and populate the list:
        for model in models:
            model_id = model.id

            # only keep models that match the filter:
            if not included or any(f in model_id for f in included):
                model_list.append(model_id)
                if verbose:
                    aprint(f"Included: {model_id}")

            # Only keep models that do not match the excluded:
            if excluded and any(e in model_id for e in excluded):
                model_list.remove(model_id)
                if verbose:
                    aprint(f"Excluded: {model_id}")

        # Next we sort models so the best ones are at the beginning of the list:
        def model_key(model):
            # Split the model name into parts
            parts = model.split('-')

            # If a part is '4o' or 'o1', replace it with 'o1.25' or '4.25' respectively:
            parts = [part if part not in ['4o', 'o1'] else part.replace('o',
                                                                        '.25')
                     for part in parts]

            # Remove all the parts that are not numbers (integer or float):
            parts = [part for part in parts if
                     part.replace('.', '', 1).isdigit()]

            # Remove parts that are integers but that lead with a zero:
            if len(parts) > 1:
                parts = [part for part in parts if not part.startswith('0')]

            # Remove parts that are numbers (float or int) that are too big (>10.0)
            if len(parts) > 1:
                parts = [part for part in parts if float(part) < 5]

            # Get the main version (e.g., '3.5' or '4' from 'gpt-3.5' or 'gpt-4')
            main_version = float(parts[-1])

            # If we find 'mini' in the model name then subtract 0.25 from the main version:
            if 'mini' in model:
                main_version -= 0.12

            # Use the length of the model name as a secondary sorting criterion
            length = len(model)
            # Sort by main version (descending), then by length (ascending)
            return -main_version, length

        # Actual sorting:
        sorted_model_list = sorted(model_list, key=model_key)

        return sorted_model_list

    except Exception as e:
        # Error message:
        aprint(
            f"Error: {type(e).__name__} with message: '{str(e)}' occured while trying to get the list of OpenAI models. ")
        # print stacktrace:
        traceback.print_exc()

        return []


@lru_cache()
def _get_raw_openai_model_list():
    from openai import OpenAI

    # Instantiate API entry point
    client = OpenAI()

    # Raw model list:
    models = client.models.list().data
    return models


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
        bad_models_filters = {'0613', 'vision',
                              'turbo-instruct',
                              'gpt-3.5-turbo',
                              'gpt-3.5-turbo-0613',
                              'gpt-3.5-turbo-0301',
                              'gpt-3.5-turbo-1106',
                              'gpt-3.5-turbo-0125',
                              'gpt-3.5-turbo-16k',
                              'chatgpt-4o-latest'}

        # get list of best models for main LLM:
        best_models_filters = {'0314', '0301', '1106', 'gpt-4', 'gpt-4o'}

        # Ensure that some 'bad' or unsupported models are excluded:
        bad_models = [m for m in model_list if
                      any(bm in m for bm in bad_models_filters)]
        for bad_model in bad_models:
            if bad_model in model_list:
                model_list.remove(bad_model)
                # model_list.append(bad_model)

        # Ensure that the best models are at the top of the list:
        best_models = [m for m in model_list if
                       any(bm in m for bm in best_models_filters)]
        model_list = best_models + [m for m in model_list if
                                    m not in best_models]

        # Ensure that the very best models are at the top of the list:
        very_best_models = [m for m in model_list if
                            ('gpt-4o' in m and 'mini' not in m)]
        model_list = very_best_models + [m for m in model_list if
                                         m not in very_best_models]

    except Exception as exc:
        aprint(f"Error occurred: {exc}")

        # print stacktrace:
        traceback.print_exc()

    return model_list
