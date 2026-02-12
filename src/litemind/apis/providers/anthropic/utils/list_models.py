"""Utilities for listing and enumerating Anthropic model variants."""

from typing import List

# Models that support 1M context via the context-1m beta header:
_LONG_CONTEXT_PATTERNS = ["claude-opus-4-6", "claude-sonnet-4-5", "claude-sonnet-4-2"]

# Models that support -thinking-max (adaptive thinking with max effort):
_MAX_THINKING_PATTERNS = ["claude-opus-4-6"]


def _get_anthropic_models_list(
    client, max_num_models: int = 100, enable_long_context: bool = True
) -> List[str]:
    """Fetch available Anthropic models and generate thinking/long-context variants.

    Queries the Anthropic models API, removes deprecated models, and appends
    thinking suffix variants (``-thinking-low``, ``-thinking-mid``,
    ``-thinking-high``, ``-thinking-max``) and long-context ``-1m`` variants
    for supported model families.

    Parameters
    ----------
    client : Anthropic
        An initialized Anthropic client instance.
    max_num_models : int
        Maximum number of models to retrieve from the API.
    enable_long_context : bool
        If True, adds ``-1m`` variants for models that support 1M context.

    Returns
    -------
    List[str]
        List of model ID strings including generated variants.
    """
    # List Anthropic models:
    from anthropic.types import ModelInfo

    # Get the first n models, to be safe:
    models_info: List[ModelInfo] = client.models.list(limit=max_num_models).data

    # Extract model IDs
    model_list: List[str] = list([str(info.id) for info in models_info])

    # remove deprecated models: claude-3-sonnet-20240229
    if "claude-3-sonnet-20240229" in model_list:
        model_list.remove("claude-3-sonnet-20240229")

    # Add thinking variants for each model family:
    _add_thinking_variants(model_list)

    # Add -1m (long context) variants for models that support 1M context:
    if enable_long_context:
        _add_long_context_variants(model_list)

    # Return model list:
    return model_list


def _add_thinking_variants(model_list: List[str]) -> None:
    """Add thinking suffix variants for supported models.

    All supported models get ``-thinking-low``, ``-thinking-mid``, and
    ``-thinking-high`` variants. Opus 4.6 also gets ``-thinking-max``
    (adaptive thinking with max effort). Variants are inserted after the
    base model so that the base model remains the default (first) choice
    when auto-selecting.

    Parameters
    ----------
    model_list : List[str]
        Model list to modify in place by inserting thinking variants.
    """
    for pattern in [
        "claude-opus-4-6",
        "claude-opus-4-5",
        "claude-sonnet-4-5",
        "claude-3-7",
    ]:
        matches = [m for m in model_list if pattern in m]
        if matches:
            model = matches[0]
            idx = model_list.index(model)
            # Insert after base model: base, low, mid, high, [max]
            model_list.insert(idx + 1, model + "-thinking-low")
            model_list.insert(idx + 2, model + "-thinking-mid")
            model_list.insert(idx + 3, model + "-thinking-high")

            # Add -thinking-max for models that support it (Opus 4.6):
            if any(p in model for p in _MAX_THINKING_PATTERNS):
                high_idx = model_list.index(model + "-thinking-high")
                model_list.insert(high_idx + 1, model + "-thinking-max")


def _add_long_context_variants(model_list: List[str]) -> None:
    """Add ``-1m`` suffix variants for models that support 1M context.

    For each matching base model, adds (after all its thinking variants):

    - ``base-1m`` (long context variant)
    - ``base-1m-thinking-low/mid/high`` (only if base has thinking variants)
    - ``base-1m-thinking-max`` (for Opus 4.6 only, if base has thinking variants)

    Parameters
    ----------
    model_list : List[str]
        Model list to modify in place by inserting long-context variants.
    """
    # Collect insertion groups: (group_end_index, [variants_to_insert])
    # Each group inserts a contiguous block at a unique position.
    insertions = []

    for model in list(model_list):
        # Skip models that are already variants (thinking or -1m):
        if "-thinking" in model or "-1m" in model:
            continue

        # Check if this model matches any long context pattern:
        if any(pattern in model for pattern in _LONG_CONTEXT_PATTERNS):
            base_index = model_list.index(model)
            has_max = any(p in model for p in _MAX_THINKING_PATTERNS)
            has_thinking = (model + "-thinking-high") in model_list

            # Find the end of this model's group (base + thinking variants):
            thinking_count = sum(
                1 for m in model_list if m.startswith(model + "-thinking-")
            )
            group_end = base_index + thinking_count + 1

            variants = [model + "-1m"]
            # Only add -1m-thinking-* variants if base model has thinking variants:
            if has_thinking:
                variants.append(model + "-1m-thinking-low")
                variants.append(model + "-1m-thinking-mid")
                variants.append(model + "-1m-thinking-high")
                if has_max:
                    variants.append(model + "-1m-thinking-max")

            insertions.append((group_end, variants))

    # Insert from the end backward so earlier insertions don't shift later ones:
    for insert_index, variants in reversed(insertions):
        for i, variant in enumerate(variants):
            model_list.insert(insert_index + i, variant)
