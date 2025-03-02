from typing import Callable, Iterable


def aggregate_chat_responses(
    chunks: Iterable["ChatResponse"], callback: Callable
) -> "ChatResponse":
    """
    Aggregate an iterable of ChatResponse objects into a single ChatResponse.

    1. If there are no items in `chunks`, returns an empty ChatResponse.
    2. Otherwise, initializes with the first chunk and merges subsequent chunks:
       - Concatenates `.message.content`
       - Extends `.message.tool_calls` and `.message.images`
       - Overwrites scalar fields with the last non-None occurrence
       - Marks `.done` as True if any chunk has `.done=True`
    3. Returns the final merged ChatResponse.

    -----------
    Parameters:
        chunks: Iterable[ChatResponse]
            An iterable of ChatResponse objects (often from streaming).
        callback: Callable
            A callable to be invoked after each chunk is processed.

    Returns:
        A single ChatResponse with merged fields.
    """

    from ollama import ChatResponse, Message

    # Create an iterator from the chunks
    iterator = iter(chunks)

    # Attempt to get the first chunk
    try:
        first_chunk = next(iterator)

        # If the first chunk has content, call the callback
        if first_chunk.message.content:
            fragment = first_chunk.message.content
            callback(fragment=fragment)

    except StopIteration:
        # No chunks provided, return an empty ChatResponse
        message = Message(role="assistant", content="")

        return ChatResponse(message=message)

    # Initialize our aggregator from the first chunk
    # Doing model_validate on the .model_dump() ensures we create a safe copy
    aggregator = ChatResponse.model_validate(first_chunk.model_dump())

    # Ensure these are not None, for safe usage below
    aggregator.message.content = aggregator.message.content or ""
    aggregator.message.tool_calls = aggregator.message.tool_calls or []
    aggregator.message.images = aggregator.message.images or []

    # Merge subsequent chunks
    for chunk in iterator:

        # -- message.content: accumulate text
        if chunk.message.content:
            fragment = chunk.message.content
            aggregator.message.content += fragment
            callback(fragment=fragment)

        # -- message.tool_calls: extend
        if chunk.message.tool_calls:
            aggregator.message.tool_calls.extend(chunk.message.tool_calls)

        # -- message.images: extend
        if chunk.message.images:
            aggregator.message.images.extend(chunk.message.images)

        # -- message.role: overwrite with last known role (if present)
        if chunk.message.role is not None:
            aggregator.message.role = chunk.message.role

        # -- model: overwrite with last non-None
        if chunk.model is not None:
            aggregator.model = chunk.model

        # -- created_at: overwrite with last non-None
        if chunk.created_at is not None:
            aggregator.created_at = chunk.created_at

        # -- done: once true, remains true (or just take last chunk's value)
        aggregator.done = aggregator.done or (chunk.done is True)

        # -- done_reason: overwrite with last non-None
        if chunk.done_reason is not None:
            aggregator.done_reason = chunk.done_reason

        # -- total_duration, load_duration, etc.: overwrite with last non-None
        if chunk.total_duration is not None:
            aggregator.total_duration = chunk.total_duration

        if chunk.load_duration is not None:
            aggregator.load_duration = chunk.load_duration

        if chunk.prompt_eval_count is not None:
            aggregator.prompt_eval_count = chunk.prompt_eval_count

        if chunk.prompt_eval_duration is not None:
            aggregator.prompt_eval_duration = chunk.prompt_eval_duration

        if chunk.eval_count is not None:
            aggregator.eval_count = chunk.eval_count

        if chunk.eval_duration is not None:
            aggregator.eval_duration = chunk.eval_duration

    return aggregator
