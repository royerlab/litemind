from typing import Callable, Iterable


def aggregate_chat_response(chunks: Iterable, callback: Callable):
    """
    Aggregate streaming response chunks from the google-genai SDK.

    This handles the new SDK's streaming response structure where chunks
    contain candidates with content parts.

    Parameters
    ----------
    chunks : Iterable
        Iterable of streaming response chunks from client.models.generate_content_stream()
    callback : Callable
        Callback function to call with each text chunk for streaming updates

    Returns
    -------
    dict
        Dictionary containing:
        - "text": Aggregated text response
        - "thinking": Native thinking content (if any)
        - "tool_calls": List of function calls extracted from the response
    """
    # Aggregated text response:
    aggregated_text = ""

    # Aggregated thinking response:
    aggregated_thinking = ""

    # Function calls:
    function_calls = []

    # Last chunk for final processing:
    last_chunk = None

    # Iterate over the chunks:
    for chunk in chunks:

        # Keep the last chunk:
        last_chunk = chunk

        # Extract text and thinking from the chunk
        chunk_text, chunk_thinking = _extract_content_from_chunk(chunk)

        # Append the text to the aggregated text:
        if chunk_text:
            aggregated_text += chunk_text
            callback(chunk_text)

        # Append thinking to aggregated thinking:
        if chunk_thinking:
            aggregated_thinking += chunk_thinking

    # Check for function calls from the final chunk's parts
    if last_chunk:
        function_calls = _extract_function_calls_from_chunk(last_chunk)

    return {
        "text": aggregated_text,
        "thinking": aggregated_thinking if aggregated_thinking else None,
        "tool_calls": function_calls,
    }


def _extract_content_from_chunk(chunk) -> tuple:
    """
    Extract text and thinking content from a streaming chunk.

    Parameters
    ----------
    chunk
        A streaming response chunk

    Returns
    -------
    tuple
        (text_content, thinking_content)
    """
    text_content = ""
    thinking_content = ""

    # Always check candidates/parts to capture both text and thinking content
    if not hasattr(chunk, "candidates") or not chunk.candidates:
        # Fall back to .text property if no candidates
        try:
            if hasattr(chunk, "text") and chunk.text:
                text_content = chunk.text
        except (AttributeError, ValueError):
            # .text may raise ValueError if response contains function calls
            pass
        return text_content, thinking_content

    for candidate in chunk.candidates:
        if not hasattr(candidate, "content") or not candidate.content:
            continue

        if not hasattr(candidate.content, "parts") or not candidate.content.parts:
            continue

        for part in candidate.content.parts:
            # Check if this part is thinking content (thought is a boolean flag)
            is_thinking_part = hasattr(part, "thought") and part.thought is True

            # Check for text content
            if hasattr(part, "text") and part.text:
                if is_thinking_part:
                    # This is thinking/reasoning content
                    thinking_content += part.text
                else:
                    # This is regular text content
                    text_content += part.text

    return text_content, thinking_content


def _extract_function_calls_from_chunk(chunk) -> list:
    """
    Extract function calls from a response chunk.

    Parameters
    ----------
    chunk
        A response chunk

    Returns
    -------
    list
        List of function call dictionaries with 'name' and 'args' keys
    """
    function_calls = []

    if not hasattr(chunk, "candidates") or not chunk.candidates:
        return function_calls

    for candidate in chunk.candidates:
        if not hasattr(candidate, "content") or not candidate.content:
            continue

        if not hasattr(candidate.content, "parts") or not candidate.content.parts:
            continue

        for part in candidate.content.parts:
            # Check for function_call attribute
            if hasattr(part, "function_call") and part.function_call:
                fc = part.function_call
                function_calls.append(
                    {
                        "name": fc.name,
                        "args": dict(fc.args) if fc.args else {},
                    }
                )

    return function_calls
