from typing import Callable, Iterable


def aggregate_chat_response(chunks: Iterable["ChatResponse"], callback: Callable):
    # Aggregated text response:
    aggregated_text = ""

    # Function calls:
    function_calls = []

    # Last response:
    last_chunk = None

    # Iterate over the chunks:
    for chunk in chunks:

        # Keep the last chunk:
        last_chunk = chunk

        # Extract the text from the chunk:
        chunk_text = _extract_text_from_parts(chunk.parts)

        # Append the text to the aggregated text:
        if chunk_text:
            aggregated_text += chunk_text
            callback(chunk_text)

    # Check for function calls from the final partial's parts
    if last_chunk and hasattr(last_chunk, "parts"):

        # Iterate over the parts of the last chunk:
        for part in last_chunk.parts:
            # Convert the part to a dict:
            dict_part = type(part).to_dict(part, use_integers_for_enums=False)

            # part might have a function_call:
            if "function_call" in dict_part:
                function_calls.append(dict_part["function_call"])

    return {"text": aggregated_text, "tool_calls": function_calls}


def _extract_text_from_parts(parts) -> str:
    """
    Convert each part to a dict, append `dict_part['text']` if present.
    (Ignore function calls, code, etc.)
    """
    out_text = []
    for p in parts:
        part_dict = type(p).to_dict(p, use_integers_for_enums=False)
        # If there's a "text" key, gather it:
        if "text" in part_dict:
            out_text.append(part_dict["text"])
    return "".join(out_text)
