# -------------------------------------------------------------------------
# Extract text from a partial response's parts by converting to a dict
# -------------------------------------------------------------------------
from typing import Callable, List

from pydantic import BaseModel

from litemind.agent.tools.toolset import ToolSet


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


# -------------------------------------------------------------------------
# streaming multi-turn logic if we have Tools
# -------------------------------------------------------------------------
def _stream_chat_with_tools(
    chat_obj: "genai.Chat",
    model_name: str,
    initial_message_parts: "List[genai.protos.Part]",
    on_text_streaming: Callable,
    toolset: ToolSet,
    response_format: BaseModel,
    max_output_tokens: int,
    temperature: float,
    **cb_kwargs,
) -> str:
    """
    Streams partial responses from chat_obj (function calling).
    After each "turn" we see if there's a function call. If so, we call it
    and feed the result back as functionResponse, repeating until no calls.
    """
    import google.generativeai as genai

    final_aggregated_text = ""
    next_input = initial_message_parts

    while True:
        # This is one "turn" in streaming mode:
        response_iter = chat_obj.send_message(next_input, stream=True)
        turn_text = ""
        last_response = None

        for partial_resp in response_iter:
            last_response = partial_resp
            chunk_text = _extract_text_from_parts(partial_resp.parts)
            if chunk_text:
                turn_text += chunk_text
                on_text_streaming(chunk_text)

        final_aggregated_text += turn_text

        # Check for function calls from the final partial's parts
        function_calls = []
        if last_response and hasattr(last_response, "parts"):
            for part in last_response.parts:
                # part might have a function_call:
                dict_part = type(part).to_dict(part, use_integers_for_enums=False)
                if "function_call" in dict_part:
                    function_calls.append(dict_part["function_call"])

        if not function_calls:
            break

        # For each function call, run the tool:
        next_input = None
        for fn_call in function_calls:
            fn_name = fn_call["name"]
            fn_args = fn_call.get("args", {})
            python_tool = toolset.get_tool(fn_name)
            if not python_tool:
                tool_result = f"[Tool {fn_name} not found]"
            else:
                # parse the arguments dict
                tool_result = python_tool(**fn_args)

            # Build a functionResponse part
            resp_part = genai.protos.Part(
                function_response=genai.protos.FunctionResponse(
                    name=fn_name, response={"result": tool_result}
                )
            )
            # The next turn is to feed back this functionResponse
            next_input = [resp_part]

    if response_format is not None:
        # Generate the prompt for the user:
        prompt = f"user: Convert the following answer to JSON:\n{final_aggregated_text.strip()}\n to adhere to the following schema:\n{response_format.model_json_schema()}\n"

        # Append to initial message parts the final aggregated text:
        initial_message_parts.append(f"tool: {prompt}")

        # Generate content with the final aggregated text:
        response = chat_obj.send_message(initial_message_parts, stream=False)

        # Extract the text from the parts:
        final_aggregated_text = response.parts[0].text

    return final_aggregated_text


def _stream_chat_no_tools(
    model_obj: "genai.Chat",
    gemini_messages: "List[str]",
    on_text_streaming: Callable,
    **cb_kwargs,
) -> str:
    """
    Single-turn streaming. Gather partial textual content from parts.
    """
    stream_iter = model_obj.generate_content(gemini_messages, stream=True)
    final_text = ""
    for partial_resp in stream_iter:
        chunk_text = _extract_text_from_parts(partial_resp.parts)
        if chunk_text:
            final_text += chunk_text
            on_text_streaming(chunk_text)
    return final_text
