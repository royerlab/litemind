import os
from typing import Optional

from litemind.agent.message import Message
from litemind.agent.tools.toolset import ToolSet
from litemind.apis.base_api import BaseApi
from litemind.apis.google.utils.messages import _convert_messages_for_gemini
from litemind.apis.google.utils.tools import create_genai_tools_from_toolset
from litemind.apis.openai.exceptions import APIError


class GeminiApi(BaseApi):
    """
    A Gemini 1.5+ API implementation conforming to the BaseApi interface.
    Uses the google.generativeai library (previously known as Google GenAI).
    """

    def __init__(self,
                 api_key: Optional[str] = None,
                 **kwargs):
        """
        Initialize the Gemini client.

        Parameters
        ----------
        api_key : Optional[str]
            The API key for Google GenAI. If not provided, reads from GOOGLE_API_KEY.
        kwargs : dict
            Additional parameters (unused here, but accepted for consistency).
        """
        if api_key is None:
            api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise APIError(
                "A valid GOOGLE_API_KEY is required for GeminiApi. "
                "Set GOOGLE_API_KEY in the environment or pass api_key explicitly."
            )

        self._api_key = api_key

        # google.generativeai references
        import google.generativeai as genai

        # Register the API key with google.generativeai
        genai.configure(api_key=self._api_key)

    def check_api_key(self, api_key: Optional[str] = None) -> bool:
        """
        Check whether the key is valid.

        Parameters
        ----------
        api_key: Optional[str]
            The API key to check. If not provided, uses the key set in the constructor.

        Returns
        -------
        bool
            True if the key is valid and the API is accessible.
        """

        # Local import to avoid loading the library if not needed:
        import google.generativeai as genai

        # Use the provided key or the default key:
        candidate_key = api_key or self._api_key
        if not candidate_key:
            return False

        # We’ll try a trivial call; if it fails, we assume invalid.
        try:
            # Minimal call: generate a short response
            model = genai.GenerativeModel(self.default_model())
            resp = model.generate_content("Hello, Gemini!")
            _ = resp.text  # Access to ensure no error
            return True
        except Exception:
            import traceback
            traceback.print_exc()
            return False

    from typing import List

    def model_list(self) -> List[str]:
        """
        Use the Generative AI API's list_models() to fetch all available models,
        then return only the ones that match the 'gemini' naming pattern.
        """
        gemini_models = []
        from google.generativeai import \
            list_models  # This is the function from your snippet
        for model_obj in list_models():
            # model_obj is a Model protobuf (or typed dict) with a .name attribute
            # e.g. "models/gemini-1.5-flash", "models/gemini-2.0-flash-exp", etc.
            if ("gemini" in model_obj.name.lower()
                    and not 'will be discontinued' in model_obj.description.lower()
                    and not 'deprecated' in model_obj.description.lower()):
                gemini_models.append(model_obj.name)
        return gemini_models

    def default_model(self,
                      require_vision: bool = False,
                      require_tools: bool = False) -> str:
        """
        Return a default model that supports vision or tools if needed.
        Currently, 'gemini-1.5-flash' supports both text and multi-modal.
        """

        # Get the full list of models:
        model_list = self.model_list()

        if require_vision:
            # Filter out models that don't support vision
            model_list = [model for model in model_list if
                          self.has_vision_support(model)]

        if require_tools:
            # Filter out models that don't support tools
            model_list = [model for model in model_list if
                          self.has_tool_support(model)]

        # last models are teh better more recent ones:
        return model_list[-1] if model_list else None

    def has_vision_support(self, model_name: Optional[str] = None) -> bool:
        """
        Return True if the model supports images or multimodal input.
        'gemini-1.5-flash' is known to support image inputs.
        """
        if not model_name:
            model_name = self.default_model(require_vision=True)
        return "gemini-1.5" in model_name.lower() or "gemini-2.0" in model_name.lower()

    def has_tool_support(self, model_name: Optional[str] = None) -> bool:
        """
        Gemini 1.5+ supports function-calling if you pass `tools=[...]`
        and `enable_automatic_function_calling=True` in a chat scenario.
        """
        if not model_name:
            model_name = self.default_model(require_tools=True)

        model_name = model_name.lower()

        # Experimental models are tricky and typically don't support tools:
        if 'exp' in model_name:
            return False

        # Check for specific models that support tools:
        return ("gemini-1.5-flash" in model_name.lower()
                or "gemini-1.5-pro" in model_name.lower()
                or "gemini-1.0-pro" in model_name.lower()
                or "gemini-2.0" in model_name.lower())

    from typing import Optional

    def max_num_input_tokens(self, model_name: Optional[str] = None) -> int:
        """
        Returns the maximum INPUT token limit for the specified Gemini model,
        based on the documentation you provided.

        If model_name is None, we call get_default_openai_model_name() as a fallback.

        For unrecognized or older models, we return 4096 as a safe default.
        """

        if model_name is None:
            model_name = self.default_model()

        name = model_name.lower()

        from google.generativeai import \
            list_models  # This is the function from your snippet
        for model_obj in list_models():
            # model_obj is a Model protobuf (or typed dict) with a .name attribute
            # e.g. "models/gemini-1.5-flash", "models/gemini-2.0-flash-exp", etc.
            if name == model_obj.name.lower():
                return model_obj.input_token_limit

    def max_num_output_tokens(self, model_name: Optional[str] = None) -> int:
        """
        Returns the maximum OUTPUT token limit for the specified Gemini model,
        based on the documentation you provided.

        If model_name is None, we call get_default_openai_model_name() as a fallback.

        For unrecognized or older models, we return 8192 as a safe default.
        """

        if model_name is None:
            model_name = self.default_model()

        name = model_name.lower()

        from google.generativeai import \
            list_models  # This is the function from your snippet
        for model_obj in list_models():
            # model_obj is a Model protobuf (or typed dict) with a .name attribute
            # e.g. "models/gemini-1.5-flash", "models/gemini-2.0-flash-exp", etc.
            if name == model_obj.name.lower():
                return model_obj.output_token_limit

    def completion(self,
                   model_name: str,
                   messages: List[Message],
                   temperature: float = 0.0,
                   max_output_tokens: Optional[int] = None,
                   toolset: Optional[ToolSet] = None,
                   **kwargs) -> Message:
        """
        Single-turn completion using google.generativeai.
        We unify all messages into a single text prompt or handle function calls if tools exist.
        """

        # 1. Fallback defaults
        if model_name is None:
            model_name = self.default_model()

        if max_output_tokens is None:
            max_output_tokens = self.max_num_output_tokens(model_name)

        # 2. Convert user messages
        gemini_messages = _convert_messages_for_gemini(messages)

        # 3. Build a GenerationConfig
        import google.generativeai as genai
        from google.generativeai import types
        generation_cfg = types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens
        )

        if toolset and self.has_tool_support(model_name):
            # Build fine-grained Tools (protos)
            proto_tools = create_genai_tools_from_toolset(toolset)

            model = genai.GenerativeModel(
                model_name=model_name,
                tools=proto_tools,  # The Protobuf definitions
                generation_config=generation_cfg
            )
            chat = model.start_chat()

            # (A) Send user's query
            response = chat.send_message(gemini_messages)

            # (B) Check if there's a FunctionCall
            # Typically for single-turn usage, you see whether response is suggesting
            # a function call. If so, do it manually.
            function_calls = []
            for part in response.parts:
                if part.function_call:
                    function_calls.append(part.function_call)

            # (C) If there's a function call, manually call the function and
            #     send the result back.
            text_output = ""
            if function_calls:
                # For simplicity, handle the first function call only
                fn_call = function_calls[0]
                fn_name = fn_call.name
                fn_args = fn_call.args

                # Find the corresponding Python tool
                # (We still need a real Python function behind the scenes.)
                python_tool = toolset.get_tool(fn_name)
                if not python_tool:
                    # The model called a function we don't have
                    text_output = f"Function {fn_name} not found."
                else:
                    # Execute the tool
                    result = python_tool(**fn_args)

                    # (D) Send the function result back to the model
                    # Build a function_response to pass back
                    response_parts = [
                        genai.protos.Part(
                            function_response=genai.protos.FunctionResponse(
                                name=fn_name,
                                response={"result": result}
                            )
                        )
                    ]
                    response2 = chat.send_message(response_parts)
                    text_output = response2.text or ""
            else:
                # If no function call, just use the model's text response
                text_output = response.text or ""
        else:
            # No tool usage
            model = genai.GenerativeModel(model_name=model_name)
            response = model.generate_content(
                gemini_messages,
                generation_config=generation_cfg
            )
            text_output = response.text or ""

        # 4. Return the final text as a single Message
        response_message = Message(role="assistant", text=text_output)
        messages.append(response_message)
        return response_message
