import os
from typing import Optional

from litemind.agent.message import Message
from litemind.agent.tools.toolset import ToolSet
from litemind.apis.base_api import BaseApi
from litemind.apis.google.utils.messages import _convert_messages_for_gemini
from litemind.apis.openai.exceptions import APIError
from litemind.apis.openai.utils.vision import has_vision_support


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
            model_list = [model for model in model_list if self.has_vision_support(model)]

        if require_tools:
            # Filter out models that don't support tools
            model_list = [model for model in model_list if self.has_tool_support(model)]

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
            return  False

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

        # Set default model if not provided
        if model_name is None:
            model_name = self.default_model()

        # Get max num of output tokens for model if not provided:
        if max_output_tokens is None:
            max_output_tokens = self.max_num_output_tokens(model_name)

        # 1) Convert user messages into Ollama's format
        gemini_messages = _convert_messages_for_gemini(messages)

        # google.generativeai references
        import google.generativeai as genai
        from google.generativeai import types

        # Use generation_config for temperature, etc.
        generation_cfg = types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens
        )

        # If user has provided tools, set them up
        # This will require a chat-based approach with function-calling if we want real tool usage
        if toolset and self.has_tool_support(model_name):
            python_functions = []
            for t in toolset.list_tools():
                # We expect Python functions that match the tool signature
                python_functions.append(t.func)  # or wrap it

            # TODO: improve tool / function support based on: https://ai.google.dev/gemini-api/docs/function-calling

            # Create a GenerativeModel with function tools
            model = genai.GenerativeModel(
                model_name=model_name,
                tools=python_functions,
                generation_config=generation_cfg
            )
            # Start a chat session that can do function calls automatically
            chat = model.start_chat(enable_automatic_function_calling=True)
            response = chat.send_message(gemini_messages)
            text_output = response.text or ""
        else:
            # No tool usage, or model doesn't support it
            model = genai.GenerativeModel(model_name=model_name)

            response = model.generate_content(gemini_messages,
                                              generation_config=generation_cfg)
            text_output = response.text or ""

        # Return as a single litemind Message
        response_message = Message(role="assistant", text=text_output)
        messages.append(response_message)
        return response_message
