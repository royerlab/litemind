import os
from typing import Optional

from arbol import aprint, asection

from litemind.agent.message import Message
from litemind.agent.tools.toolset import ToolSet
from litemind.apis.base_api import BaseApi
from litemind.apis.google.utils.messages import _convert_messages_for_gemini
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

    # def model_list(self) -> List[str]:
    #     """
    #     Return a list of currently available Gemini model IDs.
    #     In real usage, you might discover them from an API call or keep an updated list.
    #     """
    #
    #     return [
    #         # Gemini 2.0 Flash (Experimental)
    #         "gemini-2.0-flash-exp",
    #
    #         # Gemini 1.5 Flash (stable and versions)
    #         "gemini-1.5-flash-latest",
    #         "gemini-1.5-flash",
    #         "gemini-1.5-flash-001",
    #         "gemini-1.5-flash-002",
    #
    #         # Gemini 1.5 Flash-8B (stable and versions)
    #         "gemini-1.5-flash-8b-latest",
    #         "gemini-1.5-flash-8b",
    #         "gemini-1.5-flash-8b-001",
    #
    #         # Gemini 1.5 Pro (stable and versions)
    #         "gemini-1.5-pro-latest",
    #         "gemini-1.5-pro",
    #         "gemini-1.5-pro-001",
    #         "gemini-1.5-pro-002",
    #
    #         # Gemini 1.0 Pro (Deprecated)
    #         "gemini-1.0-pro-latest",
    #         "gemini-1.0-pro",
    #         "gemini-1.0-pro-001",
    #
    #         # Alias for Gemini 1.0 Pro
    #         "gemini-pro",
    #     ]

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
            if "gemini" in model_obj.name.lower():
                gemini_models.append(model_obj.name)
        return gemini_models

    def default_model(self,
                      require_vision: bool = False,
                      require_tools: bool = False) -> str:
        """
        Return a default model that supports vision or tools if needed.
        Currently, 'gemini-1.5-flash' supports both text and multi-modal.
        """
        return "gemini-1.5-flash"

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
        return "gemini" in model_name.lower()

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

        # ------------------------------
        # Gemini 2.0 Flash (Experimental)
        #   Input token limit: 1,048,576
        if "gemini-2.0-flash" in name:
            return 1_048_576

        # ------------------------------
        # Gemini 1.5 Flash & Flash-8B
        #   Input token limit: 1,048,576
        if "gemini-1.5-flash-8b" in name:
            return 1_048_576
        if "gemini-1.5-flash" in name:
            return 1_048_576

        # ------------------------------
        # Gemini 1.5 Pro
        #   Input token limit: 2,097,152
        if "gemini-1.5-pro" in name:
            return 2_097_152

        # ------------------------------
        # Gemini 1.0 Pro (Deprecated)
        #   Not explicitly stated in the docs.
        #   We'll assume 1,048,576 for consistency
        #   or fall back to a safe default.
        if "gemini-1.0-pro" in name or "gemini-pro" in name:
            return 1_048_576

        # ------------------------------
        # Fallback if unrecognized:
        return 1_048_576

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

        # ------------------------------
        # Gemini 2.0 Flash (Experimental)
        #   Output token limit: 8,192
        if "gemini-2.0-flash" in name:
            return 8192

        # ------------------------------
        # Gemini 1.5 Flash & Flash-8B
        #   Output token limit: 8,192
        if "gemini-1.5-flash-8b" in name:
            return 8192
        if "gemini-1.5-flash" in name:
            return 8192

        # ------------------------------
        # Gemini 1.5 Pro
        #   Output token limit: 8,192
        if "gemini-1.5-pro" in name:
            return 8192

        # ------------------------------
        # Gemini 1.0 Pro (Deprecated)
        #   Not explicitly stated. We'll default to 8,192.
        if "gemini-1.0-pro" in name or "gemini-pro" in name:
            return 8192

        # ------------------------------
        # Fallback if unrecognized:
        return 8192

    # ----------------------------------------------------------------------
    # Single-turn completion (text or function-calling)
    # ----------------------------------------------------------------------
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

    # ----------------------------------------------------------------------
    # Describe an image (single-turn). Uses multi-modal input
    # ----------------------------------------------------------------------
    def describe_image(self,
                       image_path: str,
                       query: str = 'Here is an image, please carefully describe it in detail.',
                       model_name: str = "gemini-1.5-flash",
                       max_tokens: int = 4096,
                       number_of_tries: int = 4,
                       ) -> str:
        """
        Provide an image + text prompt to Gemini for a single-turn description.

        Parameters
        ----------
        image_path : str
            Local path to an image file.
        query : str
            Prompt describing what we want.
        model_name : str
            Gemini model to use.
        max_tokens : int
            Approx. max tokens in the output (passed via generation_config).
        number_of_tries : int
            Retry if the model refuses or returns an empty response.

        Returns
        -------
        str
            The text describing the image.
        """
        if not self.has_vision_support(model_name):
            return "This model does not support vision features."

        with asection(f"Asking Gemini to analyze image at '{image_path}'"):
            aprint(f"Query: {query}")
            aprint(f"Model: {model_name}")
            aprint(f"Max tokens: {max_tokens}")

            # Load the image using PIL
            try:
                from PIL import Image

                with Image.open(image_path) as img:
                    # We can pass it directly to model.generate_content([..., PIL.Image]).
                    # Optionally, you could resize or transform, e.g.:
                    #   img = img.resize((w, h), Image.Resampling.LANCZOS)
                    pass
            except Exception as e:
                raise APIError(f"Could not open image '{image_path}': {e}")

            # Retry loop
            for attempt in range(number_of_tries):
                try:
                    # google.generativeai references
                    import google.generativeai as genai
                    from google.generativeai import types

                    # Create the model
                    model = genai.GenerativeModel(model_name=model_name)
                    # The prompt can be [query, PIL.Image] to do a single-turn multi-modal request
                    generation_cfg = types.GenerationConfig(
                        max_output_tokens=max_tokens,
                        temperature=0.0,
                    )

                    with Image.open(image_path) as pil_img:
                        # We pass the query string and the PIL image in an array:
                        response = model.generate_content(
                            [query, pil_img],
                            generation_config=generation_cfg
                        )

                    text = (response.text or "").strip()
                    if len(text) < 5:
                        # Possibly the model refused or gave a short answer
                        aprint(
                            f"Response too short, attempt {attempt + 1}/{number_of_tries}")
                        continue

                    # Check for refusal phrases
                    lower_text = text.lower()
                    if any(r in lower_text for r in
                           ["i can't", "i cannot", "i am unable", "sorry"]):
                        aprint(
                            f"Model refused. Attempt {attempt + 1}/{number_of_tries}")
                        continue

                    return text

                except Exception as e:
                    aprint(
                        f"Error describing image (attempt {attempt + 1}/{number_of_tries}): {e}")

            # If all attempts fail:
            return "Gemini failed to provide a valid image description after multiple tries."
