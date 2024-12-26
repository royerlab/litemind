import os
from typing import List, Optional

from arbol import aprint, asection

from litemind.agent.message import Message
from litemind.agent.tools.toolset import ToolSet
from litemind.apis.base_api import BaseApi
from litemind.apis.openai.exceptions import APIError  # or create a Gemini-specific error class
# For demonstration, we’ll reuse the same APIError.
# In production, you might define your own GeminiApiError.

# google-genai packages:
from google import genai
from google.genai import types


def _get_gemini_model_list() -> List[str]:
    """
    In real usage, you'd list available models from Google GenAI
    or maintain your own curated list. For now, we return a fixed list.
    """
    return [
        "gemini-2.0-flash-exp",
        # Add other variants if available
    ]


def _get_default_gemini_model_name(require_vision: bool = False,
                                   require_tools: bool = False) -> str:
    """
    Return a default Gemini 2.0 model name.
    For simplicity, we always return "gemini-2.0-flash-exp"
    if you need either vision or tools.
    """
    return "gemini-2.0-flash-exp"


def _is_vision_model(model_name: str) -> bool:
    """
    Check if the given model name supports vision.
    For now, assume "gemini-2.0" in name => vision support.
    """
    return "gemini-2.0" in model_name.lower()


def _is_tool_model(model_name: str) -> bool:
    """
    Check if the given model name supports function-calling (tools).
    For this example, treat all gemini-2.0 models as tool-capable.
    """
    return "gemini-2.0" in model_name.lower()


class GeminiApi(BaseApi):
    """
    Implementation of BaseApi for Google's Gemini 2.0 Flash models,
    following a style similar to OpenAIApi.
    """

    def __init__(self,
                 api_key: Optional[str] = None,
                 **kwargs):
        # Retrieve the API key from environment if not provided
        if api_key is None:
            api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise APIError(
                "A valid GOOGLE_API_KEY is required for GeminiApi. "
                "Set the GOOGLE_API_KEY environment variable or pass api_key explicitly."
            )

        self.api_key = api_key

        # Create a google-genai client
        self.client = genai.Client(
            api_key=self.api_key,
            http_options={"api_version": "v1alpha"},
            **kwargs
        )

    def check_api_key(self, api_key: Optional[str] = None) -> bool:
        """
        Tries a simple completion call to verify the key.
        If it succeeds, return True; if it fails, return False.
        """
        candidate_key = api_key or self.api_key
        if not candidate_key:
            return False

        try:
            # We'll do a minimal call to see if the key is valid.
            # If no error is raised, assume success.
            messages = [Message(role='user', text="Hello, Gemini!")]
            _ = self.completion(messages=messages, temperature=0)
            return True
        except Exception:
            return False

    def model_list(self) -> List[str]:
        """
        Return a list of available Gemini models.
        """
        return _get_gemini_model_list()

    def default_model(self,
                      require_vision: bool = False,
                      require_tools: bool = False) -> str:
        """
        Return a default Gemini model,
        possibly factoring in whether we need vision or tool support.
        """
        return _get_default_gemini_model_name(require_vision=require_vision,
                                              require_tools=require_tools)

    def has_vision_support(self, model_name: Optional[str] = None) -> bool:
        """
        Return True if the specified Gemini model supports vision.
        """
        if not model_name:
            model_name = self.default_model(require_vision=True)
        return _is_vision_model(model_name)

    def has_tool_support(self, model_name: Optional[str] = None) -> bool:
        """
        Return True if the specified model supports function-calling (tools).
        """
        if not model_name:
            model_name = self.default_model(require_tools=True)
        return _is_tool_model(model_name)

    def max_num_input_token(self, model_name: Optional[str] = None) -> int:
        """
        Return max input tokens for the given model. This is
        a placeholder. Adjust to actual known limits.
        """
        # Hard-code an example limit
        return 8192

    def completion(self,
                   messages: List[Message],
                   model_name: Optional[str] = None,
                   temperature: float = 0.0,
                   toolset: Optional[ToolSet] = None,
                   **kwargs) -> Message:
        """
        Single-turn synchronous call to Gemini 2.0.
        For multi-turn chat or streaming, you'd do something more advanced.
        """
        if model_name is None:
            model_name = self.default_model()

        # Convert our messages into a single text prompt (simple approach).
        # If you want multi-modal conversation or function calls, expand here.
        text_prompt = []
        for msg in messages:
            if msg.text:
                # e.g., "system: Hello" or "user: Hello"
                text_prompt.append(f"{msg.role}: {msg.text}")
            # If there's an image or function data, handle it separately.

        final_prompt = "\n".join(text_prompt)

        # The google-genai synchronous method:
        config = types.GenerateContentConfig(
            temperature=temperature,
            # If using function-calling: config.tools = ...
            # If you want textual response only: response_modalities=["TEXT"]
            # etc.
        )

        try:
            response = self.client.models.generate_content(
                model=model_name,
                contents=[final_prompt],
                config=config,
            )
        except Exception as e:
            raise APIError(f"Gemini completion call failed: {e}")

        # The result object has a `.text` containing the model's response
        text_response = response.text
        return Message(role="assistant", text=text_response)

    def describe_image(self,
                       image_path: str,
                       query: str = 'Here is an image, please carefully describe it in detail.',
                       model_name: str = "gemini-2.0-flash-exp",
                       temperature: float = 0,
                       max_tokens: int = 4096,
                       number_of_tries: int = 4,
                       ) -> str:
        """
        Describe an image using Gemini 2.0.
        For now, we'll do a single-turn synchronous request with PIL image object.
        """
        from PIL import Image

        with asection(f"Asking Gemini to analyze an image at path '{image_path}':"):
            aprint(f"Query: '{query}'")
            aprint(f"Model: '{model_name}'")
            aprint(f"Max tokens: '{max_tokens}'")

            # Load and optionally resize image
            try:
                with Image.open(image_path) as img:
                    # If you prefer to resize or modify:
                    #   resized_img = img.resize((width, height), Image.Resampling.LANCZOS)
                    # For now, use original image
                    pil_image = img
            except Exception as e:
                raise APIError(f"Cannot open image '{image_path}': {e}")

            # We can do a simple retry loop:
            for attempt in range(number_of_tries):
                try:
                    config = types.GenerateContentConfig(
                        temperature=temperature,
                        # Possibly set system instructions to be "vision" centric:
                        system_instruction="You are an image-description assistant.",
                    )
                    response = self.client.models.generate_content(
                        model=model_name,
                        contents=[query, pil_image],
                        config=config,
                    )
                    text_response = response.text.strip()

                    # If it refused or is empty, try again
                    if not text_response or len(text_response) < 5:
                        aprint(f"Response is too short or empty, attempt {attempt+1}/{number_of_tries}")
                        continue

                    # Check for refusal phrases or disclaimers
                    lowered = text_response.lower()
                    if "sorry" in lowered and any(
                            phrase in lowered for phrase in ["i cannot", "i can't", "i am unable"]
                    ):
                        aprint("Gemini refused to describe the image. Retrying...")
                        continue

                    return text_response

                except Exception as e:
                    aprint(f"Error describing image (attempt {attempt+1}/{number_of_tries}): {e}")

            # If all attempts fail, return an error-like message
            return "Gemini could not generate a valid image description after multiple tries."
