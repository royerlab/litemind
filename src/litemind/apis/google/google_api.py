import os
from typing import Optional, Sequence, Union

from PIL import Image

from litemind.agent.message import Message
from litemind.agent.tools.toolset import ToolSet
from litemind.apis.base_api import BaseApi, ModelFeatures
from litemind.apis.exceptions import APIError, APINotAvailableError
from litemind.apis.google.utils.messages import _convert_messages_for_gemini, \
    _list_and_delete_uploaded_videos
from litemind.apis.google.utils.tools import create_genai_tools_from_toolset
from litemind.apis.utils.document_processing import is_pymupdf_available


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

        try:
            # google.generativeai references
            import google.generativeai as genai

            # Register the API key with google.generativeai
            genai.configure(api_key=self._api_key)
        except Exception as e:
            # Print stack trace:
            import traceback
            traceback.print_exc()
            raise APINotAvailableError(f"Error initializing Gemini client: {e}")

    def check_availability_and_credentials(self, api_key: Optional[
        str] = None) -> bool:

        # Local import to avoid loading the library if not needed:
        import google.generativeai as genai

        # Use the provided key or the default key:
        candidate_key = api_key or self._api_key
        if not candidate_key:
            return False

        # We’ll try a trivial call; if it fails, we assume invalid.
        try:
            # Minimal call: generate a short response
            model = genai.GenerativeModel(self.get_best_model())
            resp = model.generate_content("Hello, Gemini!")
            _ = resp.text  # Access to ensure no error
            return True
        except Exception:
            import traceback
            traceback.print_exc()
            return False

    from typing import List

    def model_list(self, features: Optional[Sequence[ModelFeatures]] = None) -> \
            List[str]:

        model_list = []
        from google.generativeai import \
            list_models  # This is the function from your snippet
        for model_obj in list_models():
            # model_obj is a Model protobuf (or typed dict) with a .name attribute
            # e.g. "models/gemini-1.5-flash", "models/gemini-2.0-flash-exp", etc.
            if ((
                    "gemini" in model_obj.name.lower() or "imagen" in model_obj.name.lower())
                    and not 'will be discontinued' in model_obj.description.lower()
                    and not 'deprecated' in model_obj.description.lower()):
                model_list.append(model_obj.name)

        # Filter the models based on the features:
        if features:
            model_list = self._filter_models(model_list, features=features)

        # Reverse the list so that the best models are at the beginning:
        model_list.reverse()

        return model_list

    def get_best_model(self, features: Optional[Union[
        str, List[str], ModelFeatures, Sequence[ModelFeatures]]] = None) -> \
            Optional[str]:

        # Normalise the features:
        features = ModelFeatures.normalise(features)

        # Get the full list of models:
        model_list = self.model_list()

        # Add 'models/text-embedding-004' to the end of the list:
        model_list.append('models/text-embedding-004')

        # Filter the models based on the requirements:
        model_list = self._filter_models(model_list,
                                         features=features)

        if model_list:
            # If we have any models left, return the first one:
            return model_list[0]
        else:
            return None

    def has_model_support_for(self,
                              features: Union[
                                  str, List[str], ModelFeatures, Sequence[
                                      ModelFeatures]],
                              model_name: Optional[str] = None) -> bool:

        # Get the best model if not provided:
        if model_name is None:
            model_name = self.get_best_model()

        # Normalise the features:
        features = ModelFeatures.normalise(features)

        # Check that the model has all the required features:
        for feature in features:

            if feature == ModelFeatures.TextGeneration:
                pass

            elif feature == ModelFeatures.ImageGeneration:
                return False

            elif feature == ModelFeatures.TextEmbeddings:
                if not 'text-embedding' in model_name.lower():
                    return False

            elif feature == ModelFeatures.ImageEmbeddings:
                if not 'text-embedding' in model_name.lower():
                    return False

            elif feature == ModelFeatures.AudioEmbeddings:
                if not 'text-embedding' in model_name.lower():
                    return False

            elif feature == ModelFeatures.VideoEmbeddings:
                if not 'text-embedding' in model_name.lower():
                    return False

            elif feature == ModelFeatures.Image:
                if not self._has_image_support(model_name):
                    return False

            elif feature == ModelFeatures.Audio:
                if not self._has_audio_support(model_name):
                    return False

            elif feature == ModelFeatures.Video:
                if not self._has_image_support(model_name):
                    return False

            elif feature == ModelFeatures.Tools:
                if not self._has_tool_support(model_name):
                    return False

            else:
                if not super().has_model_support_for(feature, model_name):
                    return False

        return True

    def _has_image_support(self, model_name: Optional[str] = None) -> bool:

        if not model_name:
            model_name = self.get_best_model()
        return "gemini-1.5" in model_name.lower() or "gemini-2.0" in model_name.lower()

    def _has_audio_support(self, model_name: Optional[str] = None) -> bool:

        if model_name is None:
            model_name = self.get_best_model()
        # Curent assumption: Gemini models support audio if they are multimodal:
        return self.has_model_support_for(model_name=model_name,
                                          features=ModelFeatures.Image)

    def _has_video_support(self, model_name: Optional[str] = None) -> bool:

        if model_name is None:
            model_name = self.get_best_model()
        # Curent assumption: Gemini models support video if they are multimodal:
        return self.has_image_support(
            model_name) and not 'thinking' in model_name

    def _has_tool_support(self, model_name: Optional[str] = None) -> bool:

        if not model_name:
            model_name = self.get_best_model()

        model_name = model_name.lower()

        # Experimental models are tricky and typically don't support tools:
        if 'exp' in model_name:
            return False

        # Check for specific models that support tools:
        return ("gemini-1.5-flash" in model_name.lower()
                or "gemini-1.5-pro" in model_name.lower()
                or "gemini-1.0-pro" in model_name.lower()
                or "gemini-2.0" in model_name.lower())

    def max_num_input_tokens(self, model_name: Optional[str] = None) -> int:

        if model_name is None:
            model_name = self.get_best_model()

        name = model_name.lower()

        from google.generativeai import \
            list_models  # This is the function from your snippet
        for model_obj in list_models():
            # model_obj is a Model protobuf (or typed dict) with a .name attribute
            # e.g. "models/gemini-1.5-flash", "models/gemini-2.0-flash-exp", etc.
            if name == model_obj.name.lower():
                return model_obj.input_token_limit

    def max_num_output_tokens(self, model_name: Optional[str] = None) -> int:

        if model_name is None:
            model_name = self.get_best_model()

        name = model_name.lower()

        from google.generativeai import \
            list_models  # This is the function from your snippet
        for model_obj in list_models():
            # model_obj is a Model protobuf (or typed dict) with a .name attribute
            # e.g. "models/gemini-1.5-flash", "models/gemini-2.0-flash-exp", etc.
            if name == model_obj.name.lower():
                return model_obj.output_token_limit

    def generate_text_completion(self,
                                 messages: List[Message],
                                 model_name: Optional[str] = None,
                                 temperature: float = 0.0,
                                 max_output_tokens: Optional[int] = None,
                                 toolset: Optional[ToolSet] = None,
                                 **kwargs) -> Message:

        # Get the best model if not provided:
        if model_name is None:
            model_name = self.get_best_model()

        # Get the maximum number of output tokens if not provided:
        if max_output_tokens is None:
            max_output_tokens = self.max_num_output_tokens(model_name)

        # Convert documents to markdown and images:
        if is_pymupdf_available():
            messages = self._convert_documents_to_markdown_in_messages(messages)

        # We will use _messages to process the messages:
        preprocessed_messages = messages

        # Convert messages to the gemini format:
        gemini_messages = _convert_messages_for_gemini(preprocessed_messages)

        # Build a GenerationConfig
        import google.generativeai as genai
        from google.generativeai import types
        generation_cfg = types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens
        )

        if toolset and self.has_model_support_for(model_name=model_name,
                                                  features=ModelFeatures.Tools):
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

        # Cleanup uploaded video or other files:
        _list_and_delete_uploaded_videos()

        # 4. Return the final text as a single Message
        response_message = Message(role="assistant", text=text_output)
        messages.append(response_message)
        return response_message

    def generate_image(self,
                       model_name: str,
                       positive_prompt: str,
                       negative_prompt: Optional[str] = None,
                       image_width: int = 512,
                       image_height: int = 512,
                       preserve_aspect_ratio: bool = True,
                       allow_resizing: bool = True,
                       **kwargs
                       ) -> Image:

        if model_name is None:
            model_name = self.get_best_model(require_image_generation=True)

        import google.generativeai as genai
        imagen = genai.ImageGenerationModel(model_id=model_name)

        # Computes aspect ratio from resolution:
        aspect_ratio = image_width / image_height

        # Supported values are: "1:1", "3:4", "4:3", "9:16", and "16:9", snap aspect_ratio to closest:
        if aspect_ratio == 1:
            aspect_ratio = "1:1"
        elif aspect_ratio == 0.75:
            aspect_ratio = "3:4"
        elif aspect_ratio == 1.33:
            aspect_ratio = "4:3"
        elif aspect_ratio == 0.56:
            aspect_ratio = "9:16"
        elif aspect_ratio == 1.77:
            aspect_ratio = "16:9"

        result = imagen.generate_images(
            prompt=positive_prompt,
            number_of_images=1,
            safety_filter_level="block_only_high",
            person_generation="allow_adult",
            aspect_ratio=aspect_ratio,
            negative_prompt=negative_prompt,
        )

        return result[0].image

    def embed_texts(self,
                    texts: List[str],
                    model_name: Optional[str] = None,
                    dimensions: int = 512,
                    **kwargs) -> Sequence[Sequence[float]]:

        if model_name is None:
            model_name = self.get_best_model(require_embeddings=True)

        import google.generativeai as genai
        result = genai.embed_content(
            model=model_name,
            content=texts,
            output_dimensionality=dimensions
        )

        return result['embedding']
