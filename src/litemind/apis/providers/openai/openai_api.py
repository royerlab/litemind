import base64
import os
import tempfile
from typing import List, Optional, Sequence, Type, Union

from openai import OpenAI
from pydantic import BaseModel

from litemind.agent.messages.message import Message
from litemind.agent.tools.toolset import ToolSet
from litemind.apis.base_api import ModelFeatures
from litemind.apis.callbacks.callback_manager import CallbackManager
from litemind.apis.default_api import DefaultApi
from litemind.apis.exceptions import APIError, APINotAvailableError
from litemind.apis.feature_scanner import get_default_model_feature_scanner
from litemind.apis.providers.openai.utils.check_availability import (
    check_openai_api_availability,
)
from litemind.apis.providers.openai.utils.convert_messages import (
    convert_messages_for_openai,
)
from litemind.apis.providers.openai.utils.format_tools import format_tools_for_openai
from litemind.apis.providers.openai.utils.list_models import (
    _get_raw_openai_model_list,
    get_openai_model_list,
)
from litemind.apis.providers.openai.utils.num_tokens import estimate_num_input_tokens
from litemind.apis.providers.openai.utils.preprocess_messages import (
    openai_preprocess_messages,
)
from litemind.apis.providers.openai.utils.process_response import (
    process_response_from_openai,
)
from litemind.media.media_base import MediaBase
from litemind.media.types.media_action import Action
from litemind.media.types.media_image import Image
from litemind.utils.normalise_uri_to_local_file_path import uri_to_local_file_path

# ------------------------------------------------------------------ #
#  Constants
# ------------------------------------------------------------------ #
_1M_TOKENS = 1_048_576  # 1 Mi tokens = 1024 × 1024
_O_SERIES_CTX = 200_000
_O_SERIES_OUT = 100_000


class OpenAIApi(DefaultApi):
    """
    An OpenAI API implementation conforming to the BaseApi interface.

    OpenAI models support text generation, some support image and audio inputs and thinking,
    but do not support all features natively. Support for these features
    are provided by the Litemind API via our fallback mechanism.

    Set the OPENAI_API_KEY environment variable to your OpenAI API key, or pass it as an argument to the constructor.
    You can get your API key from https://platform.openai.com/account/api-keys.

    The OpenAIApi class provides a high-level interface for interacting with the OpenAI API.
    It supports various features such as text generation, image generation, audio processing, and more.
    The class also provides methods for checking the availability of the API, listing available models, and selecting the best model based on the required features.
    The OpenAIApi class is designed to be used with the OpenAI API, but it can also be used with other APIs that are compatible with the OpenAI API.
    The OpenAIApi class is a subclass of the DefaultApi class, which provides a default implementation of the BaseApi interface.

    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        allow_media_conversions: bool = True,
        allow_media_conversions_with_models: bool = True,
        callback_manager: Optional[CallbackManager] = None,
        **kwargs,
    ):
        """
        Initialize the OpenAI API client.
        Parameters
        ----------
        api_key: Optional[str]
            The API key for OpenAI. If not provided, we'll read from OPENAI_API_KEY env var.
        base_url: Optional[str]
            The base URL for the OpenAI or compatible API.
        allow_media_conversions: bool
            If True, the API will allow media conversions using the default media converter.
        allow_media_conversions_with_models: bool
            If True, the API will allow media conversions using models that support the required features in addition to the default media converter.
            To use this the allow_media_conversions parameter must be True.
        kwargs: dict
            Additional options (e.g. `timeout=...`, `max_retries=...`) passed to `OpenAI(...)`.
        """

        super().__init__(
            allow_media_conversions=allow_media_conversions,
            allow_media_conversions_with_models=allow_media_conversions_with_models,
            callback_manager=callback_manager,
        )

        # get key from environmental variables:
        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY")

        # raise error if api_key is None:
        if api_key is None:
            raise APIError(
                "The api_key client option must be set either by passing api_key to the client or by setting the OPENAI_API_KEY environment variable"
            )

        # Save the API key:
        self.api_key = api_key

        # Save base URL:
        self.base_url = base_url

        # Save additional kwargs:
        self.kwargs = kwargs

        try:
            # Initialize the feature scanner:
            self.feature_scanner = get_default_model_feature_scanner()

            # Create an OpenAI client:
            from openai import OpenAI

            self.client: OpenAI = OpenAI(
                api_key=self.api_key, base_url=self.base_url, **self.kwargs
            )

            # Get the raw list of models:
            self._raw_model_list = _get_raw_openai_model_list(self.client)

            # Get the list of models from OpenAI:
            self._model_list = get_openai_model_list(self._raw_model_list)

        except Exception as e:
            # Print stack trace:
            import traceback

            traceback.print_exc()
            raise APINotAvailableError(f"Error initializing OpenAI client: {e}")

    def check_availability_and_credentials(self, api_key: Optional[str] = None) -> bool:

        # Check if the API key is provided:
        if api_key is not None:
            client = OpenAI(api_key=api_key, base_url=self.base_url, **self.kwargs)
        else:
            client = self.client

        # Check the OpenAI API key:
        result = check_openai_api_availability(client)

        # Call the callback manager:
        self.callback_manager.on_availability_check(result)

        # Return the result:
        return result

    def list_models(
        self,
        features: Optional[
            Union[str, List[str], ModelFeatures, Sequence[ModelFeatures]]
        ] = None,
        non_features: Optional[
            Union[str, List[str], ModelFeatures, Sequence[ModelFeatures]]
        ] = None,
        media_types: Optional[Sequence[Type[MediaBase]]] = None,
    ) -> List[str]:

        try:
            # Normalise the features:
            features = ModelFeatures.normalise(features)
            non_features = ModelFeatures.normalise(non_features)

            # Get the models from OpenAI:
            model_list = list(self._model_list)

            # Add the models from the super class:
            model_list += super().list_models()

            # Filter the models based on the features:
            if features:
                model_list = self._filter_models(
                    model_list,
                    features=features,
                    non_features=non_features,
                    media_types=media_types,
                )

            # Call _callbacks:
            self.callback_manager.on_model_list(model_list)

            return model_list

        except Exception:
            raise APIError("Error fetching model list from OpenAI.")

    def get_best_model(
        self,
        features: Optional[
            Union[str, List[str], ModelFeatures, Sequence[ModelFeatures]]
        ] = None,
        non_features: Optional[
            Union[str, List[str], ModelFeatures, Sequence[ModelFeatures]]
        ] = None,
        media_types: Optional[Sequence[Type[MediaBase]]] = None,
        exclusion_filters: Optional[Union[str, List[str]]] = None,
    ) -> Optional[str]:

        # Normalise the features:
        features = ModelFeatures.normalise(features)
        non_features = ModelFeatures.normalise(non_features)

        # Get model list:
        model_list = self.list_models()

        # Filter the models based on the requirements:
        model_list = self._filter_models(
            model_list,
            features=features,
            non_features=non_features,
            media_types=media_types,
            exclusion_filters=exclusion_filters,
        )

        # By default, result is None:
        best_model = None

        if len(model_list) == 0:
            pass
        else:
            best_model = model_list[0]

        # Call the callback manager:
        self.callback_manager.on_best_model_selected(best_model)

        return best_model

    def has_model_support_for(
        self,
        features: Union[str, List[str], ModelFeatures, Sequence[ModelFeatures]],
        media_types: Optional[Sequence[Type[MediaBase]]] = None,
        model_name: Optional[str] = None,
    ) -> bool:

        # Normalise the features:
        features = ModelFeatures.normalise(features)

        # Get the best model if not provided:
        if model_name is None:
            model_name = self.get_best_model(features=features, media_types=media_types)

        # If model_name is None then we return False:
        if model_name is None:
            return False

        # We check if the superclass says that the model supports the features:
        if super().has_model_support_for(
            features=features, media_types=media_types, model_name=model_name
        ):
            return True

        for feature in features:
            if not self.feature_scanner.supports_feature(
                self.__class__, model_name, feature
            ):
                # If the model does not support the feature, we return False:
                return False

        return True

    from typing import Optional

    # ------------------------------------------------------------------ #
    def max_num_input_tokens(self, model_name: Optional[str] = None) -> int:
        if model_name is None:
            model_name = self.get_best_model()
        name = model_name.lower()

        # ---------- 4.1 family ----------
        if "gpt-4.1" in name:
            return _1M_TOKENS

        # ---------- o-series ----------
        if any(tag in name for tag in ("o4-mini", "o3", "o1")):
            if "o1-mini" in name or "o1-preview" in name:
                return 128_000
            return _O_SERIES_CTX

        # ---------- GPT-4o ----------
        if "gpt-4o" in name:
            return 128_000

        # ---------- GPT-4 turbo & other 128 K previews ----------
        if "gpt-4-turbo" in name or "-preview" in name:
            return 128_000

        # ---------- GPT-4.5 (check **before** plain gpt-4!) ----------
        if "gpt-4.5" in name:
            return 128_000

        # ---------- GPT-4 standard ----------
        if "gpt-4-32k" in name:
            return 32_768
        if "gpt-4" in name:
            return 8_192

        # ---------- GPT-3.5 ----------
        if "gpt-3.5-turbo-instruct" in name:
            return 4_097
        if "gpt-3.5-turbo" in name:
            return 16_385

        # ---------- Base GPT completions ----------
        if "babbage-002" in name or "davinci-002" in name:
            return 16_384

        return 4_096

    # ------------------------------------------------------------------ #
    def max_num_output_tokens(self, model_name: Optional[str] = None) -> int:
        if model_name is None:
            model_name = self.get_best_model()
        name = model_name.lower()

        # ---------- 4.1 family ----------
        if "gpt-4.1" in name:
            return 32_768

        # ---------- o-series ----------
        if "o4-mini" in name or "o3" in name:
            return _O_SERIES_OUT
        if "o1-mini" in name:
            return 65_536
        if "o1-preview" in name:
            return 32_768
        if "o1" in name:
            return _O_SERIES_OUT

        # ---------- GPT-4o snaps ----------
        if any(tag in name for tag in ("-realtime", "-audio")):
            return 4_096
        if "2024-05-13" in name:
            return 4_096
        if "gpt-4o-mini" in name or "chatgpt-4o-latest" in name:
            return 16_384
        if "gpt-4o" in name:
            return 16_384  # ≥ 2024-08-06 snapshots

        # ---------- GPT-4 turbo & previews ----------
        if "gpt-4-turbo" in name or "-preview" in name:
            return 4_096

        # ---------- GPT-4.5 ----------
        if "gpt-4.5" in name:
            return 16_384

        # ---------- GPT-4 standard ----------
        if "gpt-4-32k" in name:
            return 32_768
        if "gpt-4" in name:
            return 8_192

        # ---------- GPT-3.5 ----------
        if "gpt-3.5-turbo-instruct" in name:
            return 4_097
        if "gpt-3.5-turbo" in name:
            return 4_096

        # ---------- Base completions ----------
        if "babbage-002" in name or "davinci-002" in name:
            return 16_384

        return 4_096

    def generate_text(
        self,
        messages: List[Message],
        model_name: Optional[str] = None,
        temperature: Optional[float] = 0.0,
        max_num_output_tokens: Optional[int] = None,
        toolset: Optional[ToolSet] = None,
        use_tools: bool = True,
        response_format: Optional[BaseModel] = None,
        **kwargs,
    ) -> List[Message]:

        # validate inputs:
        super().generate_text(
            messages=messages,
            model_name=model_name,
            temperature=temperature,
            max_num_output_tokens=max_num_output_tokens,
            toolset=toolset,
            use_tools=use_tools,
            response_format=response_format,
            **kwargs,
        )

        from openai import NotGiven

        # Set default model if not provided
        if model_name is None:
            model_name = self._get_best_model_for_text_generation(
                messages, toolset if use_tools else None, response_format
            )

        if "o1" in model_name or "o3" in model_name:
            # 'ox' models do not support any temperature except the default 1:
            temperature = 1.0

        # OpenAI specific preprocessing:
        preprocessed_messages = openai_preprocess_messages(
            model_name=model_name, messages=messages
        )

        # Preprocess messages:
        preprocessed_messages = self._preprocess_messages(
            messages=preprocessed_messages,
            allowed_media_types=self._get_allowed_media_types_for_text_generation(
                model_name=model_name
            ),
            deepcopy=False,  # We have already made a deepcopy
        )

        # Get max num of output tokens for model if not provided:
        if max_num_output_tokens is None:
            max_num_output_tokens = self.max_num_output_tokens(model_name)

        # Get max num of input tokens for model if not provided:
        max_num_input_tokens = self.max_num_input_tokens(model_name)

        # Estimate the number of tokens given a conversion of approx. 0.75 words per token:
        effective_max_output_tokens = estimate_num_input_tokens(
            max_num_input_tokens=max_num_input_tokens,
            max_num_output_tokens=max_num_output_tokens,
            preprocessed_messages=preprocessed_messages,
        )

        # Convert ToolSet to OpenAI-compatible JSON schema if toolset is provided
        openai_tools = format_tools_for_openai(toolset) if toolset else NotGiven()

        # Reasoning effort:
        # get what is after the last '-' in the name:
        reasoning_effort = model_name.split("-")[-1]

        # Check if it is a valid reasoning effort:
        if reasoning_effort not in ["low", "medium", "high"]:
            # If not, we set it to NotGiven:
            reasoning_effort = NotGiven()
        else:
            # Remove that postfix from the name:
            model_name = model_name.replace(f"-{reasoning_effort}", "")

        # Make sure that reponse_format is NotGiven if not set:
        if response_format is None:
            response_format = NotGiven()

        # List of tool_use blocks to append to the response message:
        new_messages = []

        try:
            # Loop until we get a response that doesn't require tool use:
            while True:

                # Format messages for OpenAI:
                openai_formatted_messages = convert_messages_for_openai(
                    preprocessed_messages
                )

                # Call OpenAI Chat Completions API
                with self.client.beta.chat.completions.stream(
                    model=model_name,
                    messages=openai_formatted_messages,
                    temperature=temperature,
                    max_completion_tokens=effective_max_output_tokens,
                    tools=openai_tools,
                    response_format=response_format,
                    reasoning_effort=reasoning_effort,
                    **kwargs,
                ) as streaming_response:
                    # Process the stream
                    for event in streaming_response:
                        # pprint(event)
                        if event.type == "content.delta":
                            self.callback_manager.on_text_streaming(
                                fragment=event.delta, **kwargs
                            )

                    # Get the final completion
                    openai_response = streaming_response.get_final_completion()

                # Process the response from OpenAI:
                response = process_response_from_openai(
                    openai_response=openai_response, response_format=response_format
                )

                # Append response message to original, preprocessed, and new messages:
                messages.append(response)
                preprocessed_messages.append(response)
                new_messages.append(response)

                # If the model wants to use a tool, parse out the tool calls:
                if not response.has(Action):
                    # Break out of the loop if no tool use is required anymore:
                    break

                if not use_tools:
                    # Break out of the loop if we're not using tools:
                    break

                # Process tool calls:
                self._process_tool_calls(
                    response, messages, new_messages, preprocessed_messages, toolset
                )

            # Add all parameters to kwargs:
            kwargs.update(
                {
                    "model_name": model_name,
                    "temperature": temperature,
                    "max_output_tokens": max_num_output_tokens,
                    "toolset": toolset,
                    "response_format": response_format,
                }
            )

            # Call the callback manager:
            self.callback_manager.on_text_generation(
                messages=messages, response=response, **kwargs
            )

        except Exception as e:
            # print stack trace:
            import traceback

            traceback.print_exc()
            raise APIError(f"OpenAI generate text error: {e}")

        return new_messages

    def transcribe_audio(
        self, audio_uri: str, model_name: Optional[str] = None, **kwargs
    ) -> str:

        # If no model name is provided, use the best model:
        if model_name is None:
            model_name = self.get_best_model(features=ModelFeatures.AudioTranscription)

        # If no model is available then we raise an error:
        if model_name is None:
            raise APIError("No model available for audio transcription.")

        # If the model is provided by the super class then delegate transcription to it:
        if model_name in super().list_models():
            return super().transcribe_audio(
                audio_uri=audio_uri, model_name=model_name, **kwargs
            )

        # We check that the model name contains 'whisper':
        if "whisper" not in model_name:
            raise APIError(f"Model {model_name} does not support audio transcription.")

        # Get the local path to the audio file:
        audio_file_path = uri_to_local_file_path(audio_uri)

        # Open the audio file:
        with open(audio_file_path, "rb") as audio_file:

            # Transcribe the audio file:
            transcription = self.client.audio.transcriptions.create(
                model=model_name, file=audio_file, response_format="text"
            )

        # Update kwargs with other parameters:
        kwargs.update({"model_name": model_name})

        # Call the callback manager:
        self.callback_manager.on_audio_transcription(
            audio_uri=audio_uri, transcription=transcription, **kwargs
        )

        return transcription

    def generate_audio(
        self,
        text: str,
        voice: Optional[str] = None,
        audio_format: Optional[str] = None,
        model_name: Optional[str] = None,
        **kwargs,
    ) -> str:

        # If no model name is provided, use the best model:
        if model_name is None:
            model_name = self.get_best_model(features=ModelFeatures.AudioGeneration)

        # If no model is available then we raise an error:
        if model_name is None:
            raise APIError("No model available for audio generation.")

        # If the model is provided by the super class then delegate transcription to it:
        if model_name in super().list_models():
            return super().generate_audio(
                text=text,
                voice=voice,
                audio_format=audio_format,
                model_name=model_name,
                **kwargs,
            )

        # If no voice is provided, use the default voice:
        if voice is None:
            voice = "onyx"

        # If no file format is provided, use mp3:
        if audio_format is None:
            audio_format = "mp3"

        # Call the OpenAI API to generate audio:
        response = self.client.audio.speech.create(
            model=model_name,
            voice=voice,
            response_format=audio_format,
            input=text,
            **kwargs,
        )

        # Get the path of a temporary file to save the audio using tempfile, file should not be created:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
            # Save the audio to the temporary file:
            response.write_to_file(temp_file.name)

            # Return the URI to the temporary file:
            audio_file_uri = "file://" + temp_file.name

            # Update kwargs with other parameters:
            kwargs.update(
                {"voice": voice, "audio_format": audio_format, "model_name": model_name}
            )

            # Call the callback manager:
            self.callback_manager.on_audio_generation(
                text=text, audio_uri=audio_file_uri, **kwargs
            )

            return audio_file_uri

    def generate_image(
        self,
        positive_prompt: str,
        negative_prompt: Optional[str] = None,
        model_name: str = None,
        image_width: int = 512,
        image_height: int = 512,
        preserve_aspect_ratio: bool = True,
        allow_resizing: bool = True,
        **kwargs,
    ) -> "Image":

        # Make prompt from positive and negative prompts:
        if negative_prompt:
            prompt = f"Image must consist in: {positive_prompt}\nBut NOT of: {negative_prompt}"
        else:
            prompt = positive_prompt

        # Get the best model if not provided:
        if model_name is None:
            model_name = self.get_best_model(features=ModelFeatures.ImageGeneration)

        # Allowed sizes by OpenAI:
        if "dall-e-2" in model_name:
            allowed_sizes = [
                "256x256",
                "512x512",
                "1024x1024",
                "1792x1024",
                "1024x1792",
            ]
        elif "dall-e-3" in model_name:
            allowed_sizes = ["1024x1024", "1024x1792", "1792x1024"]
        elif "gpt-image-1" in model_name:
            allowed_sizes = ["1024x1024", "1024x1536", "1536x1024"]
        else:
            raise ValueError(f"Model {model_name} is not supported.")

        # Requested size:
        requested_size = f"{image_width}x{image_height}"

        # Check if requested size is allowed:
        if requested_size not in allowed_sizes:
            if not allow_resizing:
                raise ValueError(
                    f"Requested resolution {requested_size} is not allowed and resizing is not permitted."
                )

            # Find the closest higher resolution
            allowed_widths = sorted(
                set(int(size.split("x")[0]) for size in allowed_sizes)
            )
            allowed_heights = sorted(
                set(int(size.split("x")[1]) for size in allowed_sizes)
            )

            closest_width = next(
                (w for w in allowed_widths if w >= image_width), allowed_widths[-1]
            )
            closest_height = next(
                (h for h in allowed_heights if h >= image_height), allowed_heights[-1]
            )

            closest_size = f"{closest_width}x{closest_height}"
        else:
            closest_size = requested_size

        # If model is gpt-image-1, add output format to the kwargs:
        if "gpt-image-1" in model_name:
            kwargs["output_format"] = "png"

        # Se quality to max based on model:
        #  - `auto` (default value) will automatically select the best quality for the given model.
        #  - `high`, `medium` and `low` are supported for `gpt-image-1`.
        #  - `hd` and `standard` are supported for `dall-e-3`.
        #  - `standard` is the only option for `dall-e-2`.
        if "gpt-image-1" in model_name:
            kwargs["quality"] = "auto"
        elif "dall-e-3" in model_name:
            kwargs["quality"] = "hd"
        # elif "dall-e-2" in model_name:
        #    kwargs["quality"] = "standard"

        # If Dall-E model set the response format to...
        if "dall-e" in model_name:
            kwargs["response_format"] = "b64_json"

        # Generate image:
        response = self.client.images.generate(
            model=model_name,
            prompt=prompt,
            size=closest_size,
            n=1,
            **kwargs,
        )

        # Get the image URL and download the image:
        image_url = response.data[0].b64_json

        # Decode the base64 image:
        image_data = base64.b64decode(image_url)

        # Create a PIL image from the decoded data:
        from io import BytesIO

        from PIL import Image
        from PIL.Image import Resampling

        image = Image.open(BytesIO(image_data))

        # Resize the image if needed:
        if closest_size != requested_size:
            if preserve_aspect_ratio:
                # Preserve aspect ratio:
                image.thumbnail((image_width, image_height), Resampling.LANCZOS)
            else:
                # Resize without preserving aspect ratio:
                image = image.resize((image_width, image_height), Resampling.LANCZOS)

        # Update kwargs with other parameters:
        kwargs.update(
            {
                "model_name": model_name,
                "negative_prompt": negative_prompt,
                "image_width": image_width,
                "image_height": image_height,
                "preserve_aspect_ratio": preserve_aspect_ratio,
                "allow_resizing": allow_resizing,
            }
        )

        # Call the callback manager:
        self.callback_manager.on_image_generation(image=image, prompt=prompt, **kwargs)

        return image

    def embed_texts(
        self,
        texts: Sequence[str],
        model_name: Optional[str] = None,
        dimensions: int = 512,
        **kwargs,
    ) -> Sequence[Sequence[float]]:

        # Get the best model if not provided:
        if model_name is None:
            model_name = self.get_best_model(features=ModelFeatures.TextEmbeddings)

        # If the model belongs to the superclass, we call the superclass method:
        if model_name in super().list_models():
            return super().embed_texts(
                texts=texts, model_name=model_name, dimensions=dimensions, **kwargs
            )

        # Call OpenAI API to generate embeddings:
        response = self.client.embeddings.create(
            model=model_name, dimensions=dimensions, input=texts, **kwargs
        )

        # Extract embeddings from the response:
        embeddings = list([e.embedding for e in response.data])

        # Update kwargs with other parameters:
        kwargs.update({"model_name": model_name, "dimensions": dimensions})

        # Call the callback manager:
        self.callback_manager.on_text_embedding(
            texts=texts, embeddings=embeddings, **kwargs
        )

        return embeddings
