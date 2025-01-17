import os
import tempfile
from io import BytesIO
from typing import List, Optional, Sequence, Union

import requests
from PIL import Image
from arbol import aprint

from litemind.agent.message import Message
from litemind.agent.tools.toolset import ToolSet
from litemind.apis.base_api import BaseApi, ModelFeatures
from litemind.apis.exceptions import APIError, APINotAvailableError
from litemind.apis.openai.utils.messages import convert_messages_for_openai
from litemind.apis.openai.utils.model_list import get_openai_model_list
from litemind.apis.openai.utils.process_response import _process_response
from litemind.apis.openai.utils.tools import _format_tools_for_openai
from litemind.apis.utils.document_processing import is_pymupdf_available
from litemind.apis.utils.dowload_audio_to_tempfile import \
    download_audio_to_temp_file
from litemind.apis.utils.write_base64_to_temp_file import \
    write_base64_to_temp_file


class OpenAIApi(BaseApi):

    def __init__(self,
                 api_key: Optional[str] = None,
                 use_local_whisper: bool = True,
                 **kwargs):

        """
        Initialize the OpenAI API client.
        Parameters
        ----------
        api_key: Optional[str]
            The API key for OpenAI. If not provided, we'll read from OPENAI_API_KEY env var.
        use_whisper_for_audio_if_needed: bool
            If True, use whisper to transcribe audio if the model does not support audio.
        use_local_whisper: bool
            If True, use the local whisper instance for audio transcription.
        kwargs: dict
            Additional options (e.g. `timeout=...`, `max_retries=...`) passed to `OpenAI(...)`.
        """

        self.use_local_whisper = use_local_whisper

        # get key from environmental variables:
        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY")
        if api_key is None:
            raise APIError(
                "The api_key client option must be set either by passing api_key to the client or by setting the OPENAI_API_KEY environment variable"
            )

        try:
            # Create an OpenAI client:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key, **kwargs)
        except Exception as e:
            # Print stack trace:
            import traceback
            traceback.print_exc()
            raise APINotAvailableError(f"Error initializing Gemini client: {e}")

    def check_availability_and_credentials(self, api_key: Optional[
        str] = None) -> bool:
        messages = []

        user_message = Message(role='user')
        user_message.append_text('Hello!')
        messages.append(user_message)

        try:
            self.generate_text_completion(messages)
            return True

        except Exception as e:
            # print error message from exception:
            aprint(f"Error while trying to check the OpenAI API key: {e}")
            import traceback
            traceback.print_exc()
            return False

    def model_list(self, features: Optional[Sequence[ModelFeatures]] = None) -> \
            List[str]:
        # Base filtering of models:
        filters = ['dall-e', 'audio', 'gpt', 'text-embedding']

        # Get the list of models:
        model_list = get_openai_model_list(filters=filters)

        # Filter the models based on the features:
        if features:
            model_list = self._filter_models(model_list, features=features)

        return model_list

    def get_best_model(self, features: Optional[Union[
        str, List[str], ModelFeatures, Sequence[ModelFeatures]]] = None) -> \
            Optional[str]:

        # Normalise the features:
        features = ModelFeatures.normalise(features)

        # Get model list:
        model_list = self.model_list()

        # Filter the models based on the requirements:
        model_list = self._filter_models(model_list,
                                         features=features)

        if len(model_list) == 0:
            return None
        elif len(model_list) == 1:
            return model_list[0]
        else:
            # Next we sort models so the best ones are at the beginning of the list:
            def model_key(model):
                # Split the model name into parts
                parts = model.split('-')

                # If a part is '4o' or 'o1', replace it with 'o1.25' or '4.25' respectively:
                parts = [part if part not in ['4o', 'o1'] else part.replace('o',
                                                                            '.25')
                         for part in parts]

                # Remove all the parts that are not numbers (integer or float):
                parts = [part for part in parts if
                         part.replace('.', '', 1).isdigit()]

                # Remove parts that are integers but that lead with a zero:
                if len(parts) > 1:
                    parts = [part for part in parts if not part.startswith('0')]

                # Remove parts that are numbers (float or int) that are too big (>10.0)
                if len(parts) > 1:
                    parts = [part for part in parts if float(part) < 5]

                # Get the main version (e.g., '3.5' or '4' from 'gpt-3.5' or 'gpt-4')
                main_version = float(parts[-1])

                # If we find 'mini' in the model name then substract 0.25 from the main version:
                if 'mini' in model:
                    main_version -= 0.12

                # Use the length of the model name as a secondary sorting criterion
                length = len(model)
                # Sort by main version (descending), then by length (ascending)
                return (-(main_version), length)

            # Actual sorting:
            sorted_model_list = sorted(model_list, key=model_key)

            # return the first in the list which we assume to eb the best:
            return sorted_model_list[0]

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
                if 'text-embedding' in model_name or 'dall-e' in model_name:
                    return False

            elif feature == ModelFeatures.AudioGeneration:
                pass

            elif feature == ModelFeatures.ImageGeneration:
                if 'dall-e' not in model_name:
                    return False

            elif feature == ModelFeatures.TextEmbeddings:
                if 'text-embedding' not in model_name:
                    return False

            elif feature == ModelFeatures.ImageEmbeddings:

                if 'dall-e' not in model_name:
                    return False

                if not self.has_model_support_for(
                        [ModelFeatures.Image, ModelFeatures.TextEmbeddings],
                        model_name):
                    return False

            elif feature == ModelFeatures.AudioEmbeddings:

                if 'dall-e' not in model_name:
                    return False

                if not self.has_model_support_for(
                        [ModelFeatures.Audio, ModelFeatures.TextEmbeddings],
                        model_name):
                    return False

            elif feature == ModelFeatures.VideoEmbeddings:

                if 'dall-e' not in model_name:
                    return False

                if not self.has_model_support_for(
                        [ModelFeatures.Video, ModelFeatures.TextEmbeddings],
                        model_name):
                    return False

            elif feature == ModelFeatures.Image:

                if 'text-embedding' in model_name or 'dall-e' in model_name:
                    return False

                if not self._has_image_support(model_name):
                    return False

            elif feature == ModelFeatures.Audio:

                if 'text-embedding' in model_name or 'dall-e' in model_name:
                    return False

                if not self._has_audio_support(model_name):
                    return False

            elif feature == ModelFeatures.Video:

                if 'text-embedding' in model_name or 'dall-e' in model_name:
                    return False

                if not self.has_model_support_for(ModelFeatures.Image,
                                                  model_name):
                    return False

            elif feature == ModelFeatures.Tools:
                if 'gpt-3.5' in model_name or 'dall-e' in model_name or 'text-embedding' in model_name:
                    return False

            else:
                if not super().has_model_support_for(feature, model_name):
                    return False

        return True

    def _has_image_support(self, model_name: Optional[str] = None) -> bool:

        if model_name is None:
            model_name = self.get_best_model()

        # Then, we check if it is a vision model:
        if 'vision' in model_name or 'gpt-4o' in model_name or 'gpt-o1' in model_name:
            return True

        # then, we check if it is an audio model:
        if 'audio' in model_name:
            return False

        # Any other model is not a vision model:
        return False

    def _has_audio_support(self, model_name: Optional[str] = None) -> bool:

        if model_name is None:
            model_name = self.get_best_model()

        try:
            if self.use_local_whisper:
                return super().has_model_support_for(model_name=model_name,
                                                     features=ModelFeatures.AudioTranscription) or 'audio' in model_name
            else:
                # Check that whisper-1 is in the list:
                openai_models = self.client.models.list()

                return 'audio' in model_name or 'whisper-1' in [model.id for
                                                                model in
                                                                openai_models.data]

        except Exception:
            return False

    def max_num_input_tokens(self, model_name: Optional[str] = None) -> int:

        if model_name is None:
            model_name = self.get_best_model()

        if ('gpt-4-1106-preview' in model_name
                or 'gpt-4-0125-preview' in model_name
                or 'gpt-4-vision-preview' in model_name):
            max_token_limit = 128000
        elif '32k' in model_name:
            max_token_limit = 32000
        elif '16k' in model_name:
            max_token_limit = 16385
        elif 'gpt-4' in model_name:
            max_token_limit = 8192
        elif 'gpt-3.5-turbo-1106' in model_name:
            max_token_limit = 16385
        elif 'gpt-3.5' in model_name:
            max_token_limit = 4096
        else:
            max_token_limit = 4096
        return max_token_limit

    from typing import Optional

    def max_num_input_tokens(self, model_name: Optional[str] = None) -> int:
        """
        Return the maximum context window (in tokens) for a given model,
        based on the current OpenAI model documentation.
        (Despite the function name, this is the total token capacity — input + output.)

        If model_name is None, this uses the default model name as a fallback.
        """
        if model_name is None:
            model_name = self.get_best_model()

        name = model_name.lower()

        # ============================
        # o1 family
        # ============================
        # o1: 200k token context
        # o1-mini: 128k token context
        # o1-preview: 128k token context

        if "o1-mini" in name:
            return 128000

        if "o1-preview" in name:
            return 128000

        if "o1" in name:
            # e.g. "o1-2024-12-17"
            return 200000

        # ============================
        # GPT-4o family (omni)
        # ============================
        # GPT-4o, GPT-4o-mini, GPT-4o-audio, etc.
        # All standard GPT-4o versions: 128k token context

        if "gpt-4o-mini-realtime" in name:
            # Realtime previews do not change the total context limit
            # (still 128k context overall).
            return 128000

        if "gpt-4o-realtime" in name:
            return 128000

        if "gpt-4o-audio" in name:
            return 128000

        if "gpt-4o-mini" in name:
            return 128000

        if "chatgpt-4o-latest" in name:
            return 128000

        if "gpt-4o" in name:
            return 128000

        # ============================
        # GPT-4 Turbo
        # ============================
        # GPT-4 Turbo (and previews) have a 128k token context

        if "gpt-4-turbo" in name:
            return 128000

        # GPT-4 older previews sometimes end with “-preview”:
        # gpt-4-0125-preview, gpt-4-1106-preview => 128k tokens
        if "gpt-4-1106-preview" in name or "gpt-4-0125-preview" in name:
            return 128000

        # ============================
        # GPT-4 (Standard)
        # ============================
        # The standard GPT-4: 8,192 tokens

        if "gpt-4" in name:
            return 8192

        # ============================
        # GPT-3.5 Turbo
        # ============================
        # GPT-3.5 turbo-based models: 16,385 tokens

        if "gpt-3.5-turbo-instruct" in name:
            # Special case: gpt-3.5-turbo-instruct has a 4k context
            return 4096

        if "gpt-3.5-turbo" in name:
            return 16385

        # ============================
        # GPT base families
        # ============================
        # The newer GPT base models have a 16k context
        # e.g. babbage-002, davinci-002
        if "babbage-002" in name or "davinci-002" in name:
            return 16384

        # ============================
        # Default / Fallback
        # ============================
        # For any unknown model, return a safe 4k context
        return 4096

    from typing import Optional

    def max_num_output_tokens(self, model_name: Optional[str] = None) -> int:
        """
        Return the model's 'Max output tokens' as defined in OpenAI's documentation.

        Note: This is distinct from the context window. The 'Max output tokens' refers
        to how many tokens can be generated by the model in its response (beyond inputs
        and any chain-of-thought tokens). The actual limit in practice may vary based
        on your request parameters (e.g., 'max_tokens', 'messages', etc.), but this
        function returns the documented cap for each model.

        If model_name is None, this uses get_default_openai_model_name() as a fallback.
        """

        if model_name is None:
            model_name = self.get_best_model()

        name = model_name.lower()

        # -----------------------------------
        # o1 and o1-mini
        # -----------------------------------
        #  • o1 => 200,000 tokens context, up to 100,000 for output
        #  • o1-mini => 128,000 context, up to 65,536 for output
        #  • o1-preview => 128,000 context, up to 32,768 for output
        if "o1-mini" in name:
            return 65536
        if "o1-preview" in name:
            return 32768
        if "o1" in name:
            return 100000

        # -----------------------------------
        # GPT-4o (omni) family
        # -----------------------------------
        #  • gpt-4o => 128k context, 16,384 max output
        #  • gpt-4o-mini => 128k context, 16,384 max output
        #  • gpt-4o-realtime-preview => 128k context, 4,096 max output
        #  • gpt-4o-mini-realtime-preview => 128k context, 4,096 max output
        #  • gpt-4o-audio-preview => 128k context, 16,384 max output
        #  • chatgpt-4o-latest => 128k context, 16,384 max output
        if "gpt-4o-mini-realtime" in name:
            return 4096
        if "gpt-4o-realtime" in name:
            return 4096
        if "gpt-4o-audio" in name:
            return 16384
        if "gpt-4o-mini" in name:
            return 16384
        if "chatgpt-4o-latest" in name:
            return 16384
        if "gpt-4o" in name:
            return 16384

        # -----------------------------------
        # GPT-4 Turbo and older GPT-4 previews
        # -----------------------------------
        #  • gpt-4-turbo => 128k context, 4,096 max output
        #  • gpt-4-0125-preview, gpt-4-1106-preview => 128k context, 4,096 max output
        if "gpt-4-turbo" in name:
            return 4096
        if "gpt-4-1106-preview" in name or "gpt-4-0125-preview" in name:
            return 4096

        # -----------------------------------
        # GPT-4 (standard)
        # -----------------------------------
        #  • gpt-4 => 8,192 context, 8,192 max output
        if "gpt-4" in name:
            return 8192

        # -----------------------------------
        # GPT-3.5 Turbo family
        # -----------------------------------
        #  • gpt-3.5-turbo => 16,385 context, 4,096 max output
        #  • gpt-3.5-turbo-1106 => same
        #  • gpt-3.5-turbo-instruct => 4,096 context, 4,096 max output
        if "gpt-3.5-turbo-instruct" in name:
            return 4096
        if "gpt-3.5-turbo" in name:
            return 4096

        # -----------------------------------
        # Fallback
        # -----------------------------------
        # If a model isn't recognized, return a safe default (4k).
        return 4096

    def generate_text_completion(self,
                                 messages: List[Message],
                                 model_name: Optional[str] = None,
                                 temperature: Optional[float] = 0.0,
                                 max_output_tokens: Optional[int] = None,
                                 toolset: Optional[ToolSet] = None,
                                 **kwargs) -> Message:

        from openai import NotGiven

        # Set default model if not provided
        if model_name is None:
            model_name = self.get_best_model()

        # o1 models have special needs:
        if "o1" in model_name:
            # o1 models do not support any temperature except the default 1:
            temperature = 1.0

            # o1 models do not support system messages, make a copy of messages and recast system messages as user messages:
            messages = [m.copy() for m in messages]
            for message in messages:
                if message.role == 'system':
                    message.role = 'user'

        # Get max num of output tokens for model if not provided:
        if max_output_tokens is None:
            max_output_tokens = self.max_num_output_tokens(model_name)

        # Convert ToolSet to OpenAI-compatible JSON schema if toolset is provided
        openai_tools = _format_tools_for_openai(
            toolset) if toolset else NotGiven()

        # If model does not support audio but audio transcription is available, then we use it to transcribe audio:
        if self.use_local_whisper and self.has_model_support_for(
                model_name=model_name,
                features=ModelFeatures.AudioTranscription):
            messages = self._transcribe_audio_in_messages(messages)

        # Convert documents to markdown and images:
        if is_pymupdf_available():
            messages = self._convert_documents_to_markdown_in_messages(messages)

        # Format messages for OpenAI:
        openai_formatted_messages = convert_messages_for_openai(messages)

        # Call OpenAI Chat Completions API
        response = self.client.chat.completions.create(
            model=model_name,
            messages=openai_formatted_messages,
            temperature=temperature,
            max_completion_tokens=max_output_tokens,
            tools=openai_tools,
            **kwargs
        )

        # Process API response
        response_message = _process_response(response, toolset)
        messages.append(response_message)
        return response_message

    def transcribe_audio(self, audio_uri: str, model_name: Optional[str] = None,
                         **kwargs) -> str:

        if self.use_local_whisper:
            transcription = super().transcribe_audio(audio_uri, model_name,
                                                     **kwargs)
        else:
            # Save original filename fromm URI:
            original_filename = audio_uri.split("/")[-1]

            # Remove the "file://" prefix if it exists:
            if audio_uri.startswith("file://"):
                audio_uri = audio_uri.replace("file://", "")

            # if the audio_uri is a remote url:
            if audio_uri.startswith("http://") or audio_uri.startswith(
                    "https://"):
                # Download the audio file:
                audio_uri = download_audio_to_temp_file(audio_uri)

            # if the audio_uri is a data uri:
            elif audio_uri.startswith("data:audio/"):
                # Write the audio data to a temp file:
                audio_uri = write_base64_to_temp_file(audio_uri)

            # Open the audio file:
            with open(audio_uri, "rb") as audio_file:

                # Transcribe the audio file:
                transcription = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="text"
                )

        return transcription

    def generate_audio(self,
                       text: str,
                       voice: Optional[str] = None,
                       audio_format: Optional[str] = None,
                       model_name: Optional[str] = None,
                       **kwargs) -> str:

        # If no voice is provided, use the default voice:
        if voice is None:
            voice = 'onyx'

        # If no file format is provided, use mp3:
        if audio_format is None:
            audio_format = 'mp3'

        # Call the OpenAI API to generate audio:
        response = self.client.audio.speech.create(
            model="tts-1-hd",
            voice=voice,
            response_format=audio_format,
            input=text,
            **kwargs
        )

        # Get the path of a temporary file to save the audio using tempfile, file should not be created:
        with tempfile.NamedTemporaryFile(suffix=".mp3",
                                         delete=False) as temp_file:
            # Save the audio to the temporary file:
            response.write_to_file(temp_file.name)

            # Return the URI to the temporary file:
            return 'file://' + temp_file.name

    def generate_image(self,
                       model_name: str,
                       positive_prompt: str,
                       negative_prompt: Optional[str] = None,
                       image_width: int = 512,
                       image_height: int = 512,
                       preserve_aspect_ratio: bool = True,
                       allow_resizing: bool = True,
                       **kwargs
                       ) -> 'Image':

        # Make prompt from positive and negative prompts:
        if negative_prompt:
            prompt = f"Image must consist in: {positive_prompt}\nBut NOT of: {negative_prompt}"
        else:
            prompt = positive_prompt

        allowed_sizes = ["256x256", "512x512", "1024x1024", "1792x1024",
                         "1024x1792"]
        requested_size = f"{image_width}x{image_height}"

        if requested_size not in allowed_sizes:
            if not allow_resizing:
                raise ValueError(
                    f"Requested resolution {requested_size} is not allowed and resizing is not permitted.")

            # Find the closest higher resolution
            allowed_widths = sorted(
                set(int(size.split('x')[0]) for size in allowed_sizes))
            allowed_heights = sorted(
                set(int(size.split('x')[1]) for size in allowed_sizes))

            closest_width = next(
                (w for w in allowed_widths if w >= image_width),
                allowed_widths[-1])
            closest_height = next(
                (h for h in allowed_heights if h >= image_height),
                allowed_heights[-1])

            closest_size = f"{closest_width}x{closest_height}"
        else:
            closest_size = requested_size

        response = self.client.images.generate(
            model=model_name,
            prompt=prompt,
            size=closest_size,
            quality="hd",
            n=1,
            **kwargs
        )

        image_url = response.data[0].url
        image = Image.open(BytesIO(requests.get(image_url).content))

        if closest_size != requested_size:
            if preserve_aspect_ratio:
                image.thumbnail((image_width, image_height),
                                Image.ANTIALIAS)
            else:
                image = image.resize((image_width, image_height),
                                     Image.ANTIALIAS)

        return image

    def embed_texts(self,
                    texts: List[str],
                    model_name: Optional[str] = None,
                    dimensions: int = 512,
                    **kwargs) -> Sequence[Sequence[float]]:

        if model_name is None:
            model_name = self.get_best_model(require_embeddings=True)

        response = self.client.embeddings.create(
            model=model_name,
            dimensions=dimensions,
            input=texts,
            **kwargs
        )

        return list([e.embedding for e in response.data])
