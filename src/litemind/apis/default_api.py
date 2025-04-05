import copy
import json
from typing import List, Optional, Sequence, Union

from arbol import aprint, asection
from pydantic import BaseModel

from litemind.agent.messages.actions.tool_call import ToolCall
from litemind.agent.messages.message import Message
from litemind.agent.messages.message_block import MessageBlock
from litemind.agent.tools.base_tool import BaseTool
from litemind.agent.tools.toolset import ToolSet
from litemind.apis.base_api import BaseApi
from litemind.apis.callbacks.callback_manager import CallbackManager
from litemind.apis.exceptions import FeatureNotAvailableError
from litemind.apis.model_features import ModelFeatures
from litemind.media.types.media_action import Action
from litemind.media.types.media_audio import Audio
from litemind.media.types.media_code import Code
from litemind.media.types.media_document import Document
from litemind.media.types.media_image import Image
from litemind.media.types.media_json import Json
from litemind.media.types.media_object import Object
from litemind.media.types.media_table import Table
from litemind.media.types.media_text import Text
from litemind.media.types.media_video import Video
from litemind.utils.document_processing import is_pymupdf_available
from litemind.utils.fastembed_embeddings import fastembed_text, is_fastembed_available
from litemind.utils.ffmpeg_utils import is_ffmpeg_available
from litemind.utils.random_projector import DeterministicRandomProjector
from litemind.utils.whisper_transcribe_audio import (
    is_local_whisper_available,
    transcribe_audio_with_local_whisper,
)


class DefaultApi(BaseApi):
    """
    This is the default API that provides default local models (whisper-local) and features (Video2Image, DocumentConversion, ...).
    This class is _not_ meant to be used directly, but rather as a base class for other classes that implement BaseAPI.

    """

    # constructor:
    def __init__(self, callback_manager: Optional[CallbackManager] = None):

        super().__init__(callback_manager=callback_manager)

    def check_availability_and_credentials(
        self, api_key: Optional[str] = None
    ) -> Optional[bool]:

        # Call the callback manager if available:
        self.callback_manager.on_availability_check(True)

        # The default API is always available:
        return True

    def list_models(
        self,
        features: Optional[
            Union[str, List[str], ModelFeatures, Sequence[ModelFeatures]]
        ] = None,
        non_features: Optional[
            Union[str, List[str], ModelFeatures, Sequence[ModelFeatures]]
        ] = None,
    ) -> List[str]:

        # Normalise the features:
        features = ModelFeatures.normalise(features)
        non_features = ModelFeatures.normalise(non_features)

        # We start with an empty list:
        model_list = []

        # If whisper-local is available then add to the list:
        if is_local_whisper_available():
            model_list.append("whisper-local")

        # If fastembed is available then add to the list:
        if is_fastembed_available():
            model_list.append("fastembed")

        # If pymupdf is available then add to the list:
        if is_pymupdf_available():
            model_list.append("pymupdf")

        # If ffmpeg is available then add to the list:
        if is_ffmpeg_available():
            model_list.append("ffmpeg")

        # Filter the models based on the features:
        if features:
            model_list = self._filter_models(
                model_list, features=features, non_features=non_features
            )

        # Call _callbacks:
        self.callback_manager.on_model_list(model_list)

        return model_list

    def get_best_model(
        self,
        features: Optional[
            Union[str, List[str], ModelFeatures, Sequence[ModelFeatures]]
        ] = None,
        non_features: Optional[
            Union[List[str], ModelFeatures, Sequence[ModelFeatures]]
        ] = None,
        exclusion_filters: Optional[List[str]] = None,
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
            exclusion_filters=exclusion_filters,
        )

        # If no model is left, return None:
        if len(model_list) == 0:
            return None

        # Best model is first in list:
        best_model = model_list[0]

        # Set kwargs to other parameters:
        kwargs = {"features": features, "exclusion_filters": exclusion_filters}

        # Call _callbacks:
        self.callback_manager.on_best_model_selected(best_model, **kwargs)

        return best_model

    def _filter_models(
        self,
        model_list: List[str],
        features: Union[str, List[str], ModelFeatures, Sequence[ModelFeatures]],
        non_features: Optional[
            Union[str, List[str], ModelFeatures, Sequence[ModelFeatures]]
        ] = None,
        exclusion_filters: Optional[Union[str, List[str]]] = None,
    ) -> List[str]:
        """
        Filter the list of models based on the given features.
        Parameters
        ----------
        model_list: List[str]
            List of models.
        features: Sequence[ModelFeatures], or ModelFeatures, or str, or List[str]
            List of features to filter on.
        non_features: Optional[Union[str, List[str], ModelFeatures, Sequence[ModelFeatures]]]
            List of features to exclude.
        exclusion_filters: Optional[Union[str,List[str]]]
            List of strings that if found in the model name exclude it.

        Returns
        -------
        List[str]
            Filtered list of models.

        """
        filtered_model_list = []

        if not features:
            # Nothing to filter:
            return model_list

        # Normalise the features and non_features:
        features = ModelFeatures.normalise(features)
        non_features = ModelFeatures.normalise(non_features)

        # if exclusion_filters is a string then wrap in singleton list:
        if isinstance(exclusion_filters, str):
            exclusion_filters = [exclusion_filters]

        for model in model_list:

            # Check if the model should be excluded:
            if exclusion_filters and any([f in model for f in exclusion_filters]):
                continue

            # exclude models that have the non features:
            if non_features and any(
                [
                    self.has_model_support_for(model_name=model, features=nf)
                    for nf in non_features
                ]
            ):
                continue

            # Append models that support the given features:
            if self.has_model_support_for(model_name=model, features=features):
                filtered_model_list.append(model)

        return filtered_model_list

    def has_model_support_for(
        self,
        features: Union[str, List[str], ModelFeatures, Sequence[ModelFeatures]],
        model_name: Optional[str] = None,
    ) -> bool:

        # Normalise the features:
        features = ModelFeatures.normalise(features)

        # Get the best model if not provided:
        if model_name is None:
            model_name = self.get_best_model(features=features)

        # If model_name is None then we return False:
        if model_name is None:
            return False

        # We check if the superclass says that the model supports the features:
        if super().has_model_support_for(features=features, model_name=model_name):
            return True

        # Check that the model has all the required features:
        for feature in features:

            # The following features use 'fallback' best-of-kind libraries:

            if feature == ModelFeatures.AudioTranscription:

                # In this class we only support whisper-local:
                if not model_name == "whisper-local":
                    return False

                # We provide a default implementation for audio transcription using local whisper is available:
                if model_name == "whisper-local" and not is_local_whisper_available():
                    aprint(
                        "Audio Transcription feature: whisper is not available! \n Install with: pip install openai-whisper"
                    )
                    return False

            elif feature == ModelFeatures.DocumentConversion:

                # In this class we only support pymupdf:
                if not model_name == "pymupdf":
                    return False

                # Checks if document processing library is installed:
                if model_name == "pymupdf" and not is_pymupdf_available():
                    aprint(
                        "Documents feature: pymupdf is not available! \n Install with: pip install pymupdf pymupdf4llm"
                    )
                    return False

            elif (
                feature == ModelFeatures.TextEmbeddings
                or feature == ModelFeatures.ImageEmbeddings
                or feature == ModelFeatures.AudioEmbeddings
                or feature == ModelFeatures.VideoEmbeddings
                or feature == ModelFeatures.DocumentEmbeddings
            ):

                # In this class we only support fastembed:
                if not model_name == "fastembed":
                    return False

                # Check if the fastembed library is installed:
                if model_name == "fastembed" and not is_fastembed_available():
                    aprint(
                        "Text Embeddings feature: fastembed is not available! \n Install with: pip install fastembed"
                    )
                    return False

                # Check if there is a model that supports the corresponding modality:
                if (
                    feature == ModelFeatures.ImageEmbeddings
                    and not self.has_model_support_for(ModelFeatures.Image)
                ):
                    aprint("Image Embeddings feature: model does not support images!")
                    return False
                elif (
                    feature == ModelFeatures.AudioEmbeddings
                    and not self.has_model_support_for(ModelFeatures.Audio)
                ):
                    aprint("Audio Embeddings feature: model does not support audio!")
                    return False
                elif (
                    feature == ModelFeatures.VideoEmbeddings
                    and not self.has_model_support_for(ModelFeatures.Video)
                ):
                    aprint("Video Embeddings feature: model does not support video!")
                    return False
                elif (
                    feature == ModelFeatures.DocumentEmbeddings
                    and not self.has_model_support_for(ModelFeatures.Image)
                ):
                    aprint("Video Embeddings feature: model does not support video!")
                    return False

            elif feature == ModelFeatures.VideoConversion:

                # In this class we only support ffmpeg:
                if not model_name == "ffmpeg":
                    return False

                # Check if the ffmpeg library is installed:
                if model_name == "ffmpeg" and not is_ffmpeg_available():
                    aprint(
                        "Video Conversion feature: ffmpeg is not available! \n Install with: pip install ffmpeg-python"
                    )
                    return False

            else:
                return False

        return True

    def get_model_features(self, model_name: str) -> List[ModelFeatures]:

        # List to store model features:
        model_features: List[ModelFeatures] = []

        # Get all the features:
        for feature in ModelFeatures:
            if self.has_model_support_for(model_name=model_name, features=feature):
                model_features.append(feature)

        return model_features

    def max_num_input_tokens(self, model_name: Optional[str] = None) -> int:
        # TODO: we need to figure out what to do here:
        return 1000

    def max_num_output_tokens(self, model_name: Optional[str] = None) -> int:
        # TODO: we need to figure out what to do here:
        return 1000

    def count_tokens(self, text: str, model_name: Optional[str] = None) -> int:
        # TODO: use list of messages as input instead so we can consider multimodal tokens in count.

        # Use the following conversion from word to token: 1 word to 1.33 tokens
        # (based on GPT-3.5 model, but should be similar for other models):
        return int(len(text.split()) * 1.33)

        # # use Tiktoken to count tokens by default:
        # import tiktoken
        #
        # # If model is not provided, get the best model:
        # if model_name not in self.list_models(features=[ModelFeatures.TextGeneration]):
        #     # We pick the best model as default, could be very wrong but we need a default!
        #     model_name = self.get_best_model(features=[ModelFeatures.TextGeneration])
        #
        # # Choose the appropriate encoding:
        # try:
        #     encoding = tiktoken.encoding_for_model(model_name)
        # except KeyError as e:
        #     aprint(
        #         f"Model '{model_name}' not found. Error: {e}. Using default encoding."
        #     )
        #     encoding = tiktoken.encoding_for_model("gpt-4o")
        #
        # # Encode:
        # tokens = encoding.encode(text)
        #
        # # Return the token count:
        # return len(tokens)

    def generate_text(
        self,
        messages: List[Message],
        model_name: Optional[str] = None,
        temperature: float = 0.0,
        max_num_output_tokens: Optional[int] = None,
        toolset: Optional[ToolSet] = None,
        use_tools: bool = True,
        response_format: Optional[BaseModel] = None,
        **kwargs,
    ) -> List[Message]:

        raise NotImplementedError("Text generation is not supported by this API.")

    def _preprocess_messages(
        self,
        messages: List[Message],
        convert_videos: bool = True,
        convert_documents: bool = True,
        transcribe_audio: bool = True,
        exclude_extensions: Optional[List[str]] = None,
        deepcopy: bool = True,
    ) -> List[Message]:

        # Make a deep copy of the messages and their blocks to prevent damaging the original messages:
        if deepcopy:
            messages = copy.deepcopy(messages)

        # Convert videos to images and audio:
        if convert_videos:
            # Get the best model for video processing:
            video_conversion_model_name = self.get_best_model(
                features=ModelFeatures.VideoConversion
            )

            # Convert videos to images and audio:
            messages = self.convert_videos_to_images_and_audio_in_messages(
                messages=messages,
                exclude_extensions=exclude_extensions,
                model_name=video_conversion_model_name,
            )

        # Convert audio messages to text:
        if transcribe_audio:
            # Get the best model for audio transcription:
            audio_transcription_model_name = self.get_best_model(
                features=ModelFeatures.AudioTranscription
            )

            # Convert audio message blocks:
            messages = self.convert_audio_in_messages(
                messages=messages,
                exclude_extensions=exclude_extensions,
                model_name=audio_transcription_model_name,
            )

        # Convert documents to markdown:
        if convert_documents:
            # Get the best model for document conversion:
            document_conversion_model = self.get_best_model(
                features=ModelFeatures.DocumentConversion
            )

            # Convert documents to markdown:
            messages = self.convert_documents_to_markdown_in_messages(
                messages,
                exclude_extensions=exclude_extensions,
                model_name=document_conversion_model,
            )

        return messages

    def _process_tool_calls(
        self,
        response: Message,
        messages: List[Message],
        new_messages: List[Message],
        preprocessed_messages: List[Message],
        toolset: ToolSet,
        set_preprocessed: bool = False,
    ):
        """
        This function executes the tools in the toolset on the response and adds the results to the messages.

        Parameters
        ----------
        response: Message

        messages: List[Message]
            These are the messages passed to the text generation method.

        new_messages: List[Message]
            This list accumulates the new messages that are created during the processing.

        preprocessed_messages: List[Message]
            Preprocessed messages to add the tool uses to.

        toolset: ToolSet
            Toolset to use for the tools. Must be the same one that was used to create the response.

        set_preprocessed: bool
            If True, the preprocessed messages are set to teh new tool use message instead of being appended to.

        Returns
        -------

        """

        # Prepare message that will hold the tool uses and adds it to the original, preprocessed, and new messages:
        tool_use_message = Message()
        messages.append(tool_use_message)

        if set_preprocessed:
            # remove all items in the preprocessed_messages list:
            preprocessed_messages.clear()
            preprocessed_messages.append(tool_use_message)
        else:
            preprocessed_messages.append(tool_use_message)
        new_messages.append(tool_use_message)

        # Get the tool calls:
        tool_calls = [b.media.get_content() for b in response if b.has_type(Action)]

        # Iterate through tool calls:
        for tool_call in tool_calls:
            if isinstance(tool_call, ToolCall):

                # Get tool function name:
                tool_name = tool_call.tool_name

                # Get the corresponding tool in toolset:
                tool: Optional[BaseTool] = (
                    toolset.get_tool(tool_name) if toolset else None
                )

                # Get the input arguments:
                tool_arguments = tool_call.arguments

                if tool:
                    try:
                        # Execute the tool
                        result = tool.execute(**tool_arguments)

                        # If not a string, convert from JSON:
                        if not isinstance(result, str):
                            result = json.dumps(result, default=str)

                    except Exception as e:
                        result = f"Function '{tool_name}' error: {e}"

                    # Append the tool call result to the messages:
                    tool_use_message.append_tool_use(
                        tool_name=tool_name,
                        arguments=tool_arguments,
                        result=result,
                        id=tool_call.id,
                    )

                else:
                    # Append the tool call result to the messages:
                    tool_use_error_message = (
                        f"(Tool '{tool_name}' use requested, but tool not found.)"
                    )
                    tool_use_message.append_tool_use(
                        tool_name=tool_name,
                        arguments=tool_arguments,
                        result=tool_use_error_message,
                        id=tool_call.id,
                    )

    def transcribe_audio(
        self, audio_uri: str, model_name: Optional[str] = None, **kwargs
    ) -> str:

        # If model is not provided, get the best model:
        if model_name is None:
            model_name = self.get_best_model(
                features=[ModelFeatures.AudioTranscription]
            )

        # Check if the model is None:
        if model_name is None:
            raise FeatureNotAvailableError(
                "No model available for audio transcription."
            )

        # Check if the model is whisper-local:
        if model_name == "whisper-local":
            # Check if whisper is available:
            if not is_local_whisper_available():
                # This is redundant, but we check again:
                raise FeatureNotAvailableError(
                    "Audio Transcription feature: whisper is not available! \n Install with: pip install openai-whisper"
                )

            transcription = transcribe_audio_with_local_whisper(
                audio_uri=audio_uri, **kwargs
            )

            # Call the callback manager:
            self.callback_manager.on_audio_transcription(
                audio_uri=audio_uri, transcription=transcription, **kwargs
            )

            return transcription

        else:
            raise NotImplementedError("Unknown transcription model")

    def convert_audio_in_messages(
        self,
        messages: Sequence[Message],
        exclude_extensions: Optional[List[str]] = None,
        model_name: Optional[str] = None,
    ) -> Sequence[Message]:

        # If model is not provided, get the best model:
        if model_name is None:
            model_name = self.get_best_model(
                features=[ModelFeatures.AudioTranscription]
            )

        # Check if the model is None:
        if model_name is None:
            raise FeatureNotAvailableError(
                "No model available for audio transcription."
            )

        # Iterate over each message in the list:
        for message in messages:

            # Iterate over each block in the message:
            for block in message.blocks:
                self._convert_audio_block(
                    block, message, exclude_extensions, model_name
                )

        return messages

    def _convert_audio_block(self, block, message, exclude_extensions, model_name):
        # Check if the block is an audio block:
        if block.media and isinstance(block.media, Audio):
            try:
                # Cast to Audio:
                audio: Audio = block.media

                # Extract filename from URI:
                original_filename = audio.get_filename()

                # Extract the file extension from URI:
                file_extension = audio.get_extension()

                # Check if the file extension is in the exclude list:
                if exclude_extensions and file_extension in exclude_extensions:
                    return

                # We use a local instance of whisper instead:
                transcription = self.transcribe_audio(audio.uri, model_name=model_name)

                # Add markdown quotes ''' around the transcribed text, and
                # add prefix: "Transcription: " to the transcribed text:
                transcription = f"\nTranscription of audio file '{original_filename}': \n'''\n{transcription}\n'''\n"

                # Make a new text block:
                transcription_block = MessageBlock(Text(transcription))

                # Add the transcribed text to the message right after the audio block:
                message.insert_block(transcription_block, block_before=block)

                # Remove the audio block:
                message.blocks.remove(block)

            except Exception as e:
                # Print stacktrace:
                import traceback

                traceback.print_exc()
                raise ValueError(
                    f"Could not transcribe audio from: '{block.get_content()}', error: {e}"
                )

    def convert_documents_to_markdown_in_messages(
        self,
        messages: Sequence[Message],
        exclude_extensions: Optional[List[str]] = None,
        model_name: Optional[str] = None,
    ) -> Sequence[Message]:

        # If model is not provided, get the best model:
        if model_name is None:
            model_name = self.get_best_model(features=[ModelFeatures.Document])

        # Check if the model is None:
        if model_name is None:
            raise FeatureNotAvailableError(
                "No model available for document conversion."
            )

        # Check if the model is pymupdf:
        if model_name == "pymupdf":
            # Check if pymupdf is available:
            if not is_pymupdf_available():
                # This is redundant, but we check again:
                raise FeatureNotAvailableError(
                    "Documents feature: pymupdf is not available! \n Install with: pip install pymupdf pymupdf4llm"
                )
        else:
            raise NotImplementedError("Unknown document conversion model")

        # Iterate over each message in the list:
        for message in messages:

            # Iterate over each block in the message:
            for block in message.blocks:

                # Check if the block is a document block:
                if block.media is not None and isinstance(block.media, Document):

                    # get document
                    document: Document = block.media

                    try:
                        # Extract filename from URI:
                        original_filename = document.get_filename()

                        # Extract the file extension from URI:
                        file_extension = document.get_extension()

                        # Check if the file extension is in the exclude list:
                        if exclude_extensions and file_extension in exclude_extensions:
                            continue

                        # Extract text from each page of the document:
                        pages_text = document.extract_text_from_pages()

                        # Extract images of each page of the document:
                        pages_images = document.take_image_of_each_page()

                        # Get the number of pages from both lists:
                        num_pages = min(len(pages_text), len(pages_images))

                        # Last block added:
                        last_block_added = block

                        # Log the text and images added for the callback:
                        text_log: str = ""

                        # Iterate over the pages:
                        for page_index in range(num_pages):
                            # Get the text of the page:
                            page_text = pages_text[page_index]

                            # Get the image of the page:
                            page_image = pages_images[page_index]

                            # Add markdown quotes around the text, and add the extension, for example: ```pdf for extension 'pdf':
                            text: str = (
                                f"\nText extracted from document '{original_filename}' page {page_index + 1}: \n'''{file_extension}\n{page_text}\n'''\nImage of the page:\n"
                            )

                            # Create text block:
                            text_block = MessageBlock(Text(text))

                            # Insert the text block after the document block:
                            message.insert_block(
                                text_block, block_before=last_block_added
                            )
                            last_block_added = text_block

                            # Create an image block:
                            image_block = MessageBlock(Image(page_image))

                            # Insert image block right after the text block:
                            message.insert_block(
                                image_block, block_before=last_block_added
                            )
                            last_block_added = image_block

                            # Add the text to the log:
                            text_log += f"Page {page_index + 1}: {page_text}\n"

                        # Call the callback manager:
                        self.callback_manager.on_document_conversion(
                            document_uri=document.uri, markdown=text_log
                        )

                        # Remove the document block:
                        message.blocks.remove(block)

                    except Exception as e:
                        raise ValueError(
                            f"Could not extract text from document: '{document}', error: {e}"
                        )

                elif block.media is not None and isinstance(block.media, Json):

                    # Cast to JSON media:
                    json_media: Json = block.media

                    try:

                        # Create a text block:
                        text_block = MessageBlock(json_media.to_markdown_text_media())

                        # Insert the text block after the JSON block:
                        message.insert_block(text_block, block_before=block)

                        # Remove the JSON block:
                        message.blocks.remove(block)

                    except Exception as e:
                        raise ValueError(
                            f"Could not convert JSON to markdown: '{json_media}', error: {e}"
                        )
                elif block.media is not None and isinstance(block.media, Object):

                    # Get the object from the block:
                    obj: Object = block.media

                    try:
                        # Convert the object to markdown text media:
                        markdown: Text = obj.to_markdown_text_media()

                        # Create a text block:
                        text_block = MessageBlock(markdown)

                        # Insert the text block after the object block:
                        message.insert_block(text_block, block_before=block)

                        # Remove the object block:
                        message.blocks.remove(block)

                    except Exception as e:
                        raise ValueError(
                            f"Could not convert object to markdown: '{obj}', error: {e}"
                        )
                elif block.media is not None and isinstance(block.media, Code):

                    # Get the code:
                    code: Code = block.media

                    try:
                        # Text block:
                        text_block = MessageBlock(code.to_markdown_text_media())

                        # Insert the text block after the code block:
                        message.insert_block(text_block, block_before=block)

                        # Remove the code block:
                        message.blocks.remove(block)

                    except Exception as e:
                        raise ValueError(
                            f"Could not convert code to markdown: '{code}', error: {e}"
                        )
                elif block.media is not None and isinstance(block.media, Table):

                    # Getting the Table:
                    table: Table = block.media

                    try:
                        # Create a text block:
                        text_block = MessageBlock(Text(table.to_markdown()))

                        # Insert the text block after the table block:
                        message.insert_block(text_block, block_before=block)

                        # Remove the table block:
                        message.blocks.remove(block)

                    except Exception as e:
                        raise ValueError(
                            f"Could not convert table to markdown: '{table}', error: {e}"
                        )
                elif block.media is not None and isinstance(block.media, Text):

                    # Get the text:
                    text: Text = block.media

                    try:
                        # get the string:
                        text_str = text.text

                        # Check if the text ends with at least one '\n' if not, then add it:
                        if not text_str.endswith("\n"):
                            text_str += "\n"

                        # Set the block content to the text:
                        text.text = text_str

                    except Exception as e:
                        raise ValueError(
                            f"Could not convert text: '{block.media}', error: {e}"
                        )

                else:
                    # Other block types go through unchanged...
                    pass

        return messages

    def convert_videos_to_images_and_audio_in_messages(
        self,
        messages: Sequence[Message],
        exclude_extensions: Optional[List[str]] = None,
        model_name: Optional[str] = None,
    ) -> Sequence[Message]:

        # If model is not provided, get the best model:
        if model_name is None:
            model_name = self.get_best_model(features=[ModelFeatures.VideoConversion])

        # Check if the model is None:
        if model_name is None:
            raise FeatureNotAvailableError("No model available for video conversion.")

        # Check if the model is ffmpeg:
        if model_name == "ffmpeg":
            # Check if ffmpeg is available:
            if not is_ffmpeg_available():
                # This is redundant, but we check again:
                raise FeatureNotAvailableError(
                    "Video Conversion feature: ffmpeg is not available! \n Install with: pip install ffmpeg-python"
                )

        else:
            raise NotImplementedError(f"Unknown video conversion model: {model_name}")

        # Iterate over each message in the list:
        for message in messages:

            # Iterate over each block in the message:
            for block in message.blocks:

                # Check if the block is a video block:
                if block.media is not None and isinstance(block.media, Video):

                    video: Video = block.media

                    # extract the file extension from the URI:
                    file_extension = video.get_extension()

                    # Check if the file extension is in the exclude list:
                    if exclude_extensions and file_extension in exclude_extensions:
                        continue

                    # Append video frames and audio to the message:
                    blocks = video.convert_to_frames_and_audio()

                    # Insert this message -- al its blocks -- right after the video block:
                    message.insert_blocks(blocks, block_before=block)

                    # Remove the video block:
                    message.blocks.remove(block)

        return messages

    def generate_audio(
        self,
        text: str,
        voice: Optional[str] = None,
        audio_format: Optional[str] = None,
        model_name: Optional[str] = None,
        **kwargs,
    ) -> str:

        raise NotImplementedError("Audio generation is not supported by this API.")

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
    ) -> Image:

        raise NotImplementedError("Image generation is not supported by this API.")

    def generate_video(
        self, description: str, model_name: Optional[str] = None, **kwargs
    ) -> str:

        raise NotImplementedError("Video generation is not supported by this API.")

    def embed_texts(
        self,
        texts: Sequence[str],
        model_name: Optional[str] = None,
        dimensions: int = 512,
        **kwargs,
    ) -> Sequence[Sequence[float]]:

        # if no model is passed get a default model with text embedding support:
        if model_name is None:
            model_name = self.get_best_model(features=ModelFeatures.TextEmbeddings)

        # Check that the model requested is fastembed:
        if not model_name == "fastembed":
            raise FeatureNotAvailableError(
                f"Model '{model_name}' does not support text embedding."
            )

        # Check if fastembed is available:
        if not is_fastembed_available():
            raise FeatureNotAvailableError(
                "Text Embedding feature: fastembed is not available! \n Install with: pip install fastembed"
            )

        # Ensure texts is a list:
        texts = list(texts)

        # Embed the text using fastembed:
        embeddings = fastembed_text(texts=texts, dimensions=dimensions, **kwargs)

        # Update kwargs with other parameters:
        kwargs.update({"model_name": model_name, "dimensions": dimensions})

        # Call the callback manager:
        self.callback_manager.on_text_embedding(
            texts=texts, embeddings=embeddings, **kwargs
        )

        # Return the embeddings:
        return embeddings

    def _reduce_embeddings_dimension(
        self, embeddings: Sequence[Sequence[float]], reduced_dim: int
    ) -> Sequence[Sequence[float]]:

        # Create a DeterministicRandomProjector object:
        drp = DeterministicRandomProjector(
            original_dim=len(embeddings[0]), reduced_dim=reduced_dim
        )

        # Project the embeddings:
        return drp.transform(embeddings)

    def embed_images(
        self,
        image_uris: List[str],
        model_name: Optional[str] = None,
        dimensions: int = 512,
        **kwargs,
    ) -> Sequence[Sequence[float]]:

        # If no model is passed get a default model with image support:
        if model_name is None:
            model_name = self.get_best_model(features=ModelFeatures.ImageEmbeddings)

        # Get best image model name:
        image_text_model_name = self.get_best_model(
            [ModelFeatures.Image, ModelFeatures.TextGeneration]
        )

        # Check model is not None:
        if not image_text_model_name:
            raise FeatureNotAvailableError(
                "Can't find a text generation model that supports images."
            )

        # List to store image descriptions:
        image_descriptions = []

        # Iterate over the image_uris:
        for image_uri in image_uris:
            # Describe image and then embed the description of the image:
            description = self.describe_image(
                image_uri=image_uri,
                model_name=image_text_model_name,
                query="Please describe the following image.",
                **kwargs,
            )

            # Append the description to the list:
            image_descriptions.append(description)

        # Embed the image descriptions:
        image_embeddings = self.embed_texts(
            texts=image_descriptions, model_name=model_name, dimensions=dimensions
        )

        # Update kwargs with other parameters:
        kwargs.update(
            {
                "model_name": model_name,
                "image_descriptions": image_descriptions,
                "dimensions": dimensions,
            }
        )

        # Call the callback manager:
        self.callback_manager.on_image_embedding(
            image_uris=image_uris, embeddings=image_embeddings, **kwargs
        )

        # Return the image embeddings:
        return image_embeddings

    def embed_audios(
        self,
        audio_uris: List[str],
        model_name: Optional[str] = None,
        dimensions: int = 512,
        **kwargs,
    ) -> Sequence[Sequence[float]]:

        # If no model is passed get a default model with text embedding support:
        if model_name is None:
            model_name = self.get_best_model(features=[ModelFeatures.TextEmbeddings])

        # Get the best audio text gen model:
        audio_text_model_name = self.get_best_model(
            [ModelFeatures.Audio, ModelFeatures.TextGeneration]
        )

        # Check model is not None:
        if not audio_text_model_name:
            # We can't find a text generation model that supports audio.
            # Instead, we use the best model that supports audio transcription:
            audio_transcription_model_name = self.get_best_model(
                [ModelFeatures.AudioTranscription]
            )
        else:
            audio_transcription_model_name = None

        # List to store audio descriptions:
        audio_descriptions = []

        # Iterate over the audio_uris:
        for audio_uri in audio_uris:

            if audio_text_model_name is not None:
                # Describe video and then embed the description of the video:
                description = self.describe_audio(
                    audio_uri=audio_uri,
                    model_name=audio_text_model_name,
                    query="Please describe the following audio.",
                    **kwargs,
                )
            elif audio_transcription_model_name is not None:
                # Transcribe audio and then embed the transcription of the audio:
                description = self.transcribe_audio(
                    audio_uri=audio_uri,
                    model_name=audio_transcription_model_name,
                    **kwargs,
                )
            else:
                raise FeatureNotAvailableError(
                    "Can't find a model that supports audio in text generation or an audio transcription model."
                )

            # Append the description to the list:
            audio_descriptions.append(description)

        # Embed the audio descriptions:
        audio_embeddings = self.embed_texts(
            texts=audio_descriptions,
            model_name=model_name,
            dimensions=dimensions,
        )

        # Update kwargs with other parameters:
        kwargs.update(
            {
                "model_name": model_name,
                "audio_descriptions": audio_descriptions,
                "dimensions": dimensions,
            }
        )

        # Call the callback manager:
        self.callback_manager.on_audio_embedding(
            audio_uris=audio_uris, embeddings=audio_embeddings, **kwargs
        )

        # Return the audio embeddings:
        return audio_embeddings

    def embed_videos(
        self,
        video_uris: List[str],
        model_name: Optional[str] = None,
        dimensions: int = 512,
        **kwargs,
    ) -> Sequence[Sequence[float]]:

        # If no model is passed get a default model with text embedding support:
        if model_name is None:
            model_name = self.get_best_model(features=[ModelFeatures.TextEmbeddings])

        # Get best image model name:
        video_text_model_name = self.get_best_model(
            [ModelFeatures.Video, ModelFeatures.TextGeneration]
        )

        # Check model is not None:
        if not video_text_model_name:
            raise FeatureNotAvailableError(
                "Can't find a text generation model that supports video."
            )

        # List to store video descriptions:
        video_descriptions = []

        # Iterate over the video_uris:
        for video_uri in video_uris:
            # Describe video and then embed the description of the video:
            description = self.describe_video(
                video_uri=video_uri,
                model_name=video_text_model_name,
                query="Please describe the following video.",
                **kwargs,
            )

            # Append the description to the list:
            video_descriptions.append(description)

        # Embed the video descriptions:
        video_embeddings = self.embed_texts(
            texts=video_descriptions, model_name=model_name, dimensions=dimensions
        )

        # Update kwargs with other parameters:
        kwargs.update(
            {
                "model_name": model_name,
                "video_descriptions": video_descriptions,
                "dimensions": dimensions,
            }
        )

        # Call the callback manager:
        self.callback_manager.on_video_embedding(
            video_uris=video_uris, embeddings=video_embeddings, **kwargs
        )

        # Return the video embeddings:
        return video_embeddings

    def embed_documents(
        self,
        document_uris: List[str],
        model_name: Optional[str] = None,
        dimensions: int = 512,
        **kwargs,
    ) -> Sequence[Sequence[float]]:

        # If no model is passed get a default model with text embedding support:
        if model_name is None:
            model_name = self.get_best_model(features=[ModelFeatures.TextEmbeddings])

        # Get best image model name:
        document_text_model_name = self.get_best_model(
            [ModelFeatures.Document, ModelFeatures.TextGeneration]
        )

        # Check model is not None:
        if not document_text_model_name:
            raise FeatureNotAvailableError(
                "Can't find a text generation model that supports documents."
            )

        # List to store document descriptions:
        document_descriptions = []

        # Iterate over the video_uris:
        for document_uri in document_uris:
            # Describe document and then embed the description of the document:
            description = self.describe_document(
                document_uri=document_uri,
                model_name=document_text_model_name,
                query="Please describe the following document.",
                **kwargs,
            )

            # Append the description to the list:
            document_descriptions.append(description)

        # Embed the video descriptions:
        document_embeddings = self.embed_texts(
            texts=document_descriptions, model_name=model_name, dimensions=dimensions
        )

        # Update kwargs with other parameters:
        kwargs.update(
            {
                "model_name": model_name,
                "document_embeddings": document_embeddings,
                "dimensions": dimensions,
            }
        )

        # Call the callback manager:
        self.callback_manager.on_document_embedding(
            document_uris=document_uris, embeddings=document_embeddings, **kwargs
        )

        # Return the video embeddings:
        return document_embeddings

    def describe_image(
        self,
        image_uri: str,
        system: str = "You are a helpful AI assistant that can describe and analyse images.",
        query: str = "Here is an image, please carefully describe it completely and in detail.",
        model_name: Optional[str] = None,
        temperature: float = 0,
        max_output_tokens: Optional[int] = None,
        number_of_tries: int = 4,
    ) -> str:

        with asection(
            f"Asking model {model_name} to describe a given image: '{image_uri}':"
        ):
            aprint(f"Query: '{query}'")
            aprint(f"Model: '{model_name}'")
            aprint(f"Max tokens: '{max_output_tokens}'")

            # If no model is passed get a default model with image support:
            if model_name is None:
                model_name = self.get_best_model(
                    features=[ModelFeatures.TextGeneration, ModelFeatures.Image]
                )

            # If the model does not support vision, return an error:
            if not self.has_model_support_for(
                model_name=model_name,
                features=[ModelFeatures.TextGeneration, ModelFeatures.Image],
            ):
                raise FeatureNotAvailableError(
                    f"Model '{model_name}' does not support images."
                )

            # if no max_output_tokens is passed, get the default value:
            if max_output_tokens is None:
                max_output_tokens = self.max_num_output_tokens(model_name)
            else:
                # Limit the number of tokens to the maximum allowed by the model:
                max_output_tokens = min(
                    max_output_tokens, self.max_num_output_tokens(model_name)
                )

            try:

                # Default response:
                response = None

                # Retry in case of model refusing to answer:
                for tries in range(number_of_tries):

                    messages = []

                    # System message:
                    system_message = Message(role="system")
                    system_message.append_text(system)
                    messages.append(system_message)

                    # User message:
                    user_message = Message(role="user")
                    user_message.append_text(query)
                    user_message.append_image(image_uri)

                    messages.append(user_message)

                    # Run agent:
                    response = self.generate_text(
                        messages=messages,
                        model_name=model_name,
                        temperature=temperature,
                        max_num_output_tokens=max_output_tokens,
                    )

                    # Normalise response:
                    response = str(response)

                    # Check if the response is empty:
                    if not response:
                        aprint(f"Response is empty. Trying again...")
                        continue

                    # response in lower case and trimmed of white spaces
                    response_lc = response.lower().strip()

                    # Check if response is too short:
                    if len(response_lc) < 3:
                        aprint(f"Response is empty. Trying again...")
                        continue

                    # if the model refused to answer, we try again!
                    if (
                        (
                            "sorry" in response_lc
                            and (
                                "i cannot" in response_lc
                                or "i can't" in response_lc
                                or "i am unable" in response_lc
                            )
                        )
                        or "i cannot assist" in response_lc
                        or "i can't assist" in response_lc
                        or "i am unable to assist" in response_lc
                        or "I'm sorry" in response_lc
                    ):
                        aprint(
                            f"Model {model_name} refuses to assist (response: {response}). Trying again..."
                        )
                        continue

                # Update kwargs with other parameters:
                kwargs = {
                    "model_name": model_name,
                    "temperature": temperature,
                    "max_output_tokens": max_output_tokens,
                    "number_of_tries": number_of_tries,
                    "system": system,
                    "query": query,
                }

                # call the callback manager:
                self.callback_manager.on_image_description(
                    image_uri=image_uri, description=response, **kwargs
                )

                # print the description::
                with asection(f"Description:"):
                    aprint(response)

                return response

            except Exception as e:
                # Log the error:
                aprint(f"Error: '{e}'")
                # print stack trace:
                import traceback

                traceback.print_exc()
                return f"Error: '{e}'"

    def describe_audio(
        self,
        audio_uri: str,
        system: str = "You are a helpful AI assistant that can describe and analyse audio.",
        query: str = "Here is an audio file, please carefully describe its contents in detail. If it is speech, please transcribe completely and accurately.",
        model_name: Optional[str] = None,
        temperature: float = 0,
        max_output_tokens: Optional[int] = None,
        number_of_tries: int = 4,
    ) -> Optional[str]:

        with asection(
            f"Asking model {model_name} to describe a given audio: '{audio_uri}':"
        ):
            aprint(f"Query: '{query}'")
            aprint(f"Model: '{model_name}'")
            aprint(f"Max tokens: '{max_output_tokens}'")

            # If no model is passed get a default model with image support:
            if model_name is None:
                model_name = self.get_best_model(
                    features=[ModelFeatures.TextGeneration, ModelFeatures.Audio]
                )

            # If the model does not support audio, return an error:
            if not self.has_model_support_for(
                model_name=model_name,
                features=[ModelFeatures.TextGeneration, ModelFeatures.Audio],
            ):
                raise FeatureNotAvailableError(
                    f"Model '{model_name}' does not support audio."
                )

            # if no max_output_tokens is passed, get the default value:
            if max_output_tokens is None:
                max_output_tokens = self.max_num_output_tokens(model_name)
            else:
                # Limit the number of tokens to the maximum allowed by the model:
                max_output_tokens = min(
                    max_output_tokens, self.max_num_output_tokens(model_name)
                )

            try:

                # Retry in case of model refusing to answer:
                for tries in range(number_of_tries):

                    messages = []

                    # System message:
                    system_message = Message(role="system")
                    system_message.append_text(system)
                    messages.append(system_message)

                    # User message:
                    user_message = Message(role="user")
                    user_message.append_text(query)
                    user_message.append_audio(audio_uri)

                    messages.append(user_message)

                    # Run agent:
                    response = self.generate_text(
                        messages=messages,
                        model_name=model_name,
                        temperature=temperature,
                        max_num_output_tokens=max_output_tokens,
                    )

                    # Normalise response:
                    response = str(response)

                    # Check if the response is empty:
                    if not response:
                        aprint(f"Response is empty. Trying again...")
                        continue

                    # response in lower case and trimmed of white spaces
                    response_lc = response.lower().strip()

                    # Check if response is too short:
                    if len(response_lc) < 3:
                        aprint(f"Response is empty. Trying again...")
                        continue

                    # Update kwargs with other parameters:
                    kwargs = {
                        "model_name": model_name,
                        "temperature": temperature,
                        "max_output_tokens": max_output_tokens,
                        "number_of_tries": number_of_tries,
                        "system": system,
                        "query": query,
                    }

                    # call the callback manager:
                    self.callback_manager.on_audio_description(
                        audio_uri=audio_uri, description=response, **kwargs
                    )

                    # print the description::
                    with asection(f"Description:"):
                        aprint(response)

                    return response

            except Exception as e:
                # Log the error:
                aprint(f"Error: '{e}'")
                # print stack trace:
                import traceback

                traceback.print_exc()
                return f"Error: '{e}'"

    def describe_video(
        self,
        video_uri: str,
        system: str = "You are a helpful AI assistant that can describe and analyse videos.",
        query: str = "Here is a video file, please carefully and completely describe it in detail.",
        model_name: Optional[str] = None,
        temperature: float = 0,
        max_output_tokens: Optional[int] = None,
        number_of_tries: int = 4,
    ) -> Optional[str]:

        with asection(
            f"Asking model {model_name} to describe a given video: '{video_uri}':"
        ):
            aprint(f"Query: '{query}'")
            aprint(f"Model: '{model_name}'")
            aprint(f"Max tokens: '{max_output_tokens}'")

            # If no model is passed get a default model with video support:
            if model_name is None:
                model_name = self.get_best_model(
                    features=[ModelFeatures.TextGeneration, ModelFeatures.Video]
                )

            # If the model does not support video, return an error:
            if not self.has_model_support_for(
                model_name=model_name,
                features=[ModelFeatures.TextGeneration, ModelFeatures.Video],
            ):
                raise FeatureNotAvailableError(
                    f"Model '{model_name}' does not support video."
                )

            # if no max_output_tokens is passed, get the default value:
            if max_output_tokens is None:
                max_output_tokens = self.max_num_output_tokens(model_name)
            else:
                # Limit the number of tokens to the maximum allowed by the model:
                max_output_tokens = min(
                    max_output_tokens, self.max_num_output_tokens(model_name)
                )

            try:

                # Retry in case of model refusing to answer:
                for tries in range(number_of_tries):

                    messages = []

                    # System message:
                    system_message = Message(role="system")
                    system_message.append_text(system)
                    messages.append(system_message)

                    # User message:
                    user_message = Message(role="user")
                    user_message.append_text(query)
                    user_message.append_video(video_uri)

                    messages.append(user_message)

                    # Run agent:
                    response = self.generate_text(
                        messages=messages,
                        model_name=model_name,
                        temperature=temperature,
                        max_num_output_tokens=max_output_tokens,
                    )

                    # Normalize response:
                    response = str(response)

                    # Check if the response is empty:
                    if not response:
                        aprint(f"Response is empty. Trying again...")
                        continue

                    # response in lower case and trimmed of white spaces
                    response_lc = response.lower().strip()

                    # Check if response is too short:
                    if len(response_lc) < 3:
                        aprint(f"Response is empty. Trying again...")
                        continue

                    # Update kwargs with other parameters:
                    kwargs = {
                        "model_name": model_name,
                        "temperature": temperature,
                        "max_output_tokens": max_output_tokens,
                        "number_of_tries": number_of_tries,
                        "system": system,
                        "query": query,
                    }

                    # call the callback manager:
                    self.callback_manager.on_video_description(
                        video_uri=video_uri, description=response, **kwargs
                    )

                    # print the description::
                    with asection(f"Description:"):
                        aprint(response)

                    return response

            except Exception as e:
                # Log the error:
                aprint(f"Error: '{e}'")
                # print stack trace:
                import traceback

                traceback.print_exc()
                return f"Error: '{e}'"

    def describe_document(
        self,
        document_uri: str,
        system: str = "You are a helpful AI assistant that can describe and analyse documents.",
        query: str = "Here is a document, please carefully and completely describe it in detail.",
        model_name: Optional[str] = None,
        temperature: float = 0,
        max_output_tokens: Optional[int] = None,
        number_of_tries: int = 4,
    ) -> Optional[str]:

        with asection(
            f"Asking model {model_name} to describe a given document: '{document_uri}':"
        ):
            aprint(f"Query: '{query}'")
            aprint(f"Model: '{model_name}'")
            aprint(f"Max tokens: '{max_output_tokens}'")

            # If no model is passed get a default model with video support:
            if model_name is None:
                model_name = self.get_best_model(
                    features=[ModelFeatures.TextGeneration, ModelFeatures.Document]
                )

            # If the model does not support documents, return an error:
            if not self.has_model_support_for(
                model_name=model_name,
                features=[ModelFeatures.TextGeneration, ModelFeatures.Document],
            ):
                raise FeatureNotAvailableError(
                    f"Model '{model_name}' does not support documents."
                )

            # if no max_output_tokens is passed, get the default value:
            if max_output_tokens is None:
                max_output_tokens = self.max_num_output_tokens(model_name)
            else:
                # Limit the number of tokens to the maximum allowed by the model:
                max_output_tokens = min(
                    max_output_tokens, self.max_num_output_tokens(model_name)
                )

            try:

                # Retry in case of model refusing to answer:
                for tries in range(number_of_tries):

                    messages = []

                    # System message:
                    system_message = Message(role="system")
                    system_message.append_text(system)
                    messages.append(system_message)

                    # User message:
                    user_message = Message(role="user")
                    user_message.append_text(query)
                    user_message.append_document(document_uri)

                    messages.append(user_message)

                    # Run agent:
                    response = self.generate_text(
                        messages=messages,
                        model_name=model_name,
                        temperature=temperature,
                        max_num_output_tokens=max_output_tokens,
                    )

                    # Normalize response:
                    response = str(response)

                    # Check if the response is empty:
                    if not response:
                        aprint(f"Response is empty. Trying again...")
                        continue

                    # response in lower case and trimmed of white spaces
                    response_lc = response.lower().strip()

                    # Check if response is too short:
                    if len(response_lc) < 3:
                        aprint(f"Response is empty. Trying again...")
                        continue

                    # Update kwargs with other parameters:
                    kwargs = {
                        "model_name": model_name,
                        "temperature": temperature,
                        "max_output_tokens": max_output_tokens,
                        "number_of_tries": number_of_tries,
                        "system": system,
                        "query": query,
                    }

                    # call the callback manager:
                    self.callback_manager.on_document_description(
                        video_uri=document_uri, description=response, **kwargs
                    )

                    # print the description::
                    with asection(f"Description:"):
                        aprint(response)

                    return response

            except Exception as e:
                # Log the error:
                aprint(f"Error: '{e}'")
                # print stack trace:
                import traceback

                traceback.print_exc()
                return f"Error: '{e}'"
