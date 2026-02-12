"""Standard implementation of BaseApi providing media description, embedding, and conversion.

This module defines ``DefaultApi``, the intermediate base class that all
concrete provider implementations (OpenAI, Anthropic, Gemini, Ollama) inherit
from. It implements the ``describe_*`` methods (image, audio, video, document)
by constructing ``Message`` objects with media content and delegating to
``generate_text``. It also provides local models (whisper-local, fastembed)
and automatic media conversion capabilities via ``MediaConverter``.
"""

import copy
import json
from functools import lru_cache
from typing import List, Optional, Sequence, Set, Type, Union

from arbol import aprint, asection
from pydantic import BaseModel

from litemind.agent.messages.actions.tool_call import ToolCall
from litemind.agent.messages.message import Message
from litemind.agent.tools.base_tool import BaseTool
from litemind.agent.tools.toolset import ToolSet
from litemind.apis.base_api import BaseApi
from litemind.apis.callbacks.api_callback_manager import ApiCallbackManager
from litemind.apis.exceptions import APIError, FeatureNotAvailableError
from litemind.apis.model_features import ModelFeatures
from litemind.media.conversion.converters.media_converter_delegated_callables import (
    MediaConverterApi,
)
from litemind.media.conversion.media_converter import MediaConverter
from litemind.media.media_base import MediaBase
from litemind.media.types.media_action import Action
from litemind.media.types.media_audio import Audio
from litemind.media.types.media_document import Document
from litemind.media.types.media_image import Image
from litemind.media.types.media_text import Text
from litemind.media.types.media_video import Video
from litemind.utils.fastembed_embeddings import fastembed_text, is_fastembed_available
from litemind.utils.random_projector import DeterministicRandomProjector
from litemind.utils.whisper_transcribe_audio import (
    is_local_whisper_available,
    transcribe_audio_with_local_whisper,
)


class DefaultApi(BaseApi):
    """Standard implementation of BaseApi providing media description and embedding.

    DefaultApi extends BaseApi by implementing the ``describe_*`` methods
    (image, audio, video, document) that construct appropriate Message objects
    with media content and delegate to ``generate_text``. It also provides
    local models (whisper-local, fastembed) and media conversion capabilities.

    This class is not meant to be used directly. Concrete provider
    implementations (OpenAI, Anthropic, Gemini, Ollama) inherit from this
    class and override the abstract ``generate_text`` method.
    """

    def __init__(
        self,
        allow_media_conversions: bool = True,
        allow_media_conversions_with_models: bool = True,
        callback_manager: Optional[ApiCallbackManager] = None,
    ):
        """
        Initialize the DefaultApi with the given parameters.

        Parameters
        ----------
        allow_media_conversions : bool
            If True, the API will allow media conversions using the default media converter.
        allow_media_conversions_with_models : bool
            If True, the API will allow media conversions using models that support the required features in addition to the default media converter.
            To use this the allow_media_conversions parameter must be True.
        callback_manager : Optional[ApiCallbackManager]
            The callback manager to use for callbacks. If None, no callbacks will be used.
        """

        super().__init__(callback_manager=callback_manager)

        # Set allow_conversions flag:
        self.allow_media_conversions = allow_media_conversions

        # Instantiate the media converter:
        self.media_converter = MediaConverter()

        # Add the media converter supporting APIs. This converter can use any model from the API that supports the required features.
        if allow_media_conversions:
            self.media_converter.add_default_converters()

            if allow_media_conversions_with_models:
                media_converter_api = MediaConverterApi(api=self)
                self.media_converter.add_media_converter(media_converter_api)

    def check_availability_and_credentials(
        self, api_key: Optional[str] = None
    ) -> Optional[bool]:
        """Check API availability. The default API is always available.

        Parameters
        ----------
        api_key : Optional[str]
            Ignored for the default API.

        Returns
        -------
        bool
            Always returns True.
        """
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
        media_types: Optional[Sequence[Type[MediaBase]]] = None,
    ) -> List[str]:
        """List available local models, optionally filtered by features.

        Returns models such as ``fastembed`` and ``whisper-local`` depending
        on which optional dependencies are installed.

        Parameters
        ----------
        features : Optional[Union[str, List[str], ModelFeatures, Sequence[ModelFeatures]]]
            Required features to filter models by.
        non_features : Optional[Union[str, List[str], ModelFeatures, Sequence[ModelFeatures]]]
            Features to exclude from the results.
        media_types : Optional[Sequence[Type[MediaBase]]]
            Media types that the returned models must support.

        Returns
        -------
        List[str]
            List of available model names.
        """
        # Normalise the features:
        features = ModelFeatures.normalise(features)
        non_features = ModelFeatures.normalise(non_features)

        # We start with an empty list:
        model_list = []

        # Add fastembed model if available:
        if is_fastembed_available():
            # Add fastembed model:
            model_list.append("fastembed")

        if is_local_whisper_available():
            # Add local whisper model:
            model_list.append("whisper-local")

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
        media_types: Optional[Sequence[Type[MediaBase]]] = None,
        exclusion_filters: Optional[List[str]] = None,
    ) -> Optional[str]:
        """Select the best available local model matching the given criteria.

        Parameters
        ----------
        features : Optional[Union[str, List[str], ModelFeatures, Sequence[ModelFeatures]]]
            Required features the model must support.
        non_features : Optional[Union[List[str], ModelFeatures, Sequence[ModelFeatures]]]
            Features the model must not have.
        media_types : Optional[Sequence[Type[MediaBase]]]
            Media types the model must be able to process.
        exclusion_filters : Optional[List[str]]
            Substrings that, if found in a model name, exclude it.

        Returns
        -------
        Optional[str]
            The best matching model name, or None if no model matches.
        """
        # Normalise the features, non-features, and media types:
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
        media_types: Optional[Sequence[Type[MediaBase]]] = None,
        exclusion_filters: Optional[Union[str, List[str]]] = None,
    ) -> List[str]:
        """Filter the list of models based on the given features.

        Parameters
        ----------
        model_list : List[str]
            List of models.
        features : Sequence[ModelFeatures], or ModelFeatures, or str, or List[str]
            List of features to filter on.
        non_features : Optional[Union[str, List[str], ModelFeatures, Sequence[ModelFeatures]]]
            List of features to exclude.
        media_types : Optional[Sequence[Type[MediaBase]]]
            Set of media types that the models must support.
        exclusion_filters : Optional[Union[str, List[str]]]
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

            # exclude models that have the non-features:
            if non_features and any(
                [
                    self.has_model_support_for(model_name=model, features=nf)
                    for nf in non_features
                ]
            ):
                continue

            # Append models that support the given features and media types:
            if self.has_model_support_for(
                model_name=model, features=features, media_types=media_types
            ):
                filtered_model_list.append(model)

        return filtered_model_list

    def has_model_support_for(
        self,
        features: Union[str, List[str], ModelFeatures, Sequence[ModelFeatures]],
        media_types: Optional[Sequence[Type[MediaBase]]] = None,
        model_name: Optional[str] = None,
    ) -> bool:
        """Check whether a model supports the given features and media types.

        Extends the base class check by also considering media conversion
        capabilities (e.g., a model supports images if it supports
        ``Image`` directly or via ``ImageConversion``).

        Parameters
        ----------
        features : Union[str, List[str], ModelFeatures, Sequence[ModelFeatures]]
            Features to check support for.
        media_types : Optional[Sequence[Type[MediaBase]]]
            Media types to check support for.
        model_name : Optional[str]
            Model to check. If None, checks any available model.

        Returns
        -------
        bool
            True if the model supports all requested features and media types.
        """
        # Get the features required for the given media types:
        if media_types is not None:

            # We get the features needed for the media types:
            features_for_media = ModelFeatures.get_features_needed_for_media_types(
                media_types=media_types
            )

            # For each convertible media type we need to check if the model supports the media directly OR via conversion:
            if ModelFeatures.Image in features_for_media:
                # First we remove the Image feature from the list of features:
                features_for_media.remove(ModelFeatures.Image)
                # The we check recursively whether the model supports Images or ImageConversion:
                if not self.has_model_support_for(
                    features=ModelFeatures.Image,
                    model_name=model_name,
                ) and not self.has_model_support_for(
                    features=ModelFeatures.ImageConversion,
                    model_name=model_name,
                ):
                    return False
            if ModelFeatures.Audio in features_for_media:
                # First we remove the Audio feature from the list of features:
                features_for_media.remove(ModelFeatures.Audio)
                # The we check recursively whether the model supports Audio or AudioConversion:
                if not self.has_model_support_for(
                    features=ModelFeatures.Audio,
                    model_name=model_name,
                ) and not self.has_model_support_for(
                    features=ModelFeatures.AudioConversion,
                    model_name=model_name,
                ):
                    return False
            if ModelFeatures.Video in features_for_media:
                # First we remove the Video feature from the list of features:
                features_for_media.remove(ModelFeatures.Video)
                # The we check recursively whether the model supports Video or VideoConversion:
                if not self.has_model_support_for(
                    features=ModelFeatures.Video,
                    model_name=model_name,
                ) and not self.has_model_support_for(
                    features=ModelFeatures.VideoConversion,
                    model_name=model_name,
                ):
                    return False
            if ModelFeatures.Document in features_for_media:
                # First we remove the Document feature from the list of features:
                features_for_media.remove(ModelFeatures.Document)
                # The we check recursively whether the model supports Document or DocumentConversion:
                if not self.has_model_support_for(
                    features=ModelFeatures.Document,
                    model_name=model_name,
                ) and not self.has_model_support_for(
                    features=ModelFeatures.DocumentConversion,
                    model_name=model_name,
                ):
                    return False

        # Normalise the features:
        features = ModelFeatures.normalise(features)

        # We check if the superclass says that the model supports the features:
        if super().has_model_support_for(features=features, model_name=model_name):
            return True

        # Check that the model has all the required features:
        for feature in features:

            if feature == ModelFeatures.AudioTranscription:
                if not is_local_whisper_available() or model_name != "whisper-local":
                    return False

            elif feature == ModelFeatures.ImageConversion:

                if not self.media_converter.can_convert_within(
                    source_media_type=Image, allowed_media_types={Text}
                ):
                    return False

            elif feature == ModelFeatures.AudioConversion:
                if not self.media_converter.can_convert_within(
                    source_media_type=Audio, allowed_media_types={Text}
                ):
                    return False

            elif feature == ModelFeatures.DocumentConversion:
                if not (
                    (
                        self.media_converter.can_convert_within(
                            source_media_type=Document,
                            allowed_media_types={Text, Image},
                        )
                        and self.has_model_support_for(ModelFeatures.Image)
                    )
                    or self.media_converter.can_convert_within(
                        source_media_type=Document, allowed_media_types={Text}
                    )
                ):
                    return False

            elif feature == ModelFeatures.VideoConversion:
                if not (
                    (
                        self.media_converter.can_convert_within(
                            source_media_type=Video,
                            allowed_media_types={Text, Image, Audio},
                        )
                        and self.has_model_support_for(
                            [ModelFeatures.Image, ModelFeatures.Audio]
                        )
                    )
                    or (
                        self.media_converter.can_convert_within(
                            source_media_type=Video, allowed_media_types={Text, Image}
                        )
                        and self.has_model_support_for([ModelFeatures.Image])
                    )
                    or (
                        self.media_converter.can_convert_within(
                            source_media_type=Video, allowed_media_types={Text, Audio}
                        )
                        and self.has_model_support_for([ModelFeatures.Audio])
                    )
                    or (
                        self.media_converter.can_convert_within(
                            source_media_type=Video, allowed_media_types={Text}
                        )
                    )
                ):
                    return False

            elif (
                feature == ModelFeatures.TextEmbeddings
                or feature == ModelFeatures.ImageEmbeddings
                or feature == ModelFeatures.AudioEmbeddings
                or feature == ModelFeatures.VideoEmbeddings
                or feature == ModelFeatures.DocumentEmbeddings
            ):

                # Check if the model is fastembed:
                if model_name is not None and model_name != "fastembed":
                    return False

                # Check if the fastembed library is installed:
                if not is_fastembed_available():
                    aprint(
                        "Text Embeddings feature: fastembed is not available! \n Install with: pip install fastembed"
                    )
                    return False

                # Check if there is a model that supports the corresponding modality:
                if (
                    feature == ModelFeatures.ImageEmbeddings
                    and not self.has_model_support_for(
                        features=ModelFeatures.TextGeneration, media_types=[Image]
                    )
                ):
                    aprint("Image Embeddings feature: model does not support images!")
                    return False
                elif (
                    feature == ModelFeatures.AudioEmbeddings
                    and not self.has_model_support_for(
                        features=ModelFeatures.TextGeneration, media_types=[Audio]
                    )
                ):
                    aprint("Audio Embeddings feature: model does not support audio!")
                    return False
                elif (
                    feature == ModelFeatures.VideoEmbeddings
                    and not self.has_model_support_for(
                        features=ModelFeatures.TextGeneration, media_types=[Video]
                    )
                ):
                    aprint("Video Embeddings feature: model does not support video!")
                    return False
                elif (
                    feature == ModelFeatures.DocumentEmbeddings
                    and not self.has_model_support_for(
                        features=ModelFeatures.TextGeneration, media_types=[Document]
                    )
                ):
                    aprint(
                        "Document Embeddings feature: model does not support documents!"
                    )
                    return False

            else:
                return False

        return True

    def get_model_features(self, model_name: str) -> List[ModelFeatures]:
        """Get all features supported by a given model.

        Parameters
        ----------
        model_name : str
            The model name to query.

        Returns
        -------
        List[ModelFeatures]
            List of all features the model supports.
        """
        # List to store model features:
        model_features: List[ModelFeatures] = []

        # Get all the features:
        for feature in ModelFeatures:
            if self.has_model_support_for(model_name=model_name, features=feature):
                model_features.append(feature)

        return model_features

    @lru_cache
    def _get_allowed_media_types_for_text_generation(
        self, model_name: str
    ) -> Set[Type[MediaBase]]:
        """
        Get the allowed media types for text generation.

        Parameters
        ----------
        model_name : str
            The model name to query for supported media types.

        Returns
        -------
        Set[Type[MediaBase]]
            Set of allowed media types for text generation.
        """

        # Get the model features for the given model name:
        features = self.get_model_features(model_name)

        # Get the supported media types based on the model features:
        allowed_media_types = ModelFeatures.get_supported_media_types(features)

        # Normalise to set:
        allowed_media_types = set(allowed_media_types)

        return allowed_media_types

    def max_num_input_tokens(self, model_name: Optional[str] = None) -> int:
        """Get the maximum number of input tokens for local models.

        Parameters
        ----------
        model_name : Optional[str]
            The model to query (currently ignored).

        Returns
        -------
        int
            Default token limit of 1000 for local models.
        """
        # TODO: we need to figure out what to do here:
        return 1000

    def max_num_output_tokens(self, model_name: Optional[str] = None) -> int:
        """Get the maximum number of output tokens for local models.

        Parameters
        ----------
        model_name : Optional[str]
            The model to query (currently ignored).

        Returns
        -------
        int
            Default token limit of 1000 for local models.
        """
        # TODO: we need to figure out what to do here:
        return 1000

    def count_tokens(self, text: str, model_name: Optional[str] = None) -> int:
        """Estimate token count using a word-to-token heuristic.

        Uses a ratio of 1 word to 1.33 tokens as a rough approximation.

        Parameters
        ----------
        text : str
            The text to count tokens in.
        model_name : Optional[str]
            The model whose tokenizer to use (currently ignored).

        Returns
        -------
        int
            Estimated number of tokens.
        """
        # TODO: use list of messages as input instead so we can consider multimodal tokens in count.

        # Use the following conversion from word to token: 1 word to 1.33 tokens
        # (based on GPT-3.5 model, but should be similar for other models):
        return int(len(text.split()) * 1.33)

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
        """Validate inputs for text generation.

        This base implementation only performs input validation. Concrete
        provider subclasses must call ``super().generate_text(...)`` first
        and then perform the actual model invocation.

        Parameters
        ----------
        messages : List[Message]
            The list of messages to send to the model.
        model_name : Optional[str]
            The model to use. If None, uses the best available model.
        temperature : float
            Sampling temperature (0.0 = deterministic). Must be between 0 and 2.
        max_num_output_tokens : Optional[int]
            Maximum tokens in the response. Must be a positive integer.
        toolset : Optional[ToolSet]
            The toolset available for function calling.
        use_tools : bool
            If True, executes tool calls automatically.
        response_format : Optional[BaseModel]
            Pydantic model for structured output parsing.
        **kwargs
            Additional provider-specific arguments.

        Returns
        -------
        List[Message]
            None; this base implementation returns implicitly after validation.

        Raises
        ------
        ValueError
            If any input parameters are invalid.
        """
        if not isinstance(messages, list):
            raise ValueError("Messages must be a list of Message objects.")
        if not all(isinstance(m, Message) for m in messages):
            raise ValueError("All messages must be Message objects.")
        if not messages:
            raise ValueError("Messages list cannot be empty.")
        if not isinstance(temperature, (int, float)):
            raise ValueError("Temperature must be a number.")
        if temperature < 0 or temperature > 2:
            raise ValueError("Temperature must be between 0 and 2.")
        if max_num_output_tokens is not None and not isinstance(
            max_num_output_tokens, int
        ):
            raise ValueError("max_num_output_tokens must be an integer.")
        if max_num_output_tokens is not None and max_num_output_tokens <= 0:
            raise ValueError("max_num_output_tokens must be greater than 0.")
        if toolset is not None and not isinstance(toolset, ToolSet):
            raise ValueError("toolset must be a ToolSet object.")

    def _preprocess_messages(
        self,
        messages: List[Message],
        allowed_media_types: Optional[Sequence[Type[MediaBase]]] = None,
        exclude_extensions: Optional[Sequence[str]] = None,
        deepcopy: bool = True,
    ) -> List[Message]:
        """Preprocess messages by converting media to types the model supports.

        Parameters
        ----------
        messages : List[Message]
            Messages to preprocess.
        allowed_media_types : Optional[Sequence[Type[MediaBase]]]
            Media types the target model can handle natively.
        exclude_extensions : Optional[Sequence[str]]
            File extensions to exclude from conversion.
        deepcopy : bool
            If True, deep-copies messages before modifying them.

        Returns
        -------
        List[Message]
            Preprocessed messages with converted media.
        """
        # Make a deep copy of the messages and their blocks to prevent damaging the original messages:
        if deepcopy:
            messages = copy.deepcopy(messages)

        messages = self.media_converter.convert(
            messages=messages,
            allowed_media_types=allowed_media_types,
            exclude_extensions=exclude_extensions,
        )

        return messages

    def _get_best_model_for_text_generation(self, messages, toolset, response_format):
        """Select the best model for a text generation request.

        Determines the required features based on the presence of tools,
        structured response format, and media types in the messages, then
        calls ``get_best_model`` with those requirements.

        Parameters
        ----------
        messages : List[Message]
            The messages to be sent, used to detect required media types.
        toolset : Optional[ToolSet]
            If provided, the ``Tools`` feature is required.
        response_format : Optional[BaseModel]
            If provided, the ``StructuredTextGeneration`` feature is required.

        Returns
        -------
        str
            The name of the best available model.

        Raises
        ------
        FeatureNotAvailableError
            If no model satisfies the required features.
        """
        # We Require the minimum features for text generation:
        features = [ModelFeatures.TextGeneration]
        # If tools are provided, we also require tools:
        if toolset:
            features.append(ModelFeatures.Tools)

        # if a response format is provided, we also require the response format:
        if response_format:
            features.append(ModelFeatures.StructuredTextGeneration)

        # If the messages contain media, we require the appropriate features:

        # get the list of media types in the messages:
        media_types = Message.list_media_types_in(messages)

        # Get the best model with the required features:
        model_name = self.get_best_model(features=features, media_types=media_types)

        # If no model is found, raise an error:
        if model_name is None:
            raise FeatureNotAvailableError(
                f"No suitable model with features: {features}"
            )
        return model_name

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
        response : Message
            The assistant response message containing tool call blocks.
        messages : List[Message]
            These are the messages passed to the text generation method.
        new_messages : List[Message]
            This list accumulates the new messages that are created during the processing.
        preprocessed_messages : List[Message]
            Preprocessed messages to add the tool uses to.
        toolset : ToolSet
            Toolset to use for the tools. Must be the same one that was used to create the response.
        set_preprocessed : bool
            If True, the preprocessed messages are set to the new tool use message instead of being appended to.
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
                        result = tool(**tool_arguments)

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

    def generate_audio(
        self,
        text: str,
        voice: Optional[str] = None,
        audio_format: Optional[str] = None,
        model_name: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Generate audio. Not supported by the default API.

        Raises
        ------
        NotImplementedError
            Always, since DefaultApi does not support audio generation.
        """
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
        """Generate an image. Not supported by the default API.

        Raises
        ------
        NotImplementedError
            Always, since DefaultApi does not support image generation.
        """
        raise NotImplementedError("Image generation is not supported by this API.")

    def generate_video(
        self, description: str, model_name: Optional[str] = None, **kwargs
    ) -> str:
        """Generate a video. Not supported by the default API.

        Raises
        ------
        NotImplementedError
            Always, since DefaultApi does not support video generation.
        """
        raise NotImplementedError("Video generation is not supported by this API.")

    def embed_texts(
        self,
        texts: Sequence[str],
        model_name: Optional[str] = None,
        dimensions: int = 512,
        **kwargs,
    ) -> Sequence[Sequence[float]]:
        """Embed text using the fastembed library.

        Parameters
        ----------
        texts : Sequence[str]
            The text snippets to embed.
        model_name : Optional[str]
            The model to use. Must be "fastembed" or None.
        dimensions : int
            Target dimensionality for the embedding vectors.
        **kwargs
            Additional arguments forwarded to fastembed.

        Returns
        -------
        Sequence[Sequence[float]]
            Embedding vectors, one per input text.

        Raises
        ------
        FeatureNotAvailableError
            If fastembed is not installed or the model does not support
            text embedding.
        """
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
        """Reduce embedding dimensionality using a deterministic random projection.

        Parameters
        ----------
        embeddings : Sequence[Sequence[float]]
            The embeddings to reduce.
        reduced_dim : int
            The target dimensionality.

        Returns
        -------
        Sequence[Sequence[float]]
            The reduced-dimensionality embeddings.
        """
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
        """Embed images by first describing them, then embedding the descriptions.

        Parameters
        ----------
        image_uris : List[str]
            List of image URIs or file paths.
        model_name : Optional[str]
            Embedding model to use. If None, selects the best available.
        dimensions : int
            Target dimensionality for the embedding vectors.
        **kwargs
            Additional arguments forwarded to ``describe_image``.

        Returns
        -------
        Sequence[Sequence[float]]
            Embedding vectors, one per input image.

        Raises
        ------
        FeatureNotAvailableError
            If no model supporting images or image conversion is available.
        """
        # If no model is passed get a default model with image support:
        if model_name is None:
            model_name = self.get_best_model(features=ModelFeatures.ImageEmbeddings)

        # Get best image model name:
        image_text_model_name = self.get_best_model(
            features=[ModelFeatures.TextGeneration], media_types=[Image]
        )

        # Check model is not None:
        if not image_text_model_name:
            raise FeatureNotAvailableError(
                "Can't find a text generation model that supports images or image conversion."
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
        """Embed audio files by first describing or transcribing them, then embedding.

        Parameters
        ----------
        audio_uris : List[str]
            List of audio URIs or file paths.
        model_name : Optional[str]
            Embedding model to use. If None, selects the best available.
        dimensions : int
            Target dimensionality for the embedding vectors.
        **kwargs
            Additional arguments forwarded to ``describe_audio`` or
            ``transcribe_audio``.

        Returns
        -------
        Sequence[Sequence[float]]
            Embedding vectors, one per input audio.

        Raises
        ------
        FeatureNotAvailableError
            If no model supporting audio description or transcription is
            available.
        """
        # If no model is passed get a default model with text embedding support:
        if model_name is None:
            model_name = self.get_best_model(features=[ModelFeatures.TextEmbeddings])

        # Get the best audio text gen model:
        audio_text_model_name = self.get_best_model(
            features=[ModelFeatures.TextGeneration], media_types=[Audio]
        )

        # Check model is not None:
        if not audio_text_model_name:
            # We can't find a text generation model that supports audio.
            # Instead, we use the best model that supports audio transcription:
            audio_transcription_model_name = self.get_best_model(
                [ModelFeatures.AudioConversion]
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
        """Embed videos by first describing them, then embedding the descriptions.

        Parameters
        ----------
        video_uris : List[str]
            List of video URIs or file paths.
        model_name : Optional[str]
            Embedding model to use. If None, selects the best available.
        dimensions : int
            Target dimensionality for the embedding vectors.
        **kwargs
            Additional arguments forwarded to ``describe_video``.

        Returns
        -------
        Sequence[Sequence[float]]
            Embedding vectors, one per input video.

        Raises
        ------
        FeatureNotAvailableError
            If no model supporting video description is available.
        """
        # If no model is passed get a default model with text embedding support:
        if model_name is None:
            model_name = self.get_best_model(features=[ModelFeatures.TextEmbeddings])

        # Get best image model name:
        video_text_model_name = self.get_best_model(
            features=[ModelFeatures.TextGeneration], media_types=[Video]
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
        """Embed documents by first describing them, then embedding the descriptions.

        Parameters
        ----------
        document_uris : List[str]
            List of document URIs or file paths.
        model_name : Optional[str]
            Embedding model to use. If None, selects the best available.
        dimensions : int
            Target dimensionality for the embedding vectors.
        **kwargs
            Additional arguments forwarded to ``describe_document``.

        Returns
        -------
        Sequence[Sequence[float]]
            Embedding vectors, one per input document.

        Raises
        ------
        FeatureNotAvailableError
            If no model supporting document description is available.
        """
        # If no model is passed get a default model with text embedding support:
        if model_name is None:
            model_name = self.get_best_model(features=[ModelFeatures.TextEmbeddings])

        # Get best image model name:
        document_text_model_name = self.get_best_model(
            features=[ModelFeatures.TextGeneration], media_types=[Document]
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

    def transcribe_audio(
        self, audio_uri: str, model_name: Optional[str] = None, **model_kwargs
    ) -> str:
        """Transcribe audio using the local whisper model.

        Parameters
        ----------
        audio_uri : str
            Path, URL, or base64-encoded audio to transcribe.
        model_name : Optional[str]
            The model to use. If None, selects the best available
            audio transcription model.
        **model_kwargs
            Additional arguments forwarded to the whisper transcription.

        Returns
        -------
        str
            The transcription text.

        Raises
        ------
        FeatureNotAvailableError
            If no audio transcription model is available or whisper is not
            installed.
        NotImplementedError
            If the model name is not "whisper-local".
        """
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
                audio_uri=audio_uri, **model_kwargs
            )

            # Set kwargs with other parameters:
            kwargs = {"model_name": model_name, "model_kwargs": model_kwargs}

            # Call the callback manager:
            self.callback_manager.on_audio_transcription(
                audio_uri=audio_uri, transcription=transcription, **kwargs
            )

            return transcription

        else:
            raise NotImplementedError(f"Unknown transcription model: '{model_name}'")

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
        """Describe an image by sending it to a vision-capable model.

        Constructs a message with the image and query, sends it to the
        model, and retries on refusal or empty responses.

        Parameters
        ----------
        image_uri : str
            Path, URL, or base64-encoded image to describe.
        system : str
            System message to guide the description.
        query : str
            Query prompt to use with the image.
        model_name : Optional[str]
            Model to use. If None, selects the best image-capable model.
        temperature : float
            Sampling temperature for the response.
        max_output_tokens : Optional[int]
            Maximum tokens in the response.
        number_of_tries : int
            Number of retry attempts on failure or refusal.

        Returns
        -------
        str
            Text description of the image, or None if all attempts fail.

        Raises
        ------
        FeatureNotAvailableError
            If no model supporting images is available.
        APIError
            If an unrecoverable error occurs during description.
        """
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
            if model_name is None:
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
                    response = response[-1].to_plain_text()

                    # Check if the response is empty:
                    if not response:
                        aprint("Response is empty. Trying again...")
                        continue

                    # response in lower case and trimmed of white spaces
                    response_lc = response.lower().strip()

                    # Check if response is too short:
                    if len(response_lc) < 3:
                        aprint("Response is empty. Trying again...")
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
                    with asection("Description:"):
                        aprint(response)

                    return response

                # Return None if no response was generated:
                return None

            except Exception as e:
                # Log the error:
                aprint(f"Error: '{e}'")
                # print stack trace:
                import traceback

                traceback.print_exc()
                raise APIError(f"Error describing image: {e}") from e

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
        """Describe an audio file by sending it to an audio-capable model.

        Constructs a message with the audio and query, sends it to the
        model, and retries on empty responses.

        Parameters
        ----------
        audio_uri : str
            Path, URL, or base64-encoded audio to describe.
        system : str
            System message to guide the description.
        query : str
            Query prompt to use with the audio.
        model_name : Optional[str]
            Model to use. If None, selects the best audio-capable model.
        temperature : float
            Sampling temperature for the response.
        max_output_tokens : Optional[int]
            Maximum tokens in the response.
        number_of_tries : int
            Number of retry attempts on failure.

        Returns
        -------
        Optional[str]
            Text description of the audio, or None if all attempts fail.

        Raises
        ------
        FeatureNotAvailableError
            If no model supporting audio is available.
        APIError
            If an unrecoverable error occurs during description.
        """
        with asection(
            f"Asking model {model_name} to describe a given audio: '{audio_uri}':"
        ):
            aprint(f"Query: '{query}'")
            aprint(f"Model: '{model_name}'")
            aprint(f"Max tokens: '{max_output_tokens}'")

            # If no model is passed get a default model with image support:
            if model_name is None:
                model_name = self.get_best_model(
                    features=[ModelFeatures.TextGeneration],
                    media_types=[Audio],
                )

            # If the model does not support audio, return an error:
            if model_name is None:
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
                    response = response[-1].to_plain_text()

                    # Check if the response is empty:
                    if not response:
                        aprint("Response is empty. Trying again...")
                        continue

                    # response in lower case and trimmed of white spaces
                    response_lc = response.lower().strip()

                    # Check if response is too short:
                    if len(response_lc) < 3:
                        aprint("Response is empty. Trying again...")
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
                    with asection("Description:"):
                        aprint(response)

                    return response

                # Return None if no response was generated:
                return None

            except Exception as e:
                # Log the error:
                aprint(f"Error: '{e}'")
                # print stack trace:
                import traceback

                traceback.print_exc()
                raise APIError(f"Error describing audio: {e}") from e

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
        """Describe a video by sending it to a video-capable model.

        Constructs a message with the video and query, sends it to the
        model, and retries on empty responses.

        Parameters
        ----------
        video_uri : str
            Path, URL, or base64-encoded video to describe.
        system : str
            System message to guide the description.
        query : str
            Query prompt to use with the video.
        model_name : Optional[str]
            Model to use. If None, selects the best video-capable model.
        temperature : float
            Sampling temperature for the response.
        max_output_tokens : Optional[int]
            Maximum tokens in the response.
        number_of_tries : int
            Number of retry attempts on failure.

        Returns
        -------
        Optional[str]
            Text description of the video, or None if all attempts fail.

        Raises
        ------
        FeatureNotAvailableError
            If no model supporting video is available.
        APIError
            If an unrecoverable error occurs during description.
        """
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
            if model_name is None:
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
                    response = response[-1].to_plain_text()

                    # Check if the response is empty:
                    if not response:
                        aprint("Response is empty. Trying again...")
                        continue

                    # response in lower case and trimmed of white spaces
                    response_lc = response.lower().strip()

                    # Check if response is too short:
                    if len(response_lc) < 3:
                        aprint("Response is empty. Trying again...")
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
                    with asection("Description:"):
                        aprint(response)

                    return response
                return None

            except Exception as e:
                # Log the error:
                aprint(f"Error: '{e}'")
                # print stack trace:
                import traceback

                traceback.print_exc()
                raise APIError(f"Error describing video: {e}") from e

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
        """Describe a document by sending it to a document-capable model.

        Constructs a message with the document and query, sends it to the
        model, and retries on empty responses.

        Parameters
        ----------
        document_uri : str
            Path, URL, or base64-encoded document to describe.
        system : str
            System message to guide the description.
        query : str
            Query prompt to use with the document.
        model_name : Optional[str]
            Model to use. If None, selects the best document-capable model.
        temperature : float
            Sampling temperature for the response.
        max_output_tokens : Optional[int]
            Maximum tokens in the response.
        number_of_tries : int
            Number of retry attempts on failure.

        Returns
        -------
        Optional[str]
            Text description of the document, or None if all attempts fail.

        Raises
        ------
        FeatureNotAvailableError
            If no model supporting documents is available.
        APIError
            If an unrecoverable error occurs during description.
        """
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
            if model_name is None:
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
                    response = response[-1].to_plain_text()

                    # Check if the response is empty:
                    if not response:
                        aprint("Response is empty. Trying again...")
                        continue

                    # response in lower case and trimmed of white spaces
                    response_lc = response.lower().strip()

                    # Check if response is too short:
                    if len(response_lc) < 3:
                        aprint("Response is empty. Trying again...")
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
                        document_uri=document_uri, description=response, **kwargs
                    )

                    # print the description::
                    with asection("Description:"):
                        aprint(response)

                    return response
                return None

            except Exception as e:
                # Log the error:
                aprint(f"Error: '{e}'")
                # print stack trace:
                import traceback

                traceback.print_exc()
                raise APIError(f"Error describing document: {e}") from e
