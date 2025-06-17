import base64
import json
import os
import tempfile
from typing import List, Optional, Sequence, Type, Union

from openai import OpenAI
from pydantic import BaseModel

from litemind.agent.messages.message import Message
from litemind.agent.messages.message_block import MessageBlock
from litemind.agent.tools.base_tool import BaseTool
from litemind.agent.tools.builtin_tools.mcp_tool import BuiltinMCPTool
from litemind.agent.tools.builtin_tools.web_search_tool import BuiltinWebSearchTool
from litemind.agent.tools.toolset import ToolSet
from litemind.apis.base_api import ModelFeatures
from litemind.apis.callbacks.callback_manager import CallbackManager
from litemind.apis.default_api import DefaultApi
from litemind.apis.exceptions import APIError, APINotAvailableError
from litemind.apis.feature_scanner import get_default_model_feature_scanner
from litemind.apis.providers.openai.utils.check_availability import (
    check_openai_api_availability,
)
from litemind.apis.providers.openai.utils.list_models import (
    _get_raw_openai_model_list,
    get_openai_model_list,
)
from litemind.media.media_base import MediaBase
from litemind.media.types.media_action import Action
from litemind.media.types.media_image import Image
from litemind.media.types.media_text import Text
from litemind.utils.normalise_uri_to_local_file_path import uri_to_local_file_path

# ------------------------------------------------------------------ #
#  Constants
# ------------------------------------------------------------------ #
_1M_TOKENS = 1_048_576  # 1 Mi tokens = 1024 × 1024
_O_SERIES_CTX = 200_000
_O_SERIES_OUT = 100_000


class OpenAIApi(DefaultApi):
    """
    An OpenAI Response API implementation conforming to the BaseApi interface.

    This implementation uses OpenAI's newer Response API which is stateful and provides
    built-in tools like web search, file search, and computer use. It's designed to be
    simpler and more flexible than the chat completions API.

    Key differences from chat completions:
    - Uses client.responses.create() instead of client.chat.completions.create()
    - Stateful - maintains conversation history automatically
    - Built-in tools like web_search, file_search
    - Different request/response format with 'input' parameter
    - Can chain responses using previous_response_id

    Set the OPENAI_API_KEY environment variable to your OpenAI API key, or pass it as an argument to the constructor.
    You can get your API key from https://platform.openai.com/account/api-keys.

    Note: This implementation requires models that support the Response API.
    Not compatible with older models that only support chat completions.
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
        Initialize the OpenAI Response API client.

        Parameters
        ----------
        api_key: Optional[str]
            The API key for OpenAI. If not provided, we'll read from OPENAI_API_KEY env var.
        base_url: Optional[str]
            The base URL for the OpenAI or compatible API.
        allow_media_conversions: bool
            If True, the API will allow media conversions using the default media converter.
        allow_media_conversions_with_models: bool
            If True, the API will allow media conversions using models that support the required features.
        kwargs: dict
            Additional options passed to `OpenAI(...)`.
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
            self.client: OpenAI = OpenAI(
                api_key=self.api_key, base_url=self.base_url, **self.kwargs
            )

            # Get the raw list of models:
            self._raw_model_list = _get_raw_openai_model_list(self.client)

            # Get the list of models from OpenAI (filter to only Response API compatible models):
            self._model_list = get_openai_model_list(self._raw_model_list)

        except Exception as e:
            # Print stack trace:
            import traceback

            traceback.print_exc()
            raise APINotAvailableError(
                f"Error initializing OpenAI Response API client: {e}"
            )

    def check_availability_and_credentials(self, api_key: Optional[str] = None) -> bool:
        """Check if the API is available and credentials are valid."""

        # Check if the API key is provided:
        if api_key is not None:
            client = OpenAI(api_key=api_key, base_url=self.base_url, **self.kwargs)
        else:
            client = self.client

        # Check the OpenAI API key:
        result = check_openai_api_availability(client)

        # Call the callback manager:
        self.callback_manager.on_availability_check(result)

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
        """List available models that support the Response API."""

        try:
            # Normalise the features:
            features = ModelFeatures.normalise(features)
            non_features = ModelFeatures.normalise(non_features)

            # Get the Response API compatible models:
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

            # Call callbacks:
            self.callback_manager.on_model_list(model_list)

            return model_list

        except Exception:
            raise APIError("Error fetching model list from OpenAI Response API.")

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
        """Get the best model that supports the required features."""

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

        if len(model_list) > 0:
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
        """Check if a model supports the given features."""

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

        # Check if model is Response API compatible
        if model_name not in self._model_list:
            return False

        for feature in features:
            if not self.feature_scanner.supports_feature(
                self.__class__, model_name, feature
            ):
                return False

        return True

    def max_num_input_tokens(self, model_name: Optional[str] = None) -> int:
        """Get the maximum number of input tokens for the given model."""
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

        # ---------- computer-use-preview ----------
        if "computer-use-preview" in name:
            return 128_000

        return 128_000  # Default for Response API models

    def max_num_output_tokens(self, model_name: Optional[str] = None) -> int:
        """Get the maximum number of output tokens for the given model."""
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

        # ---------- GPT-4o ----------
        if "gpt-4o-mini" in name:
            return 16_384
        if "gpt-4o" in name:
            return 16_384

        return 16_384  # Default for Response API models

    def _convert_messages_to_response_input(
        self, messages: List[Message], is_tool_followup: bool = False
    ) -> Union[str, List[dict]]:
        """
        Convert Litemind messages to Response API input format.

        The Response API can accept either:
        1. A simple string for single-turn interactions
        2. An array of conversation items for multi-turn conversations

        Parameters
        ----------
        messages : List[Message]
            The messages to convert
        is_tool_followup : bool
            True if this is a follow-up call with tool results in the same Response API session.
            When False, tool messages are converted to conversation items.
            When True, this method should not be called (use function_call_outputs directly).
        """

        if len(messages) == 1 and len(messages[0].blocks) == 1:
            # Simple case: single message with text only
            if messages[0].blocks[0].has_type(Text):
                return messages[0].blocks[0].get_content()

        # Complex case: convert to conversation items format
        conversation_items = []

        for message in messages:
            if message.role == "system":
                # System messages are converted to user messages with special formatting
                content = self._convert_message_blocks_to_response_content(
                    message.blocks, is_input=True, is_system=True
                )

                if content:  # Only add if there's content
                    conversation_items.append(
                        {"type": "message", "role": "user", "content": content}
                    )

            elif message.role == "user":
                content = self._convert_message_blocks_to_response_content(
                    message.blocks, is_input=True
                )

                if content:  # Only add if there's content
                    conversation_items.append(
                        {"type": "message", "role": "user", "content": content}
                    )

            elif message.role == "assistant":
                # Assistant messages from previous turns
                content = self._convert_message_blocks_to_response_content(
                    message.blocks, is_input=False
                )

                if content:  # Only add if there's content
                    conversation_items.append(
                        {"type": "message", "role": "assistant", "content": content}
                    )

            elif message.role == "tool":
                # Tool use results - convert to user messages describing what happened
                # DO NOT convert to function_call_outputs when starting a new conversation

                if not is_tool_followup:
                    # Convert tool results to descriptive user messages for conversation history
                    content = []
                    for block in message.blocks:
                        if block.has_type(Action):
                            action = block.get_content()
                            if hasattr(action, "tool_name") and hasattr(
                                action, "result"
                            ):
                                tool_result_text = f"[Tool {action.tool_name} returned: {action.result}]"
                                content.append(
                                    {"type": "input_text", "text": tool_result_text}
                                )

                    if content:  # Only add if there's content
                        conversation_items.append(
                            {
                                "type": "message",
                                "role": "user",  # Tool results become user messages in conversation history
                                "content": content,
                            }
                        )

        return conversation_items

    def _convert_message_blocks_to_response_content(
        self, blocks: List[MessageBlock], is_input: bool = True, is_system: bool = False
    ) -> List[dict]:
        """
        Convert message blocks to Response API content format.

        Parameters
        ----------
        blocks : List[MessageBlock]
            The message blocks to convert
        is_input : bool
            True for input content (user messages), False for output content (assistant messages)
        is_system : bool
            True if this is a system message (adds [SYSTEM] prefix to text)
        """
        from litemind.media.types.media_audio import Audio

        content = []

        for block in blocks:
            media = block.media

            if block.has_type(Text):
                text_content = media.get_content()
                if is_system:
                    text_content = f"[SYSTEM]: {text_content}"

                content.append(
                    {
                        "type": "input_text" if is_input else "output_text",
                        "text": text_content,
                    }
                )

            elif block.has_type(Image):
                if is_input:
                    # Use existing to_remote_or_data_uri method
                    image_uri = media.to_remote_or_data_uri()
                    content.append({"type": "input_image", "image_url": image_uri})
                else:
                    # For assistant output, describe the image
                    filename = (
                        media.get_filename()
                        if hasattr(media, "get_filename")
                        else "image"
                    )
                    content.append(
                        {"type": "output_text", "text": f"[Image: {filename}]"}
                    )

            elif block.has_type(Audio):
                if is_input:
                    # Use existing get_raw_data method
                    try:
                        audio_data = media.get_raw_data()
                        import base64

                        audio_b64 = base64.b64encode(audio_data).decode("utf-8")

                        # Determine format from file extension using existing method
                        filename = media.get_filename()
                        audio_format = (
                            "mp3" if filename.lower().endswith(".mp3") else "wav"
                        )

                        content.append(
                            {
                                "type": "input_audio",
                                "input_audio": {
                                    "data": audio_b64,
                                    "format": audio_format,
                                },
                            }
                        )
                    except Exception as e:
                        # Fallback to text description if audio processing fails
                        content.append(
                            {
                                "type": "input_text",
                                "text": f"[Audio file: {media.get_filename()}]",
                            }
                        )
                else:
                    # For assistant output, describe the audio
                    content.append(
                        {
                            "type": "output_text",
                            "text": f"[Audio: {media.get_filename()}]",
                        }
                    )

            elif block.has_type(Action):
                # Handle tool calls for assistant messages
                if not is_input:
                    action = media.get_content()
                    if hasattr(action, "tool_name") and hasattr(action, "arguments"):
                        # This is a tool call - represent it as text in the conversation
                        tool_call_text = f"[Called tool {action.tool_name} with arguments: {action.arguments}]"
                        content.append({"type": "output_text", "text": tool_call_text})
                # For input (user) messages, Action blocks shouldn't normally appear

            else:
                # throw an error if unsupported media type:
                raise ValueError(
                    f"Unsupported media type in message block: {type(media).__name__}"
                )

        return content

    def _convert_response_to_messages(
        self, response, response_format: Optional[BaseModel] = None
    ) -> List[Message]:
        """Convert Response API output to Litemind messages with proper structured output handling."""

        messages = []

        for output_item in response.output:
            if output_item.type == "message" and output_item.role == "assistant":
                message = Message(role="assistant")

                for content_item in output_item.content:
                    if content_item.type == "output_text":
                        text_content = content_item.text

                        # Handle structured output - if response_format is specified and we get JSON text
                        if response_format and text_content.strip().startswith("{"):
                            try:
                                json_data = json.loads(text_content)
                                obj_instance = response_format(**json_data)
                                message.append_object(obj_instance)
                            except (json.JSONDecodeError, ValueError, TypeError) as e:
                                # If parsing fails, treat as regular text but also log the issue
                                print(
                                    f"Warning: Failed to parse structured output: {e}"
                                )
                                message.append_text(text_content)
                        else:
                            # Regular text content
                            message.append_text(text_content)

                messages.append(message)

            elif output_item.type == "function_call":
                # Handle custom function calls
                message = Message(role="assistant")
                message.append_tool_call(
                    tool_name=output_item.name,
                    arguments=(
                        json.loads(output_item.arguments)
                        if isinstance(output_item.arguments, str)
                        else output_item.arguments
                    ),
                    id=output_item.call_id,
                )
                messages.append(message)

            # Handle built-in tool calls (web search, file search, etc.)
            elif output_item.type in [
                "web_search_call",
                "file_search_call",
                "computer_use_call",
            ]:
                # Built-in tools are handled automatically by the Response API
                # We don't need to create messages for these
                pass

        return messages

    def _format_tools_for_response_api(self, toolset: ToolSet) -> List[dict]:
        """Format tools for the Response API."""
        tools = []

        for tool in toolset.list_tools():

            # Skip built-in tools as they are handled automatically by the Response API
            if tool.is_builtin():
                continue

            # Response API uses a slightly different format than Chat Completions API
            tool_def = {
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.arguments_schema,
            }
            tools.append(tool_def)

        return tools

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
        """Generate text using the Response API.

        Parameters
        ----------
        messages : List[Message]
            The input messages for the conversation.
        model_name : Optional[str]
            The name of the model to use.
        temperature : Optional[float]
            The temperature for text generation.
        max_num_output_tokens : Optional[int]
            Maximum number of output tokens.
        toolset : Optional[ToolSet]
            Custom tools to make available to the model.
        use_tools : bool
            Whether to use tools (both custom and built-in).
        response_format : Optional[BaseModel]
            Pydantic model for structured output.
        **kwargs
            Additional parameters to pass to the API.

        Returns
        -------
        List[Message]
            The generated response messages.
        """

        use_file_search: bool = False
        vector_store_ids: Optional[List[str]] = None
        use_computer_use: bool = False
        stream: bool = True

        # Validate inputs using parent method:
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

        # Set default model if not provided
        if model_name is None:
            model_name = self._get_best_model_for_text_generation(
                messages, toolset if use_tools else None, response_format
            )

        # Handle model-specific parameter restrictions
        reasoning_effort = None
        if "o1" in model_name or "o3" in model_name:
            # o-series models do not support temperature parameter
            temperature = None

            # Extract reasoning effort from model name (e.g., "o3-high" -> "high")
            reasoning_effort_suffix = model_name.split("-")[-1]

            # Check if it is a valid reasoning effort:
            if reasoning_effort_suffix in ["low", "medium", "high"]:
                reasoning_effort = reasoning_effort_suffix
                # Remove the suffix from the model name
                model_name_native = model_name.replace(
                    f"-{reasoning_effort_suffix}", ""
                )
            else:
                reasoning_effort = None
                model_name_native = model_name
        else:
            model_name_native = model_name

        # Preprocess messages:
        preprocessed_messages = self._preprocess_messages(
            messages=messages,
            allowed_media_types=self._get_allowed_media_types_for_text_generation(
                model_name=model_name
            ),
        )

        # Get max num of output tokens for model if not provided:
        if max_num_output_tokens is None:
            max_num_output_tokens = self.max_num_output_tokens(model_name)

        # Prepare tools (both custom and built-in)
        tools = []
        if use_tools:
            # Add custom function tools
            if toolset:
                tools.extend(self._format_tools_for_response_api(toolset))

                for builtinTool in toolset.list_builtin_tools():
                    if isinstance(builtinTool, BuiltinWebSearchTool):
                        web_search_tool: BuiltinWebSearchTool = builtinTool
                        # we can extract parameters from the web search tool:
                        search_context_size = web_search_tool.search_context_size
                        tools.append(
                            {
                                "type": "web_search_preview",
                                "search_context_size": search_context_size,
                            }
                        )
                    elif isinstance(builtinTool, BuiltinMCPTool):
                        mcp_tool: BuiltinMCPTool = builtinTool
                        # we can extract parameters from the MCP tool:
                        tools.append(
                            {
                                "type": "mcp",
                                "server_label": mcp_tool.server_name,
                                "server_url": mcp_tool.server_url,
                                "headers": mcp_tool.headers,
                                "require_approval": "never",
                                "allowed_tools": mcp_tool.allowed_tools,
                            }
                        )
                    else:
                        # Throw an exception if unknown built-in tool is used:
                        raise ValueError(
                            "Unknown built-in tool: {}".format(builtinTool.name)
                        )

            if use_file_search:
                if not vector_store_ids:
                    raise ValueError(
                        "vector_store_ids must be provided when use_file_search=True"
                    )
                tools.append(
                    {"type": "file_search", "vector_store_ids": vector_store_ids}
                )

            if use_computer_use:
                tools.append({"type": "computer_use"})

        try:
            # List of new messages to append to the response:
            new_messages = []
            previous_response_id = None  # Track for stateful API
            result_messages = []

            # Loop until we get a response that doesn't require tool use:
            while True:
                if previous_response_id is None:
                    # Initial request: convert full conversation history
                    response_input = self._convert_messages_to_response_input(
                        preprocessed_messages, is_tool_followup=False
                    )

                    # Prepare initial API parameters
                    api_params = {
                        "model": model_name_native,
                        "input": response_input,
                        "max_output_tokens": max_num_output_tokens,
                    }

                    # Add structured output support - THIS IS THE KEY FIX
                    if response_format:
                        api_params["text_format"] = response_format

                    # Only add temperature if it's not None (o-series models don't support it)
                    if temperature is not None:
                        api_params["temperature"] = temperature

                    # Add reasoning effort if specified (for o-series models)
                    if reasoning_effort is not None:
                        api_params["reasoning"] = {"effort": reasoning_effort}

                    # Only add tools if there are any
                    if tools:
                        api_params["tools"] = tools

                    # Add any additional kwargs
                    api_params.update(kwargs)

                else:
                    # Subsequent request: use previous_response_id with tool outputs
                    # response_input should already be set to function_call_outputs from previous iteration
                    api_params = {
                        "model": model_name_native,
                        "input": response_input,  # This will be function_call_outputs
                        "previous_response_id": previous_response_id,
                        "max_output_tokens": max_num_output_tokens,
                    }

                    # Add structured output support - THIS IS THE KEY FIX
                    if response_format:
                        api_params["text_format"] = response_format

                    # Only add temperature if it's not None (o-series models don't support it)
                    if temperature is not None:
                        api_params["temperature"] = temperature

                    # Add reasoning effort if specified (for o-series models)
                    if reasoning_effort is not None:
                        api_params["reasoning"] = {"effort": reasoning_effort}

                    # Only add tools if there are any
                    if tools:
                        api_params["tools"] = tools

                    # Add any additional kwargs
                    api_params.update(kwargs)

                # Create response using Response API with streaming
                try:
                    with self.client.responses.stream(
                        **api_params
                    ) as streaming_response:
                        # Process streaming events - based on official OpenAI example
                        for event in streaming_response:
                            if "output_text" in event.type:
                                # Extract the text delta for streaming callback
                                if hasattr(event, "delta"):
                                    self.callback_manager.on_text_streaming(
                                        fragment=event.delta, **kwargs
                                    )
                                elif hasattr(event, "text"):
                                    self.callback_manager.on_text_streaming(
                                        fragment=event.text, **kwargs
                                    )

                        # Get the final response - identical to non-streaming!
                        response = streaming_response.get_final_response()

                except AttributeError:
                    # Fallback if streaming method not available
                    response = self.client.responses.create(**api_params)

                # Convert response to Litemind messages
                result_messages = self._convert_response_to_messages(response)

                # Append response message to original, preprocessed, and new messages:
                if result_messages:
                    messages.extend(result_messages)
                    preprocessed_messages.extend(result_messages)
                    new_messages.extend(result_messages)

                # Check if we have tool calls to execute
                tool_calls_to_execute = []
                for output_item in response.output:
                    if output_item.type == "function_call":
                        tool_calls_to_execute.append(output_item)

                if tool_calls_to_execute and use_tools:
                    # Create tool use message to hold results for our internal tracking
                    tool_use_message = Message(role="tool")
                    messages.append(tool_use_message)
                    preprocessed_messages.append(tool_use_message)
                    new_messages.append(tool_use_message)

                    # Prepare function call outputs for next Response API call
                    function_call_outputs = []

                    # Execute each tool call
                    for function_call in tool_calls_to_execute:
                        tool_name = function_call.name
                        tool_arguments = (
                            json.loads(function_call.arguments)
                            if isinstance(function_call.arguments, str)
                            else function_call.arguments
                        )
                        call_id = function_call.call_id

                        # Get the corresponding tool in toolset:
                        tool: Optional[BaseTool] = (
                            toolset.get_tool(tool_name) if toolset else None
                        )

                        if tool:
                            try:
                                # Execute the tool
                                result = tool.execute(**tool_arguments)

                                # If not a string, convert from JSON:
                                if not isinstance(result, str):
                                    result = json.dumps(result, default=str)

                            except Exception as e:
                                result = f"Function '{tool_name}' error: {e}"
                        else:
                            result = f"(Tool '{tool_name}' use requested, but tool not found.)"

                        # Append the tool call result to our internal message:
                        tool_use_message.append_tool_use(
                            tool_name=tool_name,
                            arguments=tool_arguments,
                            result=result,
                            id=call_id,
                        )

                        # Prepare function call output for Response API
                        function_call_outputs.append(
                            {
                                "type": "function_call_output",
                                "call_id": call_id,
                                "output": result,
                            }
                        )

                    # Set up for next iteration with tool results
                    response_input = function_call_outputs
                    previous_response_id = response.id
                    # Continue the loop to send tool results and potentially get more tool calls

                else:
                    # No tool calls, we're done
                    break

            # If using structured output, convert the text to object
            if response_format and result_messages:
                last_message = result_messages[-1]
                if last_message.blocks and last_message.blocks[-1].has_type(Text):
                    text_content = last_message.blocks[-1].get_content()
                    try:
                        obj_data = json.loads(text_content)
                        obj_instance = response_format(**obj_data)
                        last_message.append_object(obj_instance)
                    except (json.JSONDecodeError, ValueError):
                        # If parsing fails, keep as text
                        pass

            # Add parameters to kwargs for callback:
            kwargs.update(
                {
                    "model_name": model_name,
                    "temperature": temperature,
                    "max_output_tokens": max_num_output_tokens,
                    "toolset": toolset,
                    "response_format": response_format,
                    "stream": stream,
                }
            )

            # Call the callback manager:
            if result_messages:
                self.callback_manager.on_text_generation(
                    messages=messages, response=result_messages[-1], **kwargs
                )

            return new_messages

        except Exception as e:
            import traceback

            traceback.print_exc()
            raise APIError(f"OpenAI Response API generate text error: {e}")

    def transcribe_audio(
        self, audio_uri: str, model_name: Optional[str] = None, **kwargs
    ) -> str:
        """Transcribe audio using OpenAI's Whisper API."""

        # If no model name is provided, use the best model:
        if model_name is None:
            model_name = self.get_best_model(features=ModelFeatures.AudioTranscription)

        # If no model is available then delegate to superclass:
        if model_name is None or model_name in super().list_models():
            return super().transcribe_audio(
                audio_uri=audio_uri, model_name=model_name, **kwargs
            )

        # Check that the model name contains 'whisper':
        if "whisper" not in model_name:
            raise APIError(f"Model {model_name} does not support audio transcription.")

        # Get the local path to the audio file:
        audio_file_path = uri_to_local_file_path(audio_uri)

        # Open the audio file and transcribe:
        with open(audio_file_path, "rb") as audio_file:
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
        """Generate audio using OpenAI's TTS API."""

        # If no model name is provided, use the best model:
        if model_name is None:
            model_name = self.get_best_model(features=ModelFeatures.AudioGeneration)

        # If no model is available then delegate to superclass:
        if model_name is None or model_name in super().list_models():
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

        # Save to temporary file:
        with tempfile.NamedTemporaryFile(
            suffix=f".{audio_format}", delete=False
        ) as temp_file:
            response.write_to_file(temp_file.name)
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
        """Generate image using OpenAI's image generation API."""

        # Make prompt from positive and negative prompts:
        if negative_prompt:
            prompt = f"Image must consist in: {positive_prompt}\nBut NOT of: {negative_prompt}"
        else:
            prompt = positive_prompt

        # Get the best model if not provided:
        if model_name is None:
            model_name = self.get_best_model(features=ModelFeatures.ImageGeneration)

        # For Response API compatible models, try to use the newer approach
        if model_name and model_name in self._model_list:
            # Use the Response API for image generation if available
            try:
                response = self.client.responses.create(
                    model=model_name,
                    input=f"Generate an image: {prompt}",
                    tools=(
                        [{"type": "image_generation"}]
                        if "gpt-image" in model_name
                        else None
                    ),
                    **kwargs,
                )

                # Extract image from response - this would need to be implemented
                # based on how the Response API returns generated images
                # For now, fall back to the standard image generation API
                pass
            except:
                pass

        # Fall back to standard DALL-E image generation API
        # (Same implementation as OpenAIApi)

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
            raise ValueError(
                f"Model {model_name} is not supported for image generation."
            )

        # Find appropriate size
        requested_size = f"{image_width}x{image_height}"
        if requested_size not in allowed_sizes:
            if not allow_resizing:
                raise ValueError(
                    f"Requested resolution {requested_size} is not allowed."
                )

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

        # Set quality and format based on model:
        if "gpt-image-1" in model_name:
            kwargs["quality"] = "auto"
            kwargs["output_format"] = "png"
        elif "dall-e-3" in model_name:
            kwargs["quality"] = "hd"
            kwargs["response_format"] = "b64_json"
        elif "dall-e-2" in model_name:
            kwargs["response_format"] = "b64_json"

        # Generate image:
        response = self.client.images.generate(
            model=model_name,
            prompt=prompt,
            size=closest_size,
            n=1,
            **kwargs,
        )

        # Get the image and decode:
        image_b64 = response.data[0].b64_json
        image_data = base64.b64decode(image_b64)

        # Create PIL image:
        from io import BytesIO

        from PIL import Image
        from PIL.Image import Resampling

        image = Image.open(BytesIO(image_data))

        # Resize if needed:
        if closest_size != requested_size:
            if preserve_aspect_ratio:
                image.thumbnail((image_width, image_height), Resampling.LANCZOS)
            else:
                image = image.resize((image_width, image_height), Resampling.LANCZOS)

        # Update kwargs and call callback:
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

        self.callback_manager.on_image_generation(image=image, prompt=prompt, **kwargs)
        return image

    def embed_texts(
        self,
        texts: Sequence[str],
        model_name: Optional[str] = None,
        dimensions: int = 512,
        **kwargs,
    ) -> Sequence[Sequence[float]]:
        """Generate text embeddings using OpenAI's embeddings API."""

        # Get the best model if not provided:
        if model_name is None:
            model_name = self.get_best_model(features=ModelFeatures.TextEmbeddings)

        # If the model belongs to the superclass, delegate:
        if model_name in super().list_models():
            return super().embed_texts(
                texts=texts, model_name=model_name, dimensions=dimensions, **kwargs
            )

        # Call OpenAI API to generate embeddings:
        response = self.client.embeddings.create(
            model=model_name, dimensions=dimensions, input=texts, **kwargs
        )

        # Extract embeddings from the response:
        embeddings = [e.embedding for e in response.data]

        # Update kwargs and call callback:
        kwargs.update({"model_name": model_name, "dimensions": dimensions})
        self.callback_manager.on_text_embedding(
            texts=texts, embeddings=embeddings, **kwargs
        )

        return embeddings
