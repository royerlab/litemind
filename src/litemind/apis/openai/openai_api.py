import os
from typing import List, Optional

from litemind.agent.message import Message
from litemind.agent.tools.toolset import ToolSet
from litemind.apis.base_api import BaseApi
from litemind.apis.openai.exceptions import APIError
from litemind.apis.openai.utils.default_model import \
    get_default_openai_model_name
from litemind.apis.openai.utils.messages import convert_messages_for_openai
from litemind.apis.openai.utils.model_list import get_openai_model_list
from litemind.apis.openai.utils.model_types import is_vision_model, \
    is_tool_model
from litemind.apis.openai.utils.process_response import _process_response
from litemind.apis.openai.utils.tools import _format_tools_for_openai


class OpenAIApi(BaseApi):

    def __init__(self,
                 api_key: Optional[str] = None,
                 **kwargs):

        # get key from environmental variables:
        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY")
        if api_key is None:
            raise APIError(
                "The api_key client option must be set either by passing api_key to the client or by setting the OPENAI_API_KEY environment variable"
            )

        # Create an OpenAI client:
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, **kwargs)

    def check_api_key(self, api_key: Optional[str] = None) -> bool:
        messages = []

        user_message = Message(role='user')
        user_message.append_text('Hello!')
        messages.append(user_message)

        try:
            self.completion(messages)
            return True

        except Exception:
            import traceback
            traceback.print_exc()
            return False

    def model_list(self) -> List[str]:
        return get_openai_model_list()

    def default_model(self,
                      require_vision: bool = False,
                      require_tools: bool = False) -> str:
        return get_default_openai_model_name(require_vision=require_vision,
                                             require_tools=require_tools)

    def has_vision_support(self, model_name: Optional[str] = None) -> bool:
        return is_vision_model(model_name)

    def has_tool_support(self, model_name: Optional[str] = None) -> bool:
        return is_tool_model(model_name)

    def max_num_input_tokens(self, model_name: Optional[str] = None) -> int:

        if model_name is None:
            model_name = get_default_openai_model_name()

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

        If model_name is None, this uses get_default_openai_model_name() as a fallback.
        """
        if model_name is None:
            model_name = get_default_openai_model_name()

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
            model_name = get_default_openai_model_name()

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

    def completion(self,
                   messages: List[Message],
                   model_name: Optional[str] = None,
                   temperature: Optional[float] = 0.0,
                   max_output_tokens: Optional[int] = None,
                   toolset: Optional[ToolSet] = None,
                   **kwargs) -> Message:

        from openai import NotGiven

        # Set default model if not provided
        if model_name is None:
            model_name = get_default_openai_model_name()

        # Get max num of output tokens for model if not provided:
        if max_output_tokens is None:
            max_output_tokens = self.max_num_output_tokens(model_name)

        # Convert ToolSet to OpenAI-compatible JSON schema if toolset is provided
        openai_tools = _format_tools_for_openai(
            toolset) if toolset else NotGiven()

        # Format messages for OpenAI:
        openai_formatted_messages = convert_messages_for_openai(messages)

        # Call OpenAI Chat Completions API
        response = self.client.chat.completions.create(
            model=model_name,
            messages=openai_formatted_messages,
            temperature=temperature,
            max_tokens=max_output_tokens,
            tools=openai_tools,
            **kwargs
        )

        # Process API response
        response_message = _process_response(response, toolset)
        messages.append(response_message)
        return response_message
