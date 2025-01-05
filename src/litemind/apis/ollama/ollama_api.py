from typing import List, Dict, Optional

from arbol import aprint

from litemind.agent.message import Message
from litemind.agent.tools.toolset import ToolSet
from litemind.apis.base_api import BaseApi
from litemind.apis.exceptions import APIError
from litemind.apis.ollama.utils.messages import _convert_messages_for_ollama
from litemind.apis.ollama.utils.process_response import _process_response
from litemind.apis.ollama.utils.tools import _format_tools_for_ollama


class OllamaApi(BaseApi):
    def __init__(self,
                 host: Optional[str] = None,
                 headers: Optional[Dict[str, str]] = None, **kwargs):
        """
        Initialize the Ollama API client.

        Parameters:
        - host (Optional[str]): The host for the Ollama client (default is `http://localhost:11434`).
        - headers (Optional[Dict[str, str]]): Additional headers to pass to the client.
        """

        from ollama import Client
        self.client = Client(host=host,
                             headers=headers,
                             **kwargs)

        self.host = host
        self.headers = headers

        self._model_list = [model.model for model in self.client.list().models]

    def model_list(self) -> List[str]:

        try:
            return list(self._model_list)
        except Exception:
            raise APIError("Error fetching model list from Ollama.")

    def default_model(self,
                      require_images: bool = False,
                      require_audio: bool = False,
                      require_tools: bool = False) -> Optional[str]:

        # Get the list of models:
        model_list = self.model_list()

        if require_images:
            # Filter out models that don't support vision:
            model_list = [model for model in model_list if
                          self.has_image_support(model)]

        if require_audio:
            # Filter out models that don't support audio:
            model_list = [model for model in model_list if
                          self.has_audio_support(model)]

        if require_tools:
            # Filter out models that don't support tools:
            model_list = [model for model in model_list if
                          self.has_tool_support(model)]

        # If we have any models left, return the first one
        if model_list:
            return model_list[0]
        else:
            return None

    def check_api_key(self, api_key: Optional[str] = None) -> bool:

        try:
            self.client.list()
            return True
        except Exception:
            return False

    def has_image_support(self, model_name: Optional[str] = None) -> bool:

        try:
            model_details_families = self.client.show(
                model_name).details.families
            return model_details_families and 'clip' in model_details_families
        except:
            return False

    def has_audio_support(self, model_name: Optional[str] = None) -> bool:

        # TODO: implement

        return False

    def has_tool_support(self, model_name: Optional[str] = None) -> bool:

        model_template = self.client.show(model_name).template
        return '$.Tools' in model_template

    def max_num_input_tokens(self, model_name: Optional[str] = None) -> int:

        model_info = self.client.show(model_name).modelinfo

        # search key hat contains 'context_length'
        for key in model_info.keys():
            if 'context_length' in key:
                return model_info[key]

        # return default value if nothing else works:
        return 2500

    def max_num_output_tokens(self, model_name: Optional[str] = None) -> int:

        model_info = self.client.show(model_name).modelinfo

        # search key hat contains 'max_tokens'
        for key in model_info.keys():
            if 'max_tokens' in key:
                return model_info[key]

        # return default value if nothing else works:
        return 4096

    def completion(self,
                   messages: List[Message],
                   model_name: Optional[str] = None,
                   temperature: Optional[float] = 0.0,
                   max_output_tokens: Optional[int] = None,
                   toolset: Optional[ToolSet] = None,
                   **kwargs) -> Message:

        # Set default model if not provided
        if model_name is None:
            model_name = self.default_model()

        # Get max num of output tokens for model if not provided:
        if max_output_tokens is None:
            max_output_tokens = self.max_num_output_tokens(model_name)

        # 1) Convert user messages into Ollama's format
        ollama_messages = _convert_messages_for_ollama(messages)

        # 2) Convert toolset (if any) to Ollama's tools schema
        ollama_tools = _format_tools_for_ollama(toolset) if toolset else None

        # 3) Make the request to Ollama
        from ollama import ResponseError
        try:
            aprint(f"Sending request to Ollama with model: {model_name}")
            response = self.client.chat(
                model=model_name,
                messages=ollama_messages,
                tools=ollama_tools,
                options={"temperature": temperature,
                         "num_predict": max_output_tokens},
                **kwargs
            )

            # 4) Process the response
            response_message = _process_response(response, toolset)
            messages.append(response_message)
            return response_message

        except ResponseError as e:
            raise APIError(
                f"Error during completion: {e.error} (status code: {e.status_code})")
