from typing import List, Dict, Optional

from arbol import aprint

from litemind.agent.message import Message
from litemind.agent.tools.toolset import ToolSet
from litemind.apis.base_api import BaseApi
from litemind.apis.ollama.utils.messages import _convert_messages_for_ollama
from litemind.apis.ollama.utils.process_response import _process_response
from litemind.apis.ollama.utils.tools import _format_tools_for_ollama
from litemind.apis.openai.exceptions import APIError


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
        """
        Get the list of models available in Ollama.
        (Calls ollama.list())

        Returns:
        - List[str]: The list of models available in Ollama.
        """

        try:
            return list(self._model_list)
        except Exception:
            raise APIError("Error fetching model list from Ollama.")

    def default_model(self,
                      require_vision: bool = False,
                      require_tools: bool = False) -> str:
        """
        Get the default model name for Ollama.
        (Calls ollama.list())

        Returns:
        - str: The default model name for Ollama.
        """

        model_list = self.model_list()
        if require_vision:
            model_list = [model for model in model_list if
                          self.has_vision_support(model)]
        if require_tools:
            model_list = [model for model in model_list if
                          self.has_tool_support(model)]
        if not model_list:
            return None

        return model_list[0]

    def check_api_key(self, api_key: Optional[str] = None) -> bool:
        """
        Check if Ollama is available by attempting a simple list query.

        Returns:
        - bool: True if Ollama is running and responding, False otherwise.
        """
        try:
            self.client.list()
            return True
        except Exception:
            return False

    def has_vision_support(self, model_name: Optional[str] = None) -> bool:
        """
        Indicates whether the API supports vision tasks.

        Parameters:
        - model_name (str): The model name.

        Returns:
        - bool: False (Ollama does not currently support vision).
        """

        try:
            model_details_families = self.client.show(
                model_name).details.families
            return model_details_families and 'clip' in model_details_families
        except:
            return False

    def has_tool_support(self, model_name: Optional[str] = None) -> bool:
        """
        Indicates whether the API supports tool usage.
        Parameters
        ----------
        model_name : Optional[str]
            The model name.

        Returns
        -------
        bool : True if the model supports tools, False otherwise.

        """
        model_template = self.client.show(model_name).template
        return '$.Tools' in model_template

    def max_num_input_tokens(self, model_name: Optional[str] = None) -> int:
        """
        Returns the maximum number of input tokens allowed for a specific model.

        Parameters:
        - model_name (str): The model name.

        Returns:
        - int: Currently returns 100000 as a placeholder. Update this as more info about Ollama's token limits is available.
        """
        model_info = self.client.show(model_name).modelinfo

        # search key hat contains 'context_length'
        for key in model_info.keys():
            if 'context_length' in key:
                return model_info[key]

        # return default value if nothing else works:
        return 2500

    def max_num_output_tokens(self, model_name: Optional[str] = None) -> int:
        """
        Return the model's maximum number of output tokens.
        If model_name is None, this uses get_default_openai_model_name() as a fallback.
        """

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
        """
        Generate a completion using the Ollama API, with optional tool usage.

        Parameters
        ----------
        messages : List[Message]
            A list of messages to send to the model.
        model_name : str
            The model to use for the request.
        temperature : float
            Sampling temperature.
        max_output_tokens : int
            Maximum number of tokens to generate.
        toolset : Optional[ToolSet]
            A set of tools that can be invoked by the model via function calls.

        Returns
        -------
        Message
            The response from the model.
        """

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
