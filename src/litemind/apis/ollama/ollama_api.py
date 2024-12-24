import traceback
from typing import List, Dict, Optional

from arbol import asection, aprint

from litemind.agent.message import Message
from litemind.agent.tools.toolset import ToolSet
from litemind.apis.base_api import BaseApi
from litemind.apis.openai.exceptions import APIError


class OllamaApi(BaseApi):
    def __init__(self, host: Optional[str] = None, headers: Optional[Dict[str, str]] = None, **kwargs):
        """
        Initialize the Ollama API client.

        Parameters:
        - host (Optional[str]): The host for the Ollama client (default is `http://localhost:11434`).
        - headers (Optional[Dict[str, str]]): Additional headers to pass to the client.
        """

        from ollama import Client
        self.client = Client(host=host, headers=headers, **kwargs)

    def model_list(self) -> List[str]:
        """
        Get the list of models available in Ollama.
        (Calls ollama.list())

        Returns:
        - List[str]: The list of models available in Ollama.
        """

        try:
            return [model.model for model in self.client.list().models]
        except Exception:
            raise APIError("Error fetching model list from Ollama.")


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

    def has_vision_support(self) -> bool:
        """
        Indicates whether the API supports vision tasks.

        Returns:
        - bool: False (Ollama does not currently support vision).
        """
        return False

    def max_num_input_token(self, model: str) -> int:
        """
        Returns the maximum number of input tokens allowed for a specific model.

        Parameters:
        - model (str): The model name.

        Returns:
        - int: Currently returns 100000 as a placeholder. Update this as more info about Ollama's token limits is available.
        """
        model_info = self.client.show(model).modelinfo

        # search key hat contains 'context_length'
        for key in model_info.keys():
            if 'context_length' in key:
                return model_info[key]

        # return default value:
        return 2500


    def completion(self,
                   messages: List[Message],
                   model_name: str = "llama3.2",
                   temperature: Optional[float] = 0.0,
                   toolset: Optional[ToolSet] = None,
                   **kwargs) -> Message:
        """
        Generate a completion using the Ollama API.

        Parameters:
        - messages (List[Message]): A list of messages to send to the model.
        - model (str): The model to use for the request.
        - temperature (Optional[float]): Sampling temperature (not directly supported by Ollama yet).
        - toolset (Optional[ToolSet]): Not supported by Ollama, included for interface adherence.

        Returns:
        - Message: The response from the model.
        """
        # Convert messages into Ollama's format
        ollama_messages = self._convert_messages_for_ollama(messages)

        # Ollama does not support external tools, so the toolset is unused
        if toolset:
            raise NotImplementedError("Tool usage is not supported by Ollama.")

        from ollama import ResponseError
        try:
            from ollama import ChatResponse
            response: ChatResponse = self.client.chat(model=model_name, messages=ollama_messages)
            response_message = Message(role="assistant", text=response.message.content)
            messages.append(response_message)
            return response_message
        except ResponseError as e:
            raise APIError(f"Error during completion: {e.error} (status code: {e.status_code})")

    def describe_image(self,
                       image_path: str,
                       query: str = 'Here is an image, please carefully describe it in detail.',
                       model_name: str = "llava",
                       temperature: float = 0,
                       max_tokens: int = 4096,
                       number_of_tries: int = 4,
                       ) -> str:
        """
        Describe an image using Ollama-based vision models.

        Parameters
        ----------
        image_path: str
            Path to the image to describe
        query: str
            Prompt/query to send to the vision model
        model_name: str
            Model to use (e.g. "llava")
        temperature: float
            Temperature setting for model
        max_tokens: int
            Maximum number of tokens (num_ctx in Ollama)
        number_of_tries: int
            Number of times to retry if the response is empty or a refusal

        Returns
        -------
        str
            A detailed description of the image
        """

        with asection(
                f"Asking Ollama to analyze a given image at path: '{image_path}':"):
            aprint(f"Query: '{query}'")
            aprint(f"Model: '{model_name}'")
            aprint(f"Max tokens: '{max_tokens}'")

            # Basic file format check (optional but consistent with your original function)
            if not (image_path.endswith('.png') or
                    image_path.endswith('.jpg') or
                    image_path.endswith('.jpeg')):
                raise NotImplementedError(
                    f"Image format not supported: '{image_path}' (only .png, .jpg, and .jpeg are supported)")

            try:
                for tries in range(number_of_tries):
                    # Send a request to Ollama
                    result = self.client.chat(
                        model=model_name,
                        messages=[
                            {
                                'role': 'user',
                                'content': query,
                                'images': [image_path]
                            }
                        ],
                    )

                    # Extract the content from the response
                    response = result['message']['content']
                    aprint(f"Response: '{response}'")

                    # Check if the response is empty
                    if not response:
                        aprint("Response is empty. Trying again...")
                        continue

                    # Trim and lower-case for refusal checks
                    response_lc = response.lower().strip()

                    # If the response is too short, treat that as an error and retry
                    if len(response_lc) < 3:
                        aprint("Response is too short. Trying again...")
                        continue

                    # Check if the model has refused or is unable to assist
                    if (
                            "sorry" in response_lc and
                            (
                                    "i cannot" in response_lc or "i can't" in response_lc or "i am unable" in response_lc)
                    ) or "i cannot assist" in response_lc or "i can't assist" in response_lc or "i am unable to assist" in response_lc or "i'm sorry" in response_lc:
                        aprint(
                            f"Vision model refuses to assist (response: {response}). Trying again...")
                        continue

                    # If we get here, we have a valid non-empty, non-refusal response
                    return response

                # If all retries are exhausted
                return "No valid response obtained after multiple attempts."

            except Exception as e:
                aprint(f"Error: '{e}'")
                traceback.print_exc()
                return f"Error: '{e}'"

    @staticmethod
    def _convert_messages_for_ollama(messages: List[Message]) -> List[Dict[str, str]]:
        """
        Convert messages into Ollama's format.

        Parameters:
        - messages (List[Message]): A list of messages.

        Returns:
        - List[Dict[str, str]]: Converted messages.
        """
        return [{"role": msg.role, "content": msg.text} for msg in messages]
