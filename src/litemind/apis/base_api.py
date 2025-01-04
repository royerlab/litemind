from abc import ABC, abstractmethod
from typing import List, Optional

from arbol import asection, aprint

from litemind.agent.message import Message
from litemind.agent.tools.toolset import ToolSet


class BaseApi(ABC):

    @abstractmethod
    def check_api_key(self, api_key: Optional[str]) -> bool:
        """
        Check if the API key is valid.

        Parameters
        ----------
        api_key: Optional[str]
            The API key to check.

        Returns
        -------
        bool
            True if the API key is valid, False otherwise.

        """
        pass

    @abstractmethod
    def model_list(self) -> List[str]:
        """
        Get the list of models available.

        Returns
        -------
        List[str]
            The list of models available

        """
        pass

    @abstractmethod
    def default_model(self,
                      require_vision: bool = False,
                      require_tools: bool = False
                      ) -> str:
        """
        Get the default model to use.

        Parameters
        ----------
        require_vision: bool
            Whether the model requires vision support.
        require_tools: bool
            Whether the model requires tool support.

        Returns
        -------
        str
            The default model to use.

        """
        pass

    @abstractmethod
    def has_vision_support(self, model_name: Optional[str] = None) -> bool:
        """
        Check if the model has vision support.

        Parameters
        ----------
        model_name: Optional[str]
            The name of the model to use.

        Returns
        -------
        bool
            True if the model has vision support, False otherwise.

        """
        pass

    @abstractmethod
    def has_tool_support(self, model_name: Optional[str] = None) -> bool:
        """
        Check if the model has tool support.

        Parameters
        ----------
        model_name: Optional[str]
            The name of the model to use.

        Returns
        -------
        bool
            True if the model has tool support, False otherwise.

        """
        pass

    @abstractmethod
    def max_num_input_tokens(self, model_name: Optional[str] = None) -> int:
        """
        Get the maximum number of input tokens for the given model.

        Parameters
        ----------
        model_name: Optional[str]
            The name of the model to use.

        Returns
        -------
        int
            The maximum number of input tokens for the model.

        """
        pass

    @abstractmethod
    def max_num_output_tokens(self, model_name: Optional[str] = None) -> int:
        """
        Get the maximum number of output tokens for the given model.
        Parameters
        ----------
        model_name: Optional[str]
            The name of the model to use.

        Returns
        -------
        int
            The maximum number of output tokens for the model.

        """
        pass

    @abstractmethod
    def completion(self,
                   model_name: str,
                   messages: List[Message],
                   temperature: float = 0.0,
                   max_output_tokens: Optional[int] = None,
                   toolset: Optional[ToolSet] = None,
                   **kwargs) -> Message:
        """
        Perform a text completion using the given model.

        Parameters
        ----------
        model_name: str
            The name of the model to use.
        messages: List[Message]
            The list of messages to send to the model.
        temperature: float
            The temperature to use.
        max_output_tokens: Optional[int]
            The maximum number of tokens to use.
        toolset: Optional[ToolSet]
            The toolset to use.
        kwargs: dict
            Additional arguments to pass to the completion function.

        Returns
        -------
        Message
            The response from the model.

        """
        pass

    def describe_image(self,
                       image_uri: str,
                       system: str = 'You are a helpful AI assistant that can describe/analyse images.',
                       query: str = 'Here is an image, please carefully describe it in detail.',
                       model_name: str = "gpt-4o",
                       temperature: float = 0,
                       max_output_tokens: int = 4096,
                       number_of_tries: int = 4,
                       ) -> str:
        """
        Describe an image using GPT-vision.

        Parameters
        ----------
        image_uri: str
            Can be: a path to an image file, a URL to an image, or a base64 encoded image.
        system:
            System message to use
        query  : str
            Query to use with image
        model_name   : str
            Model to use
        temperature: float
            Temperature to use
        max_output_tokens  : int
            Maximum number of tokens to use
        number_of_tries : int
            Number of times to try to send the request to the model

        Returns
        -------
        str
            Description of the image

        """

        with (asection(
                f"Asking model {model_name} to descrbe a given image: '{image_uri}':")):
            aprint(f"Query: '{query}'")
            aprint(f"Model: '{model_name}'")
            aprint(f"Max tokens: '{max_output_tokens}'")

            # If the model does not support vision, return an error:
            if not self.has_vision_support(model_name):
                return f"Model '{model_name}' does not support vision."

            try:

                # Retry in case of model refusing to answer:
                for tries in range(number_of_tries):

                    vision_model_name = self.default_model(
                        require_vision=True)

                    messages = []

                    # System message:
                    system_message = Message(role='system')
                    system_message.append_text(system)
                    messages.append(system_message)

                    # User message:
                    user_message = Message(role='user')
                    user_message.append_text(query)
                    user_message.append_image_uri(image_uri)

                    messages.append(user_message)

                    # Run agent:
                    response = self.completion(messages=messages,
                                               model_name=vision_model_name,
                                               temperature=temperature)

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
                    if ("sorry" in response_lc and (
                            "i cannot" in response_lc or "i can't" in response_lc or 'i am unable' in response_lc)) \
                            or "i cannot assist" in response_lc or "i can't assist" in response_lc or 'i am unable to assist' in response_lc or "I'm sorry" in response_lc:
                        aprint(
                            f"Vision model {model_name} refuses to assist (response: {response}). Trying again...")
                        continue
                    else:
                        return response

            except Exception as e:
                # Log the error:
                aprint(f"Error: '{e}'")
                # print stack trace:
                import traceback
                traceback.print_exc()
                return f"Error: '{e}'"
