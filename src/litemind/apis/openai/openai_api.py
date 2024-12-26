import os
from typing import List, Optional

from arbol import aprint, asection

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

    def max_num_input_token(self, model_name: Optional[str] = None) -> int:

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

    def completion(self,
                   messages: List[Message],
                   model_name: Optional[str] = None,
                   temperature: Optional[float] = 0.0,
                   toolset: Optional[ToolSet] = None,
                   **kwargs) -> Message:

        from openai import NotGiven

        # Set default model if not provided
        if model_name is None:
            model_name = get_default_openai_model_name()

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
            tools=openai_tools,
            **kwargs
        )

        # Process API response
        response_message = _process_response(response, toolset)
        messages.append(response_message)
        return response_message

    def describe_image(self,
                       image_path: str,
                       query: str = 'Here is an image, please carefully describe it in detail.',
                       model_name: str = "gpt-4o",
                       temperature: float = 0,
                       max_tokens: int = 4096,
                       number_of_tries: int = 4,
                       ) -> str:
        """
        Describe an image using GPT-vision.

        Parameters
        ----------
        image_path: str
            Path to the image to describe
        query  : str
            Query to send to GPT
        model_name   : str
            Model to use
        max_tokens  : int
            Maximum number of tokens to use
        number_of_tries : int
            Number of times to try to send the request to GPT.

        Returns
        -------
        str
            Description of the image

        """

        with (asection(
                f"Asking GPT-vision to analyse a given image at path: '{image_path}':")):
            aprint(f"Query: '{query}'")
            aprint(f"Model: '{model_name}'")
            aprint(f"Max tokens: '{max_tokens}'")

            if image_path.endswith('.png'):
                image_format = 'png'
            elif image_path.endswith('.jpg') or image_path.endswith('.jpeg'):
                image_format = 'jpeg'
            else:
                raise NotImplementedError(
                    f"Image format not supported: '{image_path}' (only .png and .jpg are supported)")

            import base64

            # Read and encode the image in base64
            with open(image_path, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode(
                    "utf-8")

                # Craft the prompt for GPT
                prompt_messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": query
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/{image_format};base64,{encoded_image}"
                                }
                            }
                        ]
                    }
                ]

                from openai import OpenAI
                from openai.resources.chat import Completions

                try:
                    for tries in range(number_of_tries):

                        # Instantiate API entry points:
                        client = OpenAI()
                        completions = Completions(client)

                        # Send a request to GPT:
                        result = completions.create(model=model_name,
                                                    temperature=temperature,
                                                    messages=prompt_messages,
                                                    max_tokens=max_tokens)

                        # Actual response:
                        response = result.choices[0].message.content
                        aprint(f"Response: '{response}'")

                        # Check if the response is empty:
                        if not response:
                            aprint(f"Response is empty. Trying again...")
                            continue

                        # response in lower case and trimmed of white spaces
                        response_lc = response.lower().strip()

                        # Check if response is too short:
                        if len(response) < 3:
                            aprint(f"Response is empty. Trying again...")
                            continue

                        # if the response contains these words: "sorry" and ("I cannot" or "I can't")  then try again:
                        if ("sorry" in response_lc and (
                                "i cannot" in response_lc or "i can't" in response_lc or 'i am unable' in response_lc)) \
                                or "i cannot assist" in response_lc or "i can't assist" in response_lc or 'i am unable to assist' in response_lc or "I'm sorry" in response_lc:
                            aprint(
                                f"Vision model refuses to assist (response: {response}). Trying again...")
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
