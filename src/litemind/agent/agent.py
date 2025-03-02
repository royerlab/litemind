from typing import List, Optional, Sequence, Union

from arbol import aprint, asection

from litemind.agent.messages.conversation import Conversation
from litemind.agent.messages.message import Message
from litemind.agent.tools.toolset import ToolSet
from litemind.apis.base_api import BaseApi
from litemind.apis.model_features import ModelFeatures


class Agent:

    def __init__(
        self,
        api: BaseApi,
        model_name: Optional[str] = None,
        model_features: Optional[
            Union[str, List[str], ModelFeatures, Sequence[ModelFeatures]]
        ] = None,
        temperature: float = 0.0,
        toolset: Optional[ToolSet] = None,
        name: str = "Agent",
        **kwargs,
    ):
        """
        Create a new agent.

        Parameters
        ----------
        api: BaseApi
            The API to use.
        model_name: str
            The model to use from the provided API. Model must support at least text generation and tools.
        model_features: Union[str, List[str], ModelFeatures, Sequence[ModelFeatures]]
            The features to require for a model. If features are provided, the model will be selected based on the features.
            A ValueError will be raised if a model has also been specified as this is mutually exclusive.
            Minimal required features are text generation and tools and are added automatically if not provided.
        temperature: float
            The temperature to use for text generation.
        toolset: ToolSet
            The toolset to use.
        name: str
            The name of the agent.
        kwargs: Any
            Additional keyword arguments to pass to the API.

        """

        # get key from environmental variables:
        self.api = api

        # Normalise the model features:
        model_features = ModelFeatures.normalise(model_features)

        # If model features are provided, get the best model based on the features:
        if model_features is not None:

            # Make sure that model is not also provided:
            if model_name:
                raise ValueError("You cannot provide both model and model_features")

            # Make sure that text generation and tools are included in the model features:
            if ModelFeatures.TextGeneration not in model_features:
                model_features = list(model_features) + [ModelFeatures.TextGeneration]
            if ModelFeatures.Tools not in model_features:
                model_features = list(model_features) + [ModelFeatures.Tools]

            # Get the best model based on the features:
            self.model = api.get_best_model(features=model_features)
        else:
            # Use the provided model or get the best model based on text generation and tools:
            self.model = model_name or api.get_best_model(
                features=[ModelFeatures.TextGeneration, ModelFeatures.Tools]
            )

        # Let's make sure the model is not None,
        if self.model is None:
            raise ValueError("No model was found with the required features")

        # ... and has the minimal required features:
        if not self.api.has_model_support_for(
            model_name=self.model, features=ModelFeatures.TextGeneration
        ):
            raise ValueError("The model does not support text generation")
        if not self.api.has_model_support_for(
            model_name=self.model, features=ModelFeatures.Tools
        ):
            raise ValueError("The model does not support tools")

        # Set the temperature, toolset, and name:
        self.temperature = temperature
        self.toolset = toolset
        self.name = name
        self._agent_kwargs = kwargs

        # Initialise conversation:
        self.conversation = Conversation()

    def append_system_message(self, message: Union[str, Message]):
        """
        Set a system message for the agent.

        Parameters
        ----------
        message: Union[str,Message]
            The system message to set.

        """

        # If a string is provided, convert it to a message:
        if isinstance(message, str):
            message = Message(role="system", text=message)

        # Append the message to the conversation:
        self.conversation.system_messages.append(message)

    def __getitem__(self, item) -> Message:
        return self.conversation[item]

    def __len__(self):
        return len(self.conversation)

    def __iadd__(self, other: Message):
        self.conversation.append(other)
        return self

    def __call__(self, *args, **kwargs) -> List[Message]:

        # Prepare the call by normalising parameters:
        messages = self._prepare_call(*args, **kwargs)

        # Get last message in conversation
        last_message = self.conversation.get_last_message()

        with asection(f"Calling agent: '{self.name}'"):

            with asection("API and model:"):
                aprint(f"API: {self.api.__class__.__name__}")
                aprint(f"Model: {self.model}")

            with asection("Available tools"):
                if self.toolset:
                    for tool in self.toolset:
                        aprint(tool.pretty_string())

            with asection("Last message in conversation:"):
                aprint(last_message)

            # Call the OpenAI API:
            response = self.api.generate_text(
                model_name=self.model,
                messages=self.conversation.get_all_messages(),
                temperature=self.temperature,
                toolset=self.toolset,
                **kwargs,
            )

            with asection("Reponse:"):
                for message in response:
                    aprint(message)

        # Append response messages to conversation:
        self.conversation.extend(response)

        # Return response
        return response

    def _prepare_call(self, *args, **kwargs) -> List[Message]:

        messages: List[Message] = []

        # if a role is provided as a keyword argument, append it:
        role = kwargs["role"] if "role" in kwargs else "user"
        # if conversation is provided as a positional argument, append it:
        if args:
            conversation = args[0]
            if isinstance(conversation, Conversation):
                self.conversation += conversation
                messages.extend(conversation.get_all_messages())

        # if conversation is provided as a keyword argument, append it:
        if "conversation" in kwargs:
            conversation = kwargs["conversation"]
            # remove conversation from kwargs:
            del kwargs["conversation"]
            if isinstance(conversation, Conversation):
                self.conversation += conversation
                messages.extend(conversation.get_all_messages())
            else:
                raise ValueError("conversation must be a Conversation object")

        # if a message is provided as a positional argument, append it:
        if args:
            for message in args:
                if isinstance(message, Message):
                    self.conversation.append(message)
                    messages.append(message)

        # if a message is provided as a keyword argument, append it:
        if "message" in kwargs:
            message = kwargs["message"]
            # remove message from kwargs:
            del kwargs["message"]
            if isinstance(message, Message):
                self.conversation.append(message)
                messages.append(message)
            else:
                raise ValueError("message must be a Message object")

        # if text is provided as a positional argument, append it:
        if args:
            for text in args:
                if isinstance(text, str):
                    message = Message(role=role)
                    message.append_text(text)
                    self.conversation.append(message)
                    messages.append(message)

        # if text is provided as a keyword argument, append it:
        if "text" in kwargs:
            text = kwargs["text"]
            # remove text from kwargs:
            del kwargs["text"]
            if isinstance(text, str):
                message = Message(role=role, text=text)
                self.conversation.append(message)
                messages.append(message)
            else:
                raise ValueError("text must be a string")

        return messages
