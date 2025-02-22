from typing import Optional

from arbol import asection, aprint

from litemind.agent.conversation import Conversation
from litemind.agent.message import Message
from litemind.agent.tools.toolset import ToolSet
from litemind.apis.base_api import BaseApi


class Agent:

    def __init__(self,
                 api: BaseApi,
                 model: Optional[str] = None,
                 temperature: float = 0.0,
                 toolset: Optional[ToolSet] = None,
                 name: str = "Agent",
                 **kwargs):

        # get key from environmental variables:
        self.api = api
        self.model = model or api.get_best_model()
        self.temperature = temperature
        self.toolset = toolset
        self.name = name
        self._agent_kwargs = kwargs

        # Initialise conversation:
        self.conversation = Conversation()

    def __getitem__(self, item) -> Message:
        return self.conversation[item]

    def __len__(self):
        return len(self.conversation)

    def __iadd__(self, other: Message):
        self.conversation.append(other)
        return self

    def __call__(self,
                 *args,
                 **kwargs) -> Message:

        self._prepare_call(args, kwargs)

        # Get the last message in conversation:
        last_message = self.conversation[-1]

        with asection(f"Calling agent: '{self.name}'"):
            with asection("With message"):
                aprint(last_message)

            # Call the OpenAI API:
            response = self.api.generate_text(
                model_name=self.model,
                messages=self.conversation.get_all_messages(),
                temperature=self.temperature,
                toolset=self.toolset,
                **kwargs
            )

            with asection("Reponse:"):
                aprint(response)

        # Append assistant message to messages:
        self.conversation.append(response)

        # Return response
        return response

    def _prepare_call(self, args, kwargs):

        # if a role is provided as a keyword argument, append it:
        role = kwargs['role'] if 'role' in kwargs else 'user'
        # if conversation is provided as a positional argument, append it:
        if args:
            conversation = args[0]
            if isinstance(conversation, Conversation):
                self.conversation += conversation

        # if conversation is provided as a keyword argument, append it:
        if 'conversation' in kwargs:
            conversation = kwargs['conversation']
            # remove conversation from kwargs:
            del kwargs['conversation']
            if isinstance(conversation, Conversation):
                self.conversation += conversation
            else:
                raise ValueError("conversation must be a Conversation object")

        # if a message is provided as a positional argument, append it:
        if args:
            message = args[0]
            if isinstance(message, Message):
                self.conversation.append(message)

        # if a message is provided as a keyword argument, append it:
        if 'message' in kwargs:
            message = kwargs['message']
            # remove message from kwargs:
            del kwargs['message']
            if isinstance(message, Message):
                self.conversation.append(message)
            else:
                raise ValueError("message must be a Message object")

        # if text is provided as a positional argument, append it:
        if args:
            text = args[0]
            if isinstance(text, str):
                message = Message(role=role)
                message.append_text(text)
                self.conversation.append(message)

        # if text is provided as a keyword argument, append it:
        if 'text' in kwargs:
            text = kwargs['text']
            # remove text from kwargs:
            del kwargs['text']
            if isinstance(text, str):
                message = Message(role=role, text=text)

                self.conversation.append(message)
            else:
                raise ValueError("text must be a string")

