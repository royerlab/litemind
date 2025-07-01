from abc import ABC
from typing import List, Optional, Sequence, Union

from arbol import aprint, asection

from litemind.agent.augmentations.augmentation_base import AugmentationBase
from litemind.agent.augmentations.augmentation_set import AugmentationSet
from litemind.agent.augmentations.information.information_base import InformationBase
from litemind.agent.messages.conversation import Conversation
from litemind.agent.messages.message import Message
from litemind.agent.tools.base_tool import BaseTool
from litemind.agent.tools.toolset import ToolSet
from litemind.apis.base_api import BaseApi
from litemind.apis.model_features import ModelFeatures


class Agent:

    def __init__(
        self,
        api: BaseApi,
        name: str = "Agent",
        model_name: Optional[str] = None,
        model_features: Optional[
            Union[str, List[str], ModelFeatures, Sequence[ModelFeatures]]
        ] = None,
        temperature: float = 0.0,
        toolset: Optional[ToolSet] = None,
        augmentation_set: Optional[AugmentationSet] = None,
        augmentation_k: int = 5,
        augmentation_context_position: str = "before_query",
        messages_stdout_enabled: bool = False,
        **kwargs,
    ):
        """
        Create a new agent.

        Parameters
        ----------
        api: BaseApi
            The API to use.
        name: str
            The name of the agent.
        model_name: str
            The name of the model to use.
        model_features: Union[str, List[str], ModelFeatures, Sequence[ModelFeatures]]
            The features to require for a model. If features are provided, the model will be selected based on the features.
            A ValueError will be raised if a model has also been specified as this is mutually exclusive.
            Minimal required features are text generation and tools and are added automatically if not provided.
        temperature: float
            The temperature to use for text generation.
        toolset: ToolSet
            The toolset to use.
        augmentation_set: Optional[AugmentationSet]
            The set of augmentations to use for document retrieval.
        augmentation_k: int
            The number of informations to retrieve from augmentations.
        augmentation_context_position: str
            Where to place the retrieved context. Options are:
            - "before_query": Add context right before the user query
            - "system": Add context as a system message
        messages_stdout_enabled: bool
            If True, messages will be printed to stdout.
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

        model_supports_tools = self.api.has_model_support_for(
            model_name=self.model, features=ModelFeatures.Tools
        )
        if toolset is not None and not model_supports_tools:
            raise ValueError("The model does not support tools")

        # Set the temperature, toolset, and name:
        self.temperature = temperature
        self.toolset = toolset or (ToolSet() if model_supports_tools else None)
        self.name = name
        self._agent_kwargs = kwargs

        # Initialize augmentation parameters
        self.augmentation_set = augmentation_set or AugmentationSet()
        self.augmentation_k = augmentation_k

        # Validate context_position
        valid_positions = ["before_query", "system"]
        if augmentation_context_position not in valid_positions:
            raise ValueError(
                f"Invalid augmentation_context_position: {augmentation_context_position}. Valid options are: {valid_positions}"
            )
        self.augmentation_context_position = augmentation_context_position

        # Initialise conversation:
        self.conversation = Conversation()

        self.messages_stdout_enabled = messages_stdout_enabled

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

    def add_tool(self, tool: BaseTool) -> None:
        """
        Add a tool to the agent's toolset.

        Parameters
        ----------
        tool: BaseTool
            The tool to add.
        """
        if self.toolset is None:
            raise ValueError("The model does not support tools")
        self.toolset.add_tool(tool)

    def remove_tool(self, name: str) -> bool:
        """
        Remove a tool by name from the agent's toolset.

        Parameters
        ----------
        name: str
            The name of the tool to remove.

        Returns
        -------
        bool
            True if the tool was removed, False if not found.
        """
        if self.toolset is None:
            raise ValueError("The model does not support tools")
        return self.toolset.remove_tool(name)

    def clear_tools(self) -> None:
        """Clear all tools from the agent's toolset."""
        if self.toolset is None:
            raise ValueError("The model does not support tools")
        self.toolset = ToolSet()

    def add_augmentation(
        self,
        augmentation: AugmentationBase,
        k: int = 5,
        threshold: float = 0.0,
    ) -> None:
        """
        Add an augmentation to the agent's augmentation set.

        Parameters
        ----------
        augmentation: AugmentationBase
            The augmentation to add.
        k: int
            The number of informations to retrieve from the augmentation.
        threshold: float
            The score threshold_dict for the augmentation. If the score is below this value, the augmentation will not be used.
        """
        self.augmentation_set.add_augmentation(augmentation, k=k, threshold=threshold)

    def remove_augmentation(self, name: str) -> bool:
        """
        Remove an augmentation by name from the agent's augmentation set.

        Parameters
        ----------
        name: str
            The name of the augmentation to remove.

        Returns
        -------
        bool
            True if the augmentation was removed, False if not found.
        """
        return self.augmentation_set.remove_augmentation(name)

    def clear_augmentations(self) -> None:
        """Clear all augmentations from the agent's augmentation set."""
        self.augmentation_set = AugmentationSet()

    def list_augmentations(self) -> List[AugmentationBase]:
        """
        Get all augmentations in the agent's augmentation set.

        Returns
        -------
        List[AugmentationBase]
            The list of augmentations.
        """
        return self.augmentation_set.list_augmentations()

    def _retrieve_relevant_documents(self, query_text: str) -> List[InformationBase]:
        """
        Retrieve relevant informations from the augmentation set.

        Parameters
        ----------
        query_text: str
            The query to retrieve informations for.

        Returns
        -------
        List[Document]
            A list of relevant informations.
        """
        with asection(
            f"Retrieving relevant informations for query: '{query_text[:100]}...'"
        ):
            documents = self.augmentation_set.search_combined(
                query_text, k=self.augmentation_k
            )
            with asection(
                f"Retrieved {len(documents)} informations from augmentation set:"
            ):
                for doc in documents:
                    aprint(f"Document ID: {doc.id}, Score: {doc.score:.4f}")
            return documents

    def _add_context_to_conversation(
        self,
        query_message: Message,
        documents: List[InformationBase],
        max_doc_length: int = 1000,
    ) -> None:
        """
        Add context from informations to the conversation.

        Parameters
        ----------
        query_message: Message
            The user's query message.
        documents: List[Document]
            The informations to add as context.
        """
        if not documents:
            return

        # Add the context based on the specified position
        if self.augmentation_context_position == "before_query":
            # Create a user message with the context
            augmentation_message = Message(role="user")

            # Insert the context message right before the user query
            idx = self.conversation.standard_messages.index(query_message)
            self.conversation.standard_messages.insert(idx, augmentation_message)

        elif self.augmentation_context_position == "system":
            # Add as a system message
            augmentation_message = Message(role="system")
            self.conversation.system_messages.append(augmentation_message)

        else:
            raise ValueError(
                f"Invalid augmentation_context_position: {self.augmentation_context_position}"
            )

        augmentation_message.append_text("Additional context information:\n\n")

        for i, doc in enumerate(documents):

            # prepare the context text
            context_text = ""
            context_text += f"--- Document {i + 1} "
            if doc.score is not None:
                context_text += f"(Relevance: {doc.score:.4f}) "
            if "augmentation" in doc.metadata:
                context_text += f"[Source: {doc.metadata['augmentation']}] "
            context_text += "---\n"

            # Add document context:
            augmentation_message.append_text(context_text)

            # Add document content:
            augmentation_message += doc.to_message_block()

        with asection(
            f"Added {len(documents)} informations to conversation, context message:"
        ):
            aprint(str(augmentation_message))

    def __getitem__(self, item) -> Message:
        return self.conversation[item]

    def __len__(self):
        return len(self.conversation)

    def __iadd__(self, other: Message):
        self.conversation.append(other)
        return self

    def __call__(self, *args, **kwargs) -> List[Message]:

        # Prepare the call by normalising parameters:
        self._prepare_call(*args, **kwargs)

        # Get last message in conversation
        last_message = self.conversation.get_last_message()

        with asection(f"Calling agent: '{self.name}'"):

            with asection("API and model:"):
                aprint(f"API: {self.api.__class__.__name__}")
                aprint(f"Model: {self.model}")

            if self.toolset:
                with asection("Available tools"):
                    if len(self.toolset) > 0:
                        for tool in self.toolset:
                            aprint(tool.pretty_string())
                    else:
                        aprint("No tools available")

            with asection("Available augmentations"):
                augmentations = self.augmentation_set.list_augmentations()
                if augmentations:
                    for aug in augmentations:
                        aprint(f"{aug.name}")
                else:
                    aprint("No augmentations available")

            # Quick report of the conversation (just number of messages):
            if self.conversation:
                with asection(f"Conversation report:"):
                    aprint(
                        f"Number of messages currently in conversation: {len(self.conversation)}"
                    )

            # Print the last message in conversation if enabled
            if self.messages_stdout_enabled:
                with asection("Last message in conversation:"):
                    aprint(last_message)

            # Process augmentations if we have any and there's a query message
            if len(self.augmentation_set) > 0 and last_message:
                query_text = last_message.to_plain_text()

                # Retrieve relevant informations
                relevant_documents = self._retrieve_relevant_documents(query_text)

                # Add context to the conversation
                if relevant_documents:
                    self._add_context_to_conversation(last_message, relevant_documents)

            # Call the API:
            response = self.api.generate_text(
                model_name=self.model,
                messages=self.conversation.get_all_messages(),
                temperature=self.temperature,
                toolset=self.toolset,
                **kwargs,
            )

            # Quick report of the response:
            with asection(f"Response:"):
                if response is None:
                    aprint(
                        "No response received from the API. Probably an error occurred."
                    )
                else:
                    aprint(f"Number of messages in response: {len(response)}")

            # Print response messages to stdout if enabled:
            if self.messages_stdout_enabled:
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

    def __repr__(self):
        return f"{self.name}(model={self.model}, api={self.api.__class__.__name__})"

    def __str__(self):
        return self.__repr__()
