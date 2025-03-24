from typing import List, Optional, Sequence, Union

from arbol import aprint, asection
from litemind.agent.agent import Agent
from litemind.agent.messages.message import Message
from litemind.agent.rag.prompts import rag_default_prompt
from litemind.agent.rag.vector_database import ChromaDB
from litemind.agent.tools.toolset import ToolSet
from litemind.apis.base_api import BaseApi, ModelFeatures


class RagAgent(Agent):

    def __init__(
        self,
        api: BaseApi,
        chroma_vectorestore=ChromaDB,
        model_name: Optional[str] = None,
        model_features: Optional[
            Union[str, List[str], ModelFeatures, Sequence[ModelFeatures]]
        ] = None,
        temperature: float = 0.0,
        toolset: Optional[ToolSet] = None,
        name: str = "RagAgent",
        **kwargs,
    ):
        """
        Initialize the ReAct agent.

        Parameters
        ----------
        api: BaseApi
            The API to use for generating responses.
        chroma_vectorestore: ChromaDB
            The Chroma vectorestore containing the relevant data.
        model_name: Optional[str]
            The model to use for generating responses.
        model_features: Optional[Union[str, List[str], ModelFeatures, Sequence[ModelFeatures]]
            The features to require for a model. If features are provided, the model will be selected based on the features.
            A ValueError will be raised if a model has also been specified as this is mutually exclusive.
            Minimal required features are text generation and tools and are added automatically if not provided.
        temperature: float
            The temperature to use for response generation.
        toolset: Optional[ToolSet]
            The toolset to use for response generation.
        name: str
            The name of the agent.
        chroma_vectorestore: Optional[ChromaDB]
            The maximum number of reasoning steps to take.
        kwargs: dict
            Additional keyword arguments to pass to the agent.

        """

        # Initialize the agent:
        super().__init__(
            api=api,
            model_name=model_name,
            model_features=model_features,
            temperature=temperature,
            toolset=toolset,
            name=name,
            **kwargs,
        )

        # get the RAG prompt
        self._prompt = rag_default_prompt
        # initialize Chromadb vectorestore
        self._chroma_vectorestore = chroma_vectorestore

        # get model and Feature name from chroma and embedding function
        self.chroma_collection = self._chroma_vectorestore.get_collection(
            name=self._chroma_vectorestore.get_collection_name()
        )

    def _get_rag_prompt(self) -> str:
        """Get the RAG prompt instruction"""
        return self._prompt

    def _add_context_to_prompt(self, prompt: str, context: str) -> str:
        """Join the context to the prompt"""
        return prompt + context

    def query_message(self, query: str):
        """Extract the relevant chunks from the vectorestore"""
        # transform the query into a vector
        query_embedding = self._chroma_vectorestore.embedding_function(input=[query])
        # extract vector
        query_vector = query_embedding[0]

        # filter out relevant documents
        responses = self.chroma_collection.query(query_embeddings=query_vector)

        # Extract relevant chuncks
        relevant_chuncks = [
            doc for sublist in responses["documents"] for doc in sublist
        ]

        print("getting relevant information")

        return relevant_chuncks

    def _create_system_message(self, response: str) -> Message:
        """Create the prompt with the context"""
        relevant_chunks = self.query_message(query=response)
        # join all relevant chunks togheter
        context = "\n\n".join(relevant_chunks)

        prompt = self._add_context_to_prompt(prompt=self._prompt, context=context)

        return Message(role="system", text=prompt)

    def __call__(self, *args, **kwargs) -> Message:
        """
        Process a query using Rag Agent
        """

        # Prepare call by extracting conversation, message, and text:
        self._prepare_call(*args, **kwargs)

        # Get last message in conversation
        last_message = self.conversation.get_last_message()

        # check if user custom input
        if "prompt" in kwargs:
            self._prompt = kwargs["prompt"] + "\n\n" + "### BEGINNING OF CHUNKS" + "\n"

        with asection("Calling Rag Agent"):

            with asection("API and model:"):
                aprint(f"API: {self.api.__class__.__name__}")
                aprint(f"Model: {self.model}")

            with asection("Available tools"):
                if self.toolset:
                    for tool in self.toolset:
                        aprint(tool.pretty_string())
                else:
                    aprint("No tools available")

            with asection("Last message in conversation:"):
                aprint(last_message)

            # create the context
            # use the query inserted by the user
            question = "".join(args)

            # create new Message
            message_system = self._create_system_message(response=question)

            message_user = Message(role="user", text=question)
            # create List of Messages for the text generation
            message_question = [message_system, message_user]

            # Call the OpenAI API:
            response = self.api.generate_text(
                model_name=self.model,
                messages=message_question,
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
