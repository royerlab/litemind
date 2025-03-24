from typing import List, Optional, Union

from chromadb import EmbeddingFunction
from litemind import AnthropicApi, GeminiApi, OllamaApi, OpenAIApi
from litemind.apis.model_features import ModelFeatures


class EmbeddingModel(EmbeddingFunction[List[str]]):

    def __init__(
        self,
        api: Union[OpenAIApi, GeminiApi, OllamaApi, AnthropicApi],
        model_name: Optional[str] = None,
        dimension=512,
        **kwargs,
    ) -> None:
        """
        Initialize the embedding function.

        Parameters
        ----------
        api: Union[OpenAIApi,GeminiApi,OllamaApi,AnthropicApi]
            A valid BaseAPI available for the user. If an api key is not detected
            it wont be possible to use the API.
        model_name: Optional[str] = None
            The name of the model for embedding.
        dimension: int | 512
            The dimension of the vector for the embedding. Depending of
            the model used, the vector has different max size.
        kwargs
            Other parameters given to the embedding function.

        Raises
        ------
        ValueError
            If the model name is not present into the available list of text embedding
            of the API.
        """
        if api is None:
            raise TypeError("You need to define a valid api.")
        self._api = api
        if model_name not in api.list_models(ModelFeatures.TextEmbeddings):
            raise ValueError(
                "The model is not present in the availables model of the api."
            )
        self._model_name = model_name
        # TODO check that max dimension is not reached
        self._dimension = dimension
        self._kwargs = kwargs

    def __call__(self, input: List[str]):
        """
        Process a List of chunks using the API's embedding function.

        Parameters
        ----------
        input: List[str]
            A list containing the chunked text.

        Returns
        -------
        A Sequence of vectors for each text in the List.
        """
        embedding = self._api.embed_texts(
            texts=input,
            model_name=self._model_name,
            dimensions=self._dimension,
            **self._kwargs,
        )

        return embedding
