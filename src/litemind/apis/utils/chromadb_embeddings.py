from chromadb import EmbeddingFunction
from litemind import OpenAIApi, GeminiApi, OllamaApi, AnthropicApi
from typing import Optional, List
from litemind.apis.model_features import ModelFeatures


class EmbeddingModel(EmbeddingFunction[List[str]]):

    def __init__(self, api: Optional[OpenAIApi] | Optional[GeminiApi] | Optional[OllamaApi] | Optional[AnthropicApi] | None, model_name: Optional[str]=None, dimension=512 ,**kwargs) -> None:
        
        if api is None:
            raise TypeError(
                "You need to define a valid api."
            )
        self._api = api
        if model_name not in api.list_models(ModelFeatures.TextEmbeddings):
                raise ValueError(
                    "The model is not present in the availables model of the api."
                )
        self._model_name = model_name
        # TODO check that max dimension is not reached
        self._dimension=dimension
        self._kwargs = kwargs
        

    def __call__(self, input: List[str]):
        
        embedding = self._api.embed_texts(texts=input, model_name=self._model_name, dimensions=self._dimension, **self._kwargs)
        
        return embedding