import json
from typing import Callable, List, Optional, Sequence

from pydantic import BaseModel

from litemind.agent.augmentations.augmentation_default import AugmentationDefault
from litemind.agent.augmentations.information.information import Information
from litemind.agent.augmentations.vector_db.base_vector_db import BaseVectorDatabase
from litemind.apis.base_api import BaseApi
from litemind.media.types.media_audio import Audio
from litemind.media.types.media_code import Code
from litemind.media.types.media_document import Document
from litemind.media.types.media_image import Image
from litemind.media.types.media_json import Json
from litemind.media.types.media_object import Object
from litemind.media.types.media_table import Table
from litemind.media.types.media_text import Text
from litemind.media.types.media_video import Video
from litemind.utils.uri_utils import is_uri


class DefaultVectorDatabase(AugmentationDefault, BaseVectorDatabase):
    """
    Abstract class for vector databases that use a default embedding function.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
        embedding_function: Optional[
            Callable[[List[Information]], List[List[float]]]
        ] = None,
        embedding_length: Optional[int] = None,
        api: BaseApi = None,
    ):
        """
        Initialize the Qdrant vector database.

        Parameters
        ----------
        name: str
            The name of the augmentation.
        description: Optional[str]
            A description of the augmentation.
        embedding_function: Optional[Callable[[List[Information]], List[List[float]]]]
            A function that takes a list of informations and returns a list of embedding vectors.
            If not provided, a default embedding function will be used.
        embedding_length: Optional[int]
            The length of the embedding vectors.
        api: BaseApi
            The API to use for embedding. If not provided, uses litemind's default API.

        """
        super().__init__(name=name, description=description)

        # Set the name of the augmentation to the extended class name:
        if name is None:
            name = self.__class__.__name__
        self.name = name

        # If no api is provided, use the default API:
        if api is None:
            try:
                from litemind import CombinedApi

                api = CombinedApi()
            except ValueError:
                raise ValueError(
                    "No API provided and litemind's default API is not available. "
                    "Please provide an API."
                )
        self.api = api

        # Set the embedding function:
        self.embedding_function = embedding_function or self._default_embedding_function

        # Call the embedding function to get the embedding length if not loaded from disk
        if embedding_length is None:
            # Make a simple information that contains some text:
            query_info = Information(Text("sample text"))

            sample_embedding = self.embedding_function([query_info])[0]
            embedding_length = len(sample_embedding)

        # Set the embedding length:
        self.embedding_length = embedding_length

    def _default_embedding_function(
        self, informations: Sequence[Information]
    ) -> Sequence[Sequence[float]]:
        """
        Default embedding function using litemind's API.

        Parameters
        ----------
        informations: Sequence[Information]
            List of informations to embed.

        Returns
        -------
        List[List[float]]
            The embedding vectors.
        """

        # If embedding_length does not exist in object, then we use a default value for the embedding length:
        if not hasattr(self, "embedding_length"):
            self.embedding_length = 768

        # First we check if all the informations are of the same type:
        info_types = {info.type for info in informations}
        if len(info_types) > 1:

            # Initialise the list of embeddings:
            embeddings = []

            # If the informations are of different types we must handle them separately:
            for info in informations:
                # Compute the embedding for the information:
                embeddings_for_info = self._raw_embedding_function([info])

                # Add embedding to the list of embeddings:
                embeddings.append(embeddings_for_info[0])

        else:
            # If all the informations are of the same type, we can proceed normally:
            embeddings = self._raw_embedding_function(informations)

        # Return the embeddings:
        return embeddings

    def _raw_embedding_function(
        self, informations: Sequence[Information]
    ) -> List[List[float]]:

        # Check if list is empty:
        if not informations:
            raise ValueError("No informations provided for embedding.")

        # If the information already has an embedding, we use it:
        if all(info.embedding is not None for info in informations):
            embeddings = [info.embedding for info in informations]
        else:

            if all(info.has_type(Text) for info in informations):

                # If the information is of type text, we use the text embedding function:
                texts = [info.content for info in informations]

                # Embed the texts:
                embeddings = self.api.embed_texts(
                    texts=texts, dimensions=self.embedding_length
                )
            elif all(info.has_type(Code) for info in informations):
                # If the information is of type code, we use the code embedding function:
                codes = [info.content for info in informations]

                # Embed the codes:
                embeddings = self.api.embed_texts(
                    texts=codes, dimensions=self.embedding_length
                )

            elif all(info.has_type(Json) for info in informations):
                # If the information is of type json, we use the json embedding function:
                json_strs = [
                    (
                        json.dumps(info.content)
                        if isinstance(info.content, dict)
                        else info.content
                    )
                    for info in informations
                ]

                # Embed the json:
                embeddings = self.api.embed_texts(
                    texts=json_strs, dimensions=self.embedding_length
                )

            elif all(info.has_type(Object) for info in informations):
                # If the information is of type object, we use the object embedding function:

                # First let's check that the object is a Pydantic BaseModel:
                if any(
                    not isinstance(info.content, BaseModel) for info in informations
                ):
                    raise ValueError(
                        "Information content must be a Pydantic BaseModel for object type Object."
                    )

                # Convert to JSON strings:
                json_strs = [
                    json.dumps(info.content.model_dump(mode="json"))
                    for info in informations
                ]

                # Embed the objects:
                embeddings = self.api.embed_texts(
                    texts=json_strs, dimensions=self.embedding_length
                )

            elif all(info.has_type(Image) for info in informations):
                # If the information is of type image, we use the image embedding function:
                image_uris = [info.content for info in informations]

                # Embed the images:
                embeddings = self.api.embed_images(
                    image_uris=image_uris, dimensions=self.embedding_length
                )
            elif all(info.has_type(Audio) for info in informations):
                # If the information is of type audio, we use the audio embedding function:
                audio_uris = [info.content for info in informations]

                # Embed the audio files:
                embeddings = self.api.embed_audios(
                    audio_uris=audio_uris, dimensions=self.embedding_length
                )
            elif all(info.has_type(Video) for info in informations):
                # If the information is of type video, we use the video embedding function:
                video_uris = [info.content for info in informations]

                # Embed the videos:
                embeddings = self.api.embed_videos(
                    video_uris=video_uris, dimensions=self.embedding_length
                )

            elif all(info.has_type(Document) for info in informations):
                # If the information is of type document, we use the document embedding function:
                document_uris = [info.content for info in informations]

                # Embed the videos:
                embeddings = self.api.embed_documents(
                    document_uris=document_uris, dimensions=self.embedding_length
                )

            elif all(info.has_type(Table) for info in informations):

                # First let's check that the all informations have a URIs:
                if any(not is_uri(info.content) for info in informations):
                    raise ValueError(
                        "All informations content must be a URI for table type."
                    )

                # Convert to Markdown:
                table_markdowns = [info.media.to_markdown() for info in informations]

                # Embed the table:
                embeddings = self.api.embed_texts(
                    texts=table_markdowns, dimensions=self.embedding_length
                )

            else:
                # If the document is of an unknown type, we raise an error:
                raise ValueError(
                    f"Unknown document type or document: {informations[0].type}."
                )

        return embeddings

    def __del__(self):
        """Ensure the connection is closed when the object is garbage collected."""
        try:
            self.close()
        except:
            pass
