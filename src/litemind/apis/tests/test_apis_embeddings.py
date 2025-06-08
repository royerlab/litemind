from typing import Sequence

import pytest

from litemind import API_IMPLEMENTATIONS
from litemind.apis.base_api import ModelFeatures
from litemind.ressources.media_resources import MediaResources


@pytest.mark.parametrize("api_class", API_IMPLEMENTATIONS)
class TestBaseApiImplementationsEmbeddings(MediaResources):
    """
    A tests suite that runs the same tests on each ApiClass
    implementing the abstract BaseApi interface.
    These tests are for the embedding methods of the API.
    """

    def test_text_embedding(self, api_class):
        """
        Test that text_embedding() returns a valid list of floats.
        """
        api_instance = api_class()

        # Get the best model for text embeddings:
        embedding_model_name = api_instance.get_best_model(ModelFeatures.TextEmbeddings)

        # Skip tests if the model does not support embeddings:
        if embedding_model_name is None:
            pytest.skip(
                f"{api_class.__name__} does not support embeddings. Skipping tests."
            )

        # Two sequences of texts:
        texts = ["Hello, world!", "Testing embeddings."]

        # Get the embeddings:
        embeddings = api_instance.embed_texts(
            texts=texts, model_name=embedding_model_name, dimensions=512
        )

        # Make sure that the embeddings is not None:
        assert embeddings is not None, "The embeddings should not be None."

        # Check that the returned embeddings are a list of length 2:
        assert isinstance(embeddings, Sequence), "The embeddings should be a list."
        assert len(embeddings) == 2, "The embeddings should be a list of length 2."

        # Check that each embedding is a list of floats:
        for embedding in embeddings:

            # Check that each embedding is a sequence of floats, not necessarily a list:
            assert isinstance(embedding, Sequence), "Each embedding should be a list."

            assert (
                len(embedding) == 512
            ), "Each embedding should be of length 512 as requested."

            for value in embedding:
                assert isinstance(
                    value, float
                ), "Each value in the embedding should be a float."

        # turn embeddings into a numpy array:
        import numpy as np

        embeddings = np.array(embeddings)
        print(embeddings.shape)

    def test_audio_embedding(self, api_class):
        """
        Test that embed_audio() returns a valid list of floats.
        """
        api_instance = api_class()

        # Get the best model for audio embeddings:
        embedding_model_name = api_instance.get_best_model(
            ModelFeatures.AudioEmbeddings
        )

        # Skip tests if the model does not support embeddings:
        if embedding_model_name is None:
            pytest.skip(
                f"{api_class.__name__} does not support audio embeddings. Skipping tests."
            )

        # Get the audio URIs:
        audio_1_uri = MediaResources.get_local_test_audio_uri("harvard.wav")
        audio_2_uri = MediaResources.get_local_test_audio_uri("preamble.wav")

        # Get the embeddings:
        embeddings = api_instance.embed_audios(
            audio_uris=[audio_1_uri, audio_2_uri],
            model_name=embedding_model_name,
            dimensions=512,
        )

        # Check that the returned embeddings are a list of length 2:
        assert isinstance(embeddings, list), "The embeddings should be a list."
        assert len(embeddings) == 2, "The embeddings should be a list of length 2."

        # Check that each embedding are lists of floats:
        for embedding in embeddings:
            assert isinstance(embedding, list), "Each embedding should be a list."
            assert (
                len(embedding) == 512
            ), "Each embedding should be of length 512 as reequested."

            for value in embedding:
                assert isinstance(
                    value, float
                ), "Each value in the embedding should be a float."

        # turn embeddings into a numpy array:
        import numpy as np

        embeddings = np.array(embeddings)
        print(embeddings.shape)

    def test_video_embedding(self, api_class):
        """
        Test that video_embedding() returns a valid list of floats.
        """
        api_instance = api_class()

        # Get the best model for video embeddings:
        embedding_model_name = api_instance.get_best_model(
            ModelFeatures.VideoEmbeddings
        )

        # Skip tests if the model does not support embeddings:
        if embedding_model_name is None:
            pytest.skip(
                f"{api_class.__name__} does not support video embeddings. Skipping tests."
            )

        print(f"Embedding model name: {embedding_model_name}")

        # Get the video URIs:
        video_uri_1 = self.get_local_test_video_uri("flying.mp4")
        video_uri_2 = self.get_local_test_video_uri("lunar_park.mp4")

        # Get the embeddings:
        embeddings = api_instance.embed_videos(
            video_uris=[video_uri_1, video_uri_2],
            model_name=embedding_model_name,
            dimensions=512,
        )

        # Check that the returned embeddings are a list of floats:
        assert isinstance(embeddings, list), "The embeddings should be a list."

        # Check that the embeddings are a list of length 1:
        assert len(embeddings) == 2, "The embeddings should be a list of length 1."

        # Check that each embedding is a list of floats of length 512:
        assert (
            len(embeddings[0]) == 512
        ), "The embeddings should be a list of length 512."

        # Check that the embedding is a list of floats:
        for value in embeddings[0]:
            assert isinstance(
                value, float
            ), "Each value in the embedding should be a float."

        # turn embeddings into a numpy array:
        import numpy as np

        embeddings = np.array(embeddings)
        print(embeddings.shape)

    def test_document_embedding(self, api_class):
        """
        Test that document_embedding() returns a valid list of floats.
        """
        api_instance = api_class()

        # Get the best model for video embeddings:
        embedding_model_name = api_instance.get_best_model(
            ModelFeatures.DocumentEmbeddings
        )

        # Skip tests if the model does not support embeddings:
        if embedding_model_name is None:
            pytest.skip(
                f"{api_class.__name__} does not support document embeddings. Skipping tests."
            )

        print(f"Embedding model name: {embedding_model_name}")

        # Get the video URIs:
        document_uri_1 = self.get_local_test_document_uri("intracktive_preprint.pdf")
        document_uri_2 = self.get_local_test_document_uri(
            "low_discrepancy_sequence.pdf"
        )

        # Get the embeddings:
        embeddings = api_instance.embed_documents(
            document_uris=[document_uri_1, document_uri_2],
            model_name=embedding_model_name,
            dimensions=512,
        )

        # Check that the returned embeddings are a list of floats:
        assert isinstance(embeddings, list), "The embeddings should be a list."

        # Check that the embeddings are a list of length 1:
        assert len(embeddings) == 2, "The embeddings should be a list of length 1."

        # Check that each embedding is a list of floats of length 512:
        assert (
            len(embeddings[0]) == 512
        ), "The embeddings should be a list of length 512."

        # Check that the embedding is a list of floats:
        for value in embeddings[0]:
            assert isinstance(
                value, float
            ), "Each value in the embedding should be a float."

        # turn embeddings into a numpy array:
        import numpy as np

        embeddings = np.array(embeddings)
        print(embeddings.shape)
