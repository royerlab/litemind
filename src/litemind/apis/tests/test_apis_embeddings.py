from pprint import pprint

import pytest

from litemind.apis.base_api import ModelFeatures
from litemind.apis.tests.base_test import BaseTest, API_IMPLEMENTATIONS


@pytest.mark.parametrize("ApiClass", API_IMPLEMENTATIONS)
class TestBaseApiImplementations(BaseTest):
    """
    A tests suite that runs the same tests on each ApiClass
    implementing the abstract BaseApi interface.
    """

    def test_text_embedding(self, ApiClass):
        """
        Test that text_embedding() returns a valid list of floats.
        """
        api_instance = ApiClass()

        embedding_model_name = api_instance.get_best_model(
            ModelFeatures.TextEmbeddings)

        # Skip tests if the model does not support embeddings:
        if not embedding_model_name or not api_instance.has_model_support_for(
                model_name=embedding_model_name,
                features=ModelFeatures.TextEmbeddings):
            pytest.skip(
                f"{ApiClass.__name__} does not support embeddings. Skipping tests.")

        texts = ["Hello, world!", "Testing embeddings."]
        embeddings = api_instance.embed_texts(texts=texts,
                                              model_name=embedding_model_name,
                                              dimensions=512)

        # Check that the returned embeddings are a list of length 2:
        assert isinstance(embeddings, list), "The embeddings should be a list."
        assert len(
            embeddings) == 2, "The embeddings should be a list of length 2."

        for embedding in embeddings:
            print(len(embedding))

        # Check that each embeddings are lists of floats:
        for embedding in embeddings:
            assert isinstance(embedding,
                              list), "Each embedding should be a list."
            assert len(
                embedding) == 512, "Each embedding should be of length 512 as reequested."

            for value in embedding:
                assert isinstance(value,
                                  float), "Each value in the embedding should be a float."

    def test_audio_embedding(self, ApiClass):
        """
        Test that text_embedding() returns a valid list of floats.
        """
        api_instance = ApiClass()

        embedding_model_name = api_instance.get_best_model(
            ModelFeatures.TextEmbeddings)

        # Skip tests if the model does not support embeddings:
        if not embedding_model_name or not api_instance.has_model_support_for(
                model_name=embedding_model_name,
                features=ModelFeatures.TextEmbeddings):
            pytest.skip(
                f"{ApiClass.__name__} does not support embeddings. Skipping tests.")

        texts = ["Hello, world!", "Testing embeddings."]
        embeddings = api_instance.embed_texts(texts=texts,
                                              model_name=embedding_model_name,
                                              dimensions=512)

        # Check that the returned embeddings are a list of length 2:
        assert isinstance(embeddings, list), "The embeddings should be a list."
        assert len(
            embeddings) == 2, "The embeddings should be a list of length 2."

        # Check that each embedding are lists of floats:
        for embedding in embeddings:
            assert isinstance(embedding,
                              list), "Each embedding should be a list."
            assert len(
                embedding) == 512, "Each embedding should be of length 512 as reequested."

            for value in embedding:
                assert isinstance(value,
                                  float), "Each value in the embedding should be a float."

    def test_video_embedding(self, ApiClass):
        """
        Test that video_embedding() returns a valid list of floats.
        """
        api_instance = ApiClass()

        embedding_model_name = api_instance.get_best_model(
            ModelFeatures.VideoEmbeddings)

        # Skip tests if the model does not support embeddings:
        if not embedding_model_name or not api_instance.has_model_support_for(
                model_name=embedding_model_name,
                features=ModelFeatures.VideoEmbeddings):
            pytest.skip(
                f"{ApiClass.__name__} does not support video embeddings. Skipping tests.")

        print(f"Embedding model name: {embedding_model_name}")

        video_uri = self._get_local_test_video_uri('flying.mp4')
        embeddings = api_instance.embed_videos(video_uris=[video_uri],
                                               model_name=embedding_model_name,
                                               dimensions=512)

        pprint(embeddings)

        # Check that the returned embeddings are a list of floats:
        assert isinstance(embeddings, list), "The embeddings should be a list."

        # Check that the embeddings are a list of length 1:
        assert len(
            embeddings) == 1, "The embeddings should be a list of length 1."

        # Check that each embedding is a list of floats of length 512:
        assert len(
            embeddings[
                0]) == 512, "The embeddings should be a list of length 512."

        # Check that the embedding is a list of floats:
        for value in embeddings[0]:
            assert isinstance(value,
                              float), "Each value in the embedding should be a float."
