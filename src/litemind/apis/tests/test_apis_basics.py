import pytest
from arbol import aprint

from litemind.apis.anthropic.anthropic_api import AnthropicApi
from litemind.apis.base_api import ModelFeatures
from litemind.apis.google.google_api import GeminiApi
from litemind.apis.ollama.ollama_api import OllamaApi
from litemind.apis.openai.openai_api import OpenAIApi
from litemind.apis.tests.base_test import BaseTest

# Put all your implementations in this list:
API_IMPLEMENTATIONS = [
    OpenAIApi,
    OllamaApi,
    AnthropicApi,
    GeminiApi
]


@pytest.mark.parametrize("ApiClass", API_IMPLEMENTATIONS)
class TestBaseApiImplementations(BaseTest):
    """
    A tests suite that runs the same tests on each ApiClass
    implementing the abstract BaseApi interface.
    """

    def test_availability_and_credentials(self, ApiClass):
        """
        Test that check_api_key() returns True (or behaves appropriately)
        for each implementation.
        """
        api_instance = ApiClass()
        assert api_instance.check_availability_and_credentials() is True, (
            f"{ApiClass.__name__}.check_api_key() should be True!"
        )

    def test_model_list(self, ApiClass):
        """
        Test that model_list() returns a non-empty list.
        """
        api_instance = ApiClass()
        models = api_instance.model_list()
        for model in models:
            aprint(model)
        assert isinstance(models,
                          list), f"{ApiClass.__name__}.model_list() should return a list!"
        assert len(
            models) > 0, f"{ApiClass.__name__}.model_list() should not be empty!"

    def test_has_has_model_support_for(self, ApiClass):

        api_instance = ApiClass()

        for feature in ModelFeatures:

            print(f"Checking support for feature: {feature} in {ApiClass}")

            # Get a model that does not necessarily support feature:
            default_model_name = api_instance.get_best_model()

            print('\n' + default_model_name)

            # Check if the model supports the feature:
            support = api_instance.has_model_support_for(
                model_name=default_model_name, features=feature)

            # We don't assert True or False globally, because some implementations
            # might support it, others might not. Just ensure it's a boolean.
            assert isinstance(support, bool), (
                f"{ApiClass.__name__}.has_model_support_for({feature}) should return a bool."
            )

            # Then we get a model that does necessarily support feature:
            default_model_name = api_instance.get_best_model(features=feature)

            # if a model is returned, check if it supports feature:
            if default_model_name:
                assert api_instance.has_model_support_for(
                    model_name=default_model_name, features=feature)
            else:
                aprint(f"No model in {ApiClass} supports feature: {feature}")

            # Let's list models that support feature:
            models = api_instance.model_list()
            for model in models:
                if api_instance.has_model_support_for(model_name=model,
                                                      features=feature):
                    aprint(model)

            print(
                f"Checked support for feature: {feature} in {ApiClass}: all good!")

    def test_get_model_features(self, ApiClass):
        """
        Test that get_model_features() returns a non-empty list of features for all API models.
        """

        api_instance = ApiClass()
        models = api_instance.model_list()
        for model in models:
            features = api_instance.get_model_features(model_name=model)

            print(f"\nModel: {model} Features: {features}")

            assert isinstance(features, list), (
                f"{ApiClass.__name__}.get_model_features() should return a list!"
            )
            assert len(features) > 0, (
                f"{ApiClass.__name__}.get_model_features() should not be empty!"
            )
            for feature in features:
                assert isinstance(feature, ModelFeatures), (
                    f"{ApiClass.__name__}.get_model_features() should return a list of ModelFeatures!"
                )

    def test_max_num_input_token(self, ApiClass):
        """
        Test that max_num_input_token returns a valid integer.
        """

        api_instance = ApiClass()
        default_model_name = api_instance.get_best_model()

        print('\n' + default_model_name)

        max_input_tokens = api_instance.max_num_input_tokens(
            model_name=default_model_name)

        print(f"\nMax input tokens: {max_input_tokens}")

        assert isinstance(max_input_tokens, int), (
            f"{ApiClass.__name__}.max_num_input_token() should return an int!"
        )
        assert max_input_tokens > 4096, (
            f"{ApiClass.__name__}.max_num_input_token() should be a positive integer above 4096!"
        )

    def test_max_num_output_tokens(self, ApiClass):
        """
        Test that max_num_output_tokens() returns a valid integer.
        """
        api_instance = ApiClass()
        default_model_name = api_instance.get_best_model()

        print('\n' + default_model_name)

        max_output_tokens = api_instance.max_num_output_tokens(
            model_name=default_model_name)

        print(f"\nMax output tokens: {max_output_tokens}")

        assert isinstance(max_output_tokens, int), (
            f"{ApiClass.__name__}.max_num_output_tokens() should return an int!"
        )
        assert max_output_tokens > 0, (
            f"{ApiClass.__name__}.max_num_output_tokens() should be a positive integer!"
        )
