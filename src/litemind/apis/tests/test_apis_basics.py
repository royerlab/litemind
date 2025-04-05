import pytest
from arbol import aprint

from litemind import API_IMPLEMENTATIONS
from litemind.apis.base_api import ModelFeatures
from litemind.ressources.media_resources import MediaResources


@pytest.mark.parametrize("api_class", API_IMPLEMENTATIONS)
class TestBaseApiImplementationsBasics(MediaResources):
    """
    A tests suite that runs the same tests on each ApiClass
    implementing the abstract BaseApi interface.
    These tests are for the basic methods of the API.
    """

    def test_availability_and_credentials(self, api_class):
        """
        Test that check_api_key() returns True (or behaves appropriately)
        for each implementation.
        """
        api_instance = api_class()
        assert (
            api_instance.check_availability_and_credentials() is True
        ), f"{api_class.__name__}.check_api_key() should be True!"

    def test_model_list(self, api_class):
        """
        Test that model_list() returns a non-empty list.
        """
        api_instance = api_class()

        # List all models:
        models = api_instance.list_models()

        # Print all models:
        for model in models:
            aprint(model)

        # Check that the list is not empty:
        assert isinstance(
            models, list
        ), f"{api_class.__name__}.model_list() should return a list!"

        # Check that the list is not empty:
        assert (
            len(models) > 0
        ), f"{api_class.__name__}.model_list() should not be empty!"

        # Iterate over the model features:
        for feature in ModelFeatures:

            # Get model that supports feature:
            model_that_supports_feature = api_instance.get_best_model(features=feature)

            # if a model is returned, check if it supports the feature requested:
            if model_that_supports_feature:
                assert api_instance.has_model_support_for(
                    model_name=model_that_supports_feature, features=feature
                ), f"{api_class.__name__}.get_best_model({feature}) should return a model ({model_that_supports_feature}) that supports {feature}!"

    def test_has_model_support_for(self, api_class):

        api_instance = api_class()

        # Iterate over the model features:
        for feature in ModelFeatures:

            print(f"Checking support for feature: {feature} in {api_class}")

            # Get a model that does not necessarily support feature:
            default_model_name = api_instance.get_best_model()

            print("\n" + default_model_name)

            # Check if the model supports the feature:
            support = api_instance.has_model_support_for(
                model_name=default_model_name, features=feature
            )

            # We don't assert True or False globally, because some implementations
            # might support it, others might not. Just ensure it's a boolean.
            assert isinstance(
                support, bool
            ), f"{api_class.__name__}.has_model_support_for({feature}) should return a bool."

            # Then we get a model that does necessarily support feature:
            default_model_name = api_instance.get_best_model(features=feature)

            # if a model is returned, check if it supports feature:
            if default_model_name:
                assert api_instance.has_model_support_for(
                    model_name=default_model_name, features=feature
                )
            else:
                aprint(f"No model in {api_class} supports feature: {feature}")

            # Let's list models that support feature:
            models = api_instance.list_models()
            for model in models:
                if api_instance.has_model_support_for(
                    model_name=model, features=feature
                ):
                    aprint(model)

            print(f"Checked support for feature: {feature} in {api_class}: all good!")

    def test_get_model_features(self, api_class):
        """
        Test that get_model_features() returns a non-empty list of features for all API models.
        """

        api_instance = api_class()

        # List all models:
        models = api_instance.list_models()

        # Iterate over the models:
        for model in models:

            # Get model features:
            features = api_instance.get_model_features(model_name=model)

            print(f"\nModel: {model} Features: {features}")

            assert isinstance(
                features, list
            ), f"{api_class.__name__}.get_model_features() should return a list!"
            # assert len(features) > 0, (
            #     f"{api_class.__name__}.get_model_features() should not be empty!"
            # )
            for feature in features:
                assert isinstance(
                    feature, ModelFeatures
                ), f"{api_class.__name__}.get_model_features() should return a list of ModelFeatures!"

    def test_max_num_input_token(self, api_class):
        """
        Test that max_num_input_token returns a valid integer.
        """

        api_instance = api_class()

        # Get the best model for text generation:
        default_model_name = api_instance.get_best_model(
            features=ModelFeatures.TextGeneration
        )

        # if no model supports text generation, we skip the test:
        if not default_model_name:
            pytest.skip(
                f"{api_class.__name__} does not support text generation. Skipping text generation tests."
            )

        print("\n" + default_model_name)

        # Get the max number of input tokens:
        max_input_tokens = api_instance.max_num_input_tokens(
            model_name=default_model_name
        )

        print(f"\nMax input tokens: {max_input_tokens}")

        assert isinstance(
            max_input_tokens, int
        ), f"{api_class.__name__}.max_num_input_token() should return an int!"
        assert (
            max_input_tokens >= 4096
        ), f"{api_class.__name__}.max_num_input_token() should be a positive integer equal or above 4096!"

    def test_max_num_output_tokens(self, api_class):
        """
        Test that max_num_output_tokens() returns a valid integer.
        """
        api_instance = api_class()

        # Get the best model for text generation:
        default_model_name = api_instance.get_best_model(
            features=ModelFeatures.TextGeneration
        )

        # if no model supports text generation, we skip the test:
        if default_model_name is None:
            pytest.skip(
                f"{api_class.__name__} does not support text generation. Skipping text generation tests."
            )

        print("\n" + default_model_name)

        # Get the max number of output tokens:
        max_output_tokens = api_instance.max_num_output_tokens(
            model_name=default_model_name
        )

        print(f"\nMax output tokens: {max_output_tokens}")

        assert isinstance(
            max_output_tokens, int
        ), f"{api_class.__name__}.max_num_output_tokens() should return an int!"
        assert (
            max_output_tokens > 0
        ), f"{api_class.__name__}.max_num_output_tokens() should be a positive integer!"
