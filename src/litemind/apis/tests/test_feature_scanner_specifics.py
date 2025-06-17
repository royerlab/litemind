from typing import Type

import pytest
from arbol import aprint, asection

from litemind import AnthropicApi, GeminiApi, OpenAIApi
from litemind.apis.base_api import BaseApi, ModelFeatures
from litemind.apis.feature_scanner import ModelFeatureScanner


@pytest.fixture
def scanner():
    return ModelFeatureScanner(print_exception_stacktraces=True)


def test_scan_openai_api_check_specifics(scanner):
    # Printout list o models available in OpenAI API:
    with asection("Available models in OpenAI API:"):
        for model in OpenAIApi().list_models():
            aprint(f"- {model}")

    # Query supported features for a specific model
    model_feature_map = {
        "gpt-4.1": [
            ModelFeatures.TextGeneration,
            ModelFeatures.StructuredTextGeneration,
            ModelFeatures.Image,
            ModelFeatures.Tools,
            ModelFeatures.WebSearchTool,
            ModelFeatures.MCPTool,
        ],
        "gpt-4o": [
            ModelFeatures.TextGeneration,
            ModelFeatures.StructuredTextGeneration,
            ModelFeatures.Image,
            ModelFeatures.Tools,
        ],
        "o1-medium": [
            ModelFeatures.TextGeneration,
            ModelFeatures.StructuredTextGeneration,
            ModelFeatures.Image,
            ModelFeatures.Tools,
        ],
        "o3-medium": [
            ModelFeatures.TextGeneration,
            ModelFeatures.StructuredTextGeneration,
            ModelFeatures.Image,
            ModelFeatures.Tools,
        ],
        # "gpt-4.5-preview": [
        #     ModelFeatures.TextGeneration,
        #     ModelFeatures.StructuredTextGeneration,
        #     ModelFeatures.Image,
        #     ModelFeatures.Tools,
        # ],
    }

    _check_features(scanner, OpenAIApi, model_feature_map)


def test_scan_anthropic_api_check_specifics(scanner):
    # Printout list o models available in Anthropic API:
    with asection("Available models in Anthropic API:"):
        for model in AnthropicApi().list_models():
            aprint(f"- {model}")

    # Query supported features for specific models
    model_feature_map = {
        "claude-opus-4-20250514": [
            ModelFeatures.WebSearchTool,
            ModelFeatures.MCPTool,
            ModelFeatures.TextGeneration,
            ModelFeatures.StructuredTextGeneration,
            ModelFeatures.Image,
            ModelFeatures.Tools,
        ],
        "claude-3-5-sonnet-20240620": [
            ModelFeatures.TextGeneration,
            ModelFeatures.StructuredTextGeneration,
            ModelFeatures.Image,
            ModelFeatures.Tools,
        ],
        "claude-3-7-sonnet-20250219-thinking-low": [
            ModelFeatures.TextGeneration,
            ModelFeatures.Thinking,
            ModelFeatures.StructuredTextGeneration,
            ModelFeatures.Image,
            ModelFeatures.Tools,
        ],
        "claude-3-7-sonnet-20250219": [
            ModelFeatures.TextGeneration,
            (ModelFeatures.Thinking,),  # this means 'not thinking',
            ModelFeatures.StructuredTextGeneration,
            ModelFeatures.Image,
            ModelFeatures.Tools,
        ],
    }

    _check_features(scanner, AnthropicApi, model_feature_map)


def test_scan_gemini_api_check_specifics(scanner):
    # Printout list o models available in Gemini API:
    with asection("Available models in Gemini API:"):
        for model in GeminiApi().list_models():
            aprint(f"- {model}")

    # Query supported features for specific models
    model_feature_map = {
        "models/gemini-1.5-pro": [
            ModelFeatures.TextGeneration,
            ModelFeatures.StructuredTextGeneration,
            ModelFeatures.Image,
            ModelFeatures.Tools,
        ],
        "models/gemini-1.5-flash": [
            ModelFeatures.TextGeneration,
            ModelFeatures.StructuredTextGeneration,
            ModelFeatures.Image,
        ],
    }

    _check_features(scanner, GeminiApi, model_feature_map)


def _check_features(scanner, api: Type[BaseApi], model_feature_map):
    for model_name, expected_features in model_feature_map.items():
        # Check if the model is available
        if model_name not in api().list_models():
            pytest.skip(f"Model {model_name} not available in OpenAI API.")

        # Check for expected features:
        for feature in expected_features:

            # Check if feature is a singleton tuple:
            if isinstance(feature, tuple) and len(feature) == 1:
                feature = feature[0]
                result = scanner.test_feature(api, model_name, feature)
                aprint(f"Result for {model_name} with feature {feature}: {result}")
                # Assert that the feature is Not in the model's features:
                assert (
                    not result
                ), f"NOT expected feature {feature} for model {model_name} from {api}"
                aprint
            else:
                result = scanner.test_feature(api, model_name, feature)
                aprint(f"Result for {model_name} with feature {feature}: {result}")
                assert (
                    result
                ), f"Expected feature {feature} for model {model_name} from {api}"
