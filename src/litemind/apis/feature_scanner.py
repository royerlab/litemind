import os
import traceback
from datetime import datetime
from typing import Dict, List, Optional, Type, Union

import yaml
from arbol import Arbol, acapture, aprint, asection
from pydantic import BaseModel

from litemind.agent.messages.message import Message
from litemind.agent.tools.toolset import ToolSet
from litemind.apis.base_api import BaseApi
from litemind.apis.model_features import ModelFeatures
from litemind.apis.providers.google.utils.is_reasoning import is_gemini_reasoning_model
from litemind.apis.providers.openai.utils.is_reasoning import is_openai_reasoning_model
from litemind.ressources.media_resources import MediaResources

# Instantiate a global ModelFeatureScanner instance for use in the project:
__model_feature_scanner = None


def get_default_model_feature_scanner() -> "ModelFeatureScanner":
    """
    Get the global ModelFeatureScanner instance.

    Returns
    -------
    ModelFeatureScanner
        The global ModelFeatureScanner instance.
    """
    global __model_feature_scanner
    if __model_feature_scanner is None:
        __model_feature_scanner = ModelFeatureScanner()
        # load from the scan_result folder located in the same folder as tis file:
        __model_feature_scanner.load_results(
            os.path.join(os.path.dirname(__file__), "scan_results")
        )
    return __model_feature_scanner


class ModelFeatureScanner(MediaResources):
    """
    A class for scanning and detecting supported features in different model providers.

    This class tests each model with minimal code to determine which features
    it supports, storing and retrieving this information in YAML files.
    """

    def __init__(
        self, output_dir: str = None, print_exception_stacktraces: bool = False
    ):
        """
        Initialize the scanner.

        Parameters
        ----------
        output_dir: str, optional
            Directory to store scan results. Defaults to current directory.
        print_exception_stacktraces: bool, optional
            Whether to print exception stack traces during scanning. Defaults to False.
        """

        # Set the output directory for scan results
        self.output_dir = output_dir or os.path.join(
            os.path.dirname(__file__), "scan_results"
        )

        # Set whether to print exception stack traces
        self.print_exception_stacktraces = print_exception_stacktraces

        # Ensure the output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        # Dictionary to store scan results
        self.scan_results: Dict[type, Dict[str, Dict[ModelFeatures, bool]]] = {}

    @staticmethod
    def _snake_case(name: str) -> str:
        import re

        s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
        return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()

    @staticmethod
    def _api_class_to_name(api_class: type) -> str:
        return api_class.__name__

    @staticmethod
    def _api_name_to_class(api_name: str) -> Optional[Type[BaseApi]]:
        import sys

        # BaseApi is imported at the top of the file

        priority_module_substrings = [
            "litemind.apis.providers",
            "litemind.apis.tests",
            __name__,
        ]  # __name__ for local (e.g. test) definitions

        candidate_modules = []
        other_modules = []

        # Safely iterate over sys.modules.items()
        # Some modules in sys.modules might be None or lack a __name__
        for mod_name, module in list(
            sys.modules.items()
        ):  # Use list() for safe iteration if modules change
            if module is None:
                continue

            module_name_str = getattr(module, "__name__", None)
            if module_name_str is None:  # Skip modules without a proper name
                continue

            if any(sub in module_name_str for sub in priority_module_substrings):
                candidate_modules.append(module)
            else:
                other_modules.append(module)

        for module in candidate_modules + other_modules:
            if hasattr(module, api_name):
                potential_class = getattr(module, api_name)
                if isinstance(potential_class, type) and issubclass(
                    potential_class, BaseApi
                ):
                    return potential_class

        aprint(
            f"Warning: API class '{api_name}' not found in loaded modules or not a subclass of BaseApi."
        )
        return None

    @staticmethod
    def _feature_to_name(feature: ModelFeatures) -> str:
        return feature.name

    @staticmethod
    def _name_to_feature(feature_name: str) -> ModelFeatures:
        return ModelFeatures[feature_name]

    def get_supported_features(
        self, api_class: type, model_name: str
    ) -> List[ModelFeatures]:
        """
        Return a list of supported features for a given API class and model.
        """
        if (
            api_class not in self.scan_results
            or model_name not in self.scan_results[api_class]
        ):
            return []
        return [
            feature
            for feature, supported in self.scan_results[api_class][model_name].items()
            if supported
        ]

    def supports_feature(
        self, api_class: Type[BaseApi], model_name: str, feature: ModelFeatures
    ) -> bool:
        """
        Return True if the given model of the API class supports the feature.
        """
        if (
            api_class not in self.scan_results
            or model_name not in self.scan_results[api_class]
        ):
            return False
        return self.scan_results[api_class][model_name].get(feature, False)

    def supports_any_feature(self, api_class: type, model_name: str) -> bool:
        """
        Return True if the given model of the API class supports any feature.
        """

        # Check if the API class and model name exist in the scan results:
        if (
            api_class not in self.scan_results
            or model_name not in self.scan_results[api_class]
        ):
            return False

        # Check if any feature is supported for the given model:
        return any(self.scan_results[api_class][model_name].values())

    def scan_apis(
        self,
        api_classes: List[Type[BaseApi]],
        model_names: Optional[List[str]] = None,
        models_per_api: int = None,
    ) -> Dict[type, Dict[str, Dict[ModelFeatures, bool]]]:
        """
        Scan multiple API providers for feature support.

        Parameters
        ----------
        api_classes: List[Type[BaseApi]]
            List of API classes to scan
        model_names: List[str], optional
            List of specific model names to scan. If None, all models will be scanned (up to models_per_api).
        models_per_api: int, optional
            Maximum number of models to scan per API. If None, scan all models.

        Returns
        -------
        Dict[type, Dict[str, Dict[ModelFeatures, bool]]]
            Nested dictionary with scan results in format:
            {api_class: {model_name: {feature: supported}}}
        """
        with asection(
            f"Scanning APIs: {', '.join([api_class.__name__ for api_class in api_classes])}"
        ):
            aprint(f"Scanning {len(api_classes)} APIs...")

            for api_class in api_classes:
                try:
                    api_instance = api_class(allow_media_conversions=False)

                    # Check if the API is available
                    if not api_instance.check_availability_and_credentials():
                        aprint(f"API {api_class.__name__} is not available. Skipping.")
                        continue

                    # Get the list of models
                    models = api_instance.list_models()

                    # Intersect with provided model names if any:
                    if model_names:
                        models = list(set(models) & set(model_names))
                        if not models:
                            aprint(
                                f"No matching models found for {api_class.__name__} with provided names."
                            )
                            continue

                        # Warn if some model in model_names are not found in the API:
                        missing_models = set(model_names) - set(models)
                        if missing_models:
                            aprint(
                                f"Warning: The following models were not found in {api_class.__name__}: {', '.join(missing_models)}"
                            )

                    # Limit the number of models if specified
                    if models_per_api and len(models) > models_per_api:
                        aprint(
                            f"Limiting scan to {models_per_api} models for {api_class.__name__}"
                        )
                        models = models[:models_per_api]

                    aprint(f"Scanning {len(models)} models from {api_class.__name__}:")

                    # Scan each model
                    self.scan_results[api_class] = {}
                    for i, model_name in enumerate(models, start=1):
                        self.scan_results[api_class][model_name] = (
                            self.scan_model_features(api_instance, model_name)
                        )
                        aprint(f"Done scanning model {i}/{len(models)}: {model_name}")

                except Exception as e:
                    aprint(f"Error scanning {api_class.__name__}: {e}")
                    traceback.print_exc()

        return self.scan_results

    def scan_model_features(
        self, api: BaseApi, model_name: str
    ) -> dict[ModelFeatures, bool]:
        """
        Scan a specific model for all supported features.

        Parameters
        ----------
        api: BaseApi
            API instance to use for scanning
        model_name: str
            Name of the model to scan

        Returns
        -------
        Dict[ModelFeatures, bool]
            Dictionary mapping feature to boolean indicating support
        """
        results: Dict[ModelFeatures, bool] = {}

        with asection(
            f"Scanning model: {model_name} from API: {api.__class__.__name__}"
        ):

            # Test each feature in the ModelFeatures enum
            for feature in ModelFeatures:
                feature_name = feature.name

                # Skip Conversion features as they are not directly testable:
                if "conversion" in feature_name.lower():
                    continue

                method_name = f"test_{self._snake_case(feature_name)}"
                # aprint(f"    Testing feature: {feature_name}")

                # Use the appropriate test method for each feature
                test_method = getattr(self, method_name, None)

                if test_method:
                    try:
                        arbol_state = Arbol.enable_output
                        Arbol.enable_output = False
                        with acapture():
                            result = test_method(api, model_name)
                        Arbol.enable_output = arbol_state
                        results[feature] = result
                        aprint(f"      {'✅' if result else '❌'} {feature_name}")
                    except Exception as e:
                        Arbol.enable_output = arbol_state
                        aprint(f"      ❌ {feature_name} (Error: {e})")
                        results[feature] = False

                        if self.print_exception_stacktraces:
                            # Print the stack trace if configured to do so:
                            aprint("Stack trace:")
                            with acapture():
                                traceback.print_exc()

                    Arbol.enable_output = arbol_state
                else:
                    # By default if no specific test method is found, assume the feature is not supported:
                    aprint(f"      ❌ {feature_name} (No test method found!! Check!!)")
                    results[feature] = False

        return results

    def test_feature(
        self,
        api_class: Type[BaseApi],
        model_name: str,
        feature: ModelFeatures,
        return_error_info: bool = False,
        disable_output: bool = False,
        allow_media_conversions: bool = False,
    ) -> Union[bool, Dict[str, Union[bool, str]]]:
        """
        Test a single feature for a specific API and model.

        Parameters
        ----------
        api_class: Type[BaseApi]
            The API class to use for testing
        model_name: str
            Name of the model to test
        feature: ModelFeatures
            The feature to test
        return_error_info: bool
            If True, returns a dict with result and error information
        disable_output: bool
            If True, disables Arbol output during feature testing. Default is True.

        Returns
        -------
        Union[bool, Dict[str, Union[bool, str]]]
            If return_error_info=False: True if feature is supported, False otherwise
            If return_error_info=True: Dict with 'result' (bool) and 'error' (str) keys
        """
        feature_name = feature.name

        # Skip Conversion features as they're not directly testable
        if "conversion" in feature_name.lower():
            result = False
            error = "Conversion features are not directly testable"
            return {"result": result, "error": error} if return_error_info else result

        # Get the appropriate test method for the feature
        method_name = f"test_{self._snake_case(feature_name)}"
        test_method = getattr(self, method_name, None)

        if test_method:
            try:
                arbol_state = Arbol.enable_output
                if disable_output:
                    Arbol.enable_output = False

                with acapture():
                    api_instance = api_class(
                        allow_media_conversions=allow_media_conversions
                    )
                    result = test_method(api_instance, model_name)

                if disable_output:
                    Arbol.enable_output = arbol_state

                return {"result": result, "error": ""} if return_error_info else result
            except Exception as e:
                if disable_output:
                    Arbol.enable_output = arbol_state

                error = str(e)
                return {"result": False, "error": error} if return_error_info else False
        else:
            error = f"No test method found for feature: {feature_name}"
            return {"result": False, "error": error} if return_error_info else False

    def test_text_generation(self, api: BaseApi, model_name: str) -> bool:
        """Test if model supports text generation."""
        try:
            messages = [
                Message(role="system", text="You are a helpful assistant."),
                Message(role="user", text="Say 'hello world'"),
            ]
            response = api.generate_text(
                model_name=model_name, messages=messages, temperature=0.0
            )

            # Check if we got a response
            if not response or len(response) < 1:
                return False

            # Check if the response contains text
            response_text = str(response[0]).lower()
            return "hello" in response_text or "world" in response_text
        except Exception as e:
            aprint(f"Error in text generation test: {e}")
            return False

    def test_structured_text_generation(self, api: BaseApi, model_name: str) -> bool:
        """Test if model supports structured text generation."""
        try:
            # Define a simple pydantic model
            class Person(BaseModel):
                name: str
                age: int

            messages = [
                Message(role="system", text="You are a helpful assistant."),
                Message(
                    role="user",
                    text="Return information about a person named Bob who is 30 years old.",
                ),
            ]

            response = api.generate_text(
                model_name=model_name,
                messages=messages,
                response_format=Person,  # type: ignore
                temperature=0.0,
            )

            # Check if response contains an object and if it's valid
            if not response or len(response) < 1:
                return False

            # Check if the last block is an object with the expected content
            last_block = response[0][-1]
            content = last_block.get_content()

            return (
                isinstance(content, Person)
                and hasattr(content, "name")
                and hasattr(content, "age")
            )
        except Exception as e:
            aprint(f"Error in test_structured_text_generation: {e}")
            return False

    def test_image_generation(self, api: BaseApi, model_name: str) -> bool:
        """Test if model supports image generation."""
        try:
            image = api.generate_image(
                positive_prompt="A beautiful sunset over mountains",
                model_name=model_name,
            )
            from PIL.Image import Image as PILImage

            return image is not None and isinstance(image, PILImage)
        except Exception as e:
            aprint(f"Error in test_image_generation: {e}")
            return False

    def test_audio_generation(self, api: BaseApi, model_name: str) -> bool:
        """Test if model supports audio generation."""
        try:
            audio_uri = api.generate_audio(
                text="Hello, this is a test.", model_name=model_name
            )
            return audio_uri is not None and isinstance(audio_uri, str)
        except Exception as e:
            aprint(f"Error in test_audio_generation: {e}")
            return False

    def test_video_generation(self, api: BaseApi, model_name: str) -> bool:
        """Test if model supports video generation."""
        try:
            video_uri = api.generate_video(
                description="A short video of a spinning cube", model_name=model_name
            )
            return video_uri is not None and isinstance(video_uri, str)
        except Exception as e:
            aprint(f"Error in test_video_generation: {e}")
            return False

    def test_thinking(self, api: BaseApi, model_name: str) -> bool:
        """Test if model supports thinking feature."""
        try:
            # we need to reason per API as this is unfortunately, and in general, tricky:

            # First we check if the API is the Anthropic API:
            if api.__class__.__name__.lower() == "anthropicapi":
                # We just check if 'thinlking' is in the name:
                has_thinking = "thinking" in model_name.lower()
            elif api.__class__.__name__.lower() == "ollamaapi":
                # For Ollama, we check if the model name contains 'thinking':
                has_thinking = "thinking" in model_name.lower()
            elif api.__class__.__name__.lower() == "geminiapi":
                # For Gemini, we check if the model name contains 'thinking':
                has_thinking = is_gemini_reasoning_model(model_name)
            elif api.__class__.__name__.lower() == "openaiapi":
                # For OpenAI, we know which models support thinking by their names. For example
                has_thinking = is_openai_reasoning_model(model_name)

            return has_thinking

        except Exception as e:
            aprint(f"Error in test_thinking: {e}")
            return False

    def test_text_embeddings(self, api: BaseApi, model_name: str) -> bool:
        """Test if model supports text embeddings."""
        try:
            texts = ["Hello, world!"]
            embeddings = api.embed_texts(
                texts=texts, model_name=model_name, dimensions=512
            )

            return (
                embeddings is not None
                and len(embeddings) == 1
                and len(embeddings[0]) > 0
                and isinstance(embeddings[0][0], float)
            )
        except Exception as e:
            aprint(f"Error in test_text_embeddings: {e}")
            return False

    def test_image_embeddings(self, api: BaseApi, model_name: str) -> bool:
        """Test if model supports image embeddings."""
        try:
            image_uri = self.get_local_test_image_uri("python.png")
            embeddings = api.embed_images(
                image_uris=[image_uri], model_name=model_name, dimensions=512
            )

            return (
                embeddings is not None
                and len(embeddings) == 1
                and len(embeddings[0]) > 0
                and isinstance(embeddings[0][0], float)
            )
        except Exception as e:
            aprint(f"Error in test_image_embeddings: {e}")
            return False

    def test_audio_embeddings(self, api: BaseApi, model_name: str) -> bool:
        """Test if model supports audio embeddings."""
        try:
            audio_uri = self.get_local_test_audio_uri("harvard.wav")
            embeddings = api.embed_audios(
                audio_uris=[audio_uri], model_name=model_name, dimensions=512
            )

            return (
                embeddings is not None
                and len(embeddings) == 1
                and len(embeddings[0]) > 0
                and isinstance(embeddings[0][0], float)
            )
        except Exception as e:
            aprint(f"Error in test_audio_embeddings: {e}")
            return False

    def test_video_embeddings(self, api: BaseApi, model_name: str) -> bool:
        """Test if model supports video embeddings."""
        try:
            video_uri = self.get_local_test_video_uri("flying.mp4")
            embeddings = api.embed_videos(
                video_uris=[video_uri], model_name=model_name, dimensions=512
            )

            return (
                embeddings is not None
                and len(embeddings) == 1
                and len(embeddings[0]) > 0
                and isinstance(embeddings[0][0], float)
            )
        except Exception as e:
            aprint(f"Error in test_video_embeddings: {e}")
            return False

    def test_document_embeddings(self, api: BaseApi, model_name: str) -> bool:
        """Test if model supports document embeddings."""
        try:
            document_uri = self.get_local_test_document_uri(
                "noise2self_paper_page4.pdf"
            )
            embeddings = api.embed_documents(
                document_uris=[document_uri], model_name=model_name, dimensions=512
            )

            return (
                embeddings is not None
                and len(embeddings) == 1
                and len(embeddings[0]) > 0
                and isinstance(embeddings[0][0], float)
            )
        except Exception as e:
            aprint(f"Error in test_document_embeddings: {e}")
            return False

    def test_image(self, api: BaseApi, model_name: str) -> bool:
        """Test if model supports image understanding."""
        try:
            # Test image description capability
            image_path = self.get_local_test_image_uri("python.png")

            messages = [
                Message(role="system", text="You are a helpful assistant."),
                Message(role="user", text="Describe this image."),
            ]
            messages[1].append_image(image_path)

            response = api.generate_text(
                model_name=model_name, messages=messages, temperature=0.0
            )

            # Check if we got a meaningful response about the image
            if not response or len(response) < 1:
                return False

            response_text = str(response[0]).lower()
            image_indicators = ["python", "logo", "snake", "blue", "yellow"]

            # If any indicators are found in the response
            return any(indicator in response_text for indicator in image_indicators)
        except Exception as e:
            aprint(f"Error in test_image: {e}")
            return False

    def test_audio(self, api: BaseApi, model_name: str) -> bool:
        """Test if model supports audio understanding."""
        try:
            # Test audio description capability
            audio_path = self.get_local_test_audio_uri("harvard.wav")

            messages = [
                Message(role="system", text="You are a helpful assistant."),
                Message(role="user", text="What is in this audio file?"),
            ]
            messages[1].append_audio(audio_path)

            response = api.generate_text(
                model_name=model_name, messages=messages, temperature=0.0
            )

            # Check if we got a meaningful response about the audio
            if not response or len(response) < 1:
                return False

            response_text = str(response[0]).lower()
            audio_indicators = ["harvard", "sentence", "ham", "smell", "beer"]

            # If any indicators are found in the response
            return any(indicator in response_text for indicator in audio_indicators)
        except Exception as e:
            aprint(f"Error in test_audio: {e}")
            return False

    def test_video(self, api: BaseApi, model_name: str) -> bool:
        """Test if model supports video understanding."""
        try:
            # Test video description capability
            video_uri = self.get_local_test_video_uri("video.mp4")

            messages = [
                Message(role="system", text="You are a helpful assistant."),
                Message(role="user", text="Describe what's happening in this video."),
            ]
            messages[1].append_video(video_uri)

            response = api.generate_text(
                model_name=model_name, messages=messages, temperature=0.0
            )

            # Check if we got a meaningful response about the video
            if not response or len(response) < 1:
                return False

            if "m unable to" in str(response[0]).lower():
                # If the model says it is unable to describe the video, we consider it as not supporting video understanding.
                return False

            # Check if the response contains text that indicates video understanding:
            response_text = str(response[0]).lower()
            video_indicators = [
                "fly ",
                "flying",
                "saucer",
                "hover",
                "disc ",
                "circular",
                "aircraft",
                "vehicle",
                "testing",
                "drone",
                "facility",
            ]

            # Check if any of the video indicators are found in the response:
            for indicator in video_indicators:
                if indicator in response_text:
                    # aprint(f"Found indicator '{indicator}' in response.")
                    return True

            # If we reach here, it means no indicators were found in the response
            return False

        except Exception as e:
            aprint(f"Error in test_video: {e}")
            return False

    def test_document(self, api: BaseApi, model_name: str) -> bool:
        """Test if model supports document understanding."""
        try:
            # Test document description capability
            document_path = self.get_local_test_document_uri(
                "noise2self_paper_page4.pdf"
            )

            messages = [
                Message(role="system", text="You are a helpful assistant."),
                Message(role="user", text="What is this document about?"),
            ]
            messages[1].append_document(document_path)

            response = api.generate_text(
                model_name=model_name, messages=messages, temperature=0.0
            )

            # Check if we got a meaningful response about the document
            if not response or len(response) < 1:
                return False

            response_text = str(response[0]).lower()
            document_indicators = [
                "noise",
                "noise2self",
                "j-invariance",
                "denoising",
                "paper",
            ]

            # If any indicators are found in the response
            return any(indicator in response_text for indicator in document_indicators)
        except Exception as e:
            aprint(f"Error in test_document: {e}")
            return False

    def test_tools(self, api: BaseApi, model_name: str) -> bool:
        """Test if model supports tools/function calling."""
        try:
            # Create a simple tool for testing
            def get_the_last_name_of(first_name: str) -> str:
                """Get the last name for the first name."""
                return "Gerrard"

            toolset = ToolSet()
            toolset.add_function_tool(
                get_the_last_name_of, "Get the last name for the first name."
            )

            messages = [
                Message(role="system", text="You are a helpful assistant."),
                Message(role="user", text="What is Paul's last name?"),
            ]

            response = api.generate_text(
                model_name=model_name,
                messages=messages,
                toolset=toolset,
                temperature=0.0,
            )

            # Check if we got a meaningful response about the tool
            if response is None or len(response) < 1:
                return False

            # Extract the response text:
            response_text = str(response[-1]).lower()

            # Check if the response contains the last name:
            return "gerrard" in response_text

        except Exception as e:
            aprint(f"Error in test_tools: {e}")
            return False

    def test_web_search_tool(self, api: BaseApi, model_name: str) -> bool:
        """Test if model supports built-in web search tool."""
        try:

            # Create a simple tool for testing
            toolset = ToolSet()

            # Add the built-in web search tool to the toolset:
            toolset.add_builtin_web_search_tool()

            # Prepare a message to test the web search tool:
            messages = [
                Message(role="system", text="You are a helpful assistant."),
                Message(role="user", text="What is the weather in London now?"),
            ]

            # Generate text using the API with the toolset:
            response = api.generate_text(
                model_name=model_name,
                messages=messages,
                toolset=toolset,
                temperature=0.0,
            )

            # Check if we got a meaningful response about the tool
            if response is None or len(response) < 1:
                return False

            # Extract the response text:
            response_text = str(response[-1]).lower()

            # Check if the response contains the last name:
            return (
                any(digit in response_text for digit in "0123456789")
                or "london" in response_text
            )

        except Exception as e:
            aprint(f"Error in test_web_search_tool: {e}")
            return False

    def test_mcp_tool(self, api: BaseApi, model_name: str) -> bool:
        """Test if model supports built-in web search tool."""
        try:
            # Create a ToolSet instance:
            toolset = ToolSet()
            # Add the built-in web search tool to the toolset:
            toolset.add_builtin_mcp_tool(
                server_name="deepwiki",
                server_url="https://mcp.deepwiki.com/mcp",
                allowed_tools=["ask_question"],
            )

            # Prepare a message to test the MCP tool:
            messages = [
                Message(
                    role="system",
                    text="You are a helpful assistant.",
                ),
                Message(
                    role="user",
                    text="What transport protocols does the 2025-03-26 version of the MCP spec (modelcontextprotocol/modelcontextprotocol) support?",
                ),
            ]

            # Generate text using the API with the toolset:
            response = api.generate_text(
                model_name=model_name,
                messages=messages,
                toolset=toolset,
                temperature=0.0,
            )

            # Check if we got a meaningful response from the tool
            if response is None or len(response) < 1:
                return False

            # Extract the response text:
            response_text = str(response[-1]).lower()

            # Check if the response contains keywords:
            return (
                "stdio" in response_text
                or "streamable" in response_text
                or "http" in response_text
            )

        except Exception as e:
            aprint(f"Error in test_mcp_tool: {e}")
            return False

    def test_audio_transcription(self, api: BaseApi, model_name: str) -> bool:
        """Test if model supports audio transcription."""
        try:
            audio_uri = self.get_local_test_audio_uri("harvard.wav")
            transcription = api.transcribe_audio(
                audio_uri=audio_uri, model_name=model_name
            )

            # Check if the transcription is valid and contains expected content
            if not transcription or not isinstance(transcription, str):
                return False

            transcription = transcription.lower()
            expected_words = ["harvard", "sentence", "ham"]

            # If any expected words are found in the transcription
            return any(word in transcription for word in expected_words)
        except Exception as e:
            aprint(f"Error in test_audio_transcription: {e}")
            return False

    def save_results(self, folder: str = None) -> List[str]:
        """
        Save scan results to YAML files, one per API, in the specified folder.
        Also saves a comprehensive Markdown report file.

        Parameters
        ----------
        folder: str, optional
            Folder to save YAML files to. If None, uses self.output_dir.

        Returns
        -------
        List[str]
            List of file paths to the saved files (including YAML files and the Markdown report).
        """
        save_dir = folder or self.output_dir
        os.makedirs(save_dir, exist_ok=True)
        saved_files = []

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save each API's results to a YAML file
        for api_class, api_results in self.scan_results.items():
            api_name = self._api_class_to_name(api_class)
            filename = f"model_features_{api_name}_{timestamp}.scan.yaml"
            filepath = os.path.join(save_dir, filename)
            # Convert ModelFeatures keys to names for YAML
            results_serializable = {
                model: {self._feature_to_name(f): v for f, v in features.items()}
                for model, features in api_results.items()
            }
            results_with_metadata = {
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "scan_version": "1.0",
                    "api": api_name,
                },
                "results": results_serializable,
            }
            with open(filepath, "w") as file:
                yaml.dump(results_with_metadata, file, default_flow_style=False)
            aprint(f"Results for {api_name} saved to {filepath}")
            saved_files.append(filepath)

        # Generate and save the Markdown report
        if self.scan_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_content = self.generate_markdown_report()
            report_filepath = os.path.join(save_dir, f"report_{timestamp}.scan.md")
            with open(report_filepath, "w") as report_file:
                report_file.write(report_content)
            aprint(f"Comprehensive Markdown report saved to {report_filepath}")
            saved_files.append(report_filepath)

        return saved_files

    def load_results(
        self, folder: str
    ) -> Dict[type, Dict[str, Dict[ModelFeatures, bool]]]:
        loaded = 0
        if not os.path.exists(folder) or not os.path.isdir(folder):
            aprint(
                f"Warning: Load folder '{folder}' does not exist or is not a directory."
            )
            return self.scan_results

        # get the list of files in the folder:
        file_list = os.listdir(folder)

        # If list is empty, return the current scan_results:
        if not file_list:
            aprint(f"Warning: Load folder '{folder}' is empty. No results to load.")
            return self.scan_results

        # Sort the files so the most recent ones are last:
        file_list.sort(
            key=lambda x: os.path.getmtime(os.path.join(folder, x)), reverse=True
        )

        # Iterate over the files in the folder:
        for filename in os.listdir(folder):

            # Check if the file is a YAML file:
            if filename.endswith(".yaml"):

                # Load the YAML file:
                filepath = os.path.join(folder, filename)
                try:
                    with open(filepath, "r") as file:
                        data = yaml.safe_load(file)

                    # Check if the data is a dictionary:
                    if not isinstance(data, dict):
                        aprint(
                            f"Warning: Skipping file {filepath} due to invalid YAML structure (not a dict)."
                        )
                        continue

                    # Get metadata and API name:
                    api_name_from_meta = data.get("metadata", {}).get("api")

                    # Fallback to parsing filename if metadata is missing or incomplete
                    api_name = api_name_from_meta or (
                        filename.split("_")[3]
                        if len(filename.split("_")) > 3
                        else filename.replace("model_feature_scan_", "").replace(
                            ".yaml", ""
                        )
                    )

                    # Convert API name to class
                    api_class = self._api_name_to_class(api_name)

                    # If the API class is not found, skip this file
                    if api_class is None:
                        aprint(
                            f"Warning: Skipping results from file {filepath} as API class '{api_name}' could not be resolved to a BaseApi subclass."
                        )
                        continue

                    # Initialize current_api_results
                    current_api_results = {}

                    # Check if the results key exists and is a dictionary
                    if "results" in data and isinstance(data["results"], dict):

                        # Iterate over the models and their features
                        for model_name, features_dict in data["results"].items():

                            # Check if features_dict is a dictionary
                            if isinstance(features_dict, dict):

                                # Get the features from the dictionary:
                                current_api_results[model_name] = {
                                    self._name_to_feature(f_name): v
                                    for f_name, v in features_dict.items()
                                    # Ensure f_name is a valid key for ModelFeatures before attempting conversion
                                    if f_name in ModelFeatures.__members__
                                }

                                # Log if some feature names were unrecognized for a model
                                unrecognized_features = [
                                    f_name
                                    for f_name in features_dict
                                    if f_name not in ModelFeatures.__members__
                                ]
                                if unrecognized_features:
                                    aprint(
                                        f"Warning: Unrecognized feature names {unrecognized_features} for model '{model_name}' in {filepath}. These were skipped."
                                    )
                            else:
                                aprint(
                                    f"Warning: Invalid features format for model '{model_name}' in {filepath}. Skipping model."
                                )
                    else:
                        aprint(
                            f"Warning: No valid 'results' data found in {filepath} for API '{api_name}'. Skipping file."
                        )
                        continue

                    if (
                        len(current_api_results) == 0
                    ):  # If no models or features were successfully parsed for this API
                        aprint(
                            f"Warning: No models or features successfully parsed from {filepath} for API '{api_name}'. Skipping file."
                        )
                        continue

                    # Check if the API class already exists in scan_results
                    if api_class in self.scan_results:
                        # Merge the loaded results with existing results
                        self.scan_results[api_class].update(current_api_results)
                    else:
                        # Add the new API class and its results
                        self.scan_results[api_class] = current_api_results

                    # Log the successful loading of results
                    aprint(
                        f"Results for {api_name} (class: {api_class.__name__}) loaded from {filepath}"
                    )

                    # Increment the loaded count
                    loaded += 1
                except yaml.YAMLError as e:
                    aprint(f"Error parsing YAML file {filepath}: {e}")
                except (
                    KeyError
                ) as e:  # Specifically catch KeyErrors from _name_to_feature if a feature name is invalid
                    aprint(
                        f"Error processing feature name in file {filepath}: Invalid feature key {e}. This file might be corrupted or have an old/invalid feature name."
                    )
                except Exception as e:
                    aprint(f"Error processing file {filepath}: {e}")
                    traceback.print_exc()

        if loaded == 0:
            aprint(f"No YAML results successfully loaded from {folder}.")
        return self.scan_results

    def generate_markdown_report(self) -> str:
        """
        Generates a Markdown report detailing supported features for scanned APIs and models.

        The report includes:
        - A general timestamp.
        - For each API:
            - Summary statistics (total models, average features per model).
            - Feature prevalence across its models (percentage and count).
            - Detailed feature support (✅/❌) for each model.

        Returns
        -------
        str
            A string containing the Markdown report.
        """
        md_lines = []
        all_model_features_list = list(ModelFeatures)  # Consistent order

        md_lines.append("# Model Feature Scan Report")
        md_lines.append(f"_Report generated on: {datetime.now().isoformat()}_")
        md_lines.append("")

        if not self.scan_results:
            md_lines.append(
                "No scan results available. Please run `scan_apis()` or `load_results()` first."
            )
            return "\n".join(md_lines)

        sorted_apis = sorted(
            self.scan_results.items(), key=lambda item: self._api_class_to_name(item[0])
        )

        for i, (api_class, api_data) in enumerate(sorted_apis):
            api_name = self._api_class_to_name(api_class)
            md_lines.append(f"## API: {api_name}")

            # API Summary
            md_lines.append("\n### API Summary")
            num_models = len(api_data)
            md_lines.append(f"*   **Total Models Scanned:** {num_models}")

            if num_models > 0:
                total_supported_features_across_all_models_in_api = 0
                feature_support_counts_in_api = {
                    feature: 0 for feature in all_model_features_list
                }

                for model_name, model_feature_dict in api_data.items():
                    supported_count_for_model = sum(
                        1
                        for feature in all_model_features_list
                        if model_feature_dict.get(feature, False)
                    )
                    total_supported_features_across_all_models_in_api += (
                        supported_count_for_model
                    )
                    for feature in all_model_features_list:
                        if model_feature_dict.get(feature, False):
                            feature_support_counts_in_api[feature] += 1

                avg_features_per_model = (
                    total_supported_features_across_all_models_in_api / num_models
                )
                md_lines.append(
                    f"*   **Average Supported Features per Model:** {avg_features_per_model:.2f}"
                )

                md_lines.append("\n#### Feature Support Across Models:")
                sorted_feature_support_counts = sorted(
                    feature_support_counts_in_api.items(), key=lambda item: item[0].name
                )  # Sort by feature name
                for feature, count in sorted_feature_support_counts:
                    percentage = (count / num_models) * 100
                    md_lines.append(
                        f"*   {feature.name}: {percentage:.1f}% ({count}/{num_models})"
                    )
            else:
                md_lines.append("*   No models scanned for this API.")

            # Model Details
            md_lines.append("\n### Model Details")
            if num_models > 0:
                sorted_model_names = sorted(api_data.keys())
                for model_name in sorted_model_names:
                    model_features_support = api_data[model_name]
                    num_supported_for_model = sum(
                        1
                        for feature_enum in all_model_features_list
                        if model_features_support.get(feature_enum, False)
                    )

                    md_lines.append(
                        f"\n#### Model: {model_name} ({num_supported_for_model}/{len(all_model_features_list)} features supported)"
                    )
                    for feature_enum in all_model_features_list:  # Use the fixed order
                        is_supported = model_features_support.get(feature_enum, False)
                        icon = "✅" if is_supported else "❌"
                        md_lines.append(f"*   {icon} {feature_enum.name}")
            else:
                md_lines.append(
                    "_No model details to display as no models were scanned for this API._"
                )

            if i < len(sorted_apis) - 1:
                md_lines.append("\n---")  # Separator between APIs
            md_lines.append("")  # Add a blank line for spacing

        # Add mention at the end of the document that explains that the report is auto-generated and that features are
        # not supported either because the model does not support them or because litemind can't interface with these features,
        # or finally because of a bug in lietmind:
        md_lines.append(
            "_This report is auto-generated by Litemind. Some features may not be supported due to model limitations, API restrictions, or potential bugs in Litemind._"
        )

        return "\n".join(md_lines)
