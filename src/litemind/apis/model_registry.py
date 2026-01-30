"""
Model Registry - A curated registry of model capabilities and metadata.

This module provides a ModelRegistry class that loads model information from
YAML configuration files, providing accurate feature data, token limits,
aliases, and other metadata without requiring live API calls.
"""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type

import yaml
from arbol import aprint

from litemind.apis.base_api import BaseApi
from litemind.apis.model_features import ModelFeatures

# Global ModelRegistry instance
__model_registry: Optional["ModelRegistry"] = None


def get_default_model_registry() -> "ModelRegistry":
    """
    Get the global ModelRegistry instance.

    Returns
    -------
    ModelRegistry
        The global ModelRegistry instance, loaded with curated model data.
    """
    global __model_registry
    if __model_registry is None:
        __model_registry = ModelRegistry()
        # Load from the model_registry folder located in the same folder as this file:
        registry_folder = os.path.join(os.path.dirname(__file__), "model_registry")
        __model_registry.load_registry(registry_folder)
    return __model_registry


@dataclass
class AliasInfo:
    """Information about a model alias."""

    name: str
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelInfo:
    """
    Complete information about a model including features and metadata.

    Attributes
    ----------
    name : str
        The model identifier/name.
    features : Dict[ModelFeatures, bool]
        Mapping of features to their support status.
    context_window : Optional[int]
        Maximum input token limit, if known.
    max_output_tokens : Optional[int]
        Maximum output token limit, if known.
    aliases : List[AliasInfo]
        List of alias configurations for this model.
    notes : Optional[str]
        Additional notes about the model.
    source : Optional[str]
        URL or reference to the source of this information.
    """

    name: str
    features: Dict[ModelFeatures, bool] = field(default_factory=dict)
    context_window: Optional[int] = None
    max_output_tokens: Optional[int] = None
    aliases: List[AliasInfo] = field(default_factory=list)
    notes: Optional[str] = None
    source: Optional[str] = None


class ModelRegistry:
    """
    A curated registry of model capabilities and metadata.

    This class loads model information from YAML configuration files and provides
    methods to query model features, resolve aliases, and retrieve metadata.

    The registry supports:
    - Model families with inherited features
    - Aliases with custom parameters (e.g., o3-low/medium/high)
    - Rich metadata including token limits and documentation sources
    """

    def __init__(self):
        """Initialize an empty ModelRegistry."""
        # Dictionary mapping API class to model data
        # Structure: {api_class: {model_name: ModelInfo}}
        self._registry: Dict[Type[BaseApi], Dict[str, ModelInfo]] = {}

        # Dictionary mapping API class to alias mappings
        # Structure: {api_class: {alias_name: (base_model_name, parameters)}}
        self._aliases: Dict[Type[BaseApi], Dict[str, Tuple[str, Dict[str, Any]]]] = {}

    @staticmethod
    def _api_class_to_name(api_class: Type[BaseApi]) -> str:
        """Convert an API class to its string name."""
        return api_class.__name__

    @staticmethod
    def _api_name_to_class(api_name: str) -> Optional[Type[BaseApi]]:
        """
        Convert an API class name string to the actual class.

        Parameters
        ----------
        api_name : str
            The name of the API class (e.g., "OpenAIApi").

        Returns
        -------
        Optional[Type[BaseApi]]
            The API class if found, None otherwise.
        """
        import sys

        priority_module_substrings = [
            "litemind.apis.providers",
            "litemind.apis.tests",
            __name__,
        ]

        candidate_modules = []
        other_modules = []

        for mod_name, module in list(sys.modules.items()):
            if module is None:
                continue

            module_name_str = getattr(module, "__name__", None)
            if module_name_str is None:
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

        return None

    @staticmethod
    def _feature_name_to_enum(feature_name: str) -> Optional[ModelFeatures]:
        """Convert a feature name string to ModelFeatures enum."""
        try:
            return ModelFeatures[feature_name]
        except KeyError:
            return None

    def load_registry(self, folder: str) -> None:
        """
        Load model registry data from YAML files in the specified folder.

        Parameters
        ----------
        folder : str
            Path to the folder containing registry YAML files.
        """
        if not os.path.exists(folder) or not os.path.isdir(folder):
            aprint(
                f"Warning: Registry folder '{folder}' does not exist or is not a directory."
            )
            return

        for filename in os.listdir(folder):
            if not filename.endswith(".yaml") and not filename.endswith(".yml"):
                continue

            filepath = os.path.join(folder, filename)
            try:
                self._load_registry_file(filepath)
            except Exception as e:
                aprint(f"Error loading registry file {filepath}: {e}")

    def _load_registry_file(self, filepath: str) -> None:
        """
        Load a single registry YAML file.

        Parameters
        ----------
        filepath : str
            Path to the YAML file.
        """
        with open(filepath, "r") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            aprint(f"Warning: Invalid registry file format in {filepath}")
            return

        # Get metadata
        metadata = data.get("metadata", {})
        api_name = metadata.get("provider")
        if not api_name:
            aprint(f"Warning: No provider specified in {filepath}")
            return

        api_class = self._api_name_to_class(api_name)
        if api_class is None:
            aprint(f"Warning: Could not resolve API class '{api_name}' from {filepath}")
            return

        # Initialize storage for this API if needed
        if api_class not in self._registry:
            self._registry[api_class] = {}
        if api_class not in self._aliases:
            self._aliases[api_class] = {}

        # Load model families (for inheritance)
        model_families: Dict[str, Dict[str, Any]] = data.get("model_families", {})

        # Load models
        models = data.get("models", {})
        for model_name, model_data in models.items():
            model_info = self._parse_model_data(
                model_name, model_data, model_families, api_class
            )
            self._registry[api_class][model_name] = model_info

            # Register aliases
            for alias in model_info.aliases:
                self._aliases[api_class][alias.name] = (model_name, alias.parameters)

    def _parse_model_data(
        self,
        model_name: str,
        model_data: Dict[str, Any],
        model_families: Dict[str, Dict[str, Any]],
        api_class: Type[BaseApi],
    ) -> ModelInfo:
        """
        Parse model data from YAML, handling inheritance from model families.

        Parameters
        ----------
        model_name : str
            The model identifier.
        model_data : Dict[str, Any]
            The raw model data from YAML.
        model_families : Dict[str, Dict[str, Any]]
            Available model families for inheritance.
        api_class : Type[BaseApi]
            The API class this model belongs to.

        Returns
        -------
        ModelInfo
            Parsed model information.
        """
        # Start with inherited data from family if specified
        features: Dict[ModelFeatures, bool] = {}
        context_window: Optional[int] = None
        max_output_tokens: Optional[int] = None

        extends = model_data.get("extends")
        if extends and extends in model_families:
            family_data = model_families[extends]
            # Inherit features
            for feat_name, supported in family_data.get("features", {}).items():
                feat_enum = self._feature_name_to_enum(feat_name)
                if feat_enum:
                    features[feat_enum] = supported
            # Inherit token limits
            context_window = family_data.get("context_window")
            max_output_tokens = family_data.get("max_output_tokens")

        # Override with model-specific features
        for feat_name, supported in model_data.get("features", {}).items():
            feat_enum = self._feature_name_to_enum(feat_name)
            if feat_enum:
                features[feat_enum] = supported

        # Override token limits if specified
        if "context_window" in model_data:
            context_window = model_data["context_window"]
        if "max_output_tokens" in model_data:
            max_output_tokens = model_data["max_output_tokens"]

        # Parse aliases
        aliases: List[AliasInfo] = []
        for alias_data in model_data.get("aliases", []):
            if isinstance(alias_data, dict):
                alias_name = alias_data.get("name", "")
                alias_params = alias_data.get("parameters", {})
                if alias_name:
                    aliases.append(AliasInfo(name=alias_name, parameters=alias_params))

        return ModelInfo(
            name=model_name,
            features=features,
            context_window=context_window,
            max_output_tokens=max_output_tokens,
            aliases=aliases,
            notes=model_data.get("notes"),
            source=model_data.get("source"),
        )

    def supports_feature(
        self, api_class: Type[BaseApi], model_name: str, feature: ModelFeatures
    ) -> bool:
        """
        Check if a model supports a specific feature.

        Parameters
        ----------
        api_class : Type[BaseApi]
            The API class to query.
        model_name : str
            The model name (can be an alias).
        feature : ModelFeatures
            The feature to check.

        Returns
        -------
        bool
            True if the model supports the feature, False otherwise.
        """
        # Resolve alias if applicable
        resolved_name, _ = self.resolve_alias(api_class, model_name)

        if api_class not in self._registry:
            return False
        if resolved_name not in self._registry[api_class]:
            return False

        return self._registry[api_class][resolved_name].features.get(feature, False)

    def get_supported_features(
        self, api_class: Type[BaseApi], model_name: str
    ) -> List[ModelFeatures]:
        """
        Get all features supported by a model.

        Parameters
        ----------
        api_class : Type[BaseApi]
            The API class to query.
        model_name : str
            The model name (can be an alias).

        Returns
        -------
        List[ModelFeatures]
            List of supported features.
        """
        # Resolve alias if applicable
        resolved_name, _ = self.resolve_alias(api_class, model_name)

        if api_class not in self._registry:
            return []
        if resolved_name not in self._registry[api_class]:
            return []

        model_info = self._registry[api_class][resolved_name]
        return [feat for feat, supported in model_info.features.items() if supported]

    def get_model_info(
        self, api_class: Type[BaseApi], model_name: str
    ) -> Optional[ModelInfo]:
        """
        Get complete information about a model.

        Parameters
        ----------
        api_class : Type[BaseApi]
            The API class to query.
        model_name : str
            The model name (can be an alias).

        Returns
        -------
        Optional[ModelInfo]
            The model information if found, None otherwise.
        """
        # Resolve alias if applicable
        resolved_name, _ = self.resolve_alias(api_class, model_name)

        if api_class not in self._registry:
            return None
        return self._registry[api_class].get(resolved_name)

    def resolve_alias(
        self, api_class: Type[BaseApi], alias_or_model: str
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Resolve a model alias to its base model name and parameters.

        Parameters
        ----------
        api_class : Type[BaseApi]
            The API class to query.
        alias_or_model : str
            The alias or model name to resolve.

        Returns
        -------
        Tuple[str, Dict[str, Any]]
            A tuple of (base_model_name, parameters). If the input is not
            an alias, returns (alias_or_model, {}).
        """
        if api_class not in self._aliases:
            return (alias_or_model, {})

        if alias_or_model in self._aliases[api_class]:
            return self._aliases[api_class][alias_or_model]

        return (alias_or_model, {})

    def supports_any_feature(self, api_class: Type[BaseApi], model_name: str) -> bool:
        """
        Check if a model supports any feature at all.

        Parameters
        ----------
        api_class : Type[BaseApi]
            The API class to query.
        model_name : str
            The model name (can be an alias).

        Returns
        -------
        bool
            True if the model supports at least one feature.
        """
        return len(self.get_supported_features(api_class, model_name)) > 0

    def list_models(self, api_class: Type[BaseApi]) -> List[str]:
        """
        List all registered models for an API.

        Parameters
        ----------
        api_class : Type[BaseApi]
            The API class to query.

        Returns
        -------
        List[str]
            List of model names.
        """
        if api_class not in self._registry:
            return []
        return list(self._registry[api_class].keys())

    def list_aliases(self, api_class: Type[BaseApi]) -> List[str]:
        """
        List all registered aliases for an API.

        Parameters
        ----------
        api_class : Type[BaseApi]
            The API class to query.

        Returns
        -------
        List[str]
            List of alias names.
        """
        if api_class not in self._aliases:
            return []
        return list(self._aliases[api_class].keys())
