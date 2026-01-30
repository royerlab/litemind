"""
Tests for the ModelRegistry class.

These tests verify:
- Registry loading from YAML files
- Feature lookups (supports_feature, get_supported_features)
- Model info retrieval
- Alias resolution
- Model family inheritance
"""

import os
import tempfile

import pytest
import yaml

from litemind.apis.base_api import BaseApi, ModelFeatures
from litemind.apis.model_registry import (
    AliasInfo,
    ModelInfo,
    ModelRegistry,
    get_default_model_registry,
)


class RegistryTestApi(BaseApi):
    """Dummy API for testing purposes."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self._models = ["modelA", "modelB"]

    def check_availability_and_credentials(self, api_key=None):
        return True

    def list_models(self, features=None, non_features=None, media_type=None):
        return self._models

    def get_best_model(
        self, features=None, non_features=None, media_types=None, exclusion_filters=None
    ):
        return self._models[0]

    def has_model_support_for(self, features, media_types=None, model_name=None):
        return True

    def get_model_features(self, model_name):
        return []

    def max_num_input_tokens(self, model_name=None):
        return 4096

    def max_num_output_tokens(self, model_name=None):
        return 1024

    def count_tokens(self, text, model_name=None):
        return len(text.split())

    def generate_text(self, messages, model_name=None, **kwargs):
        return ["hello world"]

    def generate_audio(
        self, text, voice=None, audio_format=None, model_name=None, **kwargs
    ):
        return "audio_uri"

    def generate_image(
        self, positive_prompt, negative_prompt=None, model_name=None, **kwargs
    ):
        from PIL import Image as PILImage

        return PILImage.new("RGB", (1, 1), color="white")

    def generate_video(self, description, model_name=None, **kwargs):
        return "video_uri"

    def embed_texts(self, texts, model_name=None, dimensions=512, **kwargs):
        return [[0.1] * dimensions for _ in texts]

    def embed_images(self, image_uris, model_name=None, dimensions=512, **kwargs):
        return [[0.2] * dimensions for _ in image_uris]

    def embed_audios(self, audio_uris, model_name=None, dimensions=512, **kwargs):
        return [[0.3] * dimensions for _ in audio_uris]

    def embed_videos(self, video_uris, model_name=None, dimensions=512, **kwargs):
        return [[0.4] * dimensions for _ in video_uris]

    def embed_documents(self, document_uris, model_name=None, dimensions=512, **kwargs):
        return [[0.5] * dimensions for _ in document_uris]

    def transcribe_audio(self, audio_uri, model_name=None, **model_kwargs):
        return "transcription"

    def describe_image(self, image_uri, **kwargs):
        return "robot in the sky"

    def describe_audio(self, audio_uri, **kwargs):
        return "smell of ham"

    def describe_video(self, video_uri, **kwargs):
        return "roller coaster in amusement park"

    def describe_document(self, document_uri, **kwargs):
        return "noise2self paper"


@pytest.fixture
def registry():
    """Create a fresh ModelRegistry instance."""
    return ModelRegistry()


@pytest.fixture
def sample_registry_yaml():
    """Create a sample registry YAML content."""
    return {
        "metadata": {
            "provider": "RegistryTestApi",
            "registry_version": "2.0",
            "last_updated": "2026-01-27",
        },
        "model_families": {
            "test-family": {
                "features": {
                    "TextGeneration": True,
                    "Image": True,
                },
                "context_window": 128000,
                "max_output_tokens": 4096,
            }
        },
        "models": {
            "modelA": {
                "extends": "test-family",
                "features": {
                    "Tools": True,
                    "StructuredTextGeneration": True,
                },
                "source": "https://example.com/modelA",
            },
            "modelB": {
                "features": {
                    "TextGeneration": True,
                    "AudioGeneration": True,
                },
                "context_window": 64000,
                "notes": "Audio-capable model",
            },
            "modelC": {
                "extends": "test-family",
                "aliases": [
                    {"name": "modelC-fast", "parameters": {"speed": "fast"}},
                    {"name": "modelC-slow", "parameters": {"speed": "slow"}},
                ],
            },
        },
    }


def test_registry_initialization(registry):
    """Test that a fresh registry initializes correctly."""
    assert registry._registry == {}
    assert registry._aliases == {}


def test_load_registry_from_yaml(registry, sample_registry_yaml):
    """Test loading registry from YAML file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write the sample YAML
        filepath = os.path.join(tmpdir, "registry_dummy.yaml")
        with open(filepath, "w") as f:
            yaml.dump(sample_registry_yaml, f)

        # Load the registry
        registry.load_registry(tmpdir)

        # Verify RegistryTestApi is registered
        assert RegistryTestApi in registry._registry


def test_supports_feature(registry, sample_registry_yaml):
    """Test supports_feature method."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "registry_dummy.yaml")
        with open(filepath, "w") as f:
            yaml.dump(sample_registry_yaml, f)

        registry.load_registry(tmpdir)

        # Test features from family + model-specific
        assert registry.supports_feature(
            RegistryTestApi, "modelA", ModelFeatures.TextGeneration
        )
        assert registry.supports_feature(RegistryTestApi, "modelA", ModelFeatures.Image)
        assert registry.supports_feature(RegistryTestApi, "modelA", ModelFeatures.Tools)
        assert registry.supports_feature(
            RegistryTestApi, "modelA", ModelFeatures.StructuredTextGeneration
        )

        # Test feature not supported
        assert not registry.supports_feature(
            RegistryTestApi, "modelA", ModelFeatures.AudioGeneration
        )

        # Test modelB features
        assert registry.supports_feature(
            RegistryTestApi, "modelB", ModelFeatures.TextGeneration
        )
        assert registry.supports_feature(
            RegistryTestApi, "modelB", ModelFeatures.AudioGeneration
        )
        assert not registry.supports_feature(
            RegistryTestApi, "modelB", ModelFeatures.Image
        )


def test_get_supported_features(registry, sample_registry_yaml):
    """Test get_supported_features method."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "registry_dummy.yaml")
        with open(filepath, "w") as f:
            yaml.dump(sample_registry_yaml, f)

        registry.load_registry(tmpdir)

        features_a = registry.get_supported_features(RegistryTestApi, "modelA")
        assert ModelFeatures.TextGeneration in features_a
        assert ModelFeatures.Image in features_a
        assert ModelFeatures.Tools in features_a
        assert ModelFeatures.StructuredTextGeneration in features_a

        features_b = registry.get_supported_features(RegistryTestApi, "modelB")
        assert ModelFeatures.TextGeneration in features_b
        assert ModelFeatures.AudioGeneration in features_b


def test_get_model_info(registry, sample_registry_yaml):
    """Test get_model_info method."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "registry_dummy.yaml")
        with open(filepath, "w") as f:
            yaml.dump(sample_registry_yaml, f)

        registry.load_registry(tmpdir)

        info_a = registry.get_model_info(RegistryTestApi, "modelA")
        assert info_a is not None
        assert info_a.name == "modelA"
        assert info_a.context_window == 128000  # Inherited from family
        assert info_a.max_output_tokens == 4096
        assert info_a.source == "https://example.com/modelA"

        info_b = registry.get_model_info(RegistryTestApi, "modelB")
        assert info_b is not None
        assert info_b.context_window == 64000  # Overridden
        assert info_b.notes == "Audio-capable model"


def test_model_family_inheritance(registry, sample_registry_yaml):
    """Test that model families are properly inherited."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "registry_dummy.yaml")
        with open(filepath, "w") as f:
            yaml.dump(sample_registry_yaml, f)

        registry.load_registry(tmpdir)

        # modelA extends test-family, should have family features + its own
        info_a = registry.get_model_info(RegistryTestApi, "modelA")
        assert info_a.features.get(ModelFeatures.TextGeneration) is True
        assert info_a.features.get(ModelFeatures.Image) is True
        assert info_a.features.get(ModelFeatures.Tools) is True

        # modelB doesn't extend family, should only have its own features
        info_b = registry.get_model_info(RegistryTestApi, "modelB")
        assert info_b.features.get(ModelFeatures.TextGeneration) is True
        assert info_b.features.get(ModelFeatures.Image, False) is False


def test_alias_resolution(registry, sample_registry_yaml):
    """Test alias resolution."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "registry_dummy.yaml")
        with open(filepath, "w") as f:
            yaml.dump(sample_registry_yaml, f)

        registry.load_registry(tmpdir)

        # Test alias resolution
        base_name, params = registry.resolve_alias(RegistryTestApi, "modelC-fast")
        assert base_name == "modelC"
        assert params == {"speed": "fast"}

        base_name, params = registry.resolve_alias(RegistryTestApi, "modelC-slow")
        assert base_name == "modelC"
        assert params == {"speed": "slow"}

        # Test non-alias (should return same name with empty params)
        base_name, params = registry.resolve_alias(RegistryTestApi, "modelA")
        assert base_name == "modelA"
        assert params == {}


def test_alias_feature_lookup(registry, sample_registry_yaml):
    """Test that feature lookups work for aliases."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "registry_dummy.yaml")
        with open(filepath, "w") as f:
            yaml.dump(sample_registry_yaml, f)

        registry.load_registry(tmpdir)

        # Lookup features using alias should resolve to base model
        assert registry.supports_feature(
            RegistryTestApi, "modelC-fast", ModelFeatures.TextGeneration
        )
        assert registry.supports_feature(
            RegistryTestApi, "modelC-fast", ModelFeatures.Image
        )


def test_list_models(registry, sample_registry_yaml):
    """Test list_models method."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "registry_dummy.yaml")
        with open(filepath, "w") as f:
            yaml.dump(sample_registry_yaml, f)

        registry.load_registry(tmpdir)

        models = registry.list_models(RegistryTestApi)
        assert "modelA" in models
        assert "modelB" in models
        assert "modelC" in models


def test_list_aliases(registry, sample_registry_yaml):
    """Test list_aliases method."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "registry_dummy.yaml")
        with open(filepath, "w") as f:
            yaml.dump(sample_registry_yaml, f)

        registry.load_registry(tmpdir)

        aliases = registry.list_aliases(RegistryTestApi)
        assert "modelC-fast" in aliases
        assert "modelC-slow" in aliases


def test_nonexistent_api_or_model(registry, sample_registry_yaml):
    """Test queries for nonexistent APIs or models."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "registry_dummy.yaml")
        with open(filepath, "w") as f:
            yaml.dump(sample_registry_yaml, f)

        registry.load_registry(tmpdir)

        # Nonexistent API
        class FakeApi:
            pass

        assert (
            registry.supports_feature(FakeApi, "modelA", ModelFeatures.TextGeneration)
            is False
        )
        assert registry.get_supported_features(FakeApi, "modelA") == []
        assert registry.get_model_info(FakeApi, "modelA") is None

        # Nonexistent model
        assert (
            registry.supports_feature(
                RegistryTestApi, "nonexistent", ModelFeatures.TextGeneration
            )
            is False
        )
        assert registry.get_supported_features(RegistryTestApi, "nonexistent") == []


def test_get_default_model_registry():
    """Test the global registry accessor."""
    registry = get_default_model_registry()
    assert registry is not None
    assert isinstance(registry, ModelRegistry)


def test_empty_registry_folder(registry):
    """Test loading from empty or nonexistent folder."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Empty folder
        registry.load_registry(tmpdir)
        assert registry._registry == {}

        # Nonexistent folder
        registry.load_registry(os.path.join(tmpdir, "nonexistent"))
        assert registry._registry == {}


def test_model_info_dataclass():
    """Test ModelInfo dataclass."""
    info = ModelInfo(
        name="test-model",
        features={ModelFeatures.TextGeneration: True, ModelFeatures.Image: False},
        context_window=128000,
        max_output_tokens=4096,
        aliases=[AliasInfo(name="test-alias", parameters={"key": "value"})],
        notes="Test notes",
        source="https://example.com",
    )

    assert info.name == "test-model"
    assert info.features[ModelFeatures.TextGeneration] is True
    assert info.features[ModelFeatures.Image] is False
    assert info.context_window == 128000
    assert info.max_output_tokens == 4096
    assert len(info.aliases) == 1
    assert info.aliases[0].name == "test-alias"
    assert info.notes == "Test notes"
    assert info.source == "https://example.com"


def test_alias_info_dataclass():
    """Test AliasInfo dataclass."""
    alias = AliasInfo(name="test-alias", parameters={"effort": "high"})
    assert alias.name == "test-alias"
    assert alias.parameters == {"effort": "high"}

    # Test default empty parameters
    alias_default = AliasInfo(name="simple-alias")
    assert alias_default.parameters == {}
