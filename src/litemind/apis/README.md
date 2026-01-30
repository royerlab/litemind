# APIs Package

The `apis` package provides a unified API wrapper and abstraction layer for multiple LLM providers (OpenAI, Anthropic, Google Gemini, Ollama) with feature-based model selection and comprehensive callback support.

## Package Structure

```
apis/
├── base_api.py              # Abstract API interface
├── default_api.py           # Default implementations (local models)
├── combined_api.py          # Multi-API chaining with fallback
├── model_features.py        # Feature enum for capabilities
├── model_registry.py        # Curated model feature data
├── feature_validator.py     # Live API validation (for testing)
├── error_handling.py        # Error handling utilities
├── exceptions.py            # API exceptions
├── model_registry/          # YAML registry files
│   ├── registry_openai.yaml
│   ├── registry_anthropic.yaml
│   ├── registry_gemini.yaml
│   └── registry_ollama.yaml
├── providers/               # Provider implementations
│   ├── openai/
│   ├── anthropic/
│   ├── google/
│   └── ollama/
└── callbacks/               # API callback system
```

## Core Components

### BaseApi (`base_api.py`)

Abstract base class defining the interface all provider APIs must implement:

- `check_availability_and_credentials()` - Validate API keys
- `list_models()` - List available models with feature filtering
- `get_best_model()` - Select best model for required features
- `generate_text()` - Text generation with tools and streaming
- `generate_image()`, `generate_audio()`, `generate_video()` - Media generation
- `embed_texts()`, `embed_images()` - Embedding generation
- `transcribe_audio()` - Audio transcription
- `describe_image()`, `describe_audio()`, `describe_video()` - Media description

### DefaultApi (`default_api.py`)

Provides default implementations for local models:
- `whisper-local` - Local Whisper for audio transcription
- `fastembed` - Local text embeddings
- Automatic media conversion pipeline

### CombinedApi (`combined_api.py`)

Chains multiple provider APIs and routes requests to the first available API supporting required features:

```python
from litemind.apis.combined_api import CombinedApi

api = CombinedApi()  # Auto-discovers available APIs
# Or specify explicit list:
api = CombinedApi(apis=[OpenAIApi(), AnthropicApi()])

# Routes to best available model
response = api.generate_text(messages, model_features=["tools", "image"])
```

### ModelFeatures (`model_features.py`)

Enum defining all supported model capabilities:

```python
from litemind.apis.model_features import ModelFeatures

# Generation features
ModelFeatures.TextGeneration
ModelFeatures.StructuredTextGeneration
ModelFeatures.ImageGeneration
ModelFeatures.AudioGeneration
ModelFeatures.VideoGeneration
ModelFeatures.Thinking  # Extended reasoning

# Input features
ModelFeatures.Image
ModelFeatures.Audio
ModelFeatures.Video
ModelFeatures.Document

# Tool features
ModelFeatures.Tools
ModelFeatures.WebSearchTool
ModelFeatures.MCPTool

# Embedding features
ModelFeatures.TextEmbeddings
ModelFeatures.ImageEmbeddings

# Conversion features (fallback)
ModelFeatures.ImageConversion
ModelFeatures.AudioConversion
ModelFeatures.VideoConversion
ModelFeatures.DocumentConversion
```

Features can be specified as enums or strings:
```python
# All equivalent:
agent = Agent(api=api, model_features=[ModelFeatures.TextGeneration, ModelFeatures.Tools])
agent = Agent(api=api, model_features=["textgeneration", "tools"])
agent = Agent(api=api, model_features="tools")
```

### ModelRegistry (`model_registry.py`)

Curated registry of model capabilities loaded from YAML files:

```python
from litemind.apis.model_registry import get_default_model_registry

registry = get_default_model_registry()

# Check feature support
supports = registry.supports_feature(OpenAIApi, "gpt-4o", ModelFeatures.Tools)

# Get all supported features
features = registry.get_supported_features(OpenAIApi, "gpt-4o")

# Get model info (context window, max tokens, etc.)
info = registry.get_model_info(OpenAIApi, "gpt-4o")

# Resolve aliases (e.g., "o3-low" → "o3")
canonical, params = registry.resolve_alias(OpenAIApi, "o3-low")
```

**Registry YAML Format:**
```yaml
metadata:
  provider: OpenAIApi
  registry_version: "2.0"
  last_updated: "2026-01-27"

model_families:
  gpt-4o-family:
    features:
      TextGeneration: true
      StructuredTextGeneration: true
      Image: true
      Tools: true
    context_window: 128000
    max_output_tokens: 16384
    models:
      - gpt-4o
      - gpt-4o-mini
```

## Provider Implementations

### OpenAIApi (`providers/openai/`)

OpenAI Response API implementation with:
- Web search and file search tools
- Computer use support
- Deep Research mode
- Automatic conversation history management

### AnthropicApi (`providers/anthropic/`)

Anthropic Claude API with:
- Web search tool integration
- Model Context Protocol (MCP) support
- Extended thinking capabilities
- Built-in code execution sandbox

### GeminiApi (`providers/google/`)

Google Gemini API with:
- Text, image, audio, video inputs
- Extended thinking support
- Image generation (supported aspect ratios)

### OllamaApi (`providers/ollama/`)

Local Ollama models with:
- Text and image inputs
- Extended thinking on supported models
- Server connection management

## Callback System

Monitor API operations with callbacks:

```python
from litemind.apis.callbacks.base_api_callbacks import BaseApiCallbacks
from litemind.apis.callbacks.api_callback_manager import ApiCallbackManager

class MyCallback(BaseApiCallbacks):
    def on_text_generation(self, messages, response, **kwargs):
        print(f"Generated {len(response)} tokens")

    def on_text_streaming(self, fragment, **kwargs):
        print(fragment, end="", flush=True)

callback_manager = ApiCallbackManager()
callback_manager.add_callback(MyCallback())

api = OpenAIApi(callback_manager=callback_manager)
```

Available callbacks:
- `on_availability_check`, `on_model_list`, `on_best_model_selected`
- `on_text_generation`, `on_text_streaming`
- `on_audio_transcription`, `on_audio_generation`
- `on_image_generation`, `on_video_generation`
- `on_text_embedding`, `on_image_embedding`
- `on_image_description`, `on_audio_description`, `on_video_description`

## Usage Examples

### Basic Text Generation

```python
from litemind import OpenAIApi

api = OpenAIApi()
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"}
]
response = api.generate_text(messages=messages, model_name="gpt-4o")
```

### Feature-Based Model Selection

```python
from litemind.apis.combined_api import CombinedApi

api = CombinedApi()

# Get best model for multimodal + tools
model = api.get_best_model(features=["image", "tools"])

# Generate with automatic model selection
response = api.generate_text(
    messages=messages,
    model_features=["textgeneration", "tools"]
)
```

### Structured Outputs

```python
from pydantic import BaseModel

class WeatherResponse(BaseModel):
    temperature: float
    condition: str

response = api.generate_text(
    messages=messages,
    response_format=WeatherResponse
)
```

## Environment Variables

- `OPENAI_API_KEY` - OpenAI API
- `ANTHROPIC_API_KEY` - Anthropic/Claude API
- `GOOGLE_GEMINI_API_KEY` - Google Gemini API
- Ollama: No key required (local server)

## Architecture

```
CombinedApi / Provider API
         ↓
    ModelRegistry (feature lookup)
         ↓
    Feature-based model selection
         ↓
    Provider-specific utilities
    (convert_messages, format_tools, process_response)
         ↓
    Message object (response)
         ↓
    ApiCallbackManager (event distribution)
```

## Docstring Coverage

| Module | Coverage |
|--------|----------|
| base_api.py | 100% |
| default_api.py | 90% |
| combined_api.py | 100% |
| model_features.py | 100% |
| model_registry.py | 100% |
| callbacks/ | 95% |

All public APIs use numpy-style docstrings with comprehensive Parameters, Returns, and Raises sections.
