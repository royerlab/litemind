# Litemind

[![PyPI version](https://badge.fury.io/py/litemind.svg)](https://pypi.org/project/litemind/)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](./LICENSE)
[![Downloads](https://static.pepy.tech/badge/litemind)](https://pepy.tech/project/litemind)
[![GitHub stars](https://img.shields.io/github/stars/royerlab/litemind?style=social)](https://github.com/royerlab/litemind)

---

## Summary

**Litemind** is a powerful, extensible Python library for building **multimodal, agentic AI applications**. It provides
a unified, high-level interface to multiple LLM providers (OpenAI, Anthropic, Google Gemini, Ollama, and more), and a
sophisticated agent framework that supports tools, augmentations (RAG), and advanced multimodal reasoning. Litemind is
designed to make it easy to create intelligent agents that can process and reason about text, images, audio, video,
documents, tables, code, and more.

Litemind's philosophy is to **empower developers** to build production-ready, multimodal AI applications with minimal
boilerplate, while remaining fully extensible and transparent. It is built with a focus on:

- **Unified, ergonomic APIs** for all major LLM providers
- **Agentic reasoning** with tool use, memory, and RAG
- **Comprehensive multimodal support** (text, images, audio, video, documents, tables, code, etc.)
- **Automatic media conversion** and feature discovery
- **Extensible architecture** for custom tools, augmentations, and media types
- **Robust command-line tools** for code generation, repo export, and model scanning

---

## Features

- **Unified API Interface**: Seamlessly interact with OpenAI, Anthropic, Google Gemini, Ollama, and more, using a
  single, consistent API.
- **Advanced Agent Framework**: Build agents that can reason, use tools, access external knowledge, and manage
  conversations.
- **Comprehensive Multimodal Support**: Effortlessly handle text, images, audio, video, documents, tables, code, JSON,
  and more.
- **Intelligent Media Conversion**: Automatic conversion between media types based on model capabilities.
- **Vector Database Integration**: Built-in support for in-memory and Qdrant vector databases for RAG and augmentation.
- **Tool Integration**: Easily add Python functions or even other agents as tools for your agent to use.
- **Structured Output Generation**: Request and receive structured outputs (e.g., Pydantic models) from LLMs.
- **Thinking/Reasoning Models**: Support for models with explicit reasoning/thinking capabilities.
- **Feature Discovery**: Automatic detection of model capabilities and supported modalities.
- **Extensible Architecture**: Add your own tools, augmentations, and media types.
- **Command-Line Tools**: Generate READMEs, scan model features, and export repositories with a single command.
- **Robust Logging**: Integrated with [Arbol](http://github.com/royerlab/arbol) for rich, hierarchical logging (can be
  disabled).

---

## Installation

Install the core package:

```bash
pip install litemind
```

For additional features (RAG, document conversion, Whisper, video/audio, tables):

```bash
pip install litemind[rag,documents,whisper,videos,audio,tables]
```

You can also install only the extras you need, e.g.:

```bash
pip install litemind[rag]
pip install litemind[documents]
pip install litemind[whisper]
pip install litemind[videos]
```

---

## Basic Usage

Below are several illustrative examples of the **agent-level API**. Each example is self-contained and demonstrates a
different aspect of Litemind's agentic capabilities.

### 1. Basic Agent Usage

```python
from litemind import OpenAIApi
from litemind.agent.agent import Agent

# Initialize the OpenAI API
api = OpenAIApi()

# Create an agent
agent = Agent(api=api)

# Add a system message to guide the agent's behavior
agent.append_system_message("You are a helpful assistant.")

# Ask a question
response = agent("What is the capital of France?")

print("Simple Agent Response:", response)
# Expected output:
# Simple Agent Response: [*assistant*:
# The capital of France is Paris.
# ]
```

---

### 2. Agent with Tools

```python
from litemind import OpenAIApi
from litemind.agent.agent import Agent
from litemind.agent.tools.toolset import ToolSet
from datetime import datetime


# Define a function to get the current date
def get_current_date() -> str:
    """Fetch the current date"""
    return datetime.now().strftime("%Y-%m-%d")


# Initialize the OpenAI API
api = OpenAIApi()

# Create a toolset and add the function tool
toolset = ToolSet()
toolset.add_function_tool(get_current_date, "Fetch the current date")

# Create the agent, passing the toolset
agent = Agent(api=api, toolset=toolset)
agent.append_system_message("You are a helpful assistant.")

# Ask a question that requires the tool
response = agent("What is the current date?")

print("Agent with Tool Response:", response)
# Expected output:
# Agent with Tool Response: [*assistant*:
# Action: get_current_date()
# , *user*:
# Action: get_current_date()=2025-05-02
# , *assistant*:
# The current date is May 2, 2025.
# ]
```

---

### 3. Agent with Tools and Augmentation (RAG)

```python
from litemind import OpenAIApi
from litemind.agent.agent import Agent
from litemind.agent.tools.toolset import ToolSet
from litemind.agent.augmentations.information.information import Information
from litemind.agent.augmentations.vector_db.in_memory_vector_db import InMemoryVectorDatabase
from litemind.media.types.media_text import Text


# Define a function tool
def get_current_date() -> str:
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d")


# Initialize the OpenAI API
api = OpenAIApi()

# Create a toolset and add the function tool
toolset = ToolSet()
toolset.add_function_tool(get_current_date, "Fetch the current date")

# Create the agent
agent = Agent(api=api, toolset=toolset)

# Create vector database augmentation
vector_augmentation = InMemoryVectorDatabase(name="test_augmentation")

# Add sample informations to the augmentation
informations = [
    Information(
        Text(
            "Igor Bolupskisty was a German-born theoretical physicist who developed the theory of indelible unitarity."),
        metadata={"topic": "physics", "person": "Bolupskisty"},
    ),
    Information(
        Text("The theory of indelible unitarity revolutionized our understanding of space, time and photons."),
        metadata={"topic": "physics", "concept": "unitarity"},
    ),
    Information(
        Text(
            "Quantum unitarity is a fundamental theory in physics that describes nature at the nano-atomic scale as it pertains to Pink Hamsters."),
        metadata={"topic": "physics", "concept": "quantum unitarity"},
    ),
]

vector_augmentation.add_informations(informations)
agent.add_augmentation(vector_augmentation)
agent.append_system_message("You are a helpful assistant.")

# Ask a question that requires both RAG and tool use
response = agent(
    "Tell me about Igor Bolupskisty's theory of indelible unitarity. Also, what is the current date?"
)

print("Agent with Tool and Augmentation Response:", response)
# Expected output:
# Agent with Tool and Augmentation Response: [*assistant*:
# Action: get_current_date()
# , *user*:
# Action: get_current_date()=2025-05-02
# , *assistant*:
# Igor Bolupskisty was a German-born theoretical physicist known for developing the theory of indelible unitarity. This groundbreaking theory significantly advanced our understanding of space, time, and photons, reshaping fundamental concepts in physics.
#
# Today's date is May 2, 2025.
# ]
```

---

### 4. More Complex Example: Multimodal Inputs, Tools, and Augmentations

```python
from litemind import CombinedApi
from litemind.agent.agent import Agent
from litemind.agent.tools.toolset import ToolSet
from litemind.agent.augmentations.vector_db.in_memory_vector_db import InMemoryVectorDatabase
from litemind.agent.augmentations.information.information import Information
from litemind.media.types.media_text import Text
from litemind.media.types.media_image import Image


# Define a tool to analyze image colors
def analyze_image_colors(image_url: str) -> str:
    """Analyze dominant colors in an image"""
    return "The image contains primarily blue and green colors"


api = CombinedApi()
toolset = ToolSet()
toolset.add_function_tool(analyze_image_colors, "Analyze colors in an image")

# Create a knowledge base for RAG
knowledge_base = InMemoryVectorDatabase(name="art_knowledge")
knowledge_base.add_informations([
    Information(Text("Impressionism emphasizes light and color over detail")),
    Information(Text("Claude Monet was a founder of French Impressionist painting"))
])

agent = Agent(
    api=api,
    model_features=["TextGeneration", "Image", "Tools"],
    toolset=toolset
)
agent.add_augmentation(knowledge_base)

from litemind.agent.messages.message import Message

message = Message(role="user")
message.append_text("What art style does this image represent? Also analyze its colors.")
message.append_image(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/5/54/Claude_Monet%2C_Impression%2C_soleil_levant.jpg/617px-Claude_Monet%2C_Impression%2C_soleil_levant.jpg")

response = agent(message)
print(response)
# Expected output: The agent will describe the art style (Impressionism) and analyze the image colors.
```

---

## Concepts

Litemind is built around several key concepts and classes:

- **APIs**: Abstracts over LLM providers (OpenAI, Anthropic, Gemini, Ollama, etc.) and exposes a unified interface.
- **Agent**: Manages conversation, tool use, and augmentations. The `Agent` class itself implements all agentic
  features (including ReAct-style reasoning, tool use, and RAG).
- **Message**: Represents a message in a conversation, supporting multimodal content (text, images, audio, etc.).
- **ToolSet**: A collection of tools (Python functions or other agents) that the agent can use.
- **AugmentationSet**: A collection of augmentations (e.g., vector databases) for RAG.
- **Information**: Represents a knowledge chunk (text, image, etc.) with metadata, for use in augmentations.
- **Media Classes**: Type-safe wrappers for different content types (Text, Image, Audio, Video, Document, Table, Code,
  Json, Object, NdImage, etc.).

### API Layers

Litemind provides two main API layers:

1. **API Wrapper Layer**: Direct access to model capabilities (text generation, embeddings, etc.), with automatic media
   conversion and feature detection.
2. **Agentic API**: High-level agent framework with conversation management, tool integration, augmentation (RAG), and
   context injection.

**Note:** The agentic API is the recommended way to build intelligent, multimodal agents. The wrapper API is useful for
low-level or advanced use cases.

---

## Multi-modality

Litemind excels at handling multimodal inputs and outputs. It supports:

- **Text**
- **Images**
- **Audio**
- **Video**
- **Documents**
- **Tables**
- **Code**
- **JSON**
- **Objects (Pydantic models)**
- **nD Images**

### Model Features

Model features describe what a model can do (e.g., TextGeneration, Image, Tools, etc.). They are used to:

- Select the right model for your task
- Ensure media compatibility
- Enable automatic media conversion

You can request features as enums or as strings (singleton or list):

```python
from litemind.apis.model_features import ModelFeatures
from litemind.agent.agent import Agent

# Using enums
agent = Agent(api=api, model_features=[ModelFeatures.TextGeneration, ModelFeatures.Image])

# Using strings
agent = Agent(api=api, model_features=["TextGeneration", "Image", "Tools"])

# Singleton string
agent = Agent(api=api, model_features="TextGeneration")
```

### Media Classes

Media classes provide type-safe handling of different content types:

- `Text`, `Image`, `Audio`, `Video`, `Document`, `Table`, `Code`, `Json`, `Object`, `NdImage`, etc.

Example:

```python
from litemind.agent.messages.message import Message

message = Message(role="user")
message.append_text("Analyze these materials:")
message.append_image(
    "https://upload.wikimedia.org/wikipedia/commons/3/3e/Einstein_1921_by_F_Schmutzer_-_restoration.jpg")
message.append_document("file:///path/to/report.pdf")
```

---

## More Examples

### 1. Using the CombinedAPI

```python
from litemind import CombinedApi

api = CombinedApi()

models = api.list_models()
print(f"Available models: {models}")

best_model = api.get_best_model(
    features=["TextGeneration", "Image", "Tools"],
    exclusion_filters=["mini", "small"]
)
print(f"Best model: {best_model}")
```

---

### 2. Agent That Can Execute Python Code

```python
from litemind import OpenAIApi
from litemind.agent.agent import Agent
from litemind.agent.tools.toolset import ToolSet
import subprocess
import tempfile


def execute_python_code(code: str) -> str:
    """Execute Python code and return the output"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        f.flush()
        try:
            result = subprocess.run(
                ['python', f.name],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.stdout if result.returncode == 0 else result.stderr
        except Exception as e:
            return f"Error: {str(e)}"


api = OpenAIApi()
toolset = ToolSet()
toolset.add_function_tool(execute_python_code, "Execute Python code")
agent = Agent(api=api, toolset=toolset)
agent.append_system_message("You are a Python programming assistant.")

response = agent("Calculate the factorial of 10 using Python")
print(response)
```

---

### 3. Use of the Wrapper API to Get Structured Outputs

```python
from litemind import OpenAIApi
from litemind.agent.messages.message import Message
from pydantic import BaseModel
from typing import List


class MovieReview(BaseModel):
    title: str
    rating: float
    pros: List[str]
    cons: List[str]
    summary: str


api = OpenAIApi()
messages = [
    Message(role="system", text="You are a movie critic."),
    Message(role="user", text="Review the movie 'Inception' (2010)")
]
response = api.generate_text(
    messages=messages,
    response_format=MovieReview,
    temperature=0.7
)

review = response[0][-1].get_content()
print(f"Title: {review.title}")
print(f"Rating: {review.rating}/10")
print(f"Pros: {', '.join(review.pros)}")
print(f"Cons: {', '.join(review.cons)}")
```

---

### 4. Agent Tool That Generates an Image Using the Wrapper API

```python
from litemind import CombinedApi
from litemind.agent.agent import Agent
from litemind.agent.tools.toolset import ToolSet


def generate_image(prompt: str) -> str:
    """Generate an image based on the prompt"""
    api = CombinedApi()
    image = api.generate_image(
        positive_prompt=prompt,
        model_name=api.get_best_model(features=["ImageGeneration"]),
        image_width=1024,
        image_height=1024
    )
    image_path = f"/tmp/generated_{hash(prompt)}.png"
    image.save(image_path)
    return f"Image generated and saved to: {image_path}"


api = CombinedApi()
toolset = ToolSet()
toolset.add_function_tool(generate_image, "Generate images from text descriptions")
agent = Agent(api=api, toolset=toolset)
agent.append_system_message("You are a creative AI assistant.")
response = agent("Create an image of a futuristic city at sunset")
print(response)
```

---

## Command Line Tools

Litemind provides several command-line tools for code generation, repo export, and model scanning.

### 1. Code Generation

Generate files (like `README.md`) for your repository:

```bash
litemind codegen
litemind codegen openai -m gpt-4-turbo
litemind codegen -f README
```

### 2. Export Repository

Export your entire repository to a single file:

```bash
litemind export -o exported_repo.txt
litemind export -f /path/to/repo -o output.txt -e .py .md -x __pycache__ .git
```

### 3. Scan Model Features

Scan available models and their features:

```bash
litemind scan
litemind scan openai anthropic
litemind scan openai -m gpt-4 gpt-3.5-turbo
litemind scan -o ./scan_results
```

### 4. Codegen YAML Template

To use codegen, create a `.codegen` folder in your repository root with one or more `*.codegen.yml` files. Example:

```yaml
prompt: |
  Generate a comprehensive README.md file for this Python repository.
  Include sections for installation, usage, features, and examples.
  Make it professional and well-structured.
folder:
  path: .
  extensions:
    - .py
    - .md
    - .yml
    - .toml
  excluded:
    - __pycache__
    - .git
    - build
    - dist
file: README.md
```

---

## Caveats and Limitations

- **API Key Requirements**: You must provide API keys for OpenAI, Anthropic, and Google Gemini. Ollama requires a local
  server.
- **Model Availability**: Not all models support all features (e.g., some models may not support images or tools).
- **Rate Limits**: Subject to provider rate limits and quotas.
- **Media Conversion**: Some conversions may require optional dependencies (see installation).
- **Memory Usage**: Large media files or documents may consume significant memory.
- **Async Support**: Litemind does not currently provide async APIs.
- **Token Limits**: Be aware of model-specific token limits.
- **Cost Considerations**: Using cloud APIs may incur costs.
- **Error Handling**: Some error handling (especially in the API wrapper layer) can be improved.
- **Ollama API**: Some multimodal features (embeddings, media description) are not fully supported by Ollama.

---

## Code Health

**Test Suite Results:**

- **Total Tests:** 234
- **Passed:** 226
- **Failed:** 8

**Failed Tests:**

- `test_apis_basics.py::TestBaseApiImplementationsBasics::test_max_num_input_token[OllamaApi]`
- `test_apis_basics.py::TestBaseApiImplementationsBasics::test_max_num_output_tokens[OllamaApi]`
- `test_apis_describe.py::TestBaseApiImplementationsDescribe::test_describe_audio_if_supported[OllamaApi]`
- `test_apis_describe.py::TestBaseApiImplementationsDescribe::test_describe_video_if_supported[OllamaApi]`
- `test_apis_describe.py::TestBaseApiImplementationsDescribe::test_describe_document_if_supported[OllamaApi]`
- `test_apis_embeddings.py::TestBaseApiImplementationsEmbeddings::test_audio_embedding[OllamaApi]`
- `test_apis_embeddings.py::TestBaseApiImplementationsEmbeddings::test_video_embedding[OllamaApi]`
- `test_apis_embeddings.py::TestBaseApiImplementationsEmbeddings::test_document_embedding[OllamaApi]`

**Assessment:**  
The failures are not critical and all relate to the Ollama API implementation and specific features (embeddings and
media description). The core functionality and other API implementations (OpenAI, Anthropic, Google) are working
correctly. These failures appear to be due to Ollama's limited support for certain multimodal features rather than
fundamental issues with Litemind itself.

For more details, see [ANALYSIS.md](./ANALYSIS.md).

---

## API Keys

To use Litemind, you need to set up API keys for the LLM providers you want to use:

- **OpenAI**: [Get your API key](https://platform.openai.com/api-keys)
    - Set environment variable: `OPENAI_API_KEY`
- **Anthropic**: [Get your API key](https://console.anthropic.com/)
    - Set environment variable: `ANTHROPIC_API_KEY`
- **Google Gemini**: [Get your API key](https://makersuite.google.com/app/apikey)
    - Set environment variable: `GOOGLE_GEMINI_API_KEY`
- **Ollama**: No API key needed, but ensure Ollama is running locally ([Ollama website](https://ollama.ai/))

### Setting Environment Variables

**macOS/Linux:**  
Edit your `~/.bashrc`, `~/.zshrc`, or `~/.bash_profile`:

```bash
export OPENAI_API_KEY="your-key-here"
export ANTHROPIC_API_KEY="your-key-here"
export GOOGLE_GEMINI_API_KEY="your-key-here"
```

**Windows:**  
Edit system environment variables or use PowerShell:

```powershell
[System.Environment]::SetEnvironmentVariable('OPENAI_API_KEY','your-key-here','User')
[System.Environment]::SetEnvironmentVariable('ANTHROPIC_API_KEY','your-key-here','User')
[System.Environment]::SetEnvironmentVariable('GOOGLE_GEMINI_API_KEY','your-key-here','User')
```

---

## Roadmap

- [x] Setup a readme with a quick start guide.
- [x] Setup continuous integration and PyPI deployment.
- [x] Improve document conversion (page per page text and video interleaving + whole page images)
- [x] Cleanup structured output with tool usage
- [x] Implement streaming callbacks
- [x] Improve folder/archive conversion: add ascii folder tree
- [x] Reorganise media files used for testing into a single media folder
- [x] Improve logging with arbol, with option to turn off.
- [x] Use specialised libraries for document type identification
- [x] Cleanup the document conversion code.
- [x] Add support for adding nD images to messages.
- [x] Automatic feature support discovery for models (which models support images as input, reasoning, etc...)
- [ ] Add support for OpenAI's new 'Response' API.
- [ ] Add support for MCP protocol.
- [ ] Add webui functionality for agents using Reflex.
- [ ] Video conversion temporal sampling should adapt to the video length, short videos should have more frames...
- [ ] RAG ingestion code for arbitrary digital objects: folders, pdf, images, urls, etc...
- [ ] Use the faster pybase64 for base64 encoding/decoding.
- [ ] Deal with message sizes in tokens sent to models
- [ ] Improve vendor api robustness features such as retry call when server errors, etc...
- [ ] Improve and uniformize exception handling
- [ ] Implement 'brainstorming' mode for text generation, possibly with API fusion.

---

## Contributing

We welcome contributions! Please see our [CONTRIBUTING.md](./CONTRIBUTING.md) file for guidelines on how to contribute
code, documentation, or other resources.

Key points:

- Fork the repository and create a feature branch
- Write tests for new features
- Follow PEP-8 style guidelines
- Update documentation as needed
- Submit a pull request with a clear description

**Code Quality:**

- Use `black`, `isort`, `flake8`, and `mypy` for code formatting and linting.
- Write tests using `pytest`.
- Use Numpy-style docstrings.
- Set up pre-commit hooks for automatic formatting.

---

## License

Litemind is licensed under the BSD-3-Clause License. See the [LICENSE](./LICENSE) file for details.

---

## Logging

Logging in Litemind is handled by [Arbol](http://github.com/royerlab/arbol).  
To disable logging output, set:

```python
from arbol import Arbol

Arbol.passthrough = True
```

---

> _This README was generated with the assistance of AI using Litemind itself._
