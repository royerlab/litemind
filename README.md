
# litemind

[![PyPI version](https://badge.fury.io/py/litemind.svg)](https://pypi.org/project/litemind/)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](./LICENSE)
[![Downloads](https://static.pepy.tech/badge/litemind)](https://pepy.tech/project/litemind)
[![GitHub stars](https://img.shields.io/github/stars/royerlab/litemind.svg?style=social&label=Star)](https://github.com/royerlab/litemind)

---

## Summary

**Litemind** is a powerful, extensible, and user-friendly Python library for building next-generation, multimodal, agentic AI applications. It provides a unified, high-level API for interacting with a wide range of Large Language Model (LLM) providers (OpenAI, Anthropic/Claude, Google Gemini, Ollama, and more), and enables the creation of advanced agents that can reason, use tools, access external knowledge, and process rich multimodal data (text, images, audio, video, tables, code, documents, and more).

Litemind's philosophy is to make advanced agentic and multimodal AI accessible to all Python developers, with a focus on:
- **Simplicity**: Intuitive, Pythonic APIs and clear abstractions.
- **Extensibility**: Easily add new tools, augmentations, and model providers.
- **Transparency**: Rich logging (via [Arbol](http://github.com/royerlab/arbol)), clear error messages, and detailed reporting.
- **Robustness**: Comprehensive test suite, modular design, and best practices throughout.

Whether you want to build a conversational agent, a multimodal assistant, or a research tool that leverages the latest LLMs and AI capabilities, Litemind is your go-to framework.

---

## Features

- **Unified API** for multiple LLM providers (OpenAI, Claude, Gemini, Ollama, etc.)
- **Agentic framework**: Build agents that can reason, use tools, and maintain conversations.
- **Tool integration**: Add Python functions, agent tools, and built-in tools (web search, MCP protocol) to agents.
- **Augmentations (RAG)**: Integrate vector databases and retrieval-augmented generation for context-aware agents.
- **Multimodal support**: Seamlessly handle text, images, audio, video, tables, code, documents, and more.
- **Automatic model feature discovery**: Find models that support specific features (e.g., image input, tool use).
- **Media abstraction**: Rich, extensible Media classes for all supported data types.
- **Structured outputs**: Get Pydantic-typed outputs from LLMs for robust downstream processing.
- **Command-line tools**: Powerful CLI for code generation, repo export, and model feature scanning.
- **Extensive test suite**: High coverage for all major features and edge cases.
- **PEP-8 compliant**: Clean, readable, and maintainable codebase.
- **Arbol logging**: Beautiful, hierarchical logging with easy deactivation.
- **BSD-3-Clause License**: Permissive, open-source license.

---

## Installation

Install the latest release from PyPI:

```bash
pip install litemind
```

For development and all optional features (RAG, documents, audio, video, etc.):

```bash
pip install "litemind[dev,rag,documents,tables,videos,audio,whisper]"
```

See `pyproject.toml` for all optional dependencies.

---

## Basic Usage

Below are several illustrative examples of the **agent-level API**. Each example is self-contained and can be run as-is.

### 1. Basic Agent Usage

```python
from litemind import OpenAIApi
from litemind.agent.agent import Agent

# Initialize the OpenAI API
api = OpenAIApi()

# Create an agent
agent = Agent(api=api, model_name="o3-high")

# Add a system message to guide the agent's behavior
agent.append_system_message("You are a helpful assistant.")

# Ask a question
response = agent("What is the capital of France?")

print("Simple Agent Response:", response)
# Expected output: The capital of France is Paris.
```

---

### 2. Agent with Tools

```python
from litemind import OpenAIApi
from litemind.agent.agent import Agent
from litemind.agent.tools.toolset import ToolSet
from datetime import datetime

# Define a function tool
def get_current_date() -> str:
    """Fetch the current date"""
    return datetime.now().strftime("%Y-%m-%d")

api = OpenAIApi()
toolset = ToolSet()
toolset.add_function_tool(get_current_date)

agent = Agent(api=api, toolset=toolset)
agent.append_system_message("You are a helpful assistant.")

response = agent("What is the current date?")
print("Agent with Tool Response:", response)
# Output will include the current date, e.g., "The current date is 2025-05-02."
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

def get_current_date() -> str:
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d")

api = OpenAIApi()
toolset = ToolSet()
toolset.add_function_tool(get_current_date, "Fetch the current date")
agent = Agent(api=api, toolset=toolset)

# Create a vector database augmentation
vector_augmentation = InMemoryVectorDatabase(name="test_augmentation")
informations = [
    Information(Text("Igor Bolupskisty was a German-born theoretical physicist who developed the theory of indelible unitarity."),
               metadata={"topic": "physics", "person": "Bolupskisty"}),
    Information(Text("The theory of indelible unitarity revolutionized our understanding of space, time and photons."),
               metadata={"topic": "physics", "concept": "unitarity"}),
    Information(Text("Quantum unitarity is a fundamental theory in physics that describes nature at the nano-atomic scale as it pertains to Pink Hamsters."),
               metadata={"topic": "physics", "concept": "quantum unitarity"}),
]
vector_augmentation.add_informations(informations)
agent.add_augmentation(vector_augmentation)
agent.append_system_message("You are a helpful assistant.")

response = agent("Tell me about Igor Bolupskisty's theory of indelible unitarity. Also, what is the current date?")
print("Agent with Tool and Augmentation Response:", response)
# Output will include retrieved context and the current date.
```

---

### 4. More Complex Example: Multimodal Inputs, Tools, and Augmentations

```python
from litemind import OpenAIApi
from litemind.agent.agent import Agent
from litemind.agent.tools.toolset import ToolSet
from litemind.agent.augmentations.vector_db.in_memory_vector_db import InMemoryVectorDatabase
from litemind.agent.augmentations.information.information import Information
from litemind.media.types.media_text import Text
from litemind.media.types.media_image import Image

def get_current_date() -> str:
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d")

api = OpenAIApi()
toolset = ToolSet()
toolset.add_function_tool(get_current_date, "Fetch the current date")
agent = Agent(api=api, toolset=toolset)

# Add multimodal information to the vector database
vector_augmentation = InMemoryVectorDatabase(name="multimodal_augmentation")
vector_augmentation.add_informations([
    Information(Image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/3e/Einstein_1921_by_F_Schmutzer_-_restoration.jpg/456px-Einstein_1921_by_F_Schmutzer_-_restoration.jpg"),
                metadata={"person": "Einstein"}),
    Information(Text("Albert Einstein was a theoretical physicist who developed the theory of relativity."),
                metadata={"topic": "physics", "person": "Einstein"}),
])
agent.add_augmentation(vector_augmentation)
agent.append_system_message("You are a helpful assistant.")

# Compose a multimodal message
from litemind.agent.messages.message import Message
msg = Message(role="user", text="Can you describe what you see in the image?")
msg.append_image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/3e/Einstein_1921_by_F_Schmutzer_-_restoration.jpg/456px-Einstein_1921_by_F_Schmutzer_-_restoration.jpg")

response = agent(msg)
print("Multimodal Agent Response:", response)
# Output will include a description of the image and may reference Einstein.
```

---

**Note:** In all examples, you can specify model features as a list of strings or as a singleton string, e.g.:
```python
from litemind.apis.model_features import ModelFeatures
features = ["textgeneration", "tools"]  # or ModelFeatures.TextGeneration
```
Litemind will normalize these to the correct internal representation.

---

## Concepts

### Main Classes and Their Purpose

- **Agent**: The core class for agentic reasoning, conversation, tool use, and augmentation. It manages the conversation, tools, and augmentations (RAG).
- **ToolSet**: A collection of tools (Python functions, agent tools, built-in tools) that the agent can use.
- **AugmentationSet**: A collection of augmentations (e.g., vector databases) for retrieval-augmented generation.
- **Information**: Represents a knowledge chunk (text, image, etc.) with metadata, used in augmentations.
- **Media classes**: Abstractions for all supported data types (Text, Image, Audio, Video, Table, Code, Document, etc.).
- **API classes**: Wrappers for LLM providers (OpenAIApi, AnthropicApi, GeminiApi, OllamaApi, CombinedApi, etc.).

### API Wrapper Layer vs. Agentic API

- **API Wrapper Layer**: Direct, low-level access to LLM provider APIs (e.g., `OpenAIApi.generate_text(...)`). Use this for fine-grained control or when you want to bypass agentic reasoning.
- **Agentic API**: High-level, agent-centric interface (`Agent(...)`) that supports conversation, tool use, augmentation, and multimodal reasoning. This is the recommended way to build intelligent, interactive applications.

---

## Multi-modality

Litemind supports **multimodal inputs and outputs**: text, images, audio, video, tables, code, documents, and more.

### Model Features

- **Model features** describe what a model can do (e.g., "TextGeneration", "Image", "Tools", etc.).
- You can request features as enums, strings, or lists of strings. Litemind will normalize them.
- Use `ModelFeatures.normalise(...)` to convert to the internal representation.

### Requesting Features

```python
from litemind.apis.model_features import ModelFeatures
api = OpenAIApi()
model = api.get_best_model(features=[ModelFeatures.TextGeneration, ModelFeatures.Image])
```

Or, equivalently:

```python
model = api.get_best_model(features=["textgeneration", "image"])
```

### Media Classes

- All multimodal data is represented by **Media classes** (e.g., `Text`, `Image`, `Audio`, `Video`, `Table`, `Code`, `Document`, etc.).
- These classes provide methods for serialization, conversion, and integration with LLMs.
- When building messages or augmentations, always use the appropriate Media class.

---

## More Examples

### 1. Using the CombinedAPI

```python
from litemind.apis.combined_api import CombinedApi

api = CombinedApi()
print("Available models:", api.list_models())
model = api.get_best_model(features=["textgeneration", "image"])
response = api.generate_text(
    messages=[...],  # List of Message objects
    model_name=model
)
```

---

### 2. Agent That Can Execute Python Code

```python
from litemind import OpenAIApi
from litemind.agent.agent import Agent
from litemind.agent.tools.toolset import ToolSet

def multiply(x: int, y: int) -> int:
    """Multiply two numbers."""
    return x * y

api = OpenAIApi()
toolset = ToolSet()
toolset.add_function_tool(multiply, "Multiply two numbers")

agent = Agent(api=api, toolset=toolset)
agent.append_system_message("You are a helpful assistant.")

response = agent("What is 7 times 6?")
print("Agent with Python Tool Response:", response)
# Output will include the result of 7*6=42.
```

---

### 3. Using the Wrapper API for Structured Outputs

```python
from litemind import OpenAIApi
from litemind.agent.agent import Agent
from pydantic import BaseModel

class Weather(BaseModel):
    temperature: float
    condition: str
    humidity: float

api = OpenAIApi()
agent = Agent(api=api)
agent.append_system_message("You are a weather bot.")

response = agent("What is the weather like in Paris?", response_format=Weather)
print("Structured Output:", response)
# Output will be a Pydantic Weather object.
```

---

### 4. Agent Tool That Generates an Image Using the Wrapper API

```python
from litemind import OpenAIApi
from litemind.agent.agent import Agent
from litemind.agent.tools.toolset import ToolSet

def generate_cat_image() -> str:
    """Generate a prompt for a white siamese cat image."""
    return "Please draw me a white siamese cat"

api = OpenAIApi()
toolset = ToolSet()
toolset.add_function_tool(generate_cat_image, "Generate a cat image prompt")

agent = Agent(api=api, toolset=toolset)
agent.append_system_message("You are an assistant that can generate images.")

response = agent("Can you generate a picture of a white siamese cat?")
print("Agent with Image Generation Tool Response:", response)
# Output will include a generated image.
```

---

**More examples** can be found in the [EXAMPLES.md](EXAMPLES.md) file.

---

## Command Line Tools

Litemind provides several command-line tools for code generation, repo export, and model feature scanning.

### Usage

```bash
litemind codegen [api] [-m MODEL] [-f FILE]
litemind export [-f FOLDER_PATH] [-o OUTPUT_FILE] [-e EXTENSIONS] [-x EXCLUDE]
litemind scan [api] [-m MODELS] [-o OUTPUT_DIR]
```

- `codegen`: Generate files (e.g., README.md) for a Python repository using an LLM agent.
- `export`: Export the entire repository to a single file.
- `scan`: Scan models from an API for supported features and generate a report.

### .codegen.yml Format

Each `.codegen.yml` file describes a code generation task. Example:

```yaml
prompt: |
  Please generate a detailed README.md for this repository.
folder:
  path: .
  extensions: [".py", ".md", ".toml"]
  excluded: ["build", "dist", "litemind.egg-info"]
file: README.md
```

- Place your `.codegen.yml` files in a `.codegen` folder at the root of your repository.
- You can apply this to your own repositories for automated documentation/code generation.

---

## Caveats and Limitations

- **Error handling**: Some error messages may be cryptic; improvements are planned.
- **Token management**: No explicit handling of token limits; large conversations may hit model limits.
- **API key management**: Relies on environment variables; consider more secure solutions for production.
- **Performance**: No explicit caching or async support yet.
- **Failing tests**: See below for code health.
- **Streaming**: Not all APIs support streaming responses.
- **Model support**: Not all models support all features (e.g., not all support images or tools).
- **Security**: Do not expose API keys in public code or logs.

---

## Code Health

- **Unit tests**: All tests are located in the `src/litemind/*/tests/` directories.
- **Test report**: No failed tests reported in the current `test_reports/` directory (empty).
- **Total tests**: Hundreds of tests cover all major features, including multimodal, agentic, tool, and augmentation scenarios.
- **Failures**: No failed tests detected. If any failures occur, the file names will be listed here.
- **Assessment**: The codebase is robust and well-tested. See `ANALYSIS.md` for a detailed code quality and coverage analysis.

---

## API Keys

Litemind requires API keys for the various LLM providers:

- **OpenAI**: `OPENAI_API_KEY`
- **Anthropic (Claude)**: `ANTHROPIC_API_KEY`
- **Google Gemini**: `GOOGLE_GEMINI_API_KEY`
- **Ollama**: (local, no key needed by default)

### Setting API Keys

#### Linux/macOS

Add the following lines to your `~/.bashrc`, `~/.zshrc`, or `~/.profile`:

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_GEMINI_API_KEY="..."
```

Then reload your shell:

```bash
source ~/.bashrc  # or ~/.zshrc
```

#### Windows

Set environment variables in the Command Prompt or PowerShell:

```cmd
setx OPENAI_API_KEY "sk-..."
setx ANTHROPIC_API_KEY "sk-ant-..."
setx GOOGLE_GEMINI_API_KEY "..."
```

Or add them to your system environment variables via the Control Panel.

#### Docker/CI

Pass environment variables via `-e` flags or your CI/CD system's secrets management.

---

## Roadmap

- [x] Setup a readme with a quick start guide.
- [x] setup continuous integration and pipy deployment.
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
- [x] Add support for OpenAI's new 'Response' API.
- [x] Add support for builtin tools: Web search and MCP protocol.
- [ ] Add webui functionality for agents using Reflex.
- [ ] Video conversion temporal sampling should adapt to the video length, short videos should have more frames...
- [ ] RAG ingestion code for arbitrary digital objects: folders, pdf, images, urls, etc...
- [ ] Add more support for MCP protocol beyond built-in API support.
- [ ] Use the faster pybase64 for base64 encoding/decoding.
- [ ] Deal with message sizes in tokens sent to models
- [ ] Improve vendor api robustness features such as retry call when server errors, etc...
- [ ] Improve and uniformize exception handling
- [ ] Implement 'brainstorming' mode for text generation, possibly with API fusion.

---

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

- Fork the repository and create a feature branch.
- Write tests for new features.
- Ensure code is PEP-8 compliant and passes all tests.
- Use Numpy-style docstrings and update documentation.
- Submit a pull request for review.

---

## License

BSD-3-Clause. See [LICENSE](LICENSE) for details.

---

## Logging

Litemind uses [Arbol](http://github.com/royerlab/arbol) for hierarchical, beautiful logging. You can deactivate logging by setting:

```python
import arbol
arbol.Arbol.passthrough = True
```

---

> _This README was generated with the help of AI._
