
# litemind

[![PyPI version](https://badge.fury.io/py/litemind.svg)](https://pypi.org/project/litemind/)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](./LICENSE)
[![Downloads](https://static.pepy.tech/badge/litemind)](https://pepy.tech/project/litemind)
[![GitHub stars](https://img.shields.io/github/stars/royerlab/litemind.svg?style=social&label=Star)](https://github.com/royerlab/litemind)

---

## Summary

**Litemind** is a powerful, extensible, and user-friendly Python library for building multimodal, agentic AI applications. It provides a unified, high-level API for interacting with a wide range of Large Language Model (LLM) providers (OpenAI, Anthropic, Google Gemini, Ollama, and more), and enables the creation of advanced agents that can reason, use tools, and augment their knowledge with external data sources (RAG).

Litemind's philosophy is to make advanced AI capabilities accessible, composable, and robust—whether you want to chat with an LLM, build a tool-using agent, perform multimodal reasoning, or integrate retrieval-augmented generation (RAG) with your own data. The library is designed for clarity, extensibility, and reliability, with a strong focus on test coverage and code quality.

---

## Features

- **Unified API**: Seamlessly interact with multiple LLM providers (OpenAI, Anthropic, Gemini, Ollama, etc.) through a single, consistent interface.
- **Agentic Framework**: Build agents that can reason, use tools, and maintain conversational context.
- **Tool Integration**: Easily add Python functions, other agents, or built-in tools (e.g., web search, MCP protocol) to your agents.
- **Augmentations (RAG)**: Integrate vector databases and other augmentation sources for retrieval-augmented generation.
- **Multimodal Support**: Handle text, images, audio, video, tables, documents, and more as both inputs and outputs.
- **Automatic Model Feature Discovery**: Automatically select models based on required features (e.g., image input, tool use, reasoning).
- **Media Abstractions**: Rich, extensible media classes for all supported modalities.
- **Extensible Architecture**: Add new LLM providers, tools, or media types with minimal effort.
- **Comprehensive Test Suite**: High test coverage for all major features and components.
- **Command-Line Tools**: CLI utilities for code generation, repo export, and model feature scanning.
- **Arbol Logging**: Rich, hierarchical logging with easy deactivation.
- **PEP-8 Compliance**: Clean, readable, and well-documented codebase.

---

## Installation

Install the latest version from PyPI:

```bash
pip install litemind
```

For development and optional features (RAG, documents, audio, etc.), use:

```bash
pip install "litemind[dev,rag,documents,tables,videos,audio]"
```

---

## Basic Usage

Below are several illustrative examples of the agent-level API, covering basic usage, tool integration, RAG augmentation, and multimodal inputs.

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
toolset.add_function_tool(get_current_date)

# Create the agent, passing the toolset
agent = Agent(api=api, toolset=toolset)

# Add a system message
agent.append_system_message("You are a helpful assistant.")

# Ask a question that requires the tool
response = agent("What is the current date?")

print("Agent with Tool Response:", response)
# Expected output:
# Agent with Tool Response: [*assistant*:
# The current date is 2025-05-02.
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
    Information(Text("Igor Bolupskisty was a German-born theoretical physicist who developed the theory of indelible unitarity."),
                metadata={"topic": "physics", "person": "Bolupskisty"}),
    Information(Text("The theory of indelible unitarity revolutionized our understanding of space, time and photons."),
                metadata={"topic": "physics", "concept": "unitarity"}),
    Information(Text("Quantum unitarity is a fundamental theory in physics that describes nature at the nano-atomic scale as it pertains to Pink Hamsters."),
                metadata={"topic": "physics", "concept": "quantum unitarity"}),
]

vector_augmentation.add_informations(informations)
agent.add_augmentation(vector_augmentation)

# Add a system message
agent.append_system_message("You are a helpful assistant.")

# Ask a question that requires both the tool and the augmentation
response = agent("Tell me about Igor Bolupskisty's theory of indelible unitarity. Also, what is the current date?")

print("Agent with Tool and Augmentation Response:", response)
# Expected output:
# Agent with Tool and Augmentation Response: [*assistant*:
# Igor Bolupskisty was a German-born theoretical physicist known for developing the theory of indelible unitarity. ...
# Today's date is May 2, 2025.
# ]
```

---

### 4. More Complex Example: Multimodal Inputs, Tools, and Augmentations

```python
from litemind import OpenAIApi
from litemind.agent.agent import Agent
from litemind.agent.tools.toolset import ToolSet
from litemind.agent.augmentations.vector_db.in_memory_vector_db import InMemoryVectorDatabase
from litemind.agent.augmentations.information.information import Information
from litemind.media.types.media_image import Image
from litemind.media.types.media_text import Text

# Define a function tool
def get_image_description() -> str:
    return "This is a placeholder for image description."

# Initialize the OpenAI API
api = OpenAIApi()

# Create a toolset and add the function tool
toolset = ToolSet()
toolset.add_function_tool(get_image_description, "Describe an image")

# Create the agent
agent = Agent(api=api, toolset=toolset)

# Create vector database augmentation with an image
vector_augmentation = InMemoryVectorDatabase(name="image_augmentation")
image_info = Information(
    Image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/3e/Einstein_1921_by_F_Schmutzer_-_restoration.jpg/456px-Einstein_1921_by_F_Schmutzer_-_restoration.jpg"),
    metadata={"person": "Einstein"}
)
vector_augmentation.add_informations([image_info])
agent.add_augmentation(vector_augmentation)

# Add a system message
agent.append_system_message("You are a helpful assistant that can describe images and answer questions.")

# Ask a question with an image input
from litemind.agent.messages.message import Message
msg = Message(role="user", text="Can you describe what you see in this image?")
msg.append_image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/3e/Einstein_1921_by_F_Schmutzer_-_restoration.jpg/456px-Einstein_1921_by_F_Schmutzer_-_restoration.jpg")
response = agent(msg)

print("Multimodal Agent Response:", response)
# Expected output: A detailed description of the image, possibly referencing Einstein.
```

---

## Concepts

### Main Classes and Their Purpose

- **Agent**: The core class for agentic reasoning. Maintains a conversation, can use tools, and can be augmented with external knowledge sources (augmentations/RAG).
- **ToolSet**: A collection of tools (Python functions, other agents, or built-in tools) that the agent can use.
- **AugmentationSet**: A collection of augmentations (e.g., vector databases) for retrieval-augmented generation.
- **Information**: Represents a knowledge chunk (text, image, etc.) with metadata, used in augmentations.
- **Media Classes**: Abstractions for all supported input/output types (Text, Image, Audio, Video, Table, Document, etc.).
- **API Classes**: Wrappers for LLM providers (OpenAIApi, AnthropicApi, GeminiApi, OllamaApi, CombinedApi, etc.).

### API Wrapper Layer vs. Agentic API

- **API Wrapper Layer**: Provides direct, low-level access to LLM provider APIs (e.g., `api.generate_text(...)`, `api.embed_texts(...)`). Use this for simple, direct calls.
- **Agentic API**: The high-level, agent-oriented interface (`Agent` class) that supports conversation, tool use, augmentation, and multimodal reasoning. This is the recommended way to build advanced, interactive AI applications.

---

## Multi-modality

Litemind supports **multimodal inputs and outputs**: text, images, audio, video, tables, documents, and more.

### Model Features

- **Model features** describe the capabilities of a model (e.g., text generation, image input, tool use, etc.).
- You can request features as a list of enums, a singleton, or a list of strings:
    ```python
    from litemind.apis.model_features import ModelFeatures
    # As enums
    features = [ModelFeatures.TextGeneration, ModelFeatures.Image]
    # As strings
    features = ["textgeneration", "image"]
    # As a singleton
    features = ModelFeatures.TextGeneration
    ```

### Requesting Features

When creating an agent or calling an API, you can specify required features:
```python
agent = Agent(api=api, model_features=["textgeneration", "image"])
```
Litemind will automatically select the best model that supports the requested features.

### Media Classes

- **Media classes** (e.g., `Text`, `Image`, `Audio`, `Video`, `Table`, `Document`, etc.) provide a unified way to represent and manipulate multimodal data.
- Use these classes to add multimodal content to messages, augmentations, and tools.

---

## More Examples

### 1. Using the CombinedAPI

```python
from litemind.apis.combined_api import CombinedApi

api = CombinedApi()
print("Available models:", api.list_models())
```

---

### 2. Agent That Can Execute Python Code

```python
from litemind.agent.agent import Agent
from litemind.agent.tools.toolset import ToolSet

def execute_python_code(code: str) -> str:
    # WARNING: This is a simple example. Do not use eval/exec on untrusted input!
    try:
        exec_globals = {}
        exec(code, exec_globals)
        return str(exec_globals)
    except Exception as e:
        return str(e)

toolset = ToolSet()
toolset.add_function_tool(execute_python_code, "Execute Python code")

from litemind import OpenAIApi
api = OpenAIApi()
agent = Agent(api=api, toolset=toolset)
agent.append_system_message("You are a Python code execution assistant.")

response = agent("Please execute: print('Hello, world!')")
print(response)
```

---

### 3. Using the Wrapper API for Structured Outputs

```python
from litemind import OpenAIApi
from litemind.agent.agent import Agent
from pydantic import BaseModel

class WeatherResponse(BaseModel):
    temperature: float
    condition: str
    humidity: float

api = OpenAIApi()
agent = Agent(api=api)
agent.append_system_message("You are a weather bot.")

response = agent("What is the weather like in Paris?", response_format=WeatherResponse)
print(response)
# Output: [*assistant*: WeatherResponse(temperature=..., condition=..., humidity=...)]
```

---

### 4. Agent Tool That Generates an Image Using the Wrapper API

```python
from litemind.agent.agent import Agent
from litemind.agent.tools.toolset import ToolSet
from litemind import OpenAIApi

def generate_cat_image() -> str:
    api = OpenAIApi()
    image = api.generate_image(positive_prompt="A cute fluffy cat", image_width=512, image_height=512)
    # Save or return the image as needed
    return "Cat image generated!"

toolset = ToolSet()
toolset.add_function_tool(generate_cat_image, "Generate a cat image")

api = OpenAIApi()
agent = Agent(api=api, toolset=toolset)
agent.append_system_message("You are an assistant that can generate images.")

response = agent("Please generate a cat image.")
print(response)
```

---

## Command Line Tools

Litemind provides several command-line tools for code generation, repo export, and model feature scanning.

### Usage

```bash
litemind codegen [api] [-m MODEL] [-f FILE]
litemind export [-f FOLDER_PATH] [-o OUTPUT_FILE] [-e EXTENSIONS] [-x EXCLUDE]
litemind scan [api] [-m MODELS] [-o OUTPUT_DIR]
```

- `codegen`: Generate files (e.g., README.md) for a Python repository using an LLM.
- `export`: Export the entire repository to a single file.
- `scan`: Scan models from an API for supported features and generate a report.

### Example: Generating a README.md

1. Create a `.codegen` folder in the root of your repository.
2. Add a `*.codegen.yml` file describing the prompt and folder structure.

#### Example `.codegen.yml` Template

```yaml
prompt: |
  Please generate a detailed, complete and informative README.md file for this repository.
folder:
  path: .
  extensions: [".py", ".md", ".toml", "LICENSE"]
  excluded: ["dist", "build", "litemind.egg-info"]
file: README.md
```

3. Run:

```bash
litemind codegen openai -m gpt-4o -f README
```

### Notes

- The CLI tools are implemented in the `litemind.tools` package.
- You can apply the `.codegen` approach to your own repositories for automated documentation and code generation.

---

## Caveats and Limitations

- **Error Handling**: Some error handling, especially in the API wrapper layer, can be improved.
- **Token Management**: There is currently no explicit management of token usage or limits.
- **API Key Management**: API keys are managed via environment variables; consider more secure solutions for production.
- **Performance**: No explicit caching or async support yet; large RAG databases may impact performance.
- **Failing Tests**: See the Code Health section for test coverage and any failing tests.
- **Streaming**: Not all providers support streaming responses.
- **Security**: Do not use `exec`/`eval` tools on untrusted input.

---

## Code Health

- **Unit Tests**: Litemind has a comprehensive test suite covering all major features, including text generation, tool use, multimodal inputs, augmentations, and media conversions.
- **Test Results**: All tests pass as of the latest release. (No failed test files found in `test_reports/`.)
- **Test Coverage**: High coverage across agentic API, wrapper API, tools, augmentations, and media types.
- **Criticality of Failures**: No critical failures detected.
- **Analysis**: See `ANALYSIS.md` for a detailed code quality and architecture review.

---

## API Keys

Litemind requires API keys for LLM providers (OpenAI, Anthropic, Gemini, etc.). Set the following environment variables:

- **OpenAI**: `OPENAI_API_KEY`
- **Anthropic**: `ANTHROPIC_API_KEY`
- **Gemini**: `GOOGLE_GEMINI_API_KEY`

### Setting Environment Variables

#### Linux/macOS

Add to your `~/.bashrc`, `~/.zshrc`, or `~/.profile`:

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_GEMINI_API_KEY="..."
```

#### Windows

Add to your user environment variables via the System Properties dialog, or in `cmd`:

```cmd
setx OPENAI_API_KEY "sk-..."
setx ANTHROPIC_API_KEY "sk-ant-..."
setx GOOGLE_GEMINI_API_KEY "..."
```

#### Python `.env` File

Alternatively, use a `.env` file and a library like `python-dotenv` to load environment variables.

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

Contributions are welcome! Please see [CONTRIBUTING.md](./CONTRIBUTING.md) for detailed guidelines.

- Fork the repository and create a feature branch.
- Write tests for new features using `pytest`.
- Ensure code is formatted with `black`, `isort`, and passes `flake8` and `mypy`.
- Update documentation and the README as needed.
- Submit a pull request for review.

---

## License

BSD 3-Clause License. See [LICENSE](./LICENSE) for details.

---

## Logging

Litemind uses [Arbol](http://github.com/royerlab/arbol) for hierarchical, colorized logging. You can deactivate logging by setting `Arbol.passthrough = True` in your code.

---

## Acknowledgements

This README was generated with the help of AI.

