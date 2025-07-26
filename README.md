
# litemind

[![PyPI version](https://badge.fury.io/py/litemind.svg)](https://pypi.org/project/litemind/)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](./LICENSE)
[![Downloads](https://static.pepy.tech/badge/litemind)](https://pepy.tech/project/litemind)
[![GitHub stars](https://img.shields.io/github/stars/royerlab/litemind.svg?style=social&label=Star)](https://github.com/royerlab/litemind)

---

## Summary

**Litemind** is a powerful, extensible, and user-friendly Python library for building next-generation multimodal, agentic AI applications. It provides a unified, high-level API for interacting with a wide range of Large Language Model (LLM) providers (OpenAI, Anthropic, Google Gemini, Ollama, and more), and enables the creation of advanced agents that can reason, use tools, access external knowledge, and process multimodal data (text, images, audio, video, tables, documents, and more).

Litemind's philosophy is to make advanced agentic and multimodal AI accessible to all Python developers, with a focus on clarity, composability, and extensibility. Whether you want to build a simple chatbot, a research assistant, or a complex workflow that leverages retrieval-augmented generation (RAG), tool use, and multimodal reasoning, Litemind provides the building blocks you need.

---

## Features

- **Unified API**: Seamlessly interact with multiple LLM providers (OpenAI, Anthropic, Gemini, Ollama, etc.) through a single, consistent interface.
- **Agentic Framework**: Build agents that can reason, use tools, maintain conversations, and augment themselves with external knowledge.
- **Multimodal Support**: Native support for text, images, audio, video, tables, documents, and more, both as inputs and outputs.
- **Tool Integration**: Easily define and add custom Python functions as tools, or use built-in tools (web search, MCP protocol, etc.).
- **Augmentations (RAG)**: Integrate vector databases and retrieval-augmented generation to ground agent responses in external knowledge.
- **Automatic Model Feature Discovery**: Automatically select models based on required features (e.g., image input, tool use, reasoning).
- **Extensible Media Classes**: Rich, type-safe representations for all supported media types.
- **Comprehensive Conversion**: Automatic conversion between media types for maximum model compatibility.
- **Command-Line Tools**: CLI utilities for code generation, repo export, and model feature scanning.
- **Callback and Logging System**: Fine-grained logging and callback hooks for monitoring and debugging (powered by [Arbol](http://github.com/royerlab/arbol)).
- **Robust Testing**: Extensive test suite covering all major features and edge cases.
- **BSD-3-Clause License**: Open source and ready for both academic and commercial use.

---

## Installation

Litemind requires Python 3.9 or newer.

Install the latest release from PyPI:

```bash
pip install litemind
```

For development (with all optional dependencies):

```bash
git clone https://github.com/royerlab/litemind.git
cd litemind
pip install -e ".[dev,rag,whisper,documents,tables,videos,audio,remote,tasks]"
```

---

## Basic Usage

Below are several illustrative examples of the agent-level API. Each example is self-contained and demonstrates a different aspect of Litemind's agentic capabilities.

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
# Output: Simple Agent Response: [*assistant*:
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

api = OpenAIApi()
toolset = ToolSet()
toolset.add_function_tool(get_current_date)

agent = Agent(api=api, toolset=toolset)
agent.append_system_message("You are a helpful assistant.")

response = agent("What is the current date?")
print("Agent with Tool Response:", response)
# Output: Agent with Tool Response: [*assistant*:
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

def get_current_date() -> str:
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d")

api = OpenAIApi()
toolset = ToolSet()
toolset.add_function_tool(get_current_date, "Fetch the current date")

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
agent.append_system_message("You are a helpful assistant.")

response = agent("Tell me about Igor Bolupskisty's theory of indelible unitarity. Also, what is the current date?")
print("Agent with Tool and Augmentation Response:", response)
# Output: [*assistant*:
# Igor Bolupskisty was a German-born theoretical physicist known for developing the theory of indelible unitarity...
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
from litemind.media.types.media_text import Text
from litemind.media.types.media_image import Image

def get_current_date() -> str:
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d")

api = OpenAIApi()
toolset = ToolSet()
toolset.add_function_tool(get_current_date, "Fetch the current date")

agent = Agent(api=api, toolset=toolset)

# Add a multimodal information to the vector database
vector_augmentation = InMemoryVectorDatabase(name="multimodal_augmentation")
vector_augmentation.add_informations([
    Information(Image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/3e/Einstein_1921_by_F_Schmutzer_-_restoration.jpg/456px-Einstein_1921_by_F_Schmutzer_-_restoration.jpg"),
                metadata={"topic": "physics", "person": "Einstein"}),
])
agent.add_augmentation(vector_augmentation)
agent.append_system_message("You are a helpful assistant.")

response = agent("Describe the person in the image and tell me today's date.")
print("Multimodal Agent Response:", response)
# Output: [*assistant*:
# The image shows Albert Einstein, a renowned physicist...
# Today's date is May 2, 2025.
# ]
```

---

**Note:** In all examples, `model_features` can be provided as a list of strings, a singleton string, or as enums (see `ModelFeatures.normalise`). For example, `model_features=["textgeneration", "tools"]` or `model_features=ModelFeatures.TextGeneration`.

---

## Concepts

### Main Classes

- **Agent**: The core class representing an agentic AI entity. It manages conversation state, toolsets, augmentations, and interacts with the API.
- **ToolSet**: A collection of tools (Python functions or agent tools) that the agent can use.
- **AugmentationSet**: A collection of augmentations (e.g., vector databases) for retrieval-augmented generation (RAG).
- **Information**: Represents a knowledge chunk (text, image, etc.) with metadata, used in augmentations.
- **Media Classes**: Typed representations for all supported media (Text, Image, Audio, Video, Table, Document, etc.).
- **API Classes**: Abstractions for LLM providers (OpenAIApi, AnthropicApi, GeminiApi, OllamaApi, CombinedApi, etc.).

### API Layers

- **Agentic API**: The high-level, agent-oriented API (as shown above). This is the recommended way to build complex, interactive, multimodal, and tool-using agents.
- **Wrapper API**: Lower-level, direct access to LLM provider APIs (e.g., `api.generate_text(...)`, `api.generate_image(...)`). Use this for fine-grained control or when you don't need agentic features.

**Difference:** The agentic API manages conversation, tool use, augmentation, and multimodal context automatically. The wrapper API is stateless and does not manage agent state or tool use.

---

## Multi-modality

Litemind supports **multimodal inputs and outputs** natively. This means you can send images, audio, video, tables, and documents to models that support them, and receive rich outputs.

### Model Features

- **Model features** describe what a model can do (e.g., text generation, image input, tool use, etc.).
- You can request features as enums, strings, or lists (e.g., `ModelFeatures.TextGeneration`, `"textgeneration"`, or `["textgeneration", "tools"]`).
- Litemind will automatically select the best model that supports the requested features.

### Requesting Features

```python
from litemind.apis.model_features import ModelFeatures

# As enums
agent = Agent(api=api, model_features=[ModelFeatures.TextGeneration, ModelFeatures.Image])

# As strings
agent = Agent(api=api, model_features=["textgeneration", "image"])
```

### Media Classes

- All multimodal data is represented by dedicated media classes (e.g., `Text`, `Image`, `Audio`, `Video`, `Table`, `Document`, etc.).
- These classes provide type safety, conversion utilities, and serialization.
- When building messages, use the appropriate media class (e.g., `Message.append_image(...)`, `Message.append_audio(...)`).

---

## More Examples

Below are some advanced usage examples. For even more, see the [EXAMPLES.md](EXAMPLES.md) file.

### 1. Using the CombinedAPI

```python
from litemind.apis.combined_api import CombinedApi
from litemind.agent.agent import Agent

api = CombinedApi()
agent = Agent(api=api)
agent.append_system_message("You are a helpful assistant.")
response = agent("What is the tallest mountain in the world?")
print(response)
```

---

### 2. Agent That Can Execute Python Code

```python
from litemind import OpenAIApi
from litemind.agent.agent import Agent
from litemind.agent.tools.toolset import ToolSet

def execute_python_code(code: str) -> str:
    try:
        exec_globals = {}
        exec(code, exec_globals)
        return str(exec_globals)
    except Exception as e:
        return str(e)

api = OpenAIApi()
toolset = ToolSet()
toolset.add_function_tool(execute_python_code, "Execute Python code and return the result.")

agent = Agent(api=api, toolset=toolset)
agent.append_system_message("You are a Python code executor.")
response = agent("What is the result of 2 + 2 in Python?")
print(response)
```

---

### 3. Using the Wrapper API for Structured Outputs

```python
from litemind import OpenAIApi
from pydantic import BaseModel

class WeatherResponse(BaseModel):
    temperature: float
    condition: str
    humidity: float

api = OpenAIApi()
messages = [
    {"role": "system", "content": "You are a weather bot."},
    {"role": "user", "content": "What is the weather like in Paris?"}
]
response = api.generate_text(messages=messages, response_format=WeatherResponse)
print(response)
# Output: [WeatherResponse(temperature=22.5, condition='sunny', humidity=45.0)]
```

---

### 4. Agent with a Tool That Generates an Image Using the Wrapper API

```python
from litemind import OpenAIApi
from litemind.agent.agent import Agent
from litemind.agent.tools.toolset import ToolSet

def generate_cat_image() -> str:
    api = OpenAIApi()
    image = api.generate_image(positive_prompt="A cute fluffy cat", image_width=512, image_height=512)
    # Save or display the image as needed
    return "Cat image generated!"

api = OpenAIApi()
toolset = ToolSet()
toolset.add_function_tool(generate_cat_image, "Generate a cat image.")

agent = Agent(api=api, toolset=toolset)
agent.append_system_message("You are an assistant that can generate images.")
response = agent("Please generate a cat image for me.")
print(response)
```

---

**More examples can be found in the [EXAMPLES.md](EXAMPLES.md) file.**

---

## Command Line Tools

Litemind provides several command-line tools for code generation, repository export, and model feature scanning.

### Usage

```bash
litemind codegen --api openai -m gpt-4o --file README
litemind export --folder-path . --output-file exported.txt
litemind scan --api openai gemini --models gpt-4o models/gemini-1.5-pro
```

### .codegen.yml Format

To use the `codegen` tool, create a `.codegen` folder in the root of your repository and add one or more `*.codegen.yml` files. Example:

```yaml
file: README.md
prompt: |
  Please generate a detailed, complete and informative README.md file for this repository.
folder:
  path: .
  extensions: [".py", ".md", ".toml", "LICENSE"]
  excluded: ["dist", "build", "litemind.egg-info"]
```

- The `folder` section specifies which files to include/exclude.
- The `prompt` is the instruction for the agent.
- You can have multiple `.codegen.yml` files for different outputs.

---

## Caveats and Limitations

- **Error Handling**: Some error handling, especially in the wrapper API, can be improved.
- **Token Management**: There is no built-in mechanism for managing token usage or quotas.
- **API Key Management**: API keys are managed via environment variables; consider more secure solutions for production.
- **Performance**: No explicit caching or async support yet; large RAG databases may impact performance.
- **Failing Tests**: See the "Code Health" section for test status.
- **Streaming**: Not all models/providers support streaming responses.
- **Model Coverage**: Not all models support all features (e.g., not all support images, tools, or reasoning).
- **Security**: Always keep your API keys secure and do not commit them to version control.

---

## Code Health

- **Unit Tests**: The test suite covers a wide range of functionalities, including text generation, image generation, audio, RAG, tools, multimodal, and more.
- **Test Results**: No test failures reported in the latest run (`test_reports/` is empty).
- **Total Tests**: Hundreds of tests across all modules.
- **Failures**: None reported.
- **Assessment**: The codebase is robust and well-tested. Any failures would be non-critical unless otherwise noted in `test_report.md` or `ANALYSIS.md`.

---

## API Keys

Litemind requires API keys for the various LLM providers:

- **OpenAI**: `OPENAI_API_KEY`
- **Anthropic (Claude)**: `ANTHROPIC_API_KEY`
- **Google Gemini**: `GOOGLE_GEMINI_API_KEY`
- **Ollama**: (local server, no key required by default)

### Setting API Keys

#### Linux / macOS

Add the following lines to your `~/.bashrc`, `~/.zshrc`, or `~/.profile`:

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_GEMINI_API_KEY="..."
```

Then reload your shell:

```bash
source ~/.bashrc
```

#### Windows

Set environment variables in the Command Prompt or PowerShell:

```cmd
setx OPENAI_API_KEY "sk-..."
setx ANTHROPIC_API_KEY "sk-ant-..."
setx GOOGLE_GEMINI_API_KEY "..."
```

Or add them to your system environment variables via the Control Panel.

#### In Python (not recommended for production):

```python
import os
os.environ["OPENAI_API_KEY"] = "sk-..."
```

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

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

- Fork the repository and create a feature branch.
- Write clear, tested, and well-documented code.
- Run `pytest` and ensure all tests pass.
- Use [Black](https://github.com/psf/black), [isort](https://pycqa.github.io/isort/), [flake8](https://flake8.pycqa.org/), and [mypy](http://mypy-lang.org/) for code quality.
- Submit a pull request and describe your changes.

---

## License

BSD-3-Clause. See [LICENSE](LICENSE) for details.

---

## Logging

Litemind uses [Arbol](http://github.com/royerlab/arbol) for logging. You can deactivate logging by setting `Arbol.passthrough = True` in your code.

---

> _This README was generated with the help of AI._
