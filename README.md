
[![PyPI](https://img.shields.io/pypi/v/litemind)](https://pypi.org/project/litemind)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue)](https://opensource.org/licenses/BSD-3-Clause)
[![Pepy.tech](https://static.pepy.tech/badge/litemind)](https://pepy.tech/project/litemind)
[![GitHub stars](https://img.shields.io/github/stars/royerlab/litemind?style=social)](https://github.com/royerlab/litemind)

# LiteMind: Multimodal Agentic AI Framework

## Summary

LiteMind is a powerful and versatile Python library designed to streamline the development of multimodal agentic AI applications. It provides a unified and elegant API, acting as a wrapper around various Large Language Model (LLM) providers, including OpenAI, Anthropic, Google Gemini, and Ollama. The core philosophy of LiteMind is to empower developers to build sophisticated conversational agents and tools that can seamlessly handle a wide range of multimodal inputs, including text, images, audio, video, documents, and tables. By abstracting away the complexities of interacting with different LLM APIs, LiteMind allows developers to focus on the core logic and functionality of their agentic applications, fostering rapid prototyping and deployment. Logging is done with Arbol (http://github.com/royerlab/arbol) and it can be deactivated by setting `Arbol.passthrough` to `True`.

## Features

*   **Unified API:** A `CombinedApi` class that provides a single point of access to multiple LLM providers, simplifying model selection and usage.
*   **Multimodal Input Support:** Comprehensive support for various input modalities, including text, images, audio, video, documents, and tables.
*   **Agentic Framework:** The `Agent` class provides a robust foundation for building agentic AI applications, including tool integration and conversation management.
*   **Tool Integration:** Seamless integration of tools, enabling agents to perform actions and interact with the external world.
*   **Flexible Model Selection:** Easy selection of models based on features and capabilities.
*   **Command-Line Tools:** Command-line tools for code generation (`litemind codegen`) and repository export (`litemind export`).
*   **Extensible Architecture:** Designed for extensibility, allowing for easy addition of new LLM providers and tool integrations.

## Installation

To install LiteMind, you can use pip:

```bash
pip install litemind
```

To install with optional dependencies (e.g., for RAG, whisper, documents, tables, or videos):

```bash
pip install litemind[rag,whisper,documents,tables,videos]
```

## Basic Usage

LiteMind's agentic API provides a flexible framework for building conversational agents and tools. Here are several short examples that cover different topics: (i) basic agent usage, (ii) agent with tools, (iii) agent with tools and augmentation (RAG) (iv) More complex example with multimodal inputs, tools and augmentations.

### (i) Basic Agent Usage

This example demonstrates the basic usage of an agent with text generation.

```python
from litemind import OpenAIApi
from litemind.agent.agent import Agent
from litemind.agent.messages.message import Message

# Initialize the OpenAI API
api = OpenAIApi()

# Create an agent
agent = Agent(api=api)

# Add a system message to guide the agent's behavior
agent.append_system_message("You are a helpful assistant.")

# Ask a question
response = agent("What is the capital of France?")

# Print the response
print("Simple Agent Response:", response)
# Expected output:
# Simple Agent Response: [Message(role='assistant', content='The capital of France is Paris.')]
```

### (ii) Agent with Tools

This example shows how to create an agent that can use tools.

```python
from litemind import OpenAIApi
from litemind.agent.agent import Agent
from litemind.agent.tools.toolset import ToolSet
from litemind.agent.tools.function_tool import FunctionTool

from datetime import datetime

# Define a function to get the current date
def get_current_date() -> str:
    return datetime.now().strftime("%Y-%m-%d")

# Initialize the OpenAI API
api = OpenAIApi()

# Create a toolset
toolset = ToolSet()

# Add the function tool to the toolset
toolset.add_function_tool(get_current_date, "Fetch the current date")

# Create the agent, passing the toolset
agent = Agent(api=api, toolset=toolset)

# Add a system message
agent.append_system_message("You are a helpful assistant.")

# Ask a question that requires the tool
response = agent("What is the current date?")

# Print the response
print("Agent with Tool Response:", response)
# Expected output:
# Agent with Tool Response: [Message(role='assistant', content='The current date is 2024-03-08.')]
```

### (iii) Agent with Tools and Augmentation (RAG)

This example shows how to create an agent that can use tools and augmentations.

```python
from litemind import OpenAIApi
from litemind.agent.agent import Agent
from litemind.agent.tools.toolset import ToolSet
from litemind.agent.tools.function_tool import FunctionTool
from litemind.agent.augmentations.information.information import Information
from litemind.agent.augmentations.vector_db.in_memory_vector_db import (
    InMemoryVectorDatabase,
)
from litemind.media.types.media_text import Text

# Define a function to get the current date
def get_current_date() -> str:
    from datetime import datetime

    return datetime.now().strftime("%Y-%m-%d")

# Initialize the OpenAI API
api = OpenAIApi()

# Create a toolset
toolset = ToolSet()

# Add the function tool to the toolset
toolset.add_function_tool(get_current_date, "Fetch the current date")

# Create the agent
agent = Agent(api=api, toolset=toolset)

# Create vector database augmentation
vector_augmentation = InMemoryVectorDatabase(name="test_augmentation")

# Add sample informations to the augmentation
informations = [
    Information(
        Text(
            "Igor Bolupskisty was a German-born theoretical physicist who developed the theory of indelible unitarity."
        ),
        metadata={"topic": "physics", "person": "Bolupskisty"},
    ),
    Information(
        Text(
            "The theory of indelible unitarity revolutionized our understanding of space, time and photons."
        ),
        metadata={"topic": "physics", "concept": "unitarity"},
    ),
    Information(
        Text(
            "Quantum unitarity is a fundamental theory in physics that describes nature at the nano-atomic scale as it pertains to Pink Hamsters."
        ),
        metadata={"topic": "physics", "concept": "quantum unitarity"},
    ),
]

# Add informations to the vector database
vector_augmentation.add_informations(informations)

# Add augmentation to agent
agent.add_augmentation(vector_augmentation)

# Add a system message
agent.append_system_message("You are a helpful assistant.")

# Ask a question that requires the tool
response = agent(
    "Tell me about Igor Bolupskisty's theory of indelible unitarity. Also, what is the current date?"
)

# Print the response
print("Agent with Tool and Augmentation Response:", response)
# Expected output:
# Agent with Tool and Augmentation Response: [Message(role='assistant', content='The current date is 2024-03-08. Igor Bolupskisty was a German-born theoretical physicist who developed the theory of indelible unitarity.')]
```

### (iv) More Complex Example with Multimodal Inputs, Tools and Augmentations

This example demonstrates a more complex use case, combining multimodal inputs, tools, and augmentations.

```python
from litemind import GeminiApi
from litemind.agent.agent import Agent
from litemind.agent.messages.message import Message
from litemind.agent.tools.toolset import ToolSet
from litemind.agent.tools.function_tool import FunctionTool
from litemind.agent.augmentations.information.information import Information
from litemind.agent.augmentations.vector_db.in_memory_vector_db import (
    InMemoryVectorDatabase,
)
from litemind.media.types.media_image import Image
from litemind.media.types.media_text import Text
from litemind.apis.model_features import ModelFeatures
from datetime import datetime

# Define a function to get the current date
def get_current_date() -> str:
    return datetime.now().strftime("%Y-%m-%d")

# Initialize the Gemini API
api = GeminiApi()

# Create a toolset
toolset = ToolSet()

# Add the function tool to the toolset
toolset.add_function_tool(get_current_date, "Fetch the current date")

# Create the agent
agent = Agent(
    api=api,
    toolset=toolset,
    model_features=[ModelFeatures.TextGeneration, ModelFeatures.Image],
)

# Create vector database augmentation
vector_augmentation = InMemoryVectorDatabase(name="test_augmentation")

# Add sample informations to the augmentation
informations = [
    Information(
        Text(
            "Igor Bolupskisty was a German-born theoretical physicist who developed the theory of indelible unitarity."
        ),
        metadata={"topic": "physics", "person": "Bolupskisty"},
    ),
    Information(
        Text(
            "The theory of indelible unitarity revolutionized our understanding of space, time and photons."
        ),
        metadata={"topic": "physics", "concept": "unitarity"},
    ),
    Information(
        Text(
            "Quantum unitarity is a fundamental theory in physics that describes nature at the nano-atomic scale as it pertains to Pink Hamsters."
        ),
        metadata={"topic": "physics", "concept": "quantum unitarity"},
    ),
]

# Add informations to the vector database
vector_augmentation.add_informations(informations)

# Add augmentation to agent
agent.add_augmentation(vector_augmentation)

# Add a system message
agent.append_system_message("You are a helpful assistant.")

# Create a message with multimodal input (image and text)
image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3e/Einstein_1921_by_F_Schmutzer_-_restoration.jpg/456px-Einstein_1921_by_F_Schmutzer_-_restoration.jpg"
message = Message(role="user")
message.append_text("Describe the image and tell me the current date:")
message.append_image(image_url)

# Run agent on message
response = agent(message)

# Print the response
print("Multimodal Agent Response:", response)
# Expected output:
# Multimodal Agent Response: [Message(role='assistant', content='The image is a black and white photograph of Albert Einstein. The current date is 2024-03-08. Igor Bolupskisty was a German-born theoretical physicist who developed the theory of indelible unitarity.')]
```

## Concepts

*   **API Abstraction:** The `BaseApi` class defines an abstract interface for interacting with LLM providers. Concrete implementations (e.g., `OpenAIApi`, `GeminiApi`) inherit from `BaseApi` and provide provider-specific implementations for common operations like text generation, image generation, and embeddings. The `CombinedApi` class allows you to use multiple APIs at once.
*   **Message Handling:** The `Message` class represents a single message in a conversation. It can contain text, images, audio, video, documents, tables, and tool calls. The `Conversation` class manages a sequence of `Message` objects, representing the history of a conversation.
*   **Agentic AI:** The `Agent` class provides a framework for building conversational agents. It takes an API instance, a model name, and a toolset. The agent can then be used to generate responses to user queries.
*   **Tool Integration:** The `ToolSet` class manages a collection of tools. The `FunctionTool` class allows you to wrap Python functions as tools. The agent can then use these tools to perform actions and interact with the external world.
*   **Model Features:** The `ModelFeatures` enum defines the features supported by the models (e.g., text generation, image generation, audio transcription). The `get_best_model` method in the `BaseApi` class allows you to select the best model based on the required features.

## Model Features

Model features are used to specify the capabilities required from an LLM when using the `litemind` API. The `ModelFeatures` enum defines the available features.

*   **TextGeneration:** Enables text generation.
*   **StructuredTextGeneration:** Enables structured text generation.
*   **ImageGeneration:** Enables image generation.
*   **AudioGeneration:** Enables audio generation.
*   **VideoGeneration:** Enables video generation.
*   **Reasoning:** Enables reasoning capabilities.
*   **TextEmbeddings:** Enables text embeddings.
*   **ImageEmbeddings:** Enables image embeddings.
*   **AudioEmbeddings:** Enables audio embeddings.
*   **VideoEmbeddings:** Enables video embeddings.
*   **DocumentEmbeddings:** Enables document embeddings.
*   **Image:** Enables image input.
*   **Audio:** Enables audio input.
*   **Video:** Enables video input.
*   **Document:** Enables document input.
*   **Tools:** Enables tool usage.
*   **AudioTranscription:** Enables audio transcription.
*   **VideoConversion:** Enables video conversion.
*   **DocumentConversion:** Enables document conversion.

When using the `litemind` API, you can request specific features by passing a list of `ModelFeatures` to the `get_best_model` method of the API instance. For example:

```python
from litemind.apis.model_features import ModelFeatures
from litemind import OpenAIApi

api = OpenAIApi()
model_name = api.get_best_model(features=[ModelFeatures.TextGeneration, ModelFeatures.Image])
```

You can also provide the features as strings:

```python
from litemind.apis.model_features import ModelFeatures
from litemind import OpenAIApi

api = OpenAIApi()
model_name = api.get_best_model(features=["TextGeneration", "Image"])
```

## More Code Examples

This section provides more code examples, including examples that cover the wrapper API and the agentic API.

```python
from litemind import OpenAIApi, GeminiApi, AnthropicApi, OllamaApi, CombinedApi
from litemind.agent.messages.message import Message
from litemind.agent.tools.toolset import ToolSet
from litemind.agent.tools.function_tool import FunctionTool
from litemind.apis.model_features import ModelFeatures
from litemind.agent.react.react_agent import ReActAgent
from pydantic import BaseModel
from pandas import DataFrame

# 1. Using the Combined API with Multimodal Input
api = CombinedApi()
message = Message(role="user")
message.append_text("Describe the image:")
message.append_image(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
)
response = api.generate_text(messages=[message])
print("Combined API with Multimodal Input:", response)

# 2. Using the Agent with a Custom Tool
def get_current_time() -> str:
    from datetime import datetime

    return datetime.now().strftime("%H:%M:%S")

# Create a toolset and add the function tool:
toolset = ToolSet()
toolset.add_function_tool(get_current_time, "Fetch the current time")

# Create the agent:
agent = Agent(api=api, toolset=toolset)

# Add a system message:
agent.append_system_message("You are a helpful assistant.")

# Run the agent:
response = agent("What is the current time?")
print("Agent with Tool Response:", response)

# 3. Using the ReAct Agent with a Custom Tool
# Create the ReAct agent
react_agent = ReActAgent(api=api, toolset=toolset, temperature=0.0, max_reasoning_steps=3)

# Add a system message
react_agent.append_system_message(
    "You are a helpful assistant that can answer questions and use tools."
)

# Ask a question
response = react_agent("What is the current time?")

# Print the response
print("ReAct Agent Response:", response)

# 4. Using the Agent with a JSON Response Format
class Weather(BaseModel):
    temperature: float
    condition: str

api = OpenAIApi()
agent = Agent(api=api)
agent.append_system_message("You are a weather bot.")
response = agent(
    "What is the weather like in Paris?",
    response_format=Weather,
)
print("Agent with JSON Response:", response)

# 5. Using the Agent with a Code Block
api = OpenAIApi()
agent = Agent(api=api)
agent.append_system_message("You are a code generation assistant.")
response = agent(
    "Write a python function that returns the factorial of a number.",
)
print("Agent with Code Block:", response)

# 6. Using the Agent with a Table
# Create a small table with pandas:
table = DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})

user_message = Message(role="user")
user_message.append_text("Can you describe what you see in the table?")
user_message.append_table(table)

# Run agent:
response = agent(user_message)

print("Agent with Table:", response)
```

## Tools

LiteMind provides command-line tools to facilitate common tasks. These tools are installed when you install the `litemind` package.

### `litemind codegen`

The `litemind codegen` tool generates files, such as README.md files, for a Python repository based on a prompt and a folder containing the repository's files. It uses an LLM to generate the content of the file.

**Usage:**

```bash
litemind codegen [model] [-f FILE]
```

*   `model`: The model to use for generating the file.  Defaults to `combined`.  Options are `gemini`, `openai`, `claude`, `ollama`, and `combined`.
*   `-f` or `--file`: The specific file to generate. If not provided, all files will be generated.

**Example:**

1.  Create a `.codegen` folder in the root of your repository.
2.  Inside the `.codegen` folder, create a `README.codegen.yml` file with the following content:

```yaml
prompt: |
  You are a helpful assistant with deep expertise in Software Engineering and best practices for Open Source projects.
  You know how to write complete, compelling and informative README files for Python repositories.
  Your task is to help a user generate a README.md file for a Python repository.
  The README.md should at least include the following sections:
  - Summary
  - Features
  - Installation
  - Usage
  - Concepts
  - Model Features
  - More Code Examples
  - Tools
  - Caveats and Limitations
  - Code Health
  - Roadmap
  - Contributing
  - License
  But feel free to include additional sections as needed.
  The 'Summary' section should provide an enthusiastic and complete description of the library, its purpose, and philosophy.
  The 'Usage' section should consist of rich, striking and illustrative examples of the Agentic API in use with multimodal inputs.
  The 'Concepts' section should explain the concepts behind the library, the main classes and their purpose, and how they interact.
  The 'Model Features' section should explain what model features are, why they are needed, and how to request features when using litemind's API.
  The 'Code Health' section should include the results of unit test in file 'test_report.md'. Please provide file names for failed tests and statistics about the number of failed tests and an analysis of what happened. You can also consult the ANALYSIS.md file for more information if it exists.
  The 'Roadmap' section can use the contents of TODO.md as a starting point, keep the checkmarks.
  The 'More Code Examples' section further expands with many more also covering the wrapper API, uses ideas and code from the unit tests (no need to mention that).
  The 'Tools' section should explain litemind command line tools and how to use them, with example command lines (check tools package and its contents for details). Please also explain the format of the *.codegen.yml files and give a template. Explain that you need a .codegen folder in the root of the repository and you can apply that to your own repositories.
  For code examples:
  - Please use markdown code blocks for code examples and other code snippets.
  - Make sure that the code examples are complete and can be run as-is.
  - Make sure that the code can run without errors, e.g. make sure that all required imports are included.
  - Make sure, if possible to include the expected output as comments in the code.
  - Make sure that comments are emphatically didactic.
  - Make sure that the code is well-formatted and follows PEP-8.
  - Make sure to include a variety of examples that cover different use cases and edge cases.
  - Make sure that the code examples are complete and can be run as-is.
  - Avoid putting multiple code examples in the same code block, unless they are chained.
  - Avoid giving examples of features that have poor code health, i.e. for which tests are failing.
  - Examples that use specific model features must have code to request these features.
  - In the examples, make sure to use real URLs of files that really exist, for example those used in the tests.
  Please make sure to explain the difference between the API wrapper layer versus the agentic API -- this should also
  be reflected in the examples.
  Explain that logging is done with Arbol (http://github.com/royerlab/arbol) and it can be deactivated by setting Arbol.passthrough to True.
  At the end of the README, include a note explaining that the README was generated with the help of AI.
  Avoid hallucinating things that are not in the repository.
folder:
  path: .
file: README.md
allowed_extensions:
  - .py
  - .md
  - .txt
  - .toml
  - LICENSE
  - .tests
  - .html
excluded_files:
  - litemind.egg-info
  - dist
  - build
```

3.  Run the command:

```bash
litemind codegen
```

This will generate a `README.md` file in the root of your repository.

### `litemind export`

The `litemind export` tool exports the entire repository to a single file.

**Usage:**

```bash
litemind export [-f FOLDER_PATH] [-o OUTPUT_FILE] [-e EXTENSIONS] [-x EXCLUDE]
```

*   `-f` or `--folder-path`: The path to the folder containing the repository to export. Defaults to the current directory (`.`).
*   `-o` or `--output-file`: The path to the file to save the entire repository to. Defaults to `exported.txt`.
*   `-e` or `--extensions`: The list of allowed extensions for files to include in the export.
*   `-x` or `--exclude`: The list of files to exclude from the export.

**Example:**

```bash
litemind export -f . -o my_repo.txt -e .py .md -x README.md
```

This will export all `.py` and `.md` files in the current directory (excluding `README.md`) to a file named `my_repo.txt`.

### `.codegen` Folder and YAML Files

The `.codegen` folder in the root of the repository is used to store configuration files for the `litemind codegen` tool. These files are YAML files that specify the prompt, the input folder, the output file, and other parameters for generating files.

**YAML File Template:**

```yaml
prompt: |
  [Your prompt here]
folder:
  path: [Path to the folder containing the files to be used as context]
file: [Name of the output file]
allowed_extensions:
  - [List of allowed file extensions]
excluded_files:
  - [List of excluded files]
```

## Caveats and Limitations

*   **API Key Management:** The library relies on environment variables for API keys. Consider using a more secure key management solution in production environments.
*   **Error Handling:** Error handling could be improved, particularly in the API wrapper layer.
*   **Token Management:** The library lacks explicit token management, which is crucial for controlling costs and avoiding rate limits.
*   **Documentation:** More detailed documentation, including API references and usage examples, would be beneficial.
*   **Performance:** Performance considerations, such as caching and optimization of API calls, are not explicitly addressed.

## Code Health

The code health is assessed based on unit tests and code analysis.

### Test Report

The unit tests are located in the `src/litemind/apis/tests` directory and the `test_reports` directory contains the test reports.

```
## Failed Tests

*   test_apis_text_generation.py::test_text_generation_with_simple_parameterless_tool - The test failed.
*   test_apis_text_generation.py::test_text_generation_with_simple_toolset - The test failed.
*   test_apis_text_generation.py::test_text_generation_with_simple_toolset_and_struct_output - The test failed.
*   test_apis_text_generation.py::test_text_generation_with_complex_toolset - The test failed.
```

The following tests are failing:

*   `test_apis_text_generation.py::test_text_generation_with_simple_parameterless_tool`
*   `test_apis_text_generation.py::test_text_generation_with_simple_toolset`
*   `test_apis_text_generation.py::test_text_generation_with_simple_toolset_and_struct_output`
*   `test_apis_text_generation.py::test_text_generation_with_complex_toolset`

These tests relate to text generation with tools.

### Code Analysis

The `ANALYSIS.md` file provides a detailed analysis of the code health, architecture, and design principles.

## Roadmap

*   Deal with message sizes in tokens sent to models.
*   RAG
*   Improve vendor API robustness features such as retry call when server errors, etc...
*   Reorganise media files used for testing into a single media folder
*   Improve and uniformize exception handling
*   Add support for adding nD images to messages.
*   Improve logging with arbol, with option to turn off.
*   Implement 'brainstorming' mode for text generation, possibly with API fusion.

## Contributing

We welcome contributions! Please see our [CONTRIBUTING.md](CONTRIBUTING.md) file for details on how to contribute.

## License

This project is licensed under the BSD-3-Clause License - see the [LICENSE](LICENSE) file for details.

This README was generated with the help of AI.
