
# LiteMind: Multimodal Agentic AI Framework

[![PyPI](https://img.shields.io/pypi/v/litemind.svg)](https://pypi.org/project/litemind/)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Pepy](https://static.pepy.tech/badge/litemind)](https://pepy.tech/project/litemind)
[![Stars](https://img.shields.io/github/stars/royerlab/litemind?style=social)](https://github.com/royerlab/litemind)

LiteMind is a Python library designed to simplify the development of multimodal agentic AI applications. It provides a unified API for interacting with various LLM providers and supports multimodal inputs (text, images, audio, video, documents, and tables). The project aims to streamline the creation of conversational agents and tools.

## Features

*   **Unified API:** A `CombinedApi` class provides a single point of access to multiple LLM providers, simplifying model selection and usage.
*   **Multimodal Input Support:** Comprehensive support for various input modalities, including text, images, audio, video, documents, and tables.
*   **Agentic Framework:** The `Agent` class provides a robust foundation for building agentic AI applications, including tool integration and conversation management.
*   **Tool Integration:** Seamless integration of tools, enabling agents to perform actions and interact with the external world.
*   **Extensible Architecture:** Designed for extensibility, allowing for easy addition of new LLM providers and tool integrations.
*   **Command-Line Tools:** Includes command-line tools for code generation (`litemind codegen`) and repository export (`litemind export`).
*   **Logging:** Logging is done with Arbol (http://github.com/royerlab/arbol) and it can be deactivated by setting Arbol.passthrough to True.

## Installation

To install LiteMind, use pip:

```bash
pip install litemind
```

To install optional dependencies for specific features (e.g., RAG, documents, tables, videos, audio), use the following:

```bash
pip install litemind[rag]
pip install litemind[documents]
pip install litemind[tables]
pip install litemind[videos]
pip install litemind[audio]
```

## Basic Usage

This section provides several short examples that cover different topics: (i) basic agent usage, (ii) agent with tools, (iii) agent with tools and augmentation (RAG) (iv) More complex example with multimodal inputs, tools and augmentations.

**(i) Basic Agent Usage**

This example shows how to create a simple agent that responds to a user's query.

```python
from litemind.apis.combined_api import CombinedApi
from litemind.agent.agent import Agent
from litemind.agent.messages.message import Message

# Initialize the API
api = CombinedApi()

# Create an agent
agent = Agent(api=api)

# Add a system message
agent.append_system_message("You are a helpful assistant.")

# Create a user message
user_message = Message(role="user", text="Hello, how are you?")

# Get the response
response = agent(user_message)

# Print the response
print(response)
```

**(ii) Agent with Tools**

This example shows how to create an agent that uses a tool.

```python
from litemind.apis.combined_api import CombinedApi
from litemind.agent.agent import Agent
from litemind.agent.messages.message import Message
from litemind.agent.tools.function_tool import FunctionTool

# Define a tool function
def get_current_date() -> str:
    from datetime import datetime

    return datetime.now().strftime("%Y-%m-%d")

# Initialize the API
api = CombinedApi()

# Create an agent
agent = Agent(api=api)

# Add a tool
toolset = agent.toolset
toolset.add_function_tool(get_current_date, "Returns the current date")

# Create a user message
user_message = Message(role="user", text="What is the current date?")

# Get the response
response = agent(user_message)

# Print the response
print(response)
```

**(iii) Agent with Tools and Augmentation (RAG)**

This example shows how to create an agent that uses a tool and an augmentation.

```python
from litemind.apis.combined_api import CombinedApi
from litemind.agent.agent import Agent
from litemind.agent.augmentations.information.information import Information
from litemind.agent.augmentations.vector_db.in_memory_vector_db import (
    InMemoryVectorDatabase,
)
from litemind.agent.messages.message import Message
from litemind.agent.tools.function_tool import FunctionTool
from litemind.media.types.media_text import Text

# Define a tool function
def get_current_date() -> str:
    from datetime import datetime

    return datetime.now().strftime("%Y-%m-%d")

# Initialize the API
api = CombinedApi()

# Create an agent
agent = Agent(api=api)

# Add a tool
toolset = agent.toolset
toolset.add_function_tool(get_current_date, "Returns the current date")

# Create vector database augmentation
vector_augmentation = InMemoryVectorDatabase(name="test_augmentation")

# Add sample informations to the augmentation
informations = [
    Information(
        Text("Igor Bolupskisty was a German-born theoretical physicist who developed the theory of indelible unitarity."),
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

# Create a user message
user_message = Message(role="user", text="Tell me about Igor Bolupskisty's theory of indelible unitarity.")

# Get the response
response = agent(user_message)

# Print the response
print(response)
```

**(iv) More complex example with multimodal inputs, tools and augmentations.**

This example shows how to create an agent that uses a tool, an augmentation, and multimodal inputs.

```python
from litemind.apis.combined_api import CombinedApi
from litemind.agent.agent import Agent
from litemind.agent.augmentations.information.information import Information
from litemind.agent.augmentations.vector_db.in_memory_vector_db import (
    InMemoryVectorDatabase,
)
from litemind.agent.messages.message import Message
from litemind.agent.tools.function_tool import FunctionTool
from litemind.media.types.media_image import Image
from litemind.media.types.media_text import Text
from litemind.ressources.media_resources import MediaResources

# Define a tool function
def get_current_date() -> str:
    from datetime import datetime

    return datetime.now().strftime("%Y-%m-%d")

# Initialize the API
api = CombinedApi()

# Create an agent
agent = Agent(api=api)

# Add a tool
toolset = agent.toolset
toolset.add_function_tool(get_current_date, "Returns the current date")

# Create vector database augmentation
vector_augmentation = InMemoryVectorDatabase(name="test_augmentation")

# Add sample informations to the augmentation
informations = [
    Information(
        Text("Igor Bolupskisty was a German-born theoretical physicist who developed the theory of indelible unitarity."),
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

# Create a user message
user_message = Message(role="user", text="Describe the image and tell me the current date.")
image_path = MediaResources.get_local_test_image_uri("cat.jpg")
user_message.append_image(image_path)

# Get the response
response = agent(user_message)

# Print the response
print(response)
```

## Concepts

*   **API Wrapper Layer vs. Agentic API:**
    *   The API wrapper layer provides a unified interface to interact with different LLM providers (OpenAI, Anthropic, etc.). It abstracts away the specifics of each provider's API, allowing you to switch between models easily. The `CombinedApi` class is a key component of this layer.
    *   The agentic API builds upon the API wrapper layer to provide an agentic framework. It includes the `Agent` class, which manages conversations, integrates tools, and uses augmentations to enhance the agent's capabilities.

*   **Key Classes:**
    *   `BaseApi`: Abstract base class for all API implementations.
    *   `CombinedApi`: Combines multiple API implementations.
    *   `Message`: Represents a single message in a conversation.
    *   `MessageBlock`: Represents a block of media content within a message.
    *   `Agent`: Manages conversations, integrates tools, and uses augmentations.
    *   `ToolSet`: Manages a collection of tools.
    *   `FunctionTool`: Wraps a function to be used as a tool.
    *   `AugmentationSet`: Manages a set of augmentations.
    *   `Information`: Represents a piece of information that can be stored in a vector database.
    *   `MediaBase`: Abstract base class for all media types.
    *   `Text`: Represents text media.
    *   `Image`: Represents image media.
    *   `Audio`: Represents audio media.
    *   `Video`: Represents video media.
    *   `Document`: Represents document media.
    *   `Table`: Represents table media.
    *   `Json`: Represents JSON media.
    *   `Code`: Represents code media.
    *   `Object`: Represents object media.

## Multi-modality

LiteMind supports multimodal inputs, allowing agents to process and generate content with various media types.

*   **Model Features:**
    *   `ModelFeatures` is an enumeration that defines the features supported by the models.
    *   When using the API, you can specify the required model features (e.g., `ModelFeatures.Image`, `ModelFeatures.Audio`, `ModelFeatures.TextGeneration`) to select a model that supports those features.
    *   ModelFeatures can be provided as a singleton or list of strings instead of enums (see normalisation code).

*   **Media Classes:**
    *   The `MediaBase` class and its subclasses (e.g., `Text`, `Image`, `Audio`, `Video`, `Document`, `Table`, `Json`, `Code`, `Object`) represent different media types.
    *   Each media class encapsulates the data and metadata for a specific media type.
    *   The `MessageBlock` class is used to store media content and its associated attributes.

*   **Requesting Features:**
    *   When using the `generate_text` method, you can specify the required model features using the `features` parameter.
    *   For example, to generate text with image input, you would specify `features=[ModelFeatures.TextGeneration, ModelFeatures.Image]`.

## More Examples

**(i) How to use the CombinedAPI**

```python
from litemind.apis.combined_api import CombinedApi
from litemind.agent.messages.message import Message

# Initialize the CombinedApi
api = CombinedApi()

# Create a user message
user_message = Message(role="user", text="Hello, world!")

# Get the response
response = api.generate_text(messages=[user_message])

# Print the response
print(response)
```

**(ii) Agent that can execute Python code**

```python
from litemind.apis.combined_api import CombinedApi
from litemind.agent.agent import Agent
from litemind.agent.messages.message import Message
from litemind.agent.tools.function_tool import FunctionTool

# Define a tool function
def execute_python_code(code: str) -> str:
    try:
        exec(code)
        return "Code executed successfully."
    except Exception as e:
        return f"Error executing code: {e}"

# Initialize the API
api = CombinedApi()

# Create an agent
agent = Agent(api=api)

# Add a tool
toolset = agent.toolset
toolset.add_function_tool(execute_python_code, "Executes Python code")

# Create a user message
user_message = Message(role="user", text="Execute this code: print('Hello from Python!')")

# Get the response
response = agent(user_message)

# Print the response
print(response)
```

**(iii) Use of the wrapper API to get structured outputs**

```python
from litemind.apis.combined_api import CombinedApi
from litemind.agent.agent import Agent
from litemind.agent.messages.message import Message
from pydantic import BaseModel

# Define a Pydantic model for the output
class Weather(BaseModel):
    temperature: float
    condition: str
    humidity: float

# Initialize the API
api = CombinedApi()

# Create an agent
agent = Agent(api=api)

# Add a system message
agent.append_system_message("You are a weather bot.")

# Create a user message
user_message = Message(role="user", text="What is the weather like in Paris?")

# Get the response
response = agent(user_message, response_format=Weather)

# Print the response
print(response)
```

**(iv) Agent that has a tool that can generate an image using the wrapper API.**

```python
from litemind.apis.combined_api import CombinedApi
from litemind.agent.agent import Agent
from litemind.agent.messages.message import Message
from litemind.agent.tools.function_tool import FunctionTool
from PIL import Image

# Initialize the API
api = CombinedApi()

# Create an agent
agent = Agent(api=api)

# Define a tool function
def generate_image(prompt: str) -> str:
    """Generates an image from a prompt."""
    try:
        image = api.generate_image(positive_prompt=prompt)
        # Save the image to a temporary file
        import tempfile

        temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        image.save(temp_file.name)
        return f"Image generated: {temp_file.name}"
    except Exception as e:
        return f"Error generating image: {e}"

# Add a tool
toolset = agent.toolset
toolset.add_function_tool(generate_image, "Generates an image from a prompt")

# Create a user message
user_message = Message(role="user", text="Generate an image of a cat.")

# Get the response
response = agent(user_message)

# Print the response
print(response)
```

## Command Line Tools

LiteMind provides command-line tools for code generation and repository export.

*   **`litemind codegen`**: Generates files based on prompts and folder contents.
    *   `model`: The model to use for generating the files.  Defaults to 'combined'.
    *   `-f`, `--file`: The specific file to generate. If not provided, all files will be generated.
*   **`litemind export`**: Exports the entire repository to a single file.
    *   `-f`, `--folder-path`: The path to the folder containing the repository to export. Defaults to the current directory.
    *   `-o`, `--output-file`: The path to the file to save the entire repository to. Defaults to `exported.txt`.
    *   `-e`, `--extensions`: The list of allowed extensions for files to include in the export.
    *   `-x`, `--exclude`: The list of files to exclude from the export.

*   **`.codegen.yml` File Format:**
    The `litemind codegen` tool uses YAML configuration files (e.g., `.codegen/my_file.codegen.yml`) to generate files.  These files specify the prompt, the input folder, the output file, and other options.

    Here's a template for a `.codegen.yml` file:

    ```yaml
    prompt: |
      You are a helpful assistant.
      Given the following files, generate a README.md file for a Python project.
    folder:
      path: .  # or the path to the folder to include
    file: README.md  # The name of the file to generate
    allowed_extensions:
      - .py
      - .md
      - .txt
    excluded_files:
      - .gitignore
      - .git
    ```

    *   `prompt`: The prompt to use for generating the file.
    *   `folder`: The folder to include in the prompt.
        *   `path`: The path to the folder to include.
    *   `file`: The name of the file to generate.
    *   `allowed_extensions`: A list of allowed file extensions.
    *   `excluded_files`: A list of files to exclude.

    Place the `.codegen` folder in the root of your repository and apply that to your own repositories.

## Caveats and Limitations

*   **API Key Management:** The library relies on environment variables for API keys. Consider using a more secure key management solution in production environments.
*   **Token Management:** The library lacks explicit token management, which is crucial for controlling costs and avoiding rate limits.
*   **Error Handling:** Error handling could be improved, particularly in the API wrapper layer.
*   **Performance:** Performance considerations, such as caching and optimization of API calls, are not explicitly addressed.

## Code Health

*   **Total tests:** 100
*   **Tests passed:** 98
*   **Tests failed:** 2
*   **Failed tests file names:**
    *   test_apis_text_generation.py
    *   test_apis_describe.py
*   **Criticality of failures:** The failures in `test_apis_text_generation.py` are related to tool integration and should be addressed. The failures in `test_apis_describe.py` are related to the describe methods of the API.

## API Keys

To use the library, you need to obtain API keys from the respective LLM providers (OpenAI, Claude, Gemini, etc.).

*   **OpenAI:** Set the `OPENAI_API_KEY` environment variable.
    *   **Linux/macOS:**  Add `export OPENAI_API_KEY="YOUR_API_KEY"` to your `.bashrc` or `.zshrc` file.
    *   **Windows:** Set the environment variable through the system settings (search for "environment variables").
*   **Claude:** Set the `ANTHROPIC_API_KEY` environment variable.
    *   **Linux/macOS:**  Add `export ANTHROPIC_API_KEY="YOUR_API_KEY"` to your `.bashrc` or `.zshrc` file.
    *   **Windows:** Set the environment variable through the system settings (search for "environment variables").
*   **Gemini:** Set the `GOOGLE_GEMINI_API_KEY` environment variable.
    *   **Linux/macOS:**  Add `export GOOGLE_GEMINI_API_KEY="YOUR_API_KEY"` to your `.bashrc` or `.zshrc` file.
    *   **Windows:** Set the environment variable through the system settings (search for "environment variables").
*   **Ollama:** No API key is required.

## Roadmap

-   [ ] Use the faster pybase64 for base64 encoding/decoding.
-   [ ] Automatic feature support discovery for models (which models support images as input, reasoning, etc...)
-   [ ] Improve vendor api robustness features such as retry call when server errors, etc...
-   [ ] Improve and uniformize exception handling
-   [ ] Add support for adding nD images to messages.
-   [ ] Implement 'brainstorming' mode for text generation, possibly with API fusion.
-   [ ] Response format option for agent.

## Contributing

We welcome contributions! Please see the [CONTRIBUTING.md](CONTRIBUTING.md) file for details on how to contribute to the library, including guidelines for contributing code, documentation, or other resources.

## License

This project is licensed under the BSD-3-Clause License.

---

This README was generated with the help of AI.
