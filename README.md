
# LiteMind

[![PyPI version](https://badge.fury.io/py/litemind.svg)](https://badge.fury.io/py/litemind)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![PyPI - Downloads](https://static.pepy.tech/badge/litemind)](https://pepy.tech/project/litemind)
[![GitHub stars](https://img.shields.io/github/stars/royerlab/litemind?style=social)](https://github.com/royerlab/litemind)

## Summary

LiteMind is a Python library designed to provide a streamlined and elegant API for building agentic AI applications. It offers a flexible wrapper layer around various Large Language Models (LLMs) and a powerful agentic API for creating conversational agents and tools. The library's core philosophy is to simplify the complexities of LLM interactions, enabling developers to focus on building innovative AI solutions.

The library is structured around two main components:

1.  **API Wrapper Layer**: This layer abstracts the complexities of interacting with different LLM providers (OpenAI, Anthropic, Google, Ollama), allowing developers to switch between them with minimal code changes. It provides a consistent interface for common operations like text generation, image generation, audio transcription, and embeddings.

2.  **Agentic API**: This high-level framework provides the building blocks for creating sophisticated conversational agents and tools. It supports multimodal inputs (text, images, audio, video, documents), tool usage, and reasoning capabilities, enabling the creation of agents that can understand and respond to complex queries, use external tools, and maintain context across conversations.

## Features

*   **API Wrapper Layer**:
    *   Supports multiple LLM providers:
        *   OpenAI (GPT models)
        *   Anthropic (Claude models)
        *   Google (Gemini models)
        *   Ollama (local open-source models)
*   **Multimodal Support**:
    *   Text generation and structured output
    *   Image, audio, and video inputs
    *   Document processing (PDFs, webpages, etc.)
    *   Table handling
*   **Agentic API**:
    *   Conversational agents with memory
    *   Tool usage and function calling
    *   ReAct agents with reasoning capabilities
    *   Agent-to-agent communication
*   **Tooling**:
    *   Function tools that wrap Python functions
    *   Agent tools that delegate to other agents
    *   Tool composition and chaining
*   **Extensible Architecture**:
    *   Easy to add new LLM providers
    *   Customizable callbacks for monitoring and logging
    *   Flexible message handling

## Installation

Install LiteMind using pip:

```bash
pip install litemind
```

To install optional dependencies, use the following:

```bash
# For RAG (Retrieval Augmented Generation) support
pip install "litemind[rag]"

# For audio transcription with Whisper
pip install "litemind[whisper]"

# For document processing
pip install "litemind[documents]"

# For table handling
pip install "litemind[tables]"

# For video processing
pip install "litemind[videos]"

# For all optional dependencies
pip install "litemind[rag,whisper,documents,tables,videos]"
```

## Usage

### API Wrapper Examples

#### Text Generation

```python
from litemind.apis.combined_api import CombinedApi
from litemind.agent.messages.message import Message

# Create an API instance
api = CombinedApi()

# Create messages
system_message = Message(role="system", text="You are a helpful assistant.")
user_message = Message(role="user", text="What is the capital of France?")

# Generate a response
response = api.generate_text(messages=[system_message, user_message])

# Print the response
print(response[0])
# Expected output: "The capital of France is Paris."
```

#### Multimodal Input (Image)

```python
from litemind.apis.combined_api import CombinedApi
from litemind.agent.messages.message import Message
from litemind.apis.model_features import ModelFeatures

# Create an API instance
api = CombinedApi()

# Get a model that supports images
model_name = api.get_best_model([ModelFeatures.TextGeneration, ModelFeatures.Image])

# Create a message with text and an image
message = Message(role="user")
message.append_text("What's in this image?")
message.append_image("https://upload.wikimedia.org/wikipedia/commons/4/4f/Eiffel_Tower_from_Champ_de_Mars%2C_Paris%2C_France.jpg")

# Generate a response
response = api.generate_text(messages=[message], model_name=model_name)

# Print the response
print(response[0])
# Expected output: A description of the Eiffel Tower
```

#### Multimodal Input (Audio)

```python
from litemind.apis.combined_api import CombinedApi
from litemind.agent.messages.message import Message
from litemind.apis.model_features import ModelFeatures

# Create an API instance
api = CombinedApi()

# Get a model that supports audio
model_name = api.get_best_model([ModelFeatures.TextGeneration, ModelFeatures.Audio])

# Create a message with text and an audio
message = Message(role="user")
message.append_text("What is said in this audio?")
message.append_audio("file:///path/to/audio.mp3")  # Replace with your audio file path

# Generate a response
response = api.generate_text(messages=[message], model_name=model_name)

# Print the response
print(response[0])
# Expected output: Transcription of the audio
```

#### Multimodal Input (Video)

```python
from litemind.apis.combined_api import CombinedApi
from litemind.agent.messages.message import Message
from litemind.apis.model_features import ModelFeatures

# Create an API instance
api = CombinedApi()

# Get a model that supports video
model_name = api.get_best_model([ModelFeatures.TextGeneration, ModelFeatures.Video])

# Create a message with text and a video
message = Message(role="user")
message.append_text("What is happening in this video?")
message.append_video("file:///path/to/video.mp4")  # Replace with your video file path

# Generate a response
response = api.generate_text(messages=[message], model_name=model_name)

# Print the response
print(response[0])
# Expected output: Description of the video content
```

#### Structured Output

```python
from litemind.apis.combined_api import CombinedApi
from litemind.agent.messages.message import Message
from pydantic import BaseModel
from typing import List

# Define a structured output model
class MovieRecommendation(BaseModel):
    title: str
    year: int
    genre: str
    director: str
    rating: float
    description: str

class MovieRecommendations(BaseModel):
    recommendations: List[MovieRecommendation]

# Create an API instance
api = CombinedApi()

# Create messages
system_message = Message(role="system", text="You are a movie recommendation assistant.")
user_message = Message(role="user", text="Recommend 3 sci-fi movies from the 1980s.")

# Generate a structured response
response = api.generate_text(
    messages=[system_message, user_message],
    response_format=MovieRecommendations
)

# Access the structured data
recommendations = response[0][-1].content.recommendations
for movie in recommendations:
    print(f"{movie.title} ({movie.year}) - {movie.rating}/10 - {movie.genre}")
    print(f"Directed by: {movie.director}")
    print(f"Description: {movie.description}")
```

### Agentic API Examples

#### Basic Agent with Tool

```python
from litemind.agent.agent import Agent
from litemind.apis.combined_api import CombinedApi
from litemind.agent.tools.toolset import ToolSet

# Define a tool function
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    # In a real application, this would call a weather API
    return f"The weather in {city} is sunny and 75°F."

# Create a toolset and add the function tool
toolset = ToolSet()
toolset.add_function_tool(get_weather)

# Create an agent with the toolset
api = CombinedApi()
agent = Agent(api=api, toolset=toolset)

# Add a system message to define the agent's behavior
agent.append_system_message("You are a helpful weather assistant.")

# Run the agent with a user query
response = agent("What's the weather like in Paris?")

# Print the response
print(response[-1])
# Expected output: Information about the weather in Paris
```

#### ReAct Agent

```python
from litemind.agent.react.react_agent import ReActAgent
from litemind.apis.combined_api import CombinedApi
from litemind.agent.tools.toolset import ToolSet
from datetime import datetime

# Define tool functions
def get_current_date() -> str:
    """Get the current date."""
    return datetime.now().strftime("%Y-%m-%d")

def calculate_days_between(start_date: str, end_date: str) -> int:
    """Calculate the number of days between two dates."""
    from datetime import datetime
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    return abs((end - start).days)

# Create a toolset and add the function tools
toolset = ToolSet()
toolset.add_function_tool(get_current_date)
toolset.add_function_tool(calculate_days_between)

# Create a ReAct agent with the toolset
api = CombinedApi()
agent = ReActAgent(api=api, toolset=toolset)

# Run the agent with a complex query
response = agent("How many days are there between today and Christmas this year?")

# Print the response
print(response)
# Expected output: A reasoned response calculating the days until Christmas
```

#### Agent-to-Agent Communication

```python
from litemind.agent.agent import Agent
from litemind.agent.tools.tool_agent import ToolAgent
from litemind.apis.combined_api import CombinedApi
from litemind.agent.tools.toolset import ToolSet

# Create a translation agent
api = CombinedApi()
translation_agent = Agent(api=api)
translation_agent.append_system_message("You are a translator that translates English to French.")

# Create a tool agent that wraps the translation agent
translation_tool = ToolAgent(
    agent=translation_agent,
    description="Translates English text to French."
)

# Create a main agent that uses the translation tool
toolset = ToolSet()
toolset.add_tool(translation_tool)

main_agent = Agent(api=api, toolset=toolset)
main_agent.append_system_message("You are a helpful assistant that can translate text when needed.")

# Run the main agent
response = main_agent("Can you translate 'Hello, how are you?' to French?")

# Print the response
print(response[-1])
# Expected output: "Bonjour, comment allez-vous?"
```

## Concepts

### API Wrapper Layer

The API wrapper layer provides a unified interface to different LLM providers:

*   **BaseApi**: Abstract base class that defines the interface for all API implementations.
*   **DefaultApi**: Provides default implementations for common functionality.
*   **Provider-specific APIs**: Implementations for OpenAI, Anthropic, Google, and Ollama.
*   **CombinedApi**: Aggregates multiple API providers and automatically selects the best one for each request.

The API wrapper handles:

*   Model selection based on required features
*   Message conversion between different formats
*   Multimodal input processing
*   Tool usage and function calling
*   Structured output generation

### Agentic API

The Agentic API builds on top of the API wrapper to provide a higher-level framework for building conversational agents:

*   **Message**: Represents a message in a conversation, containing multiple blocks of different types (text, image, audio, etc.).
*   **MessageBlock**: A single block of content within a message.
*   **Conversation**: A collection of messages that maintains context.
*   **Agent**: A conversational agent that can use tools and maintain a conversation.
*   **ReActAgent**: An agent that implements the ReAct (Reasoning + Acting) methodology for step-by-step reasoning.
*   **ToolSet**: A collection of tools that an agent can use.
*   **BaseTool**: Abstract base class for all tools.
*   **FunctionTool**: A tool that wraps a Python function.
*   **ToolAgent**: A tool that delegates to another agent.

### Model Features

LiteMind uses a feature-based approach to model selection:

*   **ModelFeatures**: An enum that defines the features a model can support (TextGeneration, Image, Audio, Video, Tools, etc.).
*   **get_best_model()**: A method that selects the best model based on required features.
*   **has_model_support_for()**: A method that checks if a model supports specific features.

## More Code Examples

### API Wrapper: Text Generation with Different Providers

```python
# Using OpenAI API
from litemind.apis.providers.openai.openai_api import OpenAIApi
from litemind.agent.messages.message import Message

api = OpenAIApi()
messages = [Message(role="user", text="Hello, world!")]
response = api.generate_text(messages=messages, model_name="gpt-4o")

print(response[0])
# Expected output: "Hello there!"
```

```python
# Using Anthropic API
from litemind.apis.providers.anthropic.anthropic_api import AnthropicApi
from litemind.agent.messages.message import Message

api = AnthropicApi()
messages = [Message(role="user", text="Hello, world!")]
response = api.generate_text(messages=messages, model_name="claude-3-opus-20240229")

print(response[0])
# Expected output: "Hello there!"
```

```python
# Using Google Gemini API
from litemind.apis.providers.google.google_api import GeminiApi
from litemind.agent.messages.message import Message

api = GeminiApi()
messages = [Message(role="user", text="Hello, world!")]
response = api.generate_text(messages=messages, model_name="models/gemini-1.5-pro")

print(response[0])
# Expected output: "Hello there!"
```

```python
# Using Ollama API (local models)
from litemind.apis.providers.ollama.ollama_api import OllamaApi
from litemind.agent.messages.message import Message

api = OllamaApi()
messages = [Message(role="user", text="Hello, world!")]
response = api.generate_text(messages=messages, model_name="llama3")

print(response[0])
# Expected output: "Hello there!"
```

### API Wrapper: Text Generation with Streaming

```python
from litemind.apis.combined_api import CombinedApi
from litemind.agent.messages.message import Message
from litemind.apis.callbacks.print_callbacks import PrintCallbacks
from litemind.apis.callbacks.callback_manager import CallbackManager

# Create a callback manager with print callbacks
callback_manager = CallbackManager()
callback_manager.add_callback(PrintCallbacks(
    print_text_generation=True,
    print_text_streaming=True
))

# Create an API instance with callbacks
api = CombinedApi(callback_manager=callback_manager)

# Create messages
messages = [Message(role="user", text="Tell me a short story about a robot.")]

# Generate a response (callbacks will print information during generation)
response = api.generate_text(messages=messages)

# Print the final response
print("\nFinal response:")
print(response[0])
```

### API Wrapper: Embedding and Similarity

```python
from litemind.apis.combined_api import CombinedApi
import numpy as np

# Create an API instance
api = CombinedApi()

# Get text embeddings
texts = [
    "The quick brown fox jumps over the lazy dog",
    "A fast auburn fox leaps above the sleepy canine",
    "The weather is nice today"
]
embeddings = api.embed_texts(texts=texts, dimensions=512)

# Calculate cosine similarity
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Compare similarities
sim_1_2 = cosine_similarity(embeddings[0], embeddings[1])
sim_1_3 = cosine_similarity(embeddings[0], embeddings[2])
sim_2_3 = cosine_similarity(embeddings[1], embeddings[2])

print(f"Similarity between text 1 and 2: {sim_1_2:.4f}")  # Should be high (similar meaning)
print(f"Similarity between text 1 and 3: {sim_1_3:.4f}")  # Should be low (different topics)
print(f"Similarity between text 2 and 3: {sim_2_3:.4f}")  # Should be low (different topics)
```

### Agentic API: Creating a Tool Agent

```python
from litemind.agent.agent import Agent
from litemind.agent.tools.toolset import ToolSet
from litemind.agent.tools.tool_agent import ToolAgent
from litemind.apis.combined_api import CombinedApi

# Create a translation agent
api = CombinedApi()
translation_agent = Agent(api=api)
translation_agent.append_system_message("You are a translator that translates English to French.")

# Create a tool agent that wraps the translation agent
translation_tool = ToolAgent(
    agent=translation_agent,
    description="Translates English text to French."
)

# Create a main agent that uses the translation tool
toolset = ToolSet()
toolset.add_tool(translation_tool)

main_agent = Agent(api=api, toolset=toolset)
main_agent.append_system_message("You are a helpful assistant that can translate text when needed.")

# Run the main agent
response = main_agent("Can you translate 'Hello, how are you?' to French?")

# Print the response
print(response[-1])
# Expected output: "Bonjour, comment allez-vous?"
```

### Agentic API: Creating a Function Tool

```python
from litemind.agent.agent import Agent
from litemind.agent.tools.toolset import ToolSet
from litemind.apis.combined_api import CombinedApi
from datetime import datetime

# Define a tool function
def get_current_date() -> str:
    """Get the current date."""
    return datetime.now().strftime("%Y-%m-%d")

# Create a toolset and add the function tool
toolset = ToolSet()
toolset.add_function_tool(get_current_date)

# Create an agent with the toolset
api = CombinedApi()
agent = Agent(api=api, toolset=toolset)

# Add a system message to define the agent's behavior
agent.append_system_message("You are a helpful assistant that can provide the current date.")

# Run the agent with a user query
response = agent("What is the current date?")

# Print the response
print(response[-1])
# Expected output: The current date
```

## Tools

LiteMind provides a command-line tool `litemind` to perform various operations.

### `litemind codegen`

This command generates files, such as README.md files, for a Python repository based on a prompt and the contents of the repository.

#### Usage

```bash
litemind codegen [model]
```

*   `model`: (Optional) The LLM model to use for generation.  Defaults to `combined`.  Available options are `gemini`, `openai`, `claude`, `ollama`, and `combined`.

This command requires a `.codegen` folder in the root of your repository.  This folder should contain `.codegen.yml` files.

#### `.codegen.yml` file format

The `.codegen.yml` files contain instructions for the code generation process.  Here's a template:

```yaml
# .codegen/README.codegen.yml
prompt: |
  You are a helpful assistant that generates a README.md file for a Python repository.
  The README.md should include the following sections:
  - Summary
  - Features
  - Installation
  - Usage
  - Concepts
  - More Code Examples
  - Tools
  - Caveats and Limitations
  - Code Health
  - Roadmap
  - Contributing
  - License
  The 'Summary' section should provide an enthusiastic and complete description of the library, its purpose, and philosophy.
  The 'Usage' section should consist of rich, striking and illustrative examples of the Agentic API in use with multimodal inputs.
  The 'Concepts' section should explain the concepts behind the library, the main classes and their purpose, and how they interact.
  The 'Code Health' section should include the results of unit test in file 'test_report.md'. Please provide file names for failed tests and statistics about the number of failed tests and an analysis of what happened.
  The 'Roadmap' section can use the contents of TODO.md as a starting point, keep the checkmarks.
  The 'More Code Examples' section further expands with many more also covering the wrapper API, uses ideas and code from the unit tests (no need to mention that).
  The 'Tools' section should explain litemind command line tools and how to use them, with example command lines (check tools package and its contents for details). Please also explain the format of the *.codegen.yml files and give a template. Explain that you need a .codegen folder in the root of the repository and you can apply that to your own repositories.
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

*   `prompt`: The prompt to use for generating the file.  This prompt should include instructions for the LLM on what to generate.
*   `folder`:  Specifies the folder to include in the context of the prompt.
    *   `path`: The path to the folder to include.  Use `.` for the current directory.
*   `file`: The name of the file to generate.
*   `allowed_extensions`: A list of file extensions to include in the context.
*   `excluded_files`: A list of files to exclude from the context.

#### Example

To generate a README.md file for a repository, create a `.codegen` folder in the root of the repository and add a `.codegen.yml` file with the appropriate prompt and file information. Then, run:

```bash
litemind codegen
```

This will generate a README.md file in the root of the repository.

### `litemind export`

This command exports the entire repository to a single file.

#### Usage

```bash
litemind export <folder_path> -o <output_file> [-e <extensions>] [-x <exclude>]
```

*   `<folder_path>`: The path to the folder containing the repository to export.
*   `-o` or `--output-file`: The path to the file to save the entire repository to.
*   `-e` or `--extensions`: (Optional) A list of allowed extensions for files to include in the export.
*   `-x` or `--exclude`: (Optional) A list of files to exclude from the export.

#### Example

To export a repository to a single file, run:

```bash
litemind export . -o repository.txt -e .py .md -x README.md
```

This will export the current directory (.), including all `.py` and `.md` files, excluding `README.md`, to a file named `repository.txt`.

## Caveats and Limitations

*   **API Keys**: You need to provide your own API keys for the various providers (OpenAI, Anthropic, Google, Ollama).
*   **Cost**: Using commercial LLM APIs incurs costs based on their pricing models.
*   **Model Availability**: Not all features are available with all models or providers.
*   **Token Limits**: Different models have different token limits that may affect the complexity of tasks they can handle.
*   **Tool Execution**: Tools are executed locally, which means they have access to your local environment.
*   **ReAct Agent Limitations**: The ReAct agent implementation has some limitations in handling complex reasoning chains and may occasionally fail to follow the correct reasoning format.
*   **Video Processing**: Video processing requires ffmpeg to be installed and may be resource-intensive.
*   **Document Processing**: Document processing requires additional dependencies and may not work perfectly with all document formats.

## Code Health

The test suite for LiteMind shows good overall health with 34 passing tests and 1 failing test. There are 3 skipped tests.

The failing tests are primarily in the ReAct agent implementation:
*   `src/litemind/agent/react/tests/test_react_agent.py`
    *   `test_react_agent_single_tool` (1 failure)

These failures indicate that the ReAct agent and tool agent implementations need further refinement to ensure consistent behavior across different LLM providers.

The skipped tests are primarily due to certain features not being available with all providers, which is expected behavior.

## Roadmap

-   [ ] Deal with message sizes in tokens sent to models
-   [ ] RAG
-   [ ] Improve vendor api robustness features such as retry call when server errors, etc...
-   [ ] Reorganise media files used for testing into a single media folder
-   [ ] Improve and uniformize exception handling
-   [ ] Add support for adding nD images to messages.
-   [ ] Improve logging with arbol, with option to turn off.
-   [ ] Implement 'brainstorming' mode for text generation, possibly with API fusion.

## Contributing

We welcome contributions to LiteMind! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to contribute.

## License

LiteMind is licensed under the BSD-3-Clause License. See [LICENSE](LICENSE) for details.

---

This README was generated with the help of AI.
