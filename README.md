
# litemind: Multimodal Agentic AI and LLM API Wrapper

litemind is a Python library designed to empower developers to build sophisticated conversational agents and tools. It provides a flexible and elegant API for interacting with Large Language Models (LLMs) from various providers, along with a powerful agentic AI framework that supports multimodal inputs and outputs. The library emphasizes ease of use, extensibility, and the ability to create intelligent systems that can reason, act, and interact with the world in a human-like manner.

The core philosophy behind litemind is to provide a unified and intuitive interface for working with LLMs, abstracting away the complexities of different API providers and model capabilities. This allows developers to focus on building their applications rather than wrestling with API-specific details. Furthermore, litemind's agentic AI framework enables the creation of agents that can perform complex tasks, reason about their actions, and leverage a variety of tools to achieve their goals.

## Features

*   **LLM API Wrapper:**
    *   Unified API for multiple LLM providers (OpenAI, Anthropic, Ollama, Google Gemini).
    *   Automatic model selection based on features (text generation, image generation, etc.).
    *   Support for various input types (text, images, audio, video, documents, tables, etc.).
    *   Support for structured output (JSON, Pydantic models).
    *   Token counting and management.
    *   Callback system for monitoring and logging API calls.
*   **Agentic AI Framework:**
    *   ReAct (Reasoning and Acting) agent implementation.
    *   Tool support (function tools, agent tools).
    *   Conversation management.
    *   Multimodal input and output support.
*   **Multimodal Capabilities:**
    *   Image generation.
    *   Audio transcription and generation.
    *   Video processing (frame extraction, audio extraction).
    *   Document conversion (PDF, CSV, etc.).
    *   Text and media embeddings.
*   **Extensibility:**
    *   Easy integration of new LLM providers.
    *   Custom tool creation.
    *   Custom message block types.
*   **Code Health:**
    *   Comprehensive unit tests.
    *   Integration with testing tools (pytest, pytest-html, pytest-cov, pytest-md-report).

## Installation

To install litemind, use pip:

```bash
pip install litemind
```

To install optional dependencies, use the following:

```bash
pip install litemind[rag,whisper,documents,tables,videos,embed]
```

## Usage

### API Wrapper Examples

#### Text Generation

```python
from litemind.apis.combined_api import CombinedApi
from litemind.agent.messages.message import Message

# Initialize the API (uses OpenAI, Anthropic, Ollama, and Gemini by default)
api = CombinedApi()

# Create a message
message = Message(role="user", text="Write a short poem about a cat.")

# Generate text
response = api.generate_text(messages=[message])

# Print the response
print(response)
# Expected output (may vary):
# *assistant*:
# In shadows deep, a feline grace,
# With eyes of emerald, time and space.
# A velvet paw, a silent tread,
# A purring song, softly spread.
```

#### Image Generation

```python
from litemind.apis.combined_api import CombinedApi

# Initialize the API
api = CombinedApi()

# Generate an image
image = api.generate_image(positive_prompt="A futuristic city at night, cyberpunk style")

# Display the image (requires PIL)
image.show()
```

#### Audio Transcription

```python
from litemind.apis.combined_api import CombinedApi

# Initialize the API
api = CombinedApi()

# Transcribe audio
transcription = api.transcribe_audio(audio_uri="file:///path/to/your/audio.wav")

# Print the transcription
print(transcription)
```

#### Multimodal Input (Text and Image)

```python
from litemind.apis.combined_api import CombinedApi
from litemind.agent.messages.message import Message

# Initialize the API
api = CombinedApi()

# Create a message with text and an image
message = Message(role="user")
message.append_text("Describe the image:")
message.append_image("file:///path/to/your/image.jpg")

# Generate text
response = api.generate_text(messages=[message])

# Print the response
print(response)
```

### Agentic API Examples

#### Simple Agent with a Tool

```python
from litemind.agent.agent import Agent
from litemind.agent.messages.message import Message
from litemind.agent.tools.function_tool import FunctionTool
from litemind.agent.tools.toolset import ToolSet
from litemind.apis.combined_api import CombinedApi

# Define a tool function
def get_current_date() -> str:
    """Returns the current date."""
    from datetime import datetime

    return datetime.now().strftime("%Y-%m-%d")

# Create a toolset and add the function tool
toolset = ToolSet()
toolset.add_function_tool(get_current_date, "Returns the current date")

# Initialize the API
api = CombinedApi()

# Create an agent
agent = Agent(api=api, toolset=toolset)

# Create a user message
message = Message(role="user", text="What is the current date?")

# Run the agent
response = agent(message)

# Print the response
print(response)
# Expected output (may vary):
# *assistant*:
# The current date is: 2024-02-23
```

#### ReAct Agent with Multiple Tools

```python
from litemind.agent.react.react_agent import ReActAgent
from litemind.agent.messages.message import Message
from litemind.agent.tools.function_tool import FunctionTool
from litemind.agent.tools.toolset import ToolSet
from litemind.apis.combined_api import CombinedApi
from datetime import datetime, timedelta

# Define tool functions
def get_current_date() -> str:
    """Returns the current date."""
    return datetime.now().strftime("%Y-%m-%d")

def add_days_to_date(date_str: str, days: int) -> str:
    """Add a certain number of days to a given date"""
    date = datetime.strptime(date_str, "%Y-%m-%d")
    new_date = date + timedelta(days=days)
    return new_date.strftime("%Y-%m-%d")

# Create a toolset and add the function tools
toolset = ToolSet()
toolset.add_function_tool(get_current_date, "Returns the current date")
toolset.add_function_tool(add_days_to_date, "Add a certain number of days to a given date")

# Initialize the API
api = CombinedApi()

# Create a ReAct agent
agent = ReActAgent(api=api, toolset=toolset, temperature=0.0, max_reasoning_steps=5)

# Use the agent to determine the date 15 days later
response = agent("What date is 15 days from today?")

# Print the response
print(response)
# Expected output (may vary):
# 2024-03-09
```

## Concepts

*   **API Wrapper:** The API wrapper layer provides a unified interface to interact with various LLM providers. The `CombinedApi` class is the main entry point, which automatically selects the best available model based on the requested features.
*   **Agentic AI:** The agentic AI framework enables the creation of intelligent agents that can reason, act, and interact with the world. The `Agent` class is the base class for creating agents.
*   **Messages:** Messages are the fundamental unit of communication between the agent and the LLM. The `Message` class represents a message, which can contain text, images, audio, video, documents, and tables.
*   **Message Blocks:** `MessageBlock` is a class that represents a block of content within a message. It can be of various types, such as text, JSON, code, image, audio, video, document, or table.
*   **Conversations:** The `Conversation` class manages the history of messages exchanged between the agent and the user.
*   **Tools:** Tools are functions that an agent can use to interact with the external world. The `ToolSet` class manages a collection of tools. The `FunctionTool` class allows wrapping a Python function as a tool. The `ToolAgent` class allows wrapping an agent as a tool.
*   **ReAct Agent:** The `ReActAgent` class implements the ReAct (Reasoning and Acting) methodology, enabling agents to reason through chain-of-thought and systematic action-observation loops.

## More Code Examples

### Multimodal Agent with Image and Text

```python
from litemind.agent.agent import Agent
from litemind.agent.messages.message import Message
from litemind.apis.combined_api import CombinedApi

# Initialize the API
api = CombinedApi()

# Create an agent
agent = Agent(api=api)

# Create a message with text and an image
message = Message(role="user")
message.append_text("Describe the image:")
message.append_image("file:///path/to/your/image.jpg")  # Replace with your image path

# Run the agent
response = agent(message)

# Print the response
print(response)
```

### Agent with Audio Transcription

```python
from litemind.agent.agent import Agent
from litemind.agent.messages.message import Message
from litemind.apis.combined_api import CombinedApi

# Initialize the API
api = CombinedApi()

# Create an agent
agent = Agent(api=api)

# Create a message with audio
message = Message(role="user")
message.append_text("Transcribe the audio:")
message.append_audio("file:///path/to/your/audio.wav")  # Replace with your audio path

# Run the agent
response = agent(message)

# Print the response
print(response)
```

### Agent with Video Processing

```python
from litemind.agent.agent import Agent
from litemind.agent.messages.message import Message
from litemind.apis.combined_api import CombinedApi

# Initialize the API
api = CombinedApi()

# Create an agent
agent = Agent(api=api)

# Create a message with video
message = Message(role="user")
message.append_text("Describe the video:")
message.append_video("file:///path/to/your/video.mp4")  # Replace with your video path

# Run the agent
response = agent(message)

# Print the response
print(response)
```

### Agent with Document Processing

```python
from litemind.agent.agent import Agent
from litemind.agent.messages.message import Message
from litemind.apis.combined_api import CombinedApi

# Initialize the API
api = CombinedApi()

# Create an agent
agent = Agent(api=api)

# Create a message with a document
message = Message(role="user")
message.append_text("Summarize the document:")
message.append_document("file:///path/to/your/document.pdf")  # Replace with your document path

# Run the agent
response = agent(message)

# Print the response
print(response)
```

### Agent with Table Processing

```python
from litemind.agent.agent import Agent
from litemind.agent.messages.message import Message
from litemind.apis.combined_api import CombinedApi
from pandas import DataFrame

# Initialize the API
api = CombinedApi()

# Create an agent
agent = Agent(api=api)

# Create a table
table = DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})

# Create a message with a table
message = Message(role="user")
message.append_text("What is the sum of column A?")
message.append_table(table)

# Run the agent
response = agent(message)

# Print the response
print(response)
```

### Agent with Code Execution

```python
from litemind.agent.agent import Agent
from litemind.agent.messages.message import Message
from litemind.apis.combined_api import CombinedApi

# Initialize the API
api = CombinedApi()

# Create an agent
agent = Agent(api=api)

# Create a message with code
message = Message(role="user")
message.append_text("Execute this Python code:")
message.append_code("print('Hello, world!')", lang="python")

# Run the agent
response = agent(message)

# Print the response
print(response)
```

### Agent with JSON Output

```python
from litemind.agent.agent import Agent
from litemind.agent.messages.message import Message
from litemind.apis.combined_api import CombinedApi
from pydantic import BaseModel

# Initialize the API
api = CombinedApi()

# Create an agent
agent = Agent(api=api)

# Define a Pydantic model for the expected output
class MyModel(BaseModel):
    name: str
    age: int

# Create a message with a prompt and a response format
message = Message(role="user")
message.append_text("Give me a JSON with a name and age.")

# Run the agent
response = api.generate_text(
    messages=[message], response_format=MyModel
)

# Print the response
print(response)
```

### ReAct Agent with a Custom Tool

```python
from litemind.agent.react.react_agent import ReActAgent
from litemind.agent.messages.message import Message
from litemind.agent.tools.function_tool import FunctionTool
from litemind.agent.tools.toolset import ToolSet
from litemind.apis.combined_api import CombinedApi

# Define a tool function
def get_current_date() -> str:
    """Returns the current date."""
    from datetime import datetime

    return datetime.now().strftime("%Y-%m-%d")

# Create a toolset and add the function tool
toolset = ToolSet()
toolset.add_function_tool(get_current_date, "Returns the current date")

# Initialize the API
api = CombinedApi()

# Create a ReAct agent
agent = ReActAgent(api=api, toolset=toolset, temperature=0.0, max_reasoning_steps=5)

# Use the agent to determine the date 15 days later
response = agent("What date is today?")

# Print the response
print(response)
```

## Caveats and Limitations

*   **API Key Management:** The library relies on environment variables for API keys. Ensure that your API keys are securely stored and accessible.
*   **Model Availability:** The availability of specific models and features depends on the LLM provider and your API key's permissions.
*   **Token Limits:** Be mindful of token limits imposed by LLM providers. The library provides token counting functionality, but you should still manage your prompts and responses to avoid exceeding these limits.
*   **Error Handling:** While the library includes error handling, it's essential to implement robust error handling in your applications to gracefully handle API errors and unexpected responses.
*   **Cost:** Using LLM APIs can incur costs. Monitor your usage and set appropriate limits to manage expenses.
*   **Dependencies:** The library has dependencies on external libraries (e.g., `openai`, `anthropic`, `ollama`, `google-generativeai`, `pillow`, `arbol`, `json-repair`, `tiktoken`, `pydantic`). Ensure that these dependencies are installed and compatible with your environment.
*   **Open Source Models:** Open source models are not as strong as proprietary models, and may not be able to perform all the tasks.

## Code Health

The code health of the litemind project is assessed through unit tests and code quality checks.

### Unit Tests

The project includes a comprehensive suite of unit tests to ensure the correctness and reliability of the code. The test reports are located in the `test_reports` directory.

*   **Test Report:** `test_report.md` provides a summary of the test results.
*   **Test Report Standard Output:** `test_report_stdout.txt` contains the standard output from the test runs, including detailed logs and error messages.

#### Test Results Summary

The test report (`test_report.md`) provides a detailed breakdown of the test results.

*   **Total Tests:** 464
*   **Passed:** 404
*   **Failed:** 5
*   **Skipped:** 55
*   **Warnings:** 78

#### Failed Tests Analysis

The following tests failed:

*   `src/litemind/apis/tests/test_apis_generate_multimodal.py::TestBaseApiImplementationsGenerateMultimodal::test_generate_image[CombinedApi]`
    *   **Reason:** The test failed because the description of the generated image did not contain the expected keywords ("white" or "blue"). This suggests an issue with the image generation or description capabilities of the `CombinedApi` with the current model configuration.
*   `src/litemind/apis/tests/test_apis_text_generation.py::TestBaseApiImplementationsTextGeneration::test_text_generation_with_simple_toolset_and_struct_output[OpenAIApi]`
    *   **Reason:** The test failed because the response was not of type `OrderInfo`. This indicates an issue with the structured output or tool usage with the `OpenAIApi`.
*   `src/litemind/apis/tests/test_apis_text_generation.py::TestBaseApiImplementationsTextGeneration::test_text_generation_with_simple_toolset_and_struct_output[OllamaApi]`
    *   **Reason:** The test failed because of a `KeyError: 0`. This suggests an issue with how the `OllamaApi` handles the response from the tool.
*   `src/litemind/apis/tests/test_apis_text_generation.py::TestBaseApiImplementationsTextGeneration::test_text_generation_with_simple_toolset_and_struct_output[AnthropicApi]`
    *   **Reason:** The test failed because the response was not of type `OrderInfo`. This indicates an issue with the structured output or tool usage with the `AnthropicApi`.
*   `src/litemind/apis/tests/test_apis_text_generation.py::TestBaseApiImplementationsTextGeneration::test_text_generation_with_simple_toolset_and_struct_output[CombinedApi]`
    *   **Reason:** The test failed because the response was not of type `OrderInfo`. This indicates an issue with the structured output or tool usage with the `CombinedApi`.

#### Skipped Tests Analysis

The tests were skipped because the models used for the tests did not support the features required for the tests.

#### Warnings Analysis

The test suite generated 78 warnings. These warnings are related to deprecation, and the use of `torch.load` with `weights_only=False`.

### Code Quality

*   The code is formatted using `black` for consistent style.
*   The code is checked for linting errors using `flake8`.

## Roadmap

*   \[ ] RAG (Retrieval-Augmented Generation) integration.
*   \[ ] Add support for adding nD images to messages.
*   \[ ] Deal with message sizes in tokens sent to models.
*   \[ ] Improve and uniformize exception handling.
*   \[ ] Improve logging with arbol, with option to turn off.
*   \[ ] Implement 'brainstorming' mode for text generation, possibly with API fusion.
*   \[ ] Improve folder/archive conversion: add ascii folder tree.

## Contributing

Contributions to litemind are welcome! If you'd like to contribute, please follow these steps:

1.  **Fork the repository.**
2.  **Create a new branch** for your feature or bug fix.
3.  **Make your changes** and commit them with clear, concise messages.
4.  **Write unit tests** for your changes.
5.  **Submit a pull request.**

## License

This project is licensed under the [BSD 3-Clause License](LICENSE).
```

## Note

This README.md file was generated with the help of AI.