
# LiteMind: Multimodal Agentic AI Framework

[![PyPI](https://img.shields.io/pypi/v/litemind.svg)](https://pypi.org/project/litemind/)
[![License](https://img.shields.io/badge/License-BSD--3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Coverage](https://codecov.io/gh/royerlab/litemind/branch/main/graph/badge.svg)](https://codecov.io/gh/royerlab/litemind)

## Summary

LiteMind is a cutting-edge Python library designed to provide a streamlined and elegant API for building sophisticated conversational agents and tools powered by Large Language Models (LLMs). It offers a dual approach: a flexible wrapper API around various LLM providers (OpenAI, Anthropic, Google, Ollama) and a powerful agentic AI framework for creating fully multimodal applications.

The library's philosophy centers on ease of use, modularity, and extensibility. It aims to abstract away the complexities of interacting with different LLM APIs, allowing developers to focus on the core logic of their agentic applications. LiteMind supports multimodal inputs, including text, images, audio, and video, enabling the creation of rich, interactive experiences.

Whether you're building a simple chatbot, a complex task automation system, or a multimodal conversational agent, LiteMind provides the necessary building blocks to bring your ideas to life.

## Features

*   **API Wrapper Layer:**
    *   Unified API for multiple LLM providers (OpenAI, Anthropic, Google, Ollama).
    *   Automatic model selection based on feature support (text generation, image generation, tools, etc.).
    *   Simplified access to model capabilities (token counting, maximum token limits).
    *   Support for structured output with Pydantic models.
    *   Callback system for monitoring and logging API calls.
*   **Agentic AI Framework:**
    *   ReAct (Reasoning and Acting) agent for chain-of-thought reasoning and tool use.
    *   Toolsets for defining and managing tools.
    *   Multimodal message handling (text, images, audio, video, documents, tables).
    *   Conversation management with system and user messages.
    *   Flexible tool integration with function tools and agent tools.
    *   Support for streaming responses.
*   **Multimodal Input Support:**
    *   Text, images (URLs, local paths, base64), audio (URLs, local paths), video (URLs, local paths), documents (PDF, CSV, etc.), and tables (Pandas DataFrames).
    *   Automatic conversion of media files to formats suitable for LLMs.
*   **Code Health:**
    *   Comprehensive unit tests using pytest.
    *   Code coverage reports.
    *   Adherence to PEP-8 style guidelines.
    *   Integration with code quality tools (Black, isort, flake8, mypy).

## Installation

To install LiteMind, use pip:

```bash
pip install litemind
```

To install with optional dependencies (e.g., for RAG, whisper, documents, tables, or videos), use:

```bash
pip install litemind[rag,whisper,documents,tables,videos]
```

## Usage

This section provides illustrative examples of the Agentic API in use with multimodal inputs.

### API Wrapper Example

This example demonstrates how to use the API wrapper to generate text using the OpenAI API.

```python
from litemind.apis.providers.openai.openai_api import OpenAIApi
from litemind.agent.messages.message import Message

# Initialize the OpenAI API
api = OpenAIApi()

# Create a message
message = Message(role="user")
message.append_text("Write a short poem about a cat.")

# Generate text
response = api.generate_text(messages=[message], model_name="gpt-3.5-turbo")

# Print the response
print(response[0].content)
# Expected output:
# The cat sits, a furry friend,
# With eyes of green, until the end.
# A purring sound, a gentle touch,
# A loving friend, we love so much.
```

### Agentic API Example with Text and Image

This example shows how to create a ReAct agent that can describe an image.

```python
from litemind.agent.react.react_agent import ReActAgent
from litemind.apis.providers.openai.openai_api import OpenAIApi
from litemind.agent.messages.message import Message

# Initialize the OpenAI API
api = OpenAIApi()

# Create the ReAct agent
agent = ReActAgent(api=api)

# Create a message with text and an image
message = Message(role="user")
message.append_text("Describe the image:")
message.append_image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/3e/Einstein_1921_by_F_Schmutzer_-_restoration.jpg/456px-Einstein_1921_by_F_Schmutzer_-_restoration.jpg")

# Run the agent
response = agent(message)

# Print the response
print(response)
# Expected output:
# *assistant*:
# The image shows a black and white photograph of Albert Einstein. He is looking at the camera with a serious expression. He has a mustache and his hair is unkempt. He is wearing a suit and tie.
```

### Agentic API Example with Tool Use

This example demonstrates how to create a ReAct agent that can use a tool to get the current date.

```python
from datetime import datetime

from litemind.agent.react.react_agent import ReActAgent
from litemind.agent.tools.toolset import ToolSet
from litemind.apis.providers.openai.openai_api import OpenAIApi

# Define a function to get the current date
def get_current_date() -> str:
    return datetime.now().strftime("%Y-%m-%d")

# Initialize the OpenAI API
api = OpenAIApi()

# Initialize ToolSet
toolset = ToolSet()

# Add a tool that returns the current date
toolset.add_function_tool(get_current_date, "Returns the current date")

# Create the ReAct agent
agent = ReActAgent(api=api, toolset=toolset, temperature=0.0, max_reasoning_steps=5)

# Use the agent to determine the date
response = agent("What date is today?")

# Print the response
print(response)
# Expected output:
# *assistant*:
# The current date is 2024-03-01.
```

## Concepts

*   **BaseApi:** The abstract base class for all API implementations. It defines the interface for interacting with LLM providers.
*   **CombinedApi:** A class that combines multiple API implementations, allowing the agent to use the first available API that supports the required features.
*   **ModelFeatures:** An enumeration that defines the features supported by the models (e.g., text generation, image generation, tools).
*   **Message:** Represents a single message in a conversation, including the role (e.g., "system", "user", "assistant") and content (text, image, audio, video, document, table, tool).
*   **MessageBlock:** Represents a block of content within a message.
*   **Conversation:** Manages a sequence of messages.
*   **ToolSet:** Manages a collection of tools.
*   **FunctionTool:** A tool that wraps a Python function.
*   **ToolAgent:** A tool that wraps an agent.
*   **Agent:** The core class for creating agentic AI applications. It manages the conversation, selects the appropriate model, and interacts with the API.
*   **ReActAgent:** An agent that implements the ReAct (Reasoning and Acting) methodology.

## More Code Examples

### API Wrapper: Audio Transcription

```python
from litemind.apis.providers.openai.openai_api import OpenAIApi

# Initialize the OpenAI API
api = OpenAIApi()

# Transcribe audio
transcription = api.transcribe_audio(
    audio_uri="file:///path/to/audio.mp3", model_name="whisper-1"
)

# Print the transcription
print(transcription)
# Expected output:
# This is a test of the audio transcription.
```

### API Wrapper: Image Generation

```python
from litemind.apis.providers.openai.openai_api import OpenAIApi
from PIL import Image

# Initialize the OpenAI API
api = OpenAIApi()

# Generate an image
image: Image = api.generate_image(
    positive_prompt="A cat wearing a hat", image_width=512, image_height=512
)

# Save the image
image.save("cat_with_hat.png")
```

### Agentic API: Chained Tool Use

```python
from datetime import datetime, timedelta

from litemind.agent.react.react_agent import ReActAgent
from litemind.agent.tools.toolset import ToolSet
from litemind.apis.providers.openai.openai_api import OpenAIApi

# Define tools
def get_current_date() -> str:
    return datetime.now().strftime("%Y-%m-%d")

def add_days_to_date(date_str: str, days: int) -> str:
    date = datetime.strptime(date_str, "%Y-%m-%d")
    new_date = date + timedelta(days=int(days))
    return new_date.strftime("%Y-%m-%d")

# Initialize the OpenAI API
api = OpenAIApi()

# Initialize ToolSet
toolset = ToolSet()

# Add tools
toolset.add_function_tool(get_current_date, "Returns the current date")
toolset.add_function_tool(
    add_days_to_date, "Add a certain number of days to a given date"
)

# Create the ReAct agent
agent = ReActAgent(api=api, toolset=toolset, temperature=0.0, max_reasoning_steps=5)

# Use the agent to determine the date 15 days from today using chained tools
response = agent("What date is 15 days from today?")

# Print the response
print(response)
# Expected output:
# *assistant*:
# The date 15 days from today is 2024-03-16.
```

### Agentic API: Structured Output

```python
from litemind.agent.react.react_agent import ReActAgent
from litemind.agent.tools.toolset import ToolSet
from litemind.apis.providers.openai.openai_api import OpenAIApi
from pydantic import BaseModel

# Define a Pydantic model for the structured output
class OrderInfo(BaseModel):
    order_id: str
    delivery_date: str

# Initialize the OpenAI API
api = OpenAIApi()

# Initialize ToolSet (empty in this case, as we're not using tools)
toolset = ToolSet()

# Create the ReAct agent
agent = ReActAgent(api=api, toolset=toolset, temperature=0.0, max_reasoning_steps=5)

# Use the agent to determine the date 15 days later
response = agent(
    "What is the delivery date for order 12345?", response_format=OrderInfo
)

# Print the response
print(response)
# Expected output:
# *assistant*:
# OrderInfo(order_id='12345', delivery_date='2024-03-16')
```

## Caveats and Limitations

*   **API Key Management:** The library assumes that API keys are available in the environment variables.
*   **Model Availability:** The availability of models and their features depends on the LLM provider and the user's API key.
*   **Error Handling:** While the library includes basic error handling, more robust error handling and retry mechanisms may be needed for production environments.
*   **Token Limits:** The library does not automatically handle token limits. Developers need to be aware of the token limits of the models they are using and handle long inputs accordingly.
*   **Cost:** Using LLM APIs can incur costs. Be mindful of the pricing of the LLM providers.
*   **Open Source Models:** Open source models are typically not as strong as the proprietary models.

## Roadmap

*   \[ ] Deal with message sizes in tokens sent to models
*   \[ ] RAG
*   \[ ] Improve vendor api robustness features such as retry call when server errors, etc...
*   \[ ] Reorganise media files used for testing into a single media folder
*   \[ ] Improve and uniformize exception handling
*   \[ ] Add support for adding nD images to messages.
*   \[ ] Improve logging with arbol, with option to turn off.
*   \[ ] Implement 'brainstorming' mode for text generation, possibly with API fusion.

## Contributing

We welcome contributions! Please read our [CONTRIBUTING.md](CONTRIBUTING.md) file for details on how to contribute to this project.

## License

This project is licensed under the BSD-3-Clause License - see the [LICENSE](LICENSE) file for details.

## Code Health

The code health is assessed based on the unit tests and code coverage reports.

### Test Report

The test report is available in `test_reports/test_report.md`.

### Test Report Standard Output

The test report standard output is available in `test_reports/test_report_stdout.txt`.

### Analysis of Failed Tests

Based on the `test_report_stdout.txt` and `test_reports/test_report.md` files, the following tests failed:

*   `src/litemind/agent/react/tests/test_react_agent.py::test_react_agent_single_tool[GeminiApi]`
*   `src/litemind/agent/react/tests/test_react_agent.py::test_react_agent_multiple_tools[OllamaApi]`
*   `src/litemind/agent/react/tests/test_react_agent.py::test_react_agent_multiple_tools[GeminiApi]`
*   `src/litemind/agent/react/tests/test_react_agent.py::test_react_agent_chained_tools[OpenAIApi]`
*   `src/litemind/agent/react/tests/test_react_agent.py::test_react_agent_chained_tools[GeminiApi]`
*   `src/litemind/agent/react/tests/test_react_agent.py::test_react_agent_longer_dialog[OllamaApi]`
*   `src/litemind/agent/react/tests/test_react_agent.py::test_react_agent_longer_dialog[GeminiApi]`
*   `src/litemind/agent/tools/test/test_agent_tool.py::test_agent_tool_translation[DefaultApi]`
*   `src/litemind/agent/tools/test/test_agent_tool.py::test_agent_tool[DefaultApi]`
*   `src/litemind/agent/tools/test/test_agent_tool.py::test_agent_tool_with_internal_tool[DefaultApi]`
*   `src/litemind/agent/tools/test/test_toolset.py::test_toolset_add_agent_tool[DefaultApi]`
*   `src/litemind/apis/tests/test_apis_documents.py::TestBaseApiImplementationsDocuments::test_text_generation_with_folder[OllamaApi]`

The failures in `test_react_agent.py` seem to be related to the ReAct agent's ability to correctly use tools and reason about the results. The failures in `test_agent_tool.py` and `test_toolset.py` indicate issues with the AgentTool and ToolSet implementations. The failure in `test_apis_documents.py` suggests a problem with the OllamaApi's ability to handle documents.

The test suite has 525 tests, with 12 failures, 452 passed, 61 skipped, and 78 warnings.

## License

This project is licensed under the BSD-3-Clause License.

---

This README was generated with the help of AI.
