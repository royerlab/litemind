
# litemind: Multimodal Agentic AI

[![PyPI](https://img.shields.io/pypi/v/litemind.svg)](https://pypi.org/project/litemind/)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](LICENSE)
[![Coverage](https://codecov.io/gh/Loic-Royer/litemind/branch/main/graph/badge.svg)](https://codecov.io/gh/Loic-Royer/litemind)

## Summary

litemind is a Python library designed to provide an elegant and powerful API for building conversational agents and tools that leverage the capabilities of Large Language Models (LLMs). It offers two primary layers of abstraction:

1.  **API Wrapper Layer:** This layer simplifies interaction with various LLM APIs (OpenAI, Anthropic, Google, Ollama) by providing a unified interface. This allows developers to switch between different LLM providers with minimal code changes, promoting flexibility and reducing vendor lock-in.
2.  **Agentic API:** This is where litemind truly shines. It provides a high-level, intuitive API for creating fully multimodal agentic AI systems. This includes support for text, images, audio, video, documents, and tables, enabling the creation of sophisticated conversational agents and tools that can understand and respond to a wide range of inputs. The agentic API is built upon the API wrapper layer, abstracting away the complexities of interacting with individual LLM providers.

The philosophy behind litemind is to empower developers to build intelligent applications with ease. It aims to abstract away the complexities of interacting with LLMs, providing a clean and intuitive API that allows developers to focus on the core logic of their applications. It emphasizes multimodal input and output, enabling the creation of rich and engaging user experiences.

## Features

*   **Unified API Wrapper:**
    *   Supports multiple LLM providers (OpenAI, Anthropic, Google, Ollama).
    *   Easy switching between providers.
    *   Handles API key management and authentication.
    *   Provides a consistent interface for text generation, embeddings, and more.
*   **Agentic API:**
    *   Multimodal input support (text, images, audio, video, documents, tables).
    *   Multimodal output support (text, images, audio, video).
    *   ReAct (Reasoning and Acting) agent implementation for complex tasks.
    *   Tool integration (function calling) for extending agent capabilities.
    *   Conversation management.
*   **Advanced Features:**
    *   Streaming callbacks for real-time interaction.
    *   Document conversion (PDF, etc.) to text.
    *   Video to image and audio conversion.
    *   Embedding generation (text, images, audio, video).
    *   Structured output with tool usage.
    *   Exception handling and logging.

## Installation

```bash
pip install litemind
```

To install optional dependencies (e.g., for RAG, audio transcription, document processing, tables, videos, or embeddings), use the following:

```bash
pip install litemind[rag]
pip install litemind[whisper]
pip install litemind[documents]
pip install litemind[tables]
pip install litemind[videos]
pip install litemind[embed]
```

## Usage

### API Wrapper Layer

This section demonstrates how to use the API wrapper layer to interact with different LLM providers.

```python
from litemind.apis.combined_api import CombinedApi
from litemind.apis.model_features import ModelFeatures

# Initialize the API (uses OpenAI by default if API key is set)
api = CombinedApi()

# Check if the API is available
if api.check_availability_and_credentials():
    print("API is available and credentials are valid.")
else:
    print("API is not available or credentials are not valid.")
    exit()

# List available models
models = api.list_models(features=[ModelFeatures.TextGeneration])
print(f"Available models: {models}")

# Get the best model for text generation
best_model = api.get_best_model(features=[ModelFeatures.TextGeneration])
print(f"Best model for text generation: {best_model}")

# Generate text
from litemind.agent.messages.message import Message

messages = [
    Message(role="system", text="You are a helpful assistant."),
    Message(role="user", text="What is the capital of France?"),
]
response = api.generate_text(messages=messages, model_name=best_model)
print(f"Response: {response}")
# Expected output:
# Response: *assistant*:
# The capital of France is Paris.

# Generate image
from PIL import Image

try:
    image = api.generate_image(
        positive_prompt="A cute fluffy cat", image_width=256, image_height=256
    )
    assert isinstance(image, Image.Image)
    print("Image generated successfully.")
except Exception as e:
    print(f"Image generation failed: {e}")

# Embed text
texts = ["This is a tests sentence.", "Another tests sentence."]
embeddings = api.embed_texts(texts=texts, model_name=best_model)
print(f"Embeddings: {embeddings}")
# Expected output:
# Embeddings: [[0.01, 0.02, ...], [0.03, 0.04, ...]]
```

### Agentic API

This section demonstrates how to use the agentic API to create a ReAct agent.

```python
from datetime import datetime, timedelta

from litemind.agent.react.react_agent import ReActAgent
from litemind.agent.tools.toolset import ToolSet
from litemind.apis.combined_api import CombinedApi
from litemind.apis.model_features import ModelFeatures

# Define a tool to get the current date
def get_current_date():
    return datetime.now().strftime("%Y-%m-%d")

# Initialize API
api = CombinedApi()

# Initialize ToolSet
toolset = ToolSet()

# Add a tool that returns the current date
toolset.add_function_tool(get_current_date, "Returns the current date")

# Create the ReAct agent
agent = ReActAgent(api=api, toolset=toolset, temperature=0.0, max_reasoning_steps=5)

# Use the agent to determine the date 15 days later
response = agent("What date is 15 days from today?")
print(f"Response: {response}")
# Expected output:
# Response: 2025-03-10

# Example with multiple tools
def multiply_by_two(x):
    return x * 2

# Add tools
toolset.add_function_tool(get_current_date, "Returns the current date")
toolset.add_function_tool(multiply_by_two, "Multiply a number by 2")

# Create the ReAct agent
agent = ReActAgent(api=api, toolset=toolset, temperature=0.0, max_reasoning_steps=5)

# Use the agent to determine the date 15 days later
response = agent("What date is 15 days from today?")
print(f"Response: {response}")
# Expected output:
# Response: 2025-03-10

# Use the agent to multiply a number by 2
response = agent("What is 21 multiplied by 2?")
print(f"Response: {response}")
# Expected output:
# Response: 42
```

### Multimodal Examples

```python
from litemind.agent.messages.message import Message
from litemind.apis.combined_api import CombinedApi

# Initialize API
api = CombinedApi()

# Text and Image
message = Message(role="user")
message.append_text("Can you describe what you see in this image?")
message.append_image(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3e/Einstein_1921_by_F_Schmutzer_-_restoration.jpg/456px-Einstein_1921_by_F_Schmutzer_-_restoration.jpg"
)
response = api.generate_text(messages=[message])
print(f"Text and Image Response: {response}")

# Text, Image, and Audio
message = Message(role="user")
message.append_text("Can you describe what you see and hear?")
message.append_image(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3e/Einstein_1921_by_F_Schmutzer_-_restoration.jpg/456px-Einstein_1921_by_F_Schmutzer_-_restoration.jpg"
)
message.append_audio(
    "https://salford.figshare.com/ndownloader/files/14630270"
)  # Example audio URL
response = api.generate_text(messages=[message])
print(f"Text, Image, and Audio Response: {response}")

# Text and Video
message = Message(role="user")
message.append_text("Can you describe what you see in the video?")
message.append_video(
    "https://ia903405.us.archive.org/27/items/archive-video-files/test.mp4"
)
response = api.generate_text(messages=[message])
print(f"Text and Video Response: {response}")

# Text and Document
message = Message(role="user")
message.append_text("Can you summarize this document?")
message.append_document("https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf")
response = api.generate_text(messages=[message])
print(f"Text and Document Response: {response}")

# Text and Table
from pandas import DataFrame

table = DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
message = Message(role="user")
message.append_text("Can you describe what you see in the table?")
message.append_table(table)
response = api.generate_text(messages=[message])
print(f"Text and Table Response: {response}")
```

## Concepts

*   **BaseApi:** An abstract base class that defines the interface for interacting with LLM APIs. It provides methods for text generation, embedding, and other common tasks.
*   **CombinedApi:** A concrete implementation of the BaseApi that combines multiple API implementations. It allows developers to use multiple LLM providers and switch between them easily.
*   **Message:** Represents a single message in a conversation. It can contain text, images, audio, video, documents, and tables.
*   **MessageBlock:** Represents a block of content within a message. It can be of various types (text, image, audio, video, document, table, etc.).
*   **Conversation:** Represents a conversation between an agent and a user. It stores a list of messages.
*   **Agent:** The core class for creating conversational agents. It takes an API, a model, and a toolset as input. It handles the conversation flow and interacts with the API to generate responses.
*   **ToolSet:** A collection of tools that an agent can use to perform tasks.
*   **FunctionTool:** A tool that wraps a Python function.
*   **ReActAgent:** An agent that uses the ReAct (Reasoning and Acting) methodology to solve complex tasks.

## More Code Examples

### ReAct Agent with a Custom Tool

```python
from datetime import datetime

from litemind.agent.react.react_agent import ReActAgent
from litemind.agent.tools.toolset import ToolSet
from litemind.apis.combined_api import CombinedApi

# Define a custom tool
def get_current_time():
    return datetime.now().strftime("%H:%M:%S")

# Initialize API
api = CombinedApi()

# Initialize ToolSet
toolset = ToolSet()

# Add the custom tool
toolset.add_function_tool(get_current_time, "Get the current time")

# Create the ReAct agent
agent = ReActAgent(api=api, toolset=toolset, temperature=0.0, max_reasoning_steps=3)

# Ask the agent a question that requires the tool
response = agent("What time is it now?")
print(f"Response: {response}")
# Expected output:
# Response: 14:30:00
```

### Using a Pydantic Model for Structured Output

```python
from pydantic import BaseModel

from litemind.agent.messages.message import Message
from litemind.apis.combined_api import CombinedApi

# Define a Pydantic model for the structured output
class ProductReview(BaseModel):
    product_name: str
    rating: int
    comments: str

# Initialize API
api = CombinedApi()

# Create a system message to guide the agent
system_message = Message(
    role="system",
    text="You are a helpful assistant that provides product reviews in a structured JSON format.",
)

# Create a user message with the product description
user_message = Message(
    role="user",
    text="Please provide a review for the following product: A very good product, I give it 5 stars.",
)

# Combine the messages
messages = [system_message, user_message]

# Generate text with the structured output
response = api.generate_text(
    messages=messages, temperature=0.0, response_format=ProductReview
)

# Print the response
print(f"Response: {response}")
# Expected output:
# Response: ProductReview(product_name='product', rating=5, comments='A very good product')
```

### Combining Text and Image Input

```python
from litemind.agent.messages.message import Message
from litemind.apis.combined_api import CombinedApi

# Initialize API
api = CombinedApi()

# Create a message with text and an image
message = Message(role="user")
message.append_text("Describe the image:")
message.append_image(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3e/Einstein_1921_by_F_Schmutzer_-_restoration.jpg/456px-Einstein_1921_by_F_Schmutzer_-_restoration.jpg"
)

# Generate text
response = api.generate_text(messages=[message])
print(f"Response: {response}")
# Expected output:
# Response: *assistant*:
# The image is a black and white photograph of Albert Einstein. He is looking at the camera.
```

### Using Audio Input

```python
from litemind.agent.messages.message import Message
from litemind.apis.combined_api import CombinedApi

# Initialize API
api = CombinedApi()

# Create a message with audio
message = Message(role="user")
message.append_text("What is said in the audio?")
message.append_audio(
    "https://salford.figshare.com/ndownloader/files/14630270"
)  # Example audio URL

# Generate text
response = api.generate_text(messages=[message])
print(f"Response: {response}")
# Expected output:
# Response: *assistant*:
# The audio says: "The quick brown fox jumps over the lazy dog."
```

### Working with Documents

```python
from litemind.agent.messages.message import Message
from litemind.apis.combined_api import CombinedApi

# Initialize API
api = CombinedApi()

# Create a message with a document
message = Message(role="user")
message.append_text("Summarize the following document:")
message.append_document(
    "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
)

# Generate text
response = api.generate_text(messages=[message])
print(f"Response: {response}")
# Expected output:
# Response: *assistant*:
# The document is a dummy PDF file.
```

### Working with Tables

```python
from pandas import DataFrame

from litemind.agent.messages.message import Message
from litemind.apis.combined_api import CombinedApi

# Initialize API
api = CombinedApi()

# Create a table
table = DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})

# Create a message with the table
message = Message(role="user")
message.append_text("What is the sum of column A and B?")
message.append_table(table)

# Generate text
response = api.generate_text(messages=[message])
print(f"Response: {response}")
# Expected output:
# Response: *assistant*:
# The sum of column A is 6 and the sum of column B is 15.
```

### Working with Folders

```python
import os
import tempfile

from litemind.agent.messages.message import Message
from litemind.apis.combined_api import CombinedApi

# Initialize API
api = CombinedApi()

# Create a temporary folder
temp_folder = tempfile.mkdtemp()

# Create a text file in the temporary folder
with open(os.path.join(temp_folder, "file.txt"), "w") as f:
    f.write("This is a tests file.")

# Create a message with the folder
message = Message(role="user")
message.append_text("Summarize the contents of this folder:")
message.append_folder(temp_folder)

# Generate text
response = api.generate_text(messages=[message])
print(f"Response: {response}")
# Expected output:
# Response: *assistant*:
# The folder contains a file named 'file.txt'. The file contains the text 'This is a tests file.'
```

## Caveats and Limitations

*   **API Key Management:** The library relies on environment variables for API keys. Securely managing API keys in production environments is the responsibility of the developer.
*   **Model Availability:** The availability of specific models and features depends on the LLM provider and the user's API key.
*   **Token Limits:** LLMs have token limits for both input and output. Developers need to be aware of these limits and design their applications accordingly.
*   **Cost:** Using LLM APIs can incur costs. Developers should monitor their usage and manage their budgets.
*   **Error Handling:** While the library provides basic error handling, developers should implement robust error handling in their applications to handle API errors and other exceptions.
*   **Multimodal Support:** Multimodal support is dependent on the capabilities of the underlying LLM APIs. Not all models support all modalities.
*   **Open Source Models:** Open source models are not always as strong as proprietary models.

## Code Health

The code health section is automatically generated from the test reports.

### Test Results

The test results are summarized in the following table:

|                                   filepath                                    |                                              function                                               | passed | failed | skipped | SUBTOTAL |
| ----------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- | -----: | -----: | ------: | -------: |
| src/litemind/agent/react/tests/test_react_agent.py                            | test_react_agent_construction_methods                                                               |      1 |      0 |       0 |        1 |
| src/litemind/agent/react/tests/test_react_agent.py                            | test_react_agent_chained_tools                                                                      |      1 |      0 |       0 |        1 |
| src/litemind/agent/tests/test_agent.py                                        | test_agent_text_with_openai                                                                         |      1 |      0 |       0 |        1 |
| src/litemind/agent/tests/test_agent.py                                        | test_message_image                                                                                  |      1 |      0 |       0 |        1 |
| src/litemind/agent/tests/test_conversation.py                                 | test_conversation                                                                                   |      1 |      0 |       0 |        1 |
| src/litemind/agent/tests/test_message.py                                      | test_message_deepcopy                                                                               |      1 |      0 |       0 |        1 |
| src/litemind/agent/tests/test_message.py                                      | test_insert_block                                                                                   |      1 |      0 |       0 |        1 |
| src/litemind/agent/tests/test_message.py                                      | test_insert_message                                                                                 |      1 |      0 |       0 |        1 |
| src/litemind/agent/tests/test_message.py                                      | test_message_text                                                                                   |      1 |      0 |       0 |        1 |
| src/litemind/agent/tests/test_message.py                                      | test_message_object                                                                                 |      1 |      0 |       0 |        1 |
| src/litemind/agent/tests/test_message.py                                      | test_message_json                                                                                   |      1 |      0 |       0 |        1 |
| src/litemind/agent/tests/test_message.py                                      | test_message_image                                                                                  |      1 |      0 |       0 |        1 |
| src/litemind/agent/tests/test_message.py                                      | test_message_audio                                                                                  |      1 |      0 |       0 |        1 |
| src/litemind/agent/tests/test_message.py                                      | test_message_video                                                                                  |      1 |      0 |       0 |        1 |
| src/litemind/agent/tests/test_message.py                                      | test_message_document                                                                               |      1 |      0 |       0 |        1 |
| src/litemind/agent/tests/test_message.py                                      | test_message_folder                                                                                 |      1 |      0 |       0 |        1 |
| src/litemind/agent/tests/test_message.py                                      | test_message_str                                                                                    |      1 |      0 |       0 |        1 |
| src/litemind/agent/tests/test_message.py                                      | test_extract_markdown_block                                                                         |      1 |      0 |       0 |        1 |
| src/litemind/agent/tests/test_message_block.py                                | test_message_block_deepcopy                                                                         |      1 |      0 |       0 |        1 |
| src/litemind/agent/tests/test_message_block.py                                | test_message_block_text                                                                             |      1 |      0 |       0 |        1 |
| src/litemind/agent/tests/test_message_block.py                                | test_message_block_json                                                                             |      1 |      0 |       0 |        1 |
| src/litemind/agent/tests/test_message_block.py                                | test_message_block_image                                                                            |      1 |      0 |       0 |        1 |
| src/litemind/agent/tests/test_message_block.py                                | test_message_block_audio                                                                            |      1 |      0 |       0 |        1 |
| src/litemind/agent/tests/test_message_block.py                                | test_message_block_video                                                                            |      1 |      0 |       0 |        1 |
| src/litemind/agent/tests/test_message_block.py                                | test_message_block_document                                                                         |      1 |      0 |       0 |        1 |
| src/litemind/agent/tests/test_message_block.py                                | test_message_block_table                                                                            |      1 |      0 |       0 |        1 |
| src/litemind/agent/tools/test/test_agent_tool.py                              | test_agent_tool_translation                                                                         |      1 |      0 |       0 |        1 |
| src/litemind/agent/tools/test/test_agent_tool.py                              | test_agent_tool                                                                                     |      1 |      0 |       0 |        1 |
| src/litemind/agent/tools/test/test_function_tool.py                           | test_tool_initialization                                                                            |      1 |      0 |       0 |        1 |
| src/litemind/agent/tools/test/test_function_tool.py                           | test_tool_initialization_automatic_description                                                      |      1 |      0 |       0 |        1 |
| src/litemind/agent/tools/test/test_function_tool.py                           | test_tool_with_no_parameters                                                                        |      1 |      0 |       0 |        1 |
| src/litemind/agent/tools/test/test_function_tool.py                           | test_tool_with_default_parameter                                                                    |      1 |      0 |       0 |        1 |
| src/litemind/agent/tools/test/test_function_tool.py                           | test_tool_execution_with_parameters                                                                 |      1 |      0 |       0 |        1 |
| src/litemind/agent/tools/test/test_function_tool.py                           | test_tool_execution_no_parameters                                                                   |      1 |      0 |       0 |        1 |
| src/litemind/agent/tools/test/test_function_tool.py                           | test_tool_execution_with_default_parameters                                                         |      1 |      0 |       0 |        1 |
| src/litemind/agent/tools/test/test_toolset.py                                 | test_toolset_initialization_and_add_tool                                                            |      1 |      0 |       0 |        1 |
| src/litemind/agent/tools/test/test_toolset.py                                 | test_toolset_add_function_tool                                                                      |      1 |      0 |       0 |        1 |
| src/litemind/agent/tools/test/test_toolset.py                                 | test_toolset_add_agent_tool                                                                         |      1 |      0 |       0 |        1 |
| src/litemind/agent/tools/test/test_toolset.py                                 | test_toolset_get_tool                                                                               |      1 |      0 |       0 |        1 |
| src/litemind/agent/tools/test/test_toolset.py                                 | test_toolset_list_tools                                                                             |      1 |      0 |       0 |        1 |
| src/litemind/agent/tools/utils/test/test_inspect_function.py                  | test_extract_docstring                                                                              |      1 |      0 |       0 |        1 |
| src/litemind/agent/tools/utils/test/test_inspect_function.py                  | test_extract_function_info                                                                          |      1 |      0 |       0 |        1 |
| src/litemind/agent/utils/tests/test_folder_description.py                     | test_human_readable_size                                                                            |      1 |      0 |       0 |        1 |
| src/litemind/agent/utils/tests/test_folder_description.py                     | test_generate_tree_structure                                                                        |      1 |      0 |       0 |        1 |
| src/litemind/agent/utils/tests/test_folder_description.py                     | test_read_file_content                                                                              |      1 |      0 |       0 |        1 |
| src/litemind/agent/utils/tests/test_folder_description.py                     | test_read_binary_file_info                                                                          |      1 |      0 |       0 |        1 |
| src/litemind/agent/utils/tests/test_folder_description.py                     | test_file_info_header                                                                               |      1 |      0 |       0 |        1 |
| src/litemind/apis/callbacks/test/test_print_callbacks.py                      | test_on_availability_check                                                                          |      1 |      0 |       0 |        1 |
| src/litemind/apis/callbacks/test/test_print_callbacks.py                      | test_on_model_list                                                                                  |      1 |      0 |       0 |        1 |
| src/litemind/apis/callbacks/test/test_print_callbacks.py                      | test_on_best_model_selected                                                                         |      1 |      0 |       0 |        1 |
| src/litemind/apis/callbacks/test/test_print_callbacks.py                      | test_on_text_generation                                                                             |      1 |      0 |       0 |        1 |
| src/litemind/apis/callbacks/test/test_print_callbacks.py                      | test_on_text_streaming                                                                              |      1 |      0 |       0 |        1 |
| src/litemind/apis/callbacks/test/test_print_callbacks.py                      | test_on_audio_transcription                                                                         |      1 |      0 |       0 |        1 |
| src/litemind/apis/callbacks/test/test_print_callbacks.py                      | test_on_document_conversion                                                                         |      1 |      0 |       0 |        1 |
| src/litemind/apis/callbacks/test/test_print_callbacks.py                      | test_on_video_conversion                                                                            |      1 |      0 |       0 |        1 |
| src/litemind/apis/callbacks/test/test_print_callbacks.py                      | test_on_audio_generation                                                                            |      1 |      0 |       0 |        1 |
| src/litemind/apis/callbacks/test/test_print_callbacks.py                      | test_on_image_generation                                                                            |      1 |      0 |       0 |        1 |
| src/litemind/apis/callbacks/test/test_print_callbacks.py                      | test_on_text_embedding                                                                              |      1 |      0 |       0 |        1 |
| src/litemind/apis/callbacks/test/test_print_callbacks.py                      | test_on_image_embedding                                                                             |      1 |      0 |       0 |        1 |
| src/litemind/apis/callbacks/test/test_print_callbacks.py                      | test_on_audio_embedding                                                                             |      1 |      0 |       0 |        1 |
| src/litemind/apis/callbacks/test/test_print_callbacks.py                      | test_on_video_embedding                                                                             |      1 |      0 |       0 |        1 |
| src/litemind/apis/callbacks/test/test_print_callbacks.py                      | test_on_image_description                                                                           |      1 |      0 |       0 |        1 |
| src/litemind/apis/callbacks/test/test_print_callbacks.py                      | test_on_audio_description                                                                           |      1 |      0 |       0 |        1 |
| src/litemind/apis/callbacks/test/test_print_callbacks.py                      | test_on_video_description                                                                           |      1 |      0 |       0 |        1 |
| src/litemind/apis/providers/ollama/test/multi_image_test.py                   | test_compare_images_with_separate_messages                                                          |      1 |      0 |       0 |        1 |
| src/litemind/apis/providers/openai/utils/test/test_messages.py                | test_convert_text_only_message                                                                      |      1 |      0 |       0 |        1 |
| src/litemind/apis/providers/openai/utils/test/test_messages.py                | test_convert_image_only_message                                                                     |      1 |      0 |       0 |        1 |
| src/litemind/apis/providers/openai/utils/test/test_messages.py                | test_convert_text_and_image_message                                                                 |      1 |      0 |       0 |        1 |
| src/litemind/apis/providers/openai/utils/test/test_messages.py                | test_convert_multiple_images_message                                                                |      1 |      0 |       0 |        1 |
| src/litemind/apis/providers/openai/utils/test/test_messages.py                | test_convert_empty_message                                                                          |      1 |      0 |       0 |        1 |
| src/litemind/apis/providers/openai/utils/test/test_messages.py                | test_convert_multiple_messages                                                                      |      1 |      0 |       0 |        1 |
| src/litemind/apis/providers/openai/utils/test/test_messages_with_api.py       | test_openai_api_text_only_message                                                                   |      1 |      0 |       0 |        1 |
| src/litemind/apis/providers/openai/utils/test/test_messages_with_api.py       | test_openai_api_text_and_image_message                                                              |      1 |      0 |       0 |        1 |
| src/litemind/apis/providers/openai/utils/test/test_messages_with_api.py       | test_openai_api_multiple_images                                                                     |      1 |      0 |       0 |        1 |
| src/litemind/apis/providers/openai/utils/test/test_messages_with_api.py       | test_openai_api_multiple_messages                                                                   |      1 |      0 |       0 |        1 |
| src/litemind/apis/providers/openai/utils/test/test_tools.py                   | test_format_tools_for_openai                                                                        |      1 |      0 |       0 |        1 |
| src/litemind/apis/providers/openai/utils/test/test_vision.py                  | test_vision_capable_models                                                                          |      9 |      0 |       0 |        9 |
| src/litemind/apis/providers/openai/utils/test/test_vision.py                  | test_non_vision_models                                                                              |      8 |      0 |       0 |        8 |
| src/litemind/apis/providers/openai/utils/test/test_vision.py                  | test_invalid_model_name                                                                             |      1 |      0 |       0 |        1 |
| src/litemind/apis/tests/test_apis_basics.py                                   | TestBaseApiImplementationsBasics.test_availability_and_credentials                                  |      6 |      0 |       0 |        6 |
| src/litemind/apis/tests/test_apis_basics.py                                   | TestBaseApiImplementationsBasics.test_model_list                                                    |