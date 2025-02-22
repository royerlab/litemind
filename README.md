
# litemind: Agentic AI and LLM API Wrapper for Python

[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![PyPI version](https://badge.fury.io/py/litemind.svg)](https://badge.fury.io/py/litemind)
[![codecov](https://codecov.io/gh/<user>/<repo>/branch/main/graph/badge.svg?token=<token>)]()
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=%2334BF49)](https://pycqa.github.io/isort/)
[![linting: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![pytest-md-report](https://raw.githubusercontent.com/lark-parser/pytest-md-report/main/assets/pytest-md-report-badge.svg)](https://github.com/lark-parser/pytest-md-report)

## Summary

**litemind** is a Python library designed to simplify the interaction with Large Language Models (LLMs) and enable the creation of sophisticated agentic AI applications. It offers two primary APIs:

1.  **Wrapper API**: A unified and intuitive interface to interact with various LLM APIs (like OpenAI, Gemini, Anthropic, Ollama) for tasks such as text generation, image generation, audio transcription, embeddings, and more. This layer abstracts away the complexities of individual API implementations, providing a consistent way to work with different models.

2.  **Agentic API**: A powerful and elegant framework for building multimodal agentic AI. This API allows you to create conversational agents capable of reasoning, using tools, and interacting with various data modalities (text, images, audio, video, documents, tables). It's designed for building complex conversational tools and agents that can understand and respond to rich, multimodal inputs.

**Key Differentiators:**

*   **Unified API Access**: Seamlessly switch between different LLM providers using a consistent Python API.
*   **Multimodal Agentic AI**: Build advanced agents that can process and generate multimodal content, enabling richer and more interactive AI applications.
*   **Tool Integration**: Easily integrate tools and functions into your agents, allowing them to perform actions and access external information.
*   **Flexibility and Extensibility**: Designed to be modular and extensible, allowing you to customize and expand its capabilities.

## Features

**Wrapper API Layer:**

*   **Unified Interface for LLMs**: Access multiple LLM APIs (OpenAI, Gemini, Anthropic, Ollama) through a single, consistent Python API.
*   **Text Generation**: Generate text completions with various models, supporting features like streaming and structured output.
*   **Multimodal Generation**: Generate images and audio using supported models.
*   **Multimodal Input Handling**: Process messages with text, images, audio, video, documents, and tables as input.
*   **Embeddings**: Generate embeddings for text, images, audio, and videos.
*   **Description/Analysis**: Describe and analyse images, audio, and videos using multimodal models.
*   **Audio Transcription**: Transcribe audio files using local or remote models.
*   **Document Conversion**: Convert documents (PDF, etc.) to markdown format, extracting text and images.
*   **Video Conversion**: Extract images and audio from video files.
*   **Callback Management**: Implement callbacks for various API events (streaming, model selection, etc.) for logging, monitoring, and custom behaviours.

**Agentic API Layer:**

*   **Conversational Agents**: Build stateful conversational agents with memory and context management.
*   **ReAct Agents**: Implement advanced ReAct agents capable of reasoning and acting in a loop, leveraging tools for complex tasks.
*   **Tool Integration**: Easily define and integrate custom tools (functions, agents) for extending agent capabilities.
*   **Multimodal Messages**: Create and process messages with multiple content blocks (text, images, audio, video, documents, tables).
*   **Message Blocks**: Represent different data modalities (text, images, audio, video, documents, tables) as modular blocks within messages.
*   **Conversation History**: Manage conversation history for stateful agents.
*   **Flexible Message Handling**: Append folders, archives, tables, documents, and various media types to messages for rich input processing.

## Installation

```bash
pip install litemind
```

**Optional Dependencies:**

For extended functionalities, install optional dependencies:

*   **RAG (Retrieval-Augmented Generation):**
    ```bash
    pip install litemind[rag]
    ```
    (Includes `chromadb`)

*   **Whisper (Local Audio Transcription):**
    ```bash
    pip install litemind[whisper]
    ```
    (Includes `openai-whisper`)

*   **Documents (Document Processing):**
    ```bash
    pip install litemind[documents]
    ```
    (Includes `pymupdf`, `pymupdf4llm`)

*   **Tables (Table Data Handling):**
    ```bash
    pip install litemind[tables]
    ```
    (Includes `pandas`, `numpy`, `chardet`, `tabulate`)

*   **Videos (Video Processing):**
    ```bash
    pip install litemind[videos]
    ```
    (Includes `ffmpeg-python`, `imageio[ffmpeg]`)

*   **Embeddings (FastEmbed Embeddings):**
    ```bash
    pip install litemind[embed]
    ```
    (Includes `fastembed`)

*   **Test Dependencies:**
    ```bash
    pip install litemind[test]
    ```
    (Includes `pytest`, `pytest-html`, `pytest-cov`, `pytest-md-report`)

*   **All Optional Features:**
    ```bash
    pip install litemind[rag,whisper,documents,tables,videos,embed,test]
    ```

## Usage

### Wrapper API Examples

**1. Text Generation with OpenAI:**

```python
from litemind.apis.openai.openai_api import OpenAIApi
from litemind.agent.message import Message

api = OpenAIApi()
messages = [
    Message(role="user", text="Write a short poem about the moon.")
]
response = api.generate_text(messages=messages)
print(response.content)
```

**2. Image Generation with Gemini:**

```python
from litemind.apis.google.google_api import GeminiApi

api = GeminiApi()
image_uri = api.generate_image(positive_prompt="A futuristic cityscape at night.")
print(f"Image saved at: {image_uri}")
```

**3. Audio Transcription with Ollama (using local Whisper model):**

```python
from litemind.apis.ollama.ollama_api import OllamaApi
from litemind.apis.default_api import DefaultApi

# Use DefaultApi to access local whisper model
api = DefaultApi()
audio_uri = "path/to/your/audiofile.wav" # Replace with your audio file path
transcription = api.transcribe_audio(audio_uri=audio_uri)
print(f"Transcription: {transcription}")
```

**4. Text Embedding with Combined API (using best available model):**

```python
from litemind.apis.combined_api import CombinedApi

api = CombinedApi()
texts = ["First text for embedding", "Second text for embedding"]
embeddings = api.embed_texts(texts=texts)
print(f"Embeddings: {embeddings}")
```

### Agentic API Examples

**1. Simple Conversational Agent:**

```python
from litemind.agent.agent import Agent
from litemind.apis.openai.openai_api import OpenAIApi
from litemind.agent.message import Message

api = OpenAIApi()
agent = Agent(api=api, name="SimpleChatAgent")

# System message to set the agent's persona
agent += Message(role="system", text="You are a helpful and friendly chatbot.")

# User query
user_query = "Hello, how can you help me today?"
response = agent(text=user_query)

print(f"Agent's Response: {response.content}")
```

**2. ReAct Agent with Tool Integration (Date Calculation):**

```python
from litemind.agent.react.react_agent import ReActAgent
from litemind.agent.tools.toolset import ToolSet
from litemind.apis.openai.openai_api import OpenAIApi
from datetime import datetime, timedelta

def add_days_to_date(date_str, days):
    """Tool function to add days to a date in YYYY-MM-DD format."""
    date = datetime.strptime(date_str, "%Y-%m-%d")
    new_date = date + timedelta(days=days)
    return new_date.strftime("%Y-%m-%d")

api = OpenAIApi()
toolset = ToolSet()
toolset.add_function_tool(
    add_days_to_date,
    "Add a certain number of days to a given date. Input date must be in format YYYY-MM-DD."
)

react_agent = ReActAgent(
    api=api,
    toolset=toolset,
    name="DateCalculatorAgent"
)

query = "What will be the date 10 days after 2024-03-15?"
response = react_agent(text=query)

print(f"ReAct Agent Response: {response.content}")
```

**3. Multimodal Agent with Image Description:**

```python
from litemind.agent.agent import Agent
from litemind.apis.google.google_api import GeminiApi
from litemind.agent.message import Message

api = GeminiApi()
agent = Agent(api=api, name="MultimodalAgent")

image_path = "path/to/your/image.jpg" # Replace with your image path

user_message = Message(role="user")
user_message.append_text("Describe this image in detail:")
user_message.append_image(image_path)

response = agent(message=user_message)

print(f"Multimodal Agent's Image Description: {response.content}")
```

**4. Agent with Conversation History:**

```python
from litemind.agent.agent import Agent
from litemind.apis.openai.openai_api import OpenAIApi
from litemind.agent.message import Message

api = OpenAIApi()
agent = Agent(api=api, name="MemoryAgent")

# First turn
response1 = agent(text="Hello, I'm interested in ordering a pizza.")
print(f"Agent: {response1.content}")

# Second turn, agent remembers the context
response2 = agent(text="What are my options for toppings?")
print(f"Agent: {response2.content}")

# Third turn, continuing the conversation
response3 = agent(text="I'll take pepperoni and mushrooms.")
print(f"Agent: {response3.content}")
```

## Code Examples

These examples are inspired by the unit tests and demonstrate various functionalities of `litemind`.

**1. Text Embedding:**

```python
from litemind.apis.combined_api import CombinedApi
api_instance = CombinedApi()
embedding_model_name = api_instance.get_best_model(
    ModelFeatures.TextEmbeddings)
texts = ["Hello, world!", "Testing embeddings."]
embeddings = api_instance.embed_texts(texts=texts,
                                      model_name=embedding_model_name,
                                      dimensions=512)
print(embeddings)
```

**2. Audio Embedding:**

```python
from litemind.apis.combined_api import CombinedApi
from litemind.apis.tests.base_test import BaseTest
api_instance = CombinedApi()
embedding_model_name = api_instance.get_best_model(
    ModelFeatures.AudioEmbeddings)
audio_1_uri = BaseTest._get_local_test_audio_uri('harvard.wav')
audio_2_uri = BaseTest._get_local_test_audio_uri('preamble.wav')
embeddings = api_instance.embed_audios(audio_uris=[audio_1_uri, audio_2_uri],
                                       model_name=embedding_model_name,
                                       dimensions=512)
print(embeddings)
```

**3. Video Embedding:**

```python
from litemind.apis.combined_api import CombinedApi
from litemind.apis.tests.base_test import BaseTest
api_instance = CombinedApi()
embedding_model_name = api_instance.get_best_model(
    ModelFeatures.VideoEmbeddings)
video_uri_1 = BaseTest._get_local_test_video_uri('flying.mp4')
video_uri_2 = BaseTest._get_local_test_video_uri('lunar_park.mp4')
embeddings = api_instance.embed_videos(video_uris=[video_uri_1, video_uri_2],
                                       model_name=embedding_model_name,
                                       dimensions=512)
print(embeddings)
```

**4. Text Generation with Simple Toolset:**

```python
from litemind.apis.combined_api import CombinedApi
from litemind.agent.message import Message
from litemind.agent.tools.toolset import ToolSet
api_instance = CombinedApi()

def get_delivery_date(order_id: str) -> str:
    """Fetch the delivery date for a given order ID."""
    return "2024-11-15"

toolset = ToolSet()
toolset.add_function_tool(
    get_delivery_date,
    "Fetch the delivery date for a given order ID"
)
default_model_name = api_instance.get_best_model(
    [ModelFeatures.Tools, ModelFeatures.TextGeneration])
user_message = Message(role="user",
                       text="When will my order order_12345 be delivered?")
messages = [user_message]
response = api_instance.generate_text(
    model_name=default_model_name,
    messages=messages,
    toolset=toolset
)
print(response.content)
```

**5. Text Generation with Image Input:**

```python
from litemind.apis.combined_api import CombinedApi
from litemind.agent.message import Message
from litemind.apis.tests.base_test import BaseTest
api_instance = CombinedApi()
default_model_name = api_instance.get_best_model(
    [ModelFeatures.TextGeneration, ModelFeatures.Image])
user_message = Message(role='user')
user_message.append_text('Can you describe what you see in the image?')
image_path = BaseTest._get_local_test_image_uri('future.jpeg')
user_message.append_image(image_path)
messages = [user_message]
response = api_instance.generate_text(messages=messages,
                                      model_name=default_model_name)
print(response.content)
```

**6. Text Generation with Audio Input:**

```python
from litemind.apis.combined_api import CombinedApi
from litemind.agent.message import Message
from litemind.apis.tests.base_test import BaseTest
api_instance = CombinedApi()
default_model_name = api_instance.get_best_model(
    [ModelFeatures.TextGeneration, ModelFeatures.Audio])
user_message = Message(role='user')
user_message.append_text(
    'Can you describe what you heard in the audio file?')
audio_path = BaseTest._get_local_test_audio_uri('harvard.wav')
user_message.append_audio(audio_path)
messages = [user_message]
response = api_instance.generate_text(messages=messages,
                                      model_name=default_model_name)
print(response.content)
```

**7. Text Generation with Document Input:**

```python
from litemind.apis.combined_api import CombinedApi
from litemind.agent.message import Message
from litemind.apis.tests.base_test import BaseTest
api_instance = CombinedApi()
default_model_name = api_instance.get_best_model(
    [ModelFeatures.TextGeneration,
     ModelFeatures.Document,
     ModelFeatures.Image])
user_message = Message(role='user')
user_message.append_text(
    'Can you write a review for the provided paper? Please break down your comments into major and minor comments.')
doc_path = BaseTest._get_local_test_document_uri('intracktive_preprint.pdf')
user_message.append_document(doc_path)
messages = [user_message]
response = api_instance.generate_text(messages=messages,
                                      model_name=default_model_name)
print(response.content)
```

## Caveats and Limitations

*   **API Key Management**: Requires users to manage API keys for different LLM providers. Ensure API keys are securely stored and handled.
*   **Model Compatibility**: Not all models support all features. Feature availability depends on the underlying API and model capabilities.
*   **Error Handling**: While `litemind` provides a unified API, underlying API errors may still surface. Robust error handling should be implemented in applications using `litemind`.
*   **Performance**: Performance may vary depending on the chosen LLM provider and model. Consider latency and throughput requirements for your application.
*   **Local Models**: Local models (like `whisper-local`, `fastembed`, `ollama`) rely on local installations and resources. Ensure these dependencies are correctly set up.
*   **Evolving APIs**: LLM APIs are constantly evolving. `litemind` aims to provide a stable abstraction layer, but API changes may require updates to the library.
*   **Structured Output and Tools**: While supported, the reliability and format of structured output and tool usage can vary across different models and APIs. Careful testing and validation are recommended.

## Code Health

Unit tests are implemented using `pytest` and test reports are generated using `pytest-md-report`.

**Test Execution Summary:**

```
============================= test session starts ==============================
platform darwin -- Python 3.12.8, pytest-8.3.4, pluggy-1.5.0
rootdir: /Users/loic.royer/workspace/python/litemind
configfile: pyproject.toml
plugins: html-4.1.1, metadata-3.1.1, anyio-4.8.0, md-report-0.6.3
collected 459 items / 423 deselected / 36 selected

src/litemind/agent/tools/test/test_agent_tool.py ...                     [  8%]
src/litemind/agent/tools/test/test_function_tool.py .......              [ 27%]
src/litemind/agent/tools/test/test_toolset.py .....                      [ 41%]
src/litemind/agent/tools/utils/test/test_inspect_function.py ..          [ 47%]
src/litemind/apis/openai/utils/test/test_tools.py .                      [ 50%]
src/litemind/apis/tests/test_apis_text_generation.py s.....s...F.s.....  [100%]

=================================== FAILURES ===================================
_ TestBaseApiImplementationsTextGeneration.test_text_generation_with_simple_toolset_and_struct_output[GeminiApi] _
...
E           google.api_core.exceptions.InvalidArgument: 400 Function calling with a response mime type: 'application/json' is unsupported
...
=============================== warnings summary ===============================
...
=========================== short test summary info ============================
FAILED src/litemind/apis/tests/test_apis_text_generation.py::TestBaseApiImplementationsTextGeneration::test_text_generation_with_simple_toolset_and_struct_output[GeminiApi]
= 1 failed, 32 passed, 3 skipped, 423 deselected, 5 warnings in 167.19s (0:02:47) =
```

**Detailed Test Report:**

```
|                           filepath                           |                                              function                                               | passed | failed | skipped | SUBTOTAL |
| ------------------------------------------------------------ | --------------------------------------------------------------------------------------------------- | -----: | -----: | ------: | -------: |
| src/litemind/agent/tools/test/test_agent_tool.py             | test_agent_tool_translation                                                                         |      1 |      0 |       0 |        1 |
| src/litemind/agent/tools/test/test_agent_tool.py             | test_agent_tool                                                                                     |      1 |      0 |       0 |        1 |
| src/litemind/agent/tools/test/test_agent_tool.py             | test_agent_tool_with_internal_tool                                                                  |      1 |      0 |       0 |        1 |
| src/litemind/agent/tools/test/test_function_tool.py          | test_tool_initialization                                                                            |      1 |      0 |       0 |        1 |
| src/litemind/agent/tools/test/test_function_tool.py          | test_tool_initialization_automatic_description                                                      |      1 |      0 |       0 |        1 |
| src/litemind/agent/tools/test/test_function_tool.py          | test_tool_with_no_parameters                                                                        |      1 |      0 |       0 |        1 |
| src/litemind/agent/tools/test/test_function_tool.py          | test_tool_with_default_parameter                                                                    |      1 |      0 |       0 |        1 |
| src/litemind/agent/tools/test/test_function_tool.py          | test_tool_execution_with_parameters                                                                 |      1 |      0 |       0 |        1 |
| src/litemind/agent/tools/test/test_function_tool.py          | test_tool_execution_no_parameters                                                                   |      1 |      0 |       0 |        1 |
| src/litemind/agent/tools/test/test_function_tool.py          | test_tool_execution_with_default_parameters                                                         |      1 |      0 |       0 |        1 |
| src/litemind/agent/tools/test/test_toolset.py                | test_toolset_initialization_and_add_tool                                                            |      1 |      0 |       0 |        1 |
| src/litemind/agent/tools/test/test_toolset.py                | test_toolset_add_function_tool                                                                      |      1 |      0 |       0 |        1 |
| src/litemind/agent/tools/test/test_toolset.py                | test_toolset_add_agent_tool                                                                         |      1 |      0 |       0 |        1 |
| src/litemind/agent/tools/test/test_toolset.py                | test_toolset_get_tool                                                                               |      1 |      0 |       0 |        1 |
| src/litemind/agent/tools/test/test_toolset.py                | test_toolset_list_tools                                                                             |      1 |      0 |       0 |        1 |
| src/litemind/agent/tools/utils/test/test_inspect_function.py | test_extract_docstring                                                                              |      1 |      0 |       0 |        1 |
| src/litemind/agent/tools/utils/test/test_inspect_function.py | test_extract_function_info                                                                          |      1 |      0 |       0 |        1 |
| src/litemind/apis/openai/utils/test/test_tools.py            | test_format_tools_for_openai                                                                        |      1 |      0 |       0 |        1 |
| src/litemind/apis/tests/test_apis_text_generation.py         | TestBaseApiImplementationsTextGeneration.test_text_generation_with_simple_toolset                   |      5 |      0 |       1 |        6 |
| src/litemind/apis/tests/test_apis_text_generation.py         | TestBaseApiImplementationsTextGeneration.test_text_generation_with_simple_toolset_and_struct_output |      4 |      1 |       1 |        6 |
| src/litemind/apis/tests/test_apis_text_generation.py         | TestBaseApiImplementationsTextGeneration.test_text_generation_with_complex_toolset                  |      5 |      0 |       1 |        6 |
| TOTAL                                                        |                                                                                                     |     32 |      1 |       3 |       36 |
```

## Roadmap

TODO:

- [x] Setup a readme with a quick start guide.
- [x] setup continuous integration and pipy deployment.
- [x] ReAct Agent
- [ ] RAG
- [ ] Add support for adding nD images to messages.
- [x] Improve document conversion (page per page text and video interleaving + whole page images)
- [ ] Deal with message sizes in tokens sent to models
- [ ] Improve and uniformize exception handling
- [ ] Improve logging with arbol, with option to turn off.
- [ ] Implement 'brainstorming' mode for text generation, possibly with API fusion.
- [x] Implement streaming callbacks
- [x] Improve folder/archive conversion: add ascii folder tree

## Contributing and Todos

Contributions are welcome! Feel free to fork the repository and submit pull requests.

**Current Todos (from Roadmap):**

- [ ] RAG (Retrieval-Augmented Generation) implementation.
- [ ] Add support for n-dimensional (nD) images to messages.
- [ ] Improve handling of message sizes in tokens sent to models to optimize API usage and prevent errors.
- [ ] Enhance and standardize exception handling across the library for better error reporting and resilience.
- [ ] Improve logging using `arbol`, with options to configure verbosity and turn off logging for production environments.
- [ ] Implement a 'brainstorming' mode for text generation, potentially leveraging API fusion for enhanced creativity and diversity in responses.

## License

This project is licensed under the terms of the BSD 3-Clause License. See the [LICENSE](LICENSE) file for details.

---

*This README.md file was generated with the help of an AI assistant.*
