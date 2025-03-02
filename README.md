# LiteMind

[![PyPI version](https://badge.fury.io/py/litemind.svg)](https://badge.fury.io/py/litemind)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Code Coverage](https://img.shields.io/badge/coverage-86%25-brightgreen.svg)](https://github.com/royerlab/litemind)

## Summary

LiteMind is a powerful and elegant Python library that provides a unified interface for working with Large Language Models (LLMs). It operates on two distinct layers:

1. **API Wrapper Layer**: A consistent interface that abstracts away the differences between various LLM providers (OpenAI, Anthropic, Google, Ollama), making it easy to switch between them without changing your code.

2. **Agentic API**: A high-level framework for building conversational agents and tools with sophisticated reasoning capabilities, multimodal inputs, and tool usage.

LiteMind's philosophy is to provide a simple yet powerful API that handles the complexities of working with different LLM providers while enabling developers to build advanced AI applications with minimal code. The library excels at handling multimodal inputs (text, images, audio, video, documents) and provides robust tools for creating agents that can reason, use tools, and maintain context across conversations.

## Features

- **Unified API Wrapper** for multiple LLM providers:
  - OpenAI (GPT models)
  - Anthropic (Claude models)
  - Google (Gemini models)
  - Ollama (local open-source models)

- **Fully Multimodal Support**:
  - Text generation and structured output
  - Image, audio, and video inputs
  - Document processing (PDFs, webpages, etc.)
  - Table handling

- **Advanced Agentic Framework**:
  - Conversational agents with memory
  - Tool usage and function calling
  - ReAct agents with reasoning capabilities
  - Agent-to-agent communication

- **Rich Tooling**:
  - Function tools that wrap Python functions
  - Agent tools that delegate to other agents
  - Tool composition and chaining

- **Extensible Architecture**:
  - Easy to add new LLM providers
  - Customizable callbacks for monitoring and logging
  - Flexible message handling

## Installation

Install LiteMind using pip:

```bash
pip install litemind
```

For additional features, you can install optional dependencies:

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

### Basic API Wrapper Usage

The API wrapper provides a consistent interface to different LLM providers:

```python
from litemind.apis.combined_api import CombinedApi
from litemind.agent.messages.message import Message

# Create an API instance (automatically uses available providers)
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

### Multimodal Inputs

LiteMind makes it easy to work with multimodal inputs:

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

### Agentic API

The Agentic API allows you to create conversational agents with tool usage:

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

### ReAct Agent with Reasoning

LiteMind includes a ReAct agent implementation for step-by-step reasoning:

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

## Concepts

### API Wrapper Layer

The API wrapper layer provides a unified interface to different LLM providers:

- **BaseApi**: Abstract base class that defines the interface for all API implementations.
- **DefaultApi**: Provides default implementations for common functionality.
- **Provider-specific APIs**: Implementations for OpenAI, Anthropic, Google, and Ollama.
- **CombinedApi**: Aggregates multiple API providers and automatically selects the best one for each request.

The API wrapper handles:
- Model selection based on required features
- Message conversion between different formats
- Multimodal input processing
- Tool usage and function calling
- Structured output generation

### Agentic API

The Agentic API builds on top of the API wrapper to provide a higher-level framework for building conversational agents:

- **Message**: Represents a message in a conversation, containing multiple blocks of different types (text, image, audio, etc.).
- **MessageBlock**: A single block of content within a message.
- **Conversation**: A collection of messages that maintains context.
- **Agent**: A conversational agent that can use tools and maintain a conversation.
- **ReActAgent**: An agent that implements the ReAct (Reasoning + Acting) methodology for step-by-step reasoning.
- **ToolSet**: A collection of tools that an agent can use.
- **BaseTool**: Abstract base class for all tools.
- **FunctionTool**: A tool that wraps a Python function.
- **ToolAgent**: A tool that delegates to another agent.

### Model Features

LiteMind uses a feature-based approach to model selection:

- **ModelFeatures**: An enum that defines the features a model can support (TextGeneration, Image, Audio, Video, Tools, etc.).
- **get_best_model()**: A method that selects the best model based on required features.
- **has_model_support_for()**: A method that checks if a model supports specific features.

## More Code Examples

### Working with Different API Providers

```python
# Using OpenAI API
from litemind.apis.providers.openai.openai_api import OpenAIApi

api = OpenAIApi()
messages = [Message(role="user", text="Hello, world!")]
response = api.generate_text(messages=messages, model_name="gpt-4o")

# Using Anthropic API
from litemind.apis.providers.anthropic.anthropic_api import AnthropicApi

api = AnthropicApi()
messages = [Message(role="user", text="Hello, world!")]
response = api.generate_text(messages=messages, model_name="claude-3-opus-20240229")

# Using Google Gemini API
from litemind.apis.providers.google.google_api import GeminiApi

api = GeminiApi()
messages = [Message(role="user", text="Hello, world!")]
response = api.generate_text(messages=messages, model_name="models/gemini-1.5-pro")

# Using Ollama API (local models)
from litemind.apis.providers.ollama.ollama_api import OllamaApi

api = OllamaApi()
messages = [Message(role="user", text="Hello, world!")]
response = api.generate_text(messages=messages, model_name="llama3")
```

### Structured Output Generation

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
    print()
```

### Working with Documents

```python
from litemind.apis.combined_api import CombinedApi
from litemind.agent.messages.message import Message
from litemind.apis.model_features import ModelFeatures

# Create an API instance
api = CombinedApi()

# Get a model that supports documents
model_name = api.get_best_model([ModelFeatures.TextGeneration, ModelFeatures.Document])

# Create a message with a document
message = Message(role="user")
message.append_text("Please summarize this PDF document:")
message.append_document("https://arxiv.org/pdf/2303.08774.pdf")

# Generate a response
response = api.generate_text(messages=[message], model_name=model_name)

# Print the response
print(response[0])
# Expected output: A summary of the PDF document
```

### Working with Videos

```python
from litemind.apis.combined_api import CombinedApi
from litemind.agent.messages.message import Message
from litemind.apis.model_features import ModelFeatures

# Create an API instance
api = CombinedApi()

# Get a model that supports videos
model_name = api.get_best_model([ModelFeatures.TextGeneration, ModelFeatures.Video])

# Create a message with a video
message = Message(role="user")
message.append_text("What's happening in this video?")
message.append_video("https://example.com/sample-video.mp4")

# Generate a response
response = api.generate_text(messages=[message], model_name=model_name)

# Print the response
print(response[0])
# Expected output: A description of the video content
```

### Using Callbacks for Monitoring

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

### Creating a Tool Agent

```python
from litemind.agent.agent import Agent
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

### Embedding and Similarity

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

## Caveats and Limitations

- **API Keys**: You need to provide your own API keys for the various providers (OpenAI, Anthropic, Google).
- **Cost**: Using commercial LLM APIs incurs costs based on their pricing models.
- **Model Availability**: Not all features are available with all models or providers.
- **Token Limits**: Different models have different token limits that may affect the complexity of tasks they can handle.
- **Tool Execution**: Tools are executed locally, which means they have access to your local environment.
- **ReAct Agent Limitations**: The ReAct agent implementation has some limitations in handling complex reasoning chains and may occasionally fail to follow the correct reasoning format.
- **Video Processing**: Video processing requires ffmpeg to be installed and may be resource-intensive.
- **Document Processing**: Document processing requires additional dependencies and may not work perfectly with all document formats.

## Code Health

The test suite for LiteMind shows good overall health with 452 passing tests out of 525 total tests (86% pass rate). There are 12 failing tests and 61 skipped tests.

The failing tests are primarily in the ReAct agent implementation:
- `test_react_agent_single_tool` (1 failure)
- `test_react_agent_multiple_tools` (2 failures)
- `test_react_agent_chained_tools` (2 failures)
- `test_react_agent_longer_dialog` (2 failures)
- `test_agent_tool` (1 failure)
- `test_agent_tool_translation` (1 failure)
- `test_agent_tool_with_internal_tool` (1 failure)
- `test_toolset_add_agent_tool` (1 failure)
- `test_text_generation_with_folder` (1 failure)

These failures indicate that the ReAct agent and tool agent implementations need further refinement to ensure consistent behavior across different LLM providers.

The skipped tests are primarily due to certain features not being available with all providers, which is expected behavior.

## Roadmap

- [x] Setup a readme with a quick start guide.
- [x] setup continuous integration and pipy deployment.
- [x] ReAct Agent
- [x] Improve document conversion (page per page text and video interleaving + whole page images)
- [x] Cleanup structured output with tool usage
- [x] Implement streaming callbacks
- [x] Improve folder/archive conversion: add ascii folder tree
- [ ] Deal with message sizes in tokens sent to models
- [ ] RAG
- [ ] Improve vendor api robustness features such as retry call when server errors, etc...
- [ ] Reorganise media files used for testing into a single media folder
- [ ] Improve and uniformize exception handling
- [ ] Add support for adding nD images to messages.
- [ ] Improve logging with arbol, with option to turn off.
- [ ] Implement 'brainstorming' mode for text generation, possibly with API fusion.

## Contributing

We welcome contributions to LiteMind! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to contribute.

## License

LiteMind is licensed under the BSD-3-Clause License. See [LICENSE](LICENSE) for details.

---

*This README was generated with the help of AI.*