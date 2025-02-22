# LiteMind: Elegant Multimodal Agentic AI Framework

## Summary
LiteMind is a powerful Python framework that provides a unified interface for building conversational AI agents with multimodal capabilities. It offers two main layers of functionality:

1. **API Wrapper Layer**: A consistent interface for multiple LLM providers (OpenAI, Anthropic, Google, Ollama) with unified handling of text generation, embeddings, and multimodal inputs/outputs.

2. **Agentic Layer**: An elegant API for building conversational agents with rich multimodal capabilities, tool integration, and structured conversations.

## Features

### API Wrapper Layer
- Unified interface for multiple LLM providers (OpenAI, Anthropic, Google, Ollama)
- Automatic model selection based on required capabilities
- Consistent handling of:
  - Text generation and chat
  - Image, audio, and video inputs
  - Document processing (PDF, Word, etc.)
  - Embeddings generation
  - Structured output formats
  - Function calling/tools

### Agentic Layer
- Rich message composition with multiple content types
- Flexible conversation management
- Tool integration framework
- Support for:
  - Text blocks
  - Images
  - Audio
  - Video
  - Documents
  - Tables
  - Structured data
  - Folders and archives

### Additional Features
- Automatic content type detection and conversion
- Streaming support for text generation
- Callback system for monitoring and logging
- Comprehensive error handling
- Extensive test coverage

## Installation

```bash
pip install litemind
```

Optional dependencies can be installed using extras:

```bash
# For document processing support
pip install litemind[documents]

# For video processing support
pip install litemind[videos]

# For audio transcription
pip install litemind[whisper]

# For embedding generation
pip install litemind[embed]

# For table processing
pip install litemind[tables]

# For RAG support
pip install litemind[rag]
```

## Usage

### Basic API Usage

```python
from litemind.apis.combined_api import CombinedApi
from litemind.apis.model_features import ModelFeatures

# Initialize the API
api = CombinedApi()

# Get best model for text generation
model = api.get_best_model(features=ModelFeatures.TextGeneration)

# Generate text
response = api.generate_text(
    messages=[{"role": "user", "text": "Hello, how are you?"}],
    model_name=model
)
```

### Multimodal Inputs

```python
from litemind.agent.message import Message

# Create a message with multiple content types
message = Message(role="user")
message.append_text("What can you tell me about this image?")
message.append_image("https://example.com/image.jpg")
message.append_document("https://example.com/document.pdf")

# Generate response
response = api.generate_text(
    messages=[message],
    model_name=api.get_best_model([ModelFeatures.TextGeneration, ModelFeatures.Image])
)
```

### Agent Creation

```python
from litemind.agent.agent import Agent
from litemind.agent.tools.toolset import ToolSet

# Create a toolset
toolset = ToolSet()

# Add a function tool
def get_weather(location: str) -> str:
    return f"Weather in {location}: Sunny"

toolset.add_function_tool(
    get_weather,
    "Get weather information for a location"
)

# Create an agent
agent = Agent(
    api=api,
    toolset=toolset,
    temperature=0.7
)

# Use the agent
response = agent("What's the weather in Paris?")
```

### Structured Output

```python
from pydantic import BaseModel

class WeatherInfo(BaseModel):
    location: str
    temperature: float
    condition: str

response = api.generate_text(
    messages=[Message(role="user", text="What's the weather in Paris?")],
    response_format=WeatherInfo
)
```

## Code Examples

### Working with Documents

```python
# Process a PDF document
message = Message(role="user")
message.append_text("Summarize this document:")
message.append_document("path/to/document.pdf")

response = api.generate_text(
    messages=[message],
    model_name=api.get_best_model([ModelFeatures.TextGeneration, ModelFeatures.Document])
)
```

### Video Processing

```python
# Process a video file
message = Message(role="user")
message.append_text("Describe what happens in this video:")
message.append_video("path/to/video.mp4")

response = api.generate_text(
    messages=[message],
    model_name=api.get_best_model([ModelFeatures.TextGeneration, ModelFeatures.Video])
)
```

### Embedding Generation

```python
# Generate embeddings for texts
texts = ["Hello world", "How are you?"]
embeddings = api.embed_texts(
    texts=texts,
    model_name=api.get_best_model(ModelFeatures.TextEmbeddings),
    dimensions=512
)
```

## Caveats and Limitations

- Different LLM providers have varying capabilities and limitations
- Video processing requires ffmpeg installation
- Document processing requires additional dependencies
- Some features may require specific model support
- Token limits vary by model and provider
- Structured output may not be supported by all models

## Code Health

Based on the test report:

- Total tests: 36
- Passed: 32
- Failed: 1
- Skipped: 3

Test coverage includes:
- Agent tools functionality
- API implementations
- Message handling
- Conversation management
- Multimodal processing

## Roadmap

- [x] Setup a readme with a quick start guide
- [x] Setup continuous integration and pipy deployment
- [ ] ReAct Agent implementation
- [ ] RAG support
- [ ] Support for adding nD images to messages
- [x] Improve document conversion
- [ ] Deal with message sizes in tokens sent to models
- [ ] Improve and uniformize exception handling
- [ ] Improve logging with arbol
- [ ] Implement 'brainstorming' mode for text generation
- [x] Implement streaming callbacks
- [x] Improve folder/archive conversion

## Contributing and Todos

Contributions are welcome! Areas that need attention:

1. ReAct Agent implementation
2. RAG support
3. Token management improvements
4. Exception handling standardization
5. Logging enhancements
6. Additional model support

Please submit pull requests and report issues on GitHub.

## License

Copyright (c) 2023, Loic A. Royer
All rights reserved.

Distributed under the BSD 3-Clause License. See LICENSE file for details.

---
*Note: This README was generated with the assistance of AI using LiteMind's readme generation tool.*