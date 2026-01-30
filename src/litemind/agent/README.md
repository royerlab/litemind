# Agent Package

The `agent` package is the core agentic AI framework in Litemind, providing orchestration for conversations with language models, tool execution, and retrieval-augmented generation (RAG).

## Package Structure

```
agent/
├── agent.py                 # Main Agent orchestrator
├── exceptions.py            # Agent-specific exceptions
├── messages/                # Message and conversation handling
│   ├── message.py           # Core Message class (multimodal)
│   ├── message_block.py     # Individual content blocks
│   ├── conversation.py      # Conversation state management
│   └── actions/             # Tool call/use action types
├── tools/                   # Tool system
│   ├── base_tool.py         # Abstract tool interface
│   ├── function_tool.py     # Python functions as tools
│   ├── agent_tool.py        # Agents as tools (nesting)
│   ├── toolset.py           # Tool collection manager
│   ├── builtin_tools/       # Web search, MCP protocol
│   └── callbacks/           # Tool execution callbacks
└── augmentations/           # RAG support
    ├── augmentation_base.py # Abstract augmentation interface
    ├── augmentation_set.py  # Augmentation collection
    ├── information/         # Knowledge units with metadata
    └── vector_db/           # Vector database implementations
```

## Core Components

### Agent (`agent.py`)

The `Agent` class is the main orchestrator that manages:
- Conversation state and history
- Tool execution and result handling
- Augmentation/RAG retrieval
- Model selection based on required features

```python
from litemind import OpenAIApi
from litemind.agent.agent import Agent

agent = Agent(
    api=OpenAIApi(),
    model_name="gpt-4o",
    model_features=["textgeneration", "tools"],
)
agent.append_system_message("You are a helpful assistant.")
response = agent("Hello!")
```

### Messages (`messages/`)

#### Message

The `Message` class is a multimodal container supporting text, images, audio, video, documents, tables, and more.

```python
from litemind.agent.messages.message import Message

message = Message(role="user")
message.append_text("Describe this image:")
message.append_image("path/to/image.jpg")
message.append_audio("path/to/audio.wav")
```

Key methods:
- `append_text()`, `append_image()`, `append_audio()`, `append_video()`
- `append_document()`, `append_table()`, `append_folder()`
- `convert_media()` - Automatic format conversion
- `compress_text()` - Text compression for token limits

#### MessageBlock

Individual content blocks within a message, each containing a single media element with optional attributes.

#### Conversation

Manages system messages and conversation history with methods for appending, clearing, and accessing messages.

### Tools (`tools/`)

#### BaseTool

Abstract base class defining the tool interface with callback support.

#### FunctionTool

Wraps Python functions as tools with automatic JSON schema generation:

```python
from litemind.agent.tools.toolset import ToolSet

def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"Weather in {city}: Sunny, 72°F"

toolset = ToolSet()
toolset.add_function_tool(get_weather)
```

#### AgentTool

Wraps an Agent as a tool for hierarchical agent composition:

```python
from litemind.agent.tools.agent_tool import AgentTool

sub_agent = Agent(api=api, name="researcher")
agent_tool = AgentTool(sub_agent, description="Research assistant")
toolset.add_tool(agent_tool)
```

#### ToolSet

Collection manager for tools with callback handling:

```python
toolset = ToolSet()
toolset.add_function_tool(my_function)
toolset.add_builtin_web_search_tool()
toolset.add_builtin_mcp_tool(server_name="my-mcp", server_url="...")
```

### Augmentations (`augmentations/`)

#### AugmentationBase

Abstract interface for augmentation sources (RAG).

#### Information

Knowledge units with metadata for retrieval:

```python
from litemind.agent.augmentations.information.information import Information
from litemind.media.types.media_text import Text

info = Information(
    Text("Einstein developed the theory of relativity."),
    metadata={"topic": "physics", "person": "Einstein"}
)
```

#### Vector Databases

- `InMemoryVectorDatabase` - KD-tree based, with persistence support
- `QdrantVectorDatabase` - Qdrant cloud/local backend

```python
from litemind.agent.augmentations.vector_db.in_memory_vector_db import InMemoryVectorDatabase

vector_db = InMemoryVectorDatabase(name="knowledge_base")
vector_db.add_informations([info1, info2, info3])

agent.add_augmentation(vector_db)
```

## Callback System

Both tools and the agent support callbacks for monitoring:

```python
from litemind.agent.tools.callbacks.base_tool_callbacks import BaseToolCallbacks

class MyToolCallback(BaseToolCallbacks):
    def on_tool_start(self, tool, args, kwargs):
        print(f"Tool {tool.name} starting...")

    def on_tool_end(self, tool, result):
        print(f"Tool {tool.name} finished: {result}")

toolset.add_tool_callback(MyToolCallback())
```

## Data Flow

```
User Message
    → Agent
    → Augmentation retrieval (RAG)
    → Message formatting
    → API call
    → Tool execution (if needed)
    → Response
    → Conversation history
```

## Docstring Coverage

| Module | Coverage |
|--------|----------|
| agent.py | 85% |
| messages/message.py | 95% |
| messages/message_block.py | 100% |
| messages/conversation.py | 100% |
| tools/toolset.py | 100% |
| tools/callbacks | 100% |
| augmentations/ | 95%+ |

The package follows numpy-style docstrings with comprehensive Parameters, Returns, and Raises sections.
