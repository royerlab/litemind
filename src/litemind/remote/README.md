# Remote Package

The `remote` package enables distributed execution of Litemind Agents via RPC (Remote Procedure Call) using the RPyC library.

## Package Structure

```
remote/
├── __init__.py
├── server.py                # RPC server implementation
├── client.py                # RPC client implementation
└── tests/
    └── test_remote_agent.py # Integration tests
```

## Overview

The remote package allows you to:
- Run Litemind Agents on a server machine
- Access those agents from client machines over the network
- Maintain full multimodal capabilities (images, audio, video) via pickle serialization

## Server

### Starting a Server

```python
from litemind.remote.server import Server
from litemind.agent.agent import Agent
from litemind import CombinedApi

# Create an agent
api = CombinedApi()
agent = Agent(api=api, name="assistant")
agent.append_system_message("You are a helpful assistant.")

# Create and start server
server = Server(host="localhost", port=18861)
server.expose("my_agent", agent)  # Register agent with a name
server.start(block=True)  # Block main thread (or block=False for background)
```

### Server Configuration

```python
Server(
    host: Optional[str] = None,  # Defaults to localhost
    port: Optional[int] = None,  # Defaults to 18861
)
```

### Server Methods

- `expose(name, obj)` - Register an object for remote access
- `start(block=True)` - Start the server
- `close()` - Stop the server
- `is_running()` - Check if server is active
- `get_port()` - Get the listening port

## Client

### Connecting to a Server

```python
from litemind.remote.client import Client

# Connect to server
client = Client(host="localhost", port=18861)

# Check connection
if client.is_connected():
    # Get remote agent
    agent = client["my_agent"]  # or client.get("my_agent")

    # Use agent normally
    response = agent("Hello, world!")
    print(response)

# Clean up
client.close()
```

### Client Configuration

```python
Client(
    host: str = "localhost",
    port: int = 18861,
)
```

### Client Methods

- `is_connected()` - Check connection status
- `get(name)` - Get remote object by name
- `__getitem__(name)` - Dictionary-style access: `client["name"]`
- `close()` - Close the connection

## Full Example

### Server Side (`server_app.py`)

```python
from litemind.remote.server import Server
from litemind.agent.agent import Agent
from litemind.agent.tools.toolset import ToolSet
from litemind import CombinedApi

def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"Weather in {city}: Sunny, 72°F"

# Create agent with tools
api = CombinedApi()
toolset = ToolSet()
toolset.add_function_tool(get_weather)

agent = Agent(api=api, toolset=toolset, name="weather_assistant")
agent.append_system_message("You are a weather assistant.")

# Start server
server = Server(host="0.0.0.0", port=18861)
server.expose("weather_agent", agent)

print("Server starting on port 18861...")
server.start(block=True)
```

### Client Side (`client_app.py`)

```python
from litemind.remote.client import Client
from litemind.agent.messages.message import Message

client = Client(host="server.example.com", port=18861)

if client.is_connected():
    agent = client["weather_agent"]

    # Simple text query
    response = agent("What's the weather in Paris?")
    print(response)

    # Multimodal query
    message = Message(role="user")
    message.append_text("Describe this image:")
    message.append_image("path/to/local/image.jpg")
    response = agent(message)
    print(response)

    # Access conversation history
    conversation = agent.conversation
    print(f"Messages: {len(conversation)}")

client.close()
```

## Multimodal Support

The remote package supports full multimodal capabilities:

```python
from litemind.agent.messages.message import Message

# Build multimodal message locally
message = Message(role="user")
message.append_text("Describe what you see and hear:")
message.append_image("photo.jpg")
message.append_audio("recording.wav")

# Send to remote agent - media is serialized via pickle
response = remote_agent(message)
```

## RPyC Configuration

The package uses specific RPyC settings for Litemind compatibility:

```python
{
    "allow_pickle": True,       # Complex objects (Messages, Media)
    "allow_all_attrs": True,    # Attribute access on remote objects
    "allow_public_attrs": True, # Public attribute access
    "sync_request_timeout": 300,# 5 minutes for API calls
}
```

## Architecture

### Server-Side Flow

```
Client Request
       ↓
ThreadedServer (RPyC)
       ↓
ObjectService.exposed_get_object()
       ↓
_ServerWrapper (materializes remote args)
       ↓
Wrapped Object (Agent)
       ↓
Response (sent back to client)
```

### Client-Side Flow

```
client["agent_name"]
       ↓
_ValueWrapper (handles class arguments)
       ↓
RPyC proxy connection
       ↓
Server processing
       ↓
Response
```

## Error Handling

```python
from litemind.remote.client import Client

client = Client(host="localhost", port=99999)

# Graceful handling of connection failures
if not client.is_connected():
    print("Failed to connect to server")

# Missing objects return None
agent = client.get("nonexistent")
if agent is None:
    print("Agent not found on server")

# Always clean up
client.close()
```

## Testing

The package includes integration tests that:
- Test basic text conversations
- Test multimodal (image + audio) conversations
- Test connection failure handling
- Test missing object handling

Run tests with:
```bash
pytest src/litemind/remote/tests/test_remote_agent.py -v
```

## Network Considerations

- **Firewall**: Ensure port 18861 (default) is open
- **Timeout**: Default 5-minute timeout for API calls
- **Serialization**: Complex objects serialized via pickle
- **Security**: Consider using SSH tunnels for production

## Docstring Coverage

| Component | Coverage |
|-----------|----------|
| Server.__init__ | Complete |
| Server.expose | Complete |
| Server.start | Complete |
| Client.__init__ | Complete |
| Client.is_connected | Complete |
| Client.get | Complete |
| ObjectService | Needs improvement |

Overall server-side coverage: 33%. Client-side coverage: 100%.
