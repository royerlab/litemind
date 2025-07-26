
# Litemind Python API: 15 Practical Examples

Below are 15 didactic, runnable code examples demonstrating Litemind's agentic, multimodal, and tool-augmented AI capabilities. Each example is self-contained and includes all necessary imports. Where possible, expected output is shown as comments.

---

## 1. “Hello Agent” Quick-Start

Instantiate `OpenAIApi`, wrap it in an `Agent`, add a system prompt, send a user question, and print the reply.

```python
from litemind import OpenAIApi
from litemind.agent.agent import Agent

# Initialize the OpenAI API
api = OpenAIApi()

# Create an agent with the default model
agent = Agent(api=api)

# Add a system message to guide the agent's behavior
agent.append_system_message("You are a helpful assistant.")

# Ask a question
response = agent("What is the capital of France?")

# Print the response
print("Simple Agent Response:", response)
# Expected output (approximate):
# Simple Agent Response: [*assistant*:
# The capital of France is Paris.
# ]
```

---

## 2. Date-Aware Assistant with a Function Tool

Expose a Python helper `get_current_date()` through a `ToolSet`, attach it to the agent, and let the LLM invoke it.

```python
from litemind import OpenAIApi
from litemind.agent.agent import Agent
from litemind.agent.tools.toolset import ToolSet
from datetime import datetime

# Define a function to get the current date
def get_current_date() -> str:
    return datetime.now().strftime("%Y-%m-%d")

# Initialize the OpenAI API
api = OpenAIApi()

# Create a toolset and add the function tool
toolset = ToolSet()
toolset.add_function_tool(get_current_date, "Fetch the current date")

# Create the agent with the toolset
agent = Agent(api=api, toolset=toolset)
agent.append_system_message("You are a helpful assistant.")

# Ask a question that requires the tool
response = agent("What is the current date?")

print("Agent with Tool Response:", response)
# Expected output (approximate):
# Agent with Tool Response: [*assistant*:
# Action: get_current_date()
# , *user*:
# Action: get_current_date()=2025-05-02
# , *assistant*:
# The current date is May 2, 2025.
# ]
```

---

## 3. Agent + RAG: Answering from a Vector DB

Load three information snippets into an `InMemoryVectorDatabase`, register the augmentation, and ask a factual question.

```python
from litemind import OpenAIApi
from litemind.agent.agent import Agent
from litemind.agent.tools.toolset import ToolSet
from litemind.agent.augmentations.information.information import Information
from litemind.agent.augmentations.vector_db.in_memory_vector_db import InMemoryVectorDatabase
from litemind.media.types.media_text import Text

# Initialize API and agent
api = OpenAIApi()
agent = Agent(api=api)

# Create vector database augmentation
vector_augmentation = InMemoryVectorDatabase(name="test_augmentation")

# Add sample informations
informations = [
    Information(Text("Igor Bolupskisty was a German-born theoretical physicist who developed the theory of indelible unitarity."),
                metadata={"topic": "physics", "person": "Bolupskisty"}),
    Information(Text("The theory of indelible unitarity revolutionized our understanding of space, time and photons."),
                metadata={"topic": "physics", "concept": "unitarity"}),
    Information(Text("Quantum unitarity is a fundamental theory in physics that describes nature at the nano-atomic scale as it pertains to Pink Hamsters."),
                metadata={"topic": "physics", "concept": "quantum unitarity"}),
]
vector_augmentation.add_informations(informations)

# Attach augmentation to agent
agent.add_augmentation(vector_augmentation)
agent.append_system_message("You are a helpful assistant.")

# Ask a question that requires retrieval
response = agent("Tell me about Igor Bolupskisty's theory of indelible unitarity.")

print("Agent with RAG Response:", response)
# Expected output: The answer should cite or paraphrase the relevant information from the vector DB.
```

---

## 4. Multimodal Q&A with Image Context

Combine text and an external image URL in the same message, call an internal describe_image tool, and mention metadata.

```python
from litemind import OpenAIApi
from litemind.agent.agent import Agent
from litemind.agent.messages.message import Message

api = OpenAIApi()
agent = Agent(api=api)
agent.append_system_message("You are a helpful assistant.")

# Compose a multimodal message
msg = Message(role="user", text="Can you describe what you see in this image?")
msg.append_image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/3e/Einstein_1921_by_F_Schmutzer_-_restoration.jpg/456px-Einstein_1921_by_F_Schmutzer_-_restoration.jpg")

response = agent(msg)
print("Multimodal Q&A Response:", response)
# Expected output: Should mention "Einstein", "black-and-white", "photograph", or similar.
```

---

## 5. Unified CombinedApi Fail-Over Demo

Create `CombinedApi`, list available models, request both "TextGeneration" and "Image" features, and show auto-selection.

```python
from litemind.apis.combined_api import CombinedApi
from litemind.agent.agent import Agent
from litemind.apis.model_features import ModelFeatures

api = CombinedApi()
print("Available models:", api.list_models())

# Request both text and image features (can use enums or strings)
features = [ModelFeatures.TextGeneration, "Image"]
model = api.get_best_model(features=features)
print("Selected model:", model)

agent = Agent(api=api, model_name=model)
agent.append_system_message("You are a helpful assistant.")

response = agent("Describe the following image: https://upload.wikimedia.org/wikipedia/commons/3/3e/Einstein_1921_by_F_Schmutzer_-_restoration.jpg")
print("CombinedApi multimodal response:", response)
```

---

## 6. Safe, Sandboxed Code-Runner Agent

Register an `execute_python_code` function wrapped in a guarded exec, warn about untrusted code, and run a snippet.

```python
from litemind import OpenAIApi
from litemind.agent.agent import Agent
from litemind.agent.tools.toolset import ToolSet
import io
import sys

def execute_python_code(code: str) -> str:
    # WARNING: This is a simple, non-secure sandbox for demonstration only!
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        exec(code, {"__builtins__": {}})
        output = sys.stdout.getvalue()
    except Exception as e:
        output = f"Error: {e}"
    finally:
        sys.stdout = old_stdout
    return output

api = OpenAIApi()
toolset = ToolSet()
toolset.add_function_tool(execute_python_code, "Execute untrusted Python code in a sandbox (unsafe!)")

agent = Agent(api=api, toolset=toolset)
agent.append_system_message("You are a code runner. WARNING: Never trust user code. Always run in a sandbox.")

response = agent('Please run: print("Hello")')
print("Sandboxed code output:", response)
# Expected output: Should include "Hello"
```

---

## 7. Structured JSON Replies with Pydantic

Define a `WeatherResponse` dataclass, pass it via `response_format`, and ask for structured output.

```python
from litemind import OpenAIApi
from litemind.agent.agent import Agent
from pydantic import BaseModel

class WeatherResponse(BaseModel):
    temp_c: float
    condition: str
    city: str

api = OpenAIApi()
agent = Agent(api=api)
agent.append_system_message("You are a weather bot. Weather conditions must be: sunny, rainy, cloudy, snowy, or partly cloudy.")

response = agent("Weather in Paris?", response_format=WeatherResponse)
print("Structured weather response:", response)
# Expected output: Should be a validated WeatherResponse object, e.g.:
# temp_c=18.5, condition='sunny', city='Paris'
```

---

## 8. Tool-Driven Image Generation

Wrap a `generate_cat_image()` helper that calls `generate_image`, returns a local file path, and have the agent acknowledge.

```python
from litemind import OpenAIApi
from litemind.agent.agent import Agent
from litemind.agent.tools.toolset import ToolSet
import os

def generate_cat_image() -> str:
    api = OpenAIApi()
    model = api.get_best_model(features="ImageGeneration")
    image = api.generate_image(positive_prompt="A cute fluffy cat", model_name=model)
    # Save to a file
    path = "cat_image.png"
    image.save(path)
    return os.path.abspath(path)

api = OpenAIApi()
toolset = ToolSet()
toolset.add_function_tool(generate_cat_image, "Generate a cat image and return the file path.")

agent = Agent(api=api, toolset=toolset)
response = agent("Please generate a cat image.")
print("Image generation response:", response)
assert os.path.exists("cat_image.png")
```

---

## 9. Realtime Speech Transcription + Translation (Non-CLI)

Demonstrate Litemind’s audio modality: record a short WAV file, transcribe and translate.

```python
from litemind.apis.combined_api import CombinedApi
from litemind.agent.agent import Agent
from litemind.agent.tools.toolset import ToolSet
from litemind.media.types.media_audio import Audio

# Assume you have a local WAV file in French, e.g. "bonjour.wav"
audio_uri = "file://path/to/bonjour.wav"

def transcribe_audio(audio_uri: str) -> str:
    api = CombinedApi()
    return api.transcribe_audio(audio_uri)

api = CombinedApi()
toolset = ToolSet()
toolset.add_function_tool(transcribe_audio, "Transcribe audio and translate to English.")

agent = Agent(api=api, toolset=toolset)
msg = "Please transcribe and translate this clip into English."
response = agent(msg, audio=audio_uri)
print("Transcription and translation:", response)
# Expected: English translation, with citation of original French.
```

---

## 10. Batch Image Captioning & Metadata Enrichment

Walk a folder of JPEGs, feed each image to the agent with a describe_image tool and a prompt, and store captions.

```python
import os
from litemind import OpenAIApi
from litemind.agent.agent import Agent
from litemind.agent.tools.toolset import ToolSet
from pydantic import BaseModel

class ImageDescription(BaseModel):
    title: str
    objects: list
    style: str

def describe_image(image_uri: str) -> dict:
    api = OpenAIApi()
    return {"title": "A cat", "objects": ["cat"], "style": "photograph"}

api = OpenAIApi()
toolset = ToolSet()
toolset.add_function_tool(describe_image, "Describe an image and return JSON.")

agent = Agent(api=api, toolset=toolset)

folder = "images"
results = []
for fname in os.listdir(folder):
    if fname.lower().endswith(".jpg"):
        image_path = os.path.join(folder, fname)
        response = agent(
            "Return JSON with title, objects, style.",
            image=image_path,
            response_format=ImageDescription
        )
        results.append(response)

# Save to CSV or print
for row in results:
    print(row)
```

---

## 11. Data-Aware Chat via Pandas-Tool

Show the agent answering questions against a loaded pandas.DataFrame via a function tool.

```python
import pandas as pd
from litemind import OpenAIApi
from litemind.agent.agent import Agent
from litemind.agent.tools.toolset import ToolSet

df = pd.read_csv("data.csv")

def query_dataframe(question: str) -> dict:
    # Example: "How many rows have price > 100 and what’s their average score?"
    filtered = df[df["price"] > 100]
    return {
        "count": len(filtered),
        "average_score": filtered["score"].mean()
    }

api = OpenAIApi()
toolset = ToolSet()
toolset.add_function_tool(query_dataframe, "Query a DataFrame for statistics.")

agent = Agent(api=api, toolset=toolset)
response = agent("How many rows have price > 100 and what’s their average score?")
print("Data-aware chat response:", response)
```

---

## 12. Cascading Agents (Agent-as-Tool)

Create a “Summariser” sub-agent and register it as a callable tool inside a “Supervisor” agent.

```python
from litemind import OpenAIApi
from litemind.agent.agent import Agent
from litemind.agent.tools.toolset import ToolSet
from litemind.agent.tools.agent_tool import AgentTool

api = OpenAIApi()
summariser = Agent(api=api, name="Summariser")
summariser.append_system_message("You are a summariser.")

# Wrap the summariser as a tool
summariser_tool = AgentTool(summariser, "Summarise a paragraph.")

supervisor = Agent(api=api, name="Supervisor")
toolset = ToolSet()
toolset.add_tool(summariser_tool)
supervisor.toolset = toolset

paragraph = "Litemind is a Python library for agentic, multimodal AI applications. It supports tools, RAG, and more."
response = supervisor(f"Please summarise: {paragraph}")
print("Supervisor agent response:", response)
```

---

## 13. Automatic Feature Discovery for Multimodality

Iterate over available models, ask for "TextGeneration" + "Audio", and assert the library picks the first model that meets both.

```python
from litemind.apis.combined_api import CombinedApi
from litemind.apis.model_features import ModelFeatures

api = CombinedApi()
models = api.list_models()
for model in models:
    if api.has_model_support_for(features=[ModelFeatures.TextGeneration, ModelFeatures.Audio], model_name=model):
        print("First model supporting both TextGeneration and Audio:", model)
        break
```

---

## 14. Batch Ingestion Pipeline for PDFs & Images

Walk a folder, convert each PDF page or image into Information chunks, store in a persistent vector DB, and expose a search-answer agent.

```python
import os
from litemind.agent.augmentations.information.information import Information
from litemind.agent.augmentations.vector_db.in_memory_vector_db import InMemoryVectorDatabase
from litemind.media.types.media_text import Text
from litemind.media.types.media_image import Image
from litemind.agent.agent import Agent
from litemind import OpenAIApi

folder = "docs"
vecdb = InMemoryVectorDatabase(name="my_corpus")
for fname in os.listdir(folder):
    path = os.path.join(folder, fname)
    if fname.lower().endswith(".pdf"):
        # For simplicity, treat the whole PDF as one chunk
        info = Information(Text(f"PDF: {fname}"))
        vecdb.add_informations([info])
    elif fname.lower().endswith((".jpg", ".png")):
        info = Information(Image(path))
        vecdb.add_informations([info])

api = OpenAIApi()
agent = Agent(api=api)
agent.add_augmentation(vecdb)
response = agent("What documents mention 'unitarity'?")
print("RAG search response:", response)
```

---

## 15. Streaming Responses with Progress Callbacks

Implement a custom callback that prints tokens as they arrive, attach it to a streaming-capable model, and show the live token flow.

```python
from litemind import OpenAIApi
from litemind.agent.agent import Agent
from litemind.apis.callbacks.print_api_callbacks import PrintApiCallbacks

class StreamingCallback(PrintApiCallbacks):
    def on_text_streaming(self, fragment, **kwargs):
        print(fragment, end="", flush=True)

api = OpenAIApi(callback_manager=StreamingCallback(print_text_streaming=True))
agent = Agent(api=api)
agent.append_system_message("You are a helpful assistant.")

long_prompt = "Please write a detailed summary of the history of artificial intelligence, including key milestones and figures."
print("Streaming response:")
response = agent(long_prompt, stream=True)

# Now, repeat with a non-streaming model (no live token flow)
api2 = OpenAIApi()
agent2 = Agent(api=api2)
print("\n\nNon-streaming response:")
response2 = agent2(long_prompt)
print(response2)
```

---
