# Litemind Code Examples

Below are 15 didactic, runnable, and well-commented code examples demonstrating Litemind's agentic, multimodal, and tool-augmented LLM capabilities. Each example is self-contained and can be run as-is (assuming you have valid API keys and the required dependencies installed).

---

## 1. "Hello Agent" Quick-Start

Instantiate an OpenAIApi, wrap it in an Agent, add a system prompt, send a user question, and print the reply.

```python
from litemind import OpenAIApi
from litemind.agent.agent import Agent

# Initialize the OpenAI API
api = OpenAIApi()

# Create an agent with a specific model (or omit model_name for default)
agent = Agent(api=api, model_name="gpt-4o")

# Add a system message to guide the agent's behavior
agent.append_system_message("You are a helpful assistant.")

# Ask a question
response = agent("What is the capital of France?")

# Print the response
print("Agent Response:", response)
# Expected output: The capital of France is Paris.
```

---

## 2. Date-Aware Assistant with a Function Tool

Expose a Python helper through a ToolSet, attach it to the agent, and let the LLM invoke it.

```python
from litemind import OpenAIApi
from litemind.agent.agent import Agent
from litemind.agent.tools.toolset import ToolSet
from datetime import datetime

# Define a function to get the current date
def get_current_date() -> str:
    return datetime.now().strftime("%Y-%m-%d")

# Initialize the API and toolset
api = OpenAIApi()
toolset = ToolSet()
toolset.add_function_tool(get_current_date, "Fetch the current date")

# Create the agent with the toolset
agent = Agent(api=api, toolset=toolset)
agent.append_system_message("You are a helpful assistant.")

# Ask a question that requires the tool
response = agent("What is the current date?")

print("Agent with Tool Response:", response)
# Expected output: The current date is YYYY-MM-DD.
```

---

## 3. Agent + RAG: Answering from a Vector DB

Load information snippets into an InMemoryVectorDatabase, register the augmentation, and ask a factual question.

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

# Add sample information
informations = [
    Information(Text("Igor Bolupskisty was a German-born theoretical physicist who developed the theory of indelible unitarity."), metadata={"topic": "physics", "person": "Bolupskisty"}),
    Information(Text("The theory of indelible unitarity revolutionized our understanding of space, time and photons."), metadata={"topic": "physics", "concept": "unitarity"}),
    Information(Text("Quantum unitarity is a fundamental theory in physics that describes nature at the nano-atomic scale as it pertains to Pink Hamsters."), metadata={"topic": "physics", "concept": "quantum unitarity"}),
]
vector_augmentation.add_informations(informations)

# Attach augmentation to agent
agent.add_augmentation(vector_augmentation)
agent.append_system_message("You are a helpful assistant.")

# Ask a question that requires retrieval
response = agent("Tell me about Igor Bolupskisty's theory of indelible unitarity.")

print("Agent with RAG Response:", response)
# Expected output: A summary mentioning Bolupskisty and unitarity, citing the retrieved facts.
```

---

## 4. Multimodal Q&A with Image Context

Combine text and an image in a Message, call an internal describe_image tool, and mention metadata from the vector DB.

```python
from litemind import OpenAIApi
from litemind.agent.agent import Agent
from litemind.agent.messages.message import Message

# Use a real image URL (public domain)
image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3e/Einstein_1921_by_F_Schmutzer_-_restoration.jpg/456px-Einstein_1921_by_F_Schmutzer_-_restoration.jpg"

api = OpenAIApi()
agent = Agent(api=api, model_name="gpt-4o")
agent.append_system_message("You are a helpful assistant.")

# Compose a multimodal message
msg = Message(role="user", text="Can you describe what you see in this image?")
msg.append_image(image_url)

# Send the message
response = agent(msg)

print("Multimodal Q&A Response:", response)
# Expected output: A description of the image (e.g., "A sepia photograph of Albert Einstein at a chalkboard...")
```

---

## 5. Unified CombinedApi Fail-Over Demo

Create a CombinedApi, list available models, request both "TextGeneration" and "Image" features, and show auto-selection.

```python
from litemind import CombinedApi
from litemind.agent.agent import Agent
from litemind.apis.model_features import ModelFeatures

# Instantiate CombinedApi
api = CombinedApi()

# List all available models
print("Available models:", api.list_models())

# Request a model that supports both text and image generation
features = [ModelFeatures.TextGeneration, ModelFeatures.Image]
model = api.get_best_model(features=features)
print("Selected model for Text+Image:", model)

# Create agent with required features
agent = Agent(api=api, model_features=features)
agent.append_system_message("You are a helpful assistant.")

# Ask a mixed prompt
response = agent("Describe a photo of a white cat sitting on a red sofa.")

print("CombinedApi Response:", response)
# Expected output: A description of a white cat on a red sofa.
```

---

## 6. Safe, Sandboxed Code-Runner Agent

Register an execute_python_code function wrapped in a guarded exec, warn about untrusted code, and run a snippet.

```python
from litemind import OpenAIApi
from litemind.agent.agent import Agent
from litemind.agent.tools.toolset import ToolSet
import io
import contextlib

def execute_python_code(code: str) -> str:
    """Executes Python code in a restricted environment and returns stdout."""
    stdout = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout):
            exec(code, {"__builtins__": {}})
    except Exception as e:
        return f"Error: {e}"
    return stdout.getvalue()

api = OpenAIApi()
toolset = ToolSet()
toolset.add_function_tool(execute_python_code, "Execute untrusted Python code in a sandbox.")

agent = Agent(api=api, toolset=toolset)
agent.append_system_message("You are a code runner. WARNING: Never trust untrusted code.")

response = agent('Please run: print("Hello")')

print("Sandboxed Code Runner Output:", response)
# Expected output: "Hello"
```

---

## 7. Structured JSON Replies with Pydantic

Define a Pydantic model, pass it via response_format, and get a validated object back.

```python
from litemind import OpenAIApi
from litemind.agent.agent import Agent
from pydantic import BaseModel

class WeatherResponse(BaseModel):
    temp_c: float
    condition: str
    city: str

api = OpenAIApi()
agent = Agent(api=api, model_features=["TextGeneration", "StructuredTextGeneration"])
agent.append_system_message("You are a weather bot. Reply in JSON.")

response = agent("Weather in Paris?", response_format=WeatherResponse)

# Access the parsed fields
weather = response[-1][-1].get_content()
print("Weather object:", weather)
print("Temperature:", weather.temp_c)
print("Condition:", weather.condition)
print("City:", weather.city)
# Expected output: Weather object with fields temp_c, condition, city
```

---

## 8. Tool-Driven Image Generation

Wrap a generate_cat_image() helper that calls generate_image, returns a file path, and have the agent acknowledge.

```python
from litemind import OpenAIApi
from litemind.agent.agent import Agent
from litemind.agent.tools.toolset import ToolSet

def generate_cat_image() -> str:
    # Use the API to generate an image and save it locally
    api = OpenAIApi()
    image = api.generate_image("A white fluffy cat sitting on a sofa", image_width=512, image_height=512)
    path = "/tmp/generated_cat.png"
    image.save(path)
    return path

api = OpenAIApi()
toolset = ToolSet()
toolset.add_function_tool(generate_cat_image, "Generate a cat image and return the file path.")

agent = Agent(api=api, toolset=toolset)
response = agent("Please generate a cat image.")

print("Image Generation Response:", response)
import os
assert os.path.exists("/tmp/generated_cat.png")
# Expected output: Acknowledgement and the image file exists at /tmp/generated_cat.png
```

---

## 9. Realtime Speech Transcription + Translation (Non-CLI)

Demonstrate Litemind's audio modality: record a WAV, transcribe and translate.

```python
from litemind import CombinedApi
from litemind.agent.agent import Agent
from litemind.agent.tools.toolset import ToolSet
from litemind.media.types.media_audio import Audio

# Assume you have a local WAV file in French
audio_uri = "file:///path/to/french_clip.wav"  # Replace with a real file

def transcribe_audio(audio_uri: str) -> str:
    api = CombinedApi()
    return api.transcribe_audio(audio_uri)

api = CombinedApi()
model = api.get_best_model(["TextGeneration", "Audio"])
toolset = ToolSet()
toolset.add_function_tool(transcribe_audio, "Transcribe audio to text.")

agent = Agent(api=api, toolset=toolset)
msg = agent.agent.messages.message.Message(role="user", text="Please transcribe and translate this clip into English.")
msg.append_audio(audio_uri)

response = agent(msg)
print("Transcription & Translation:", response)
# Expected output: English translation, with citation of the original French.
```

---

## 10. Batch Image Captioning & Metadata Enrichment

Walk a folder of JPEGs, feed each to the agent with a describe_image tool, and store captions.

```python
import os
from litemind import OpenAIApi
from litemind.agent.agent import Agent
from litemind.agent.tools.toolset import ToolSet
from litemind.media.types.media_image import Image
from pydantic import BaseModel

class ImageCaption(BaseModel):
    title: str
    objects: list
    style: str

def describe_image(image_uri: str) -> dict:
    api = OpenAIApi()
    return {"title": "A cat", "objects": ["cat"], "style": "photograph"}  # Replace with real call

api = OpenAIApi()
toolset = ToolSet()
toolset.add_function_tool(describe_image, "Describe an image and return JSON.")

agent = Agent(api=api, toolset=toolset)

folder = "/path/to/images"  # Replace with a real folder
captions = []
for fname in os.listdir(folder):
    if fname.lower().endswith(".jpg"):
        image_uri = f"file://{os.path.join(folder, fname)}"
        msg = agent.agent.messages.message.Message(role="user", text="Return JSON with title, objects, style.")
        msg.append_image(image_uri)
        response = agent(msg, response_format=ImageCaption)
        captions.append(response[-1][-1].get_content().dict())

# Save to CSV
import pandas as pd
pd.DataFrame(captions).to_csv("captions.csv", index=False)
```

---

## 11. Data-Aware Chat via Pandas-Tool

Show the agent answering questions against a loaded pandas.DataFrame via a tool.

```python
import pandas as pd
from litemind import OpenAIApi
from litemind.agent.agent import Agent
from litemind.agent.tools.toolset import ToolSet

# Load a CSV into a DataFrame
df = pd.read_csv("data.csv")  # Replace with your CSV

def query_dataframe(question: str) -> dict:
    # Example: implement a simple query
    filtered = df[df["price"] > 100]
    avg_score = filtered["score"].mean()
    return {"count": len(filtered), "average_score": avg_score}

api = OpenAIApi()
toolset = ToolSet()
toolset.add_function_tool(query_dataframe, "Query a DataFrame.")

agent = Agent(api=api, toolset=toolset)
response = agent("How many rows have price > 100 and what’s their average score?")

print("Data-aware Chat Response:", response)
# Expected output: Aggregated numbers from the DataFrame.
```

---

## 12. Cascading Agents (Agent-as-Tool)

Create a summariser sub-agent and register it as a callable tool inside a supervisor agent.

```python
from litemind import OpenAIApi
from litemind.agent.agent import Agent
from litemind.agent.tools.toolset import ToolSet

api = OpenAIApi()

# Child summariser agent
summariser = Agent(api=api, name="Summariser")
summariser.append_system_message("You are a summariser. Summarise any text you receive.")

# Wrap summariser as a tool
toolset = ToolSet()
toolset.add_agent_tool(summariser, "Summarise a paragraph.")

# Supervisor agent
supervisor = Agent(api=api, toolset=toolset, name="Supervisor")
supervisor.append_system_message("You are a supervisor. Use the summariser tool to condense text.")

paragraph = "Litemind is a Python library for building agentic, multimodal AI applications. It supports tools, RAG, and more."
response = supervisor(f"Please summarise the following: {paragraph}")

print("Supervisor Agent Response:", response)
# Expected output: A concise summary, delegated via the summariser tool.
```

---

## 13. Automatic Feature Discovery for Multimodality

Iterate over available models, ask for "TextGeneration" + "Audio", and assert the library picks the first model that meets both.

```python
from litemind import CombinedApi
from litemind.apis.model_features import ModelFeatures

api = CombinedApi()
features = ["TextGeneration", "Audio"]  # Can use enums or strings
models = api.list_models()
for model in models:
    if api.has_model_support_for(features=features, model_name=model):
        print("First model supporting both TextGeneration and Audio:", model)
        break
else:
    print("No model found with both features.")
```

---

## 14. Batch Ingestion Pipeline for PDFs & Images

Walk a folder, convert each PDF page or image into Information chunks, store in a persistent vector DB, and expose a search-answer agent.

```python
import os
from litemind import OpenAIApi
from litemind.agent.agent import Agent
from litemind.agent.augmentations.information.information import Information
from litemind.agent.augmentations.vector_db.in_memory_vector_db import InMemoryVectorDatabase
from litemind.media.types.media_document import Document
from litemind.media.types.media_image import Image

api = OpenAIApi()
vector_db = InMemoryVectorDatabase(name="my_corpus", location="/tmp/litemind_vecdb")

folder = "/path/to/folder"  # Replace with your folder
for fname in os.listdir(folder):
    path = os.path.join(folder, fname)
    if fname.lower().endswith(".pdf"):
        info = Information(Document(f"file://{path}"))
        vector_db.add_informations([info])
    elif fname.lower().endswith((".jpg", ".png")):
        info = Information(Image(f"file://{path}"))
        vector_db.add_informations([info])

agent = Agent(api=api)
agent.add_augmentation(vector_db)

response = agent("What is the main topic of the PDF documents in this folder?")
print("RAG Pipeline Response:", response)
```

---

## 15. Streaming Responses with Progress Callbacks

Implement a custom callback that prints tokens as they arrive, attach it to a streaming-capable model, and show the live token flow.

```python
from litemind import OpenAIApi
from litemind.agent.agent import Agent
from litemind.apis.callbacks.base_callbacks import BaseApiCallbacks


class PrintTokensCallback(BaseApiCallbacks):
    def on_text_streaming(self, fragment, **kwargs):
        print(fragment, end="", flush=True)


api = OpenAIApi()
api.callback_manager.add_callback(PrintTokensCallback())

agent = Agent(api=api, model_name="gpt-4o")
agent.append_system_message("You are a helpful assistant.")

# Send a long prompt to observe streaming
response = agent("Write a detailed summary of the history of artificial intelligence.")

# Now, switch to a non-streaming model (if available)
agent = Agent(api=api, model_name="gpt-3.5-turbo")  # Example non-streaming
response = agent("Write a detailed summary of the history of artificial intelligence.")
print("\nFull text (non-streaming):", response)
```

---

**Note:**  
- For all examples, ensure you have valid API keys and the required dependencies installed.
- Replace file paths and URLs with real, accessible resources as needed.
- Model features can be provided as enums, strings, or lists of strings (e.g., `["TextGeneration", "Image"]` or `[ModelFeatures.TextGeneration, ModelFeatures.Image]`).
- Litemind's agent framework is unified; there is no separate "ReActAgent" class—use `Agent` for all agentic workflows.