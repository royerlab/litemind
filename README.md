
# LiteMind

[![PyPI version](https://badge.fury.io/py/litemind.svg)](https://badge.fury.io/py/litemind)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD--3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![PyPI Downloads](https://static.pepy.tech/badge/litemind)](https://pepy.tech/project/litemind)
[![GitHub stars](https://img.shields.io/github/stars/royerlab/litemind.svg?style=social&label=Star&maxAge=2592000)](https://github.com/royerlab/litemind/stargazers)

**LiteMind is a powerful, unified, and developer-friendly Python framework for building sophisticated, multimodal, and agentic AI applications.**

It provides a clean, consistent API wrapper over multiple leading LLM providers (OpenAI, Anthropic, Google Gemini, Ollama) and a high-level agentic API for creating autonomous agents that can reason, use tools, and access external knowledge.

Our philosophy is to simplify the complex landscape of LLM APIs, enabling developers to focus on building innovative applications rather than wrestling with provider-specific implementations. With LiteMind, you can seamlessly switch between models, leverage advanced features like tool use and RAG, and handle a wide variety of media types (text, images, audio, video, documents) with minimal effort. Whether you're building a simple chatbot or a complex, multi-agent system, LiteMind provides the tools and abstractions to get you there faster.

## Features

-   **Unified API Wrapper**: A single, intuitive interface for interacting with major LLM providers including OpenAI, Anthropic, Google Gemini, and local Ollama instances.
-   **Intelligent Model Selection**: Automatically select the best model for your task based on required features like text generation, image understanding, tool use, or embedding creation.
-   **Powerful Agentic Framework**: A high-level `Agent` class that encapsulates reasoning loops, conversation history, tool execution, and RAG, inspired by the ReAct framework.
-   **Seamless Tool Integration**: Easily equip agents with custom tools by simply providing Python functions. The library handles schema generation, tool calling, and response integration.
-   **Built-in RAG/Augmentations**: Enhance your agents with external knowledge using a flexible augmentation system. LiteMind includes built-in support for vector databases (`InMemoryVectorDatabase`, `Qdrant`) for powerful Retrieval-Augmented Generation.
-   **Rich Multimodality**: Natively handle and process text, images, audio, video, PDFs, and more. The framework includes a robust `Media` abstraction layer and automatic media conversion capabilities.
-   **Structured Outputs**: Reliably get JSON outputs from models by simply providing a Pydantic model as the desired response format.
-   **Command-Line Tools**: A suite of CLI tools to help you manage your AI projects, including repository export (`export`) and model feature validation (`validate`).

## Installation

You can install LiteMind directly from PyPI:

```bash
pip install litemind
```

To include all optional dependencies for features like RAG, document processing, and multimodal capabilities, use the following command:

```bash
pip install "litemind[dev,rag,whisper,documents,tables,videos,audio,remote,tasks]"
```

## Basic Usage

This section showcases the high-level **agentic API**. These examples are designed to be striking, illustrative, and easy to run.

### Basic Agent Usage

Here's how to create a simple conversational agent.

```python
from litemind import Agent, OpenAIApi

# It's often easiest to use a specific API, but you can also use CombinedApi()
# to automatically select from available providers.
try:
    api = OpenAIApi()

    # Create an agent, LiteMind will pick the best default model
    agent = Agent(api=api, name="Helpful Assistant")

    # Set the agent's persona with a system message
    agent.append_system_message("You are an omniscient and helpful AI assistant named Ohmm.")

    # Call the agent like a function
    response = agent("Who are you and what can you do?")

    # The response is a list of Message objects
    print(response[0].to_plain_text())
    # Expected output might be:
    # I am Ohmm, an omniscient and helpful AI assistant. I can answer your questions,
    # assist with tasks, and provide information on a wide range of topics. How can I help you today?

except Exception as e:
    print(f"An error occurred: {e}")
    print("Please ensure your OpenAI API key is set in the OPENAI_API_KEY environment variable.")

```

### Agent with Tools

Equip your agent with tools to interact with the world. LiteMind makes it trivial to add any Python function as a tool.

```python
from litemind import Agent, OpenAIApi, ToolSet
from datetime import datetime

# Define a simple Python function to be used as a tool
def get_current_date() -> str:
    """Returns the current date as a string in YYYY-MM-DD format."""
    return datetime.now().strftime("%Y-%m-%d")

try:
    api = OpenAIApi()

    # Create a ToolSet and add our function to it
    toolset = ToolSet()
    toolset.add_function_tool(get_current_date, "A tool to get the current date.")

    # Create an agent and provide it with the toolset
    # LiteMind automatically selects a model that supports tool use.
    agent = Agent(api=api, toolset=toolset, name="Date Agent")

    # Ask a question that requires the tool
    response = agent("What is today's date?")

    print(response[0].to_plain_text())
    # Expected output might be:
    # Today's date is 2023-10-27.

except Exception as e:
    print(f"An error occurred: {e}")
    print("Please ensure your OpenAI API key is set for a model that supports tools (e.g., GPT-4o).")

```

### Agent with Tools and Augmentation (RAG)

Enhance your agent's knowledge with a vector database for Retrieval-Augmented Generation (RAG).

```python
from litemind import Agent, OpenAIApi, ToolSet, InMemoryVectorDatabase, Information
from litemind.media.types.media_text import Text
from datetime import datetime

# Define a tool
def get_day_of_week(date_str: str) -> str:
    """Returns the day of the week for a given date string in YYYY-MM-DD format."""
    return datetime.strptime(date_str, "%Y-%m-%d").strftime('%A')

try:
    api = OpenAIApi()

    # Create a toolset with our function
    toolset = ToolSet()
    toolset.add_function_tool(get_day_of_week)

    # Create an in-memory vector database for our knowledge base
    vector_db = InMemoryVectorDatabase(name="project_knowledge_base")

    # Create Information objects to populate the database
    project_data = [
        Information(Text("Project Alpha's deadline is 2024-08-15.")),
        Information(Text("Project Beta is scheduled to start on 2024-09-01.")),
        Information(Text("The project manager for Alpha is Alice.")),
    ]
    vector_db.add_informations(project_data)

    # Create an agent with the toolset
    agent = Agent(api=api, toolset=toolset, name="Project Manager Assistant")

    # Add the vector database as an augmentation
    agent.add_augmentation(vector_db)

    # Ask a question that requires both RAG and a tool
    response = agent("On which day of the week is Project Alpha's deadline?")

    print(response[0].to_plain_text())
    # Expected output might be:
    # Project Alpha's deadline is on a Thursday.

except Exception as e:
    print(f"An error occurred: {e}")
    print("Please ensure your OpenAI API key is set for a model that supports tools.")

```

### Agent with Multimodal Inputs, Tools, and Augmentation

This example combines multimodality (image input), tool use, and RAG in a single, powerful agent.

```python
from litemind import (
    Agent, OpenAIApi, ToolSet, InMemoryVectorDatabase,
    Information, Message, ModelFeatures
)
from litemind.media.types.media_text import Text
from litemind.media.types.media_image import Image

# A simple tool for this example
def get_location_details(location_name: str) -> str:
    """Provides fictional details about a famous location."""
    if "eiffel" in location_name.lower():
        return "The Eiffel Tower is a wrought-iron lattice tower in Paris, known for its evening light shows."
    return "I don't have details for that location."

try:
    api = OpenAIApi()

    # Create a toolset
    toolset = ToolSet()
    toolset.add_function_tool(get_location_details)

    # Create a vector database with some contextual information
    vector_db = InMemoryVectorDatabase(name="art_styles_db")
    art_data = [
        Information(Text("Impressionism is a 19th-century art movement characterized by relatively small, thin, yet visible brush strokes.")),
        Information(Text("The Eiffel Tower was initially criticized by some of France's leading artists and intellectuals for its design.")),
    ]
    vector_db.add_informations(art_data)

    # Create an agent, explicitly requesting a model that supports images and tools
    agent = Agent(
        api=api,
        toolset=toolset,
        model_features=[ModelFeatures.Image, ModelFeatures.Tools],
        name="Art & Travel Guide"
    )

    # Add the RAG augmentation
    agent.add_augmentation(vector_db)

    # Create a multimodal message with text and an image
    multimodal_message = Message(role="user")
    multimodal_message.append_text(
        "Based on the art style in this image, tell me something interesting about this landmark."
    )
    # Using a real, accessible image URL
    eiffel_tower_impressionist_url = "https://upload.wikimedia.org/wikipedia/commons/a/a3/Georges_Seurat_-_La_Tour_Eiffel.jpg"
    multimodal_message.append_image(eiffel_tower_impressionist_url)

    # Call the agent with the multimodal message
    response = agent(multimodal_message)

    print(response[0].to_plain_text())
    # Expected output might be:
    # The art style of the image is reminiscent of Impressionism or Pointillism.
    # Interestingly, while now a beloved icon, the Eiffel Tower was initially criticized by many artists.
    # It is a wrought-iron lattice tower in Paris, famous for its spectacular evening light shows.

except Exception as e:
    print(f"An error occurred: {e}")
    print("Please ensure your OpenAI API key is set for a model that supports images and tools (e.g., GPT-4o).")

```

## Concepts

LiteMind is designed with a clear, layered architecture to separate concerns and provide both ease of use and flexibility.

### API Wrapper Layer vs. Agentic API

The library has two primary layers of interaction:

1.  **API Wrapper Layer (`litemind.apis`)**: This is the low-level foundation of LiteMind. It provides a unified interface (`BaseApi`, `CombinedApi`) over various LLM providers. You use this layer when you want direct, granular control over API calls, such as generating text, creating embeddings, or generating images, without the overhead of an agentic loop. It's perfect for specific, one-off tasks.

2.  **Agentic API (`litemind.agent`)**: This is the high-level, "smart" layer. The `Agent` class uses the API Wrapper Layer to perform its tasks but adds a reasoning loop on top. This allows the agent to manage conversation history, decide when to use tools, incorporate external knowledge from augmentations (RAG), and handle complex, multi-step interactions. You should use this layer when you need an AI that can reason, plan, and act autonomously to achieve a goal.

### Main Classes

-   **`CombinedApi` / `OpenAIApi`, etc.**: These are implementations of the API Wrapper Layer. `CombinedApi` can manage multiple providers at once, while specific classes like `OpenAIApi` target a single provider.
-   **`Agent`**: The core of the agentic API. It orchestrates the entire process: receiving prompts, augmenting them with RAG, calling the LLM, executing tools, and managing the conversation.
-   **`Message`**: A container for all content sent to or received from an LLM. A `Message` can contain multiple `MessageBlock`s of different media types.
-   **`Media` Classes (e.g., `Text`, `Image`, `Audio`)**: These classes represent different types of data. They provide a consistent way to handle multimodal inputs and outputs, abstracting away the details of file paths, URLs, and data encoding.
-   **`ToolSet`**: A collection of tools (e.g., Python functions) that an `Agent` can use. LiteMind automatically generates the necessary schemas for the LLM to understand how and when to use these tools.
-   **`AugmentationSet`**: A collection of knowledge sources, typically vector databases, that an `Agent` can query to retrieve relevant information (RAG) before answering a prompt.

## Multi-modality

LiteMind is built from the ground up to support multimodal interactions, allowing you to combine text, images, audio, and more in your prompts.

### Model Features

Not all models are created equal. Some can understand images, some can generate audio, and others are specialized for tool use. To manage this, LiteMind uses an enumeration called **`ModelFeatures`**.

When you instantiate an `Agent` or call the API layer, you can specify a list of required features. LiteMind will then query its model registry to find the best available model that satisfies all your requirements.

You can provide features as `ModelFeatures` enums, or simply as strings:

```python
from litemind import CombinedApi, ModelFeatures

api = CombinedApi()

# These are all equivalent ways to request a model that can understand
# images and use tools.
best_model = api.get_best_model(features=[ModelFeatures.Image, ModelFeatures.Tools])
best_model_str = api.get_best_model(features=['Image', 'Tools'])
best_model_str_lower = api.get_best_model(features=['image', 'tools'])

print(f"Found model: {best_model}")
```

### Media Classes

To handle diverse data types, LiteMind uses a system of `Media` classes (e.g., `Image`, `Audio`, `Video`, `Document`). When you create a `Message`, you append instances of these classes to it. This provides a structured way to represent multimodal content.

```python
from litemind import Message
from litemind.media.types import Image, Audio

# Create a message with text, an image, and an audio file
message = Message(role="user")
message.append_text("Describe this image and transcribe the audio.")
message.append_image("https://upload.wikimedia.org/wikipedia/commons/a/a3/Georges_Seurat_-_La_Tour_Eiffel.jpg")
message.append_audio("https://www.learningcontainer.com/wp-content/uploads/2020/02/Kalimba.mp3")
```

The API layer automatically handles the conversion of these `Media` objects into the format required by the specific LLM provider, so you don't have to worry about the underlying details.

## More Examples

Here are some more advanced use cases.

### Using the CombinedAPI

The `CombinedApi` intelligently manages multiple API providers, allowing you to find the best model for a given task without committing to a single provider.

```python
from litemind import CombinedApi, ModelFeatures

try:
    # This will automatically discover available APIs (OpenAI, Anthropic, etc.)
    # based on your environment variables.
    api = CombinedApi()

    # Find the best model across all available providers for image generation
    image_gen_model = api.get_best_model(features=ModelFeatures.ImageGeneration)

    if image_gen_model:
        print(f"Best model for image generation: {image_gen_model}")
        # Generate an image using the selected model
        image = api.generate_image(
            positive_prompt="A photorealistic image of a red panda coding on a laptop",
            model_name=image_gen_model
        )
        image.save("red_panda_coder.png")
        print("Image saved to red_panda_coder.png")
    else:
        print("No model supporting image generation was found.")

except ValueError as e:
    print(f"An error occurred: {e}")
    print("Please ensure at least one API key (e.g., OPENAI_API_KEY) is set.")
```

### Agent that Executes Python Code

You can create a powerful agent that can write and execute Python code by giving it a tool that wraps `exec`.

```python
import io
import contextlib
from litemind import Agent, OpenAIApi, ToolSet

def execute_python_code(code: str) -> str:
    """
    Executes a string of Python code and returns its stdout output.
    Only use this for simple, safe operations.
    """
    try:
        with io.StringIO() as buf, contextlib.redirect_stdout(buf):
            exec(code)
            output = buf.getvalue()
        return f"Execution successful. Output:\n{output}"
    except Exception as e:
        return f"Error during execution: {e}"

try:
    api = OpenAIApi()
    toolset = ToolSet()
    toolset.add_function_tool(execute_python_code)

    agent = Agent(api=api, toolset=toolset, name="Python Coder")
    agent.append_system_message("You are a helpful AI assistant that can write and execute Python code.")

    response = agent("Please write and run a Python script to calculate 7 factorial.")

    print(response[-1].to_plain_text())
    # Expected output might contain:
    # Execution successful. Output:
    # 5040

except Exception as e:
    print(f"An error occurred: {e}")

```

### Using the Wrapper API for Structured Outputs

For tasks that don't require an agent, you can use the wrapper API directly to get structured JSON output matching a Pydantic model.

```python
from litemind import OpenAIApi
from litemind.agent.messages.message import Message
from pydantic import BaseModel, Field
from typing import List

# Define the desired data structure using Pydantic
class Recipe(BaseModel):
    recipe_name: str = Field(description="The name of the recipe.")
    ingredients: List[str] = Field(description="A list of ingredients.")
    instructions: List[str] = Field(description="Step-by-step instructions.")

try:
    api = OpenAIApi()
    model = api.get_best_model(features="StructuredTextGeneration")

    if model:
        messages = [
            Message(role="user", text="Provide a simple recipe for pancakes.")
        ]

        # Call generate_text with the Pydantic model as the response_format
        response = api.generate_text(
            messages=messages,
            model_name=model,
            response_format=Recipe
        )

        # The content of the last message block will be a Pydantic object
        pancake_recipe = response[0][-1].get_content()

        print(f"Recipe Name: {pancake_recipe.recipe_name}")
        print("Ingredients:")
        for item in pancake_recipe.ingredients:
            print(f"- {item}")

    else:
        print("No model supporting structured output was found.")

except Exception as e:
    print(f"An error occurred: {e}")

```

### Agent with an Image Generation Tool

An agent can have tools that call back to the LiteMind API, creating powerful, self-sufficient systems.

```python
from litemind import Agent, OpenAIApi, ToolSet

# Initialize the API once
api = OpenAIApi()

def generate_image_of_animal(animal_description: str):
    """Generates an image based on a description of an animal."""
    try:
        image = api.generate_image(
            positive_prompt=f"A photorealistic image of: {animal_description}",
            model_name=api.get_best_model(features="ImageGeneration")
        )
        image.save("generated_animal.png")
        return "Image successfully generated and saved as generated_animal.png."
    except Exception as e:
        return f"Failed to generate image: {e}"

try:
    toolset = ToolSet()
    toolset.add_function_tool(generate_image_of_animal)

    agent = Agent(api=api, toolset=toolset, name="Image Artist")
    agent.append_system_message("You are a creative assistant that can generate images of animals.")

    response = agent("Please create an image of a majestic lion with a starry nebula as its mane.")

    print(response[-1].to_plain_text())
    # Expected output:
    # Image successfully generated and saved as generated_animal.png.

except Exception as e:
    print(f"An error occurred: {e}")

```

## Command Line Tools

LiteMind comes with a set of command-line tools to streamline your workflow.

```bash
litemind --help
```

### `litemind export`

Exports your entire repository (or a subfolder) into a single text file, which is useful for providing context to LLMs.

**Usage:**

```bash
# Export the current repository to exported.txt
litemind export --folder-path . --output-file exported.txt
```

### `litemind validate`

Validates the features listed in LiteMind's model registry against live API calls to ensure they are accurate.

**Usage:**

```bash
# Validate all models from openai and gemini providers
litemind validate --api openai gemini

# Validate only specific models
litemind validate --api openai --models gpt-4o gpt-4-turbo
```

### `litemind discover`

Tests a new or unlisted model against all known features to see what it supports. The results can be saved to a YAML file.

**Usage:**

```bash
# Discover features for a new model from OpenAI
litemind discover --api openai --models new-experimental-model-v1
```

## Caveats and Limitations

-   **Token Management**: The library does not currently automatically manage or limit the number of tokens sent to the models. Long conversations can exceed model context windows.
-   **Error Handling**: While basic error handling is in place, the robustness of API calls (e.g., automatic retries on server errors) could be improved.
-   **API Key Management**: The library relies on environment variables for API keys. For production environments, a more secure key management solution is recommended.
-   **Performance**: Performance optimizations, such as caching API calls or using asynchronous operations, are not yet implemented.

## Code Health

The `test_reports` directory was empty, so a `test_report.md` file could not be found. Therefore, a detailed report on test execution cannot be provided.

However, an analysis of the repository structure reveals a comprehensive test suite located in `src/litemind/`, with test files mirroring the source code structure. This indicates a strong commitment to code quality and testing.

## API Keys

To use LiteMind with various LLM providers, you need to obtain API keys and set them as environment variables.

-   **OpenAI**: Get your key from [platform.openai.com](https://platform.openai.com/api-keys) and set `OPENAI_API_KEY`.
-   **Anthropic (Claude)**: Get your key from [console.anthropic.com](https://console.anthropic.com/dashboard) and set `ANTHROPIC_API_KEY`.
-   **Google (Gemini)**: Get your key from [makersuite.google.com](https://makersuite.google.com/app/apikey) and set `GOOGLE_GEMINI_API_KEY`.
-   **Ollama**: No key is required as it runs locally.

### Setting Environment Variables

#### macOS / Linux

Edit your shell's profile file (e.g., `~/.zshrc`, `~/.bash_profile`, or `~/.profile`):

```bash
# Open the file with a text editor
nano ~/.zshrc

# Add the following lines
export OPENAI_API_KEY="your-openai-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"
export GOOGLE_GEMINI_API_KEY="your-google-api-key"

# Save the file and reload your shell
source ~/.zshrc
```

#### Windows

1.  Search for "Edit the system environment variables" in the Start Menu and open it.
2.  Click the "Environment Variables..." button.
3.  In the "User variables" section, click "New...".
4.  For "Variable name", enter `OPENAI_API_KEY`.
5.  For "Variable value", enter your key.
6.  Click "OK".
7.  Repeat for `ANTHROPIC_API_KEY` and `GOOGLE_GEMINI_API_KEY`.
8.  Click "OK" on all windows to save. You may need to restart your terminal or IDE for the changes to take effect.

## Logging

LiteMind uses [Arbol](http://github.com/royerlab/arbol) for logging. It provides structured and colorful console output that is helpful for debugging. If you wish to disable it, you can set the `passthrough` flag to `True`:

```python
from arbol import Arbol
Arbol.passthrough = True
```

## Roadmap

- [x] Setup a readme with a quick start guide.
- [x] setup continuous integration and pipy deployment.
- [x] Improve document conversion (page per page text and video interleaving + whole page images)
- [x] Cleanup structured output with tool usage
- [x] Implement streaming callbacks
- [x] Improve folder/archive conversion: add ascii folder tree
- [x] Reorganise media files used for testing into a single media folder
- [x] Improve logging with arbol, with option to turn off.
- [x] Use specialised libraries for document type identification
- [x] Cleanup the document conversion code.
- [x] Add support for adding nD images to messages.
- [x] Automatic feature support discovery for models (which models support images as input, reasoning, etc...)
- [x] Add support for OpenAI's new 'Response' API.
- [x] Add support for builtin tools: Web search and MCP protocol.
- [ ] Add webui functionality for agents using Reflex.
- [ ] Video conversion temporal sampling should adapt to the video length, short videos should have more frames...
- [ ] RAG ingestion code for arbitrary digital objects: folders, pdf, images, urls, etc...
- [ ] Add more support for MCP protocol beyond built-in API support.
- [ ] Use the faster pybase64 for base64 encoding/decoding.
- [ ] Deal with message sizes in tokens sent to models
- [ ] Improve vendor api robustness features such as retry call when server errors, etc...
- [ ] Improve and uniformize exception handling
- [ ] Implement 'brainstorming' mode for text generation, possibly with API fusion.

## Contributing

Contributions are welcome! We appreciate any help, from bug reports and feature requests to code contributions. Please read our contributing guidelines to get started.

For more details, please see the `CONTRIBUTING.md` file.

## License

This project is licensed under the BSD-3-Clause License. See the `LICENSE` file for details.

---
*This README.md file was generated with the help of AI.*
