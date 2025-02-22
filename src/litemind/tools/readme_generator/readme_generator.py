import argparse
import os
from typing import Optional

from litemind.agent.agent import Agent
from litemind.agent.message import Message
from litemind.apis._callbacks.callback_manager import CallbackManager
from litemind.apis._callbacks.print_callbacks import PrintCallbacks
from litemind.apis.anthropic.anthropic_api import AnthropicApi
from litemind.apis.base_api import BaseApi
from litemind.apis.combined_api import CombinedApi
from litemind.apis.google.google_api import GeminiApi
from litemind.apis.model_features import ModelFeatures
from litemind.apis.ollama.ollama_api import OllamaApi
from litemind.apis.openai.openai_api import OpenAIApi


def generate_readme(folder_path: str, context_text: str, api: Optional[BaseApi] = None):

     # Initialize the API
    if api is None:
        api = CombinedApi()

    # get a model that supports text generation and documents:
    model = api.get_best_model(features=[ModelFeatures.TextGeneration, ModelFeatures.Document])

    # Initialize the agent
    agent = Agent(api=api, model=model)

    # Define the prompt for generating the README.md
    prompt = f"""
     You are a helpful assistant with deep expertise in Software Engineering and best practices for Open Source projects.
     You know how to write complete, compeling and informative README files for Python repositories.
     Your task is to help a user generate a README.md file for a Python repository.
 
     The README.md should _at least_ include the following sections:
     - Summary
     - Features
     - Installation
     - Usage
     - Code Examples
     - Caveats and Limitations
     - Code Health
     - Roadmap
     - Contributing and Todos
     - License
     But feel free to include additional sections as needed.
 
     Please use markdown code blocks for code examples and other code snippets.
     The 'Usage' section should consist of rich, striking and illustrative examples of the Agentic API in use.
     The 'Code Examples' section can use ideas and code from the unit tests.
     The 'Code Health' section should include the results of unit test in file 'test_report.md'. 
     The 'Roadmap' section can use the contents of TODO.md as a starting point, keep the checkmarks.
     The 'Code Examples' section should include may examples of the agentic and wrapper APIs covering most features.
     At the end of the README, include a note explaining that the README was generated with the help of AI.
     
 
     Generate a README.md file for the Python repository described below.
 
     Here is some context about the repository:
     {context_text}
 
     Here is the the entire repository structure and contents:
     """

    # Create a message with the prompt:
    message = Message(role="user")

    # Append the prompt and the folder contents to the message:
    message.append_text(prompt)
    message.append_folder(folder_path,
                          allowed_extensions=['.py', '.md', '.txt', '.toml', 'LICENSE', '.tests', '.html'],
                          excluded_files=['README.md'])
    message.append_text(
        "Please generate a detailed, complete and informative README.md file for this repository without any preamble or postamble.")

    print(str(message))

    # Use the agent to generate the README.md content
    response = agent(message)

    # extract the README content from the response:
    readme_content = response[0].content.strip()

    # If it starts with '```markdown', remove it:
    if readme_content.startswith('```markdown'):
        readme_content = readme_content[11:]

    # If it ends with '```', remove it:
    if readme_content.endswith('```'):
        readme_content = readme_content[:-3]

    print(str(readme_content))

    # Write the generated content to README.md:
    readme_file_path = os.path.join(folder_path, 'README.md')
    with open(readme_file_path, 'w') as readme_file:
        readme_file.write(readme_content)

    return readme_content


def main():
    """Command-line entry point for generating the README."""
    parser = argparse.ArgumentParser(description="Generate a README.md file for a Python repository.")
    parser.add_argument("model",
                        choices=["gemini", "openai", "claude", "ollama", "combined"],
                        default="combined",
                        nargs='?',
                        help="The model to use for generating the README. Default is 'combined'.")
    args = parser.parse_args()

    # Initialize the API based on the chosen model
    if args.model == "gemini":
        api = GeminiApi()
    elif args.model == "openai":
        api = OpenAIApi()
    elif args.model == "claude":
        api = AnthropicApi()
    elif args.model == "ollama":
        api = OllamaApi()
    else:
        api = CombinedApi()

    folder_path = '.'
    context_text = (
        'This repository contains a Python project called "litemind" '
        'that provides a wrapper API around LLM Apis as well as an elegant API '
        'for fully multimodal agentic AI for building conversational agents '
        'and tools built upon them. Please make sure to explain the difference '
        'between the API wrapper layer versus the agentic API -- this should also'
        'be reflected in the examples.'
    )
    generate_readme(folder_path, context_text, api=api)


if __name__ == '__main__':
    main()
