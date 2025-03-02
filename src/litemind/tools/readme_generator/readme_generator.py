import argparse
import os
from typing import Optional

from arbol import asection

from litemind.agent.messages.message import Message
from litemind.apis.base_api import BaseApi
from litemind.apis.combined_api import CombinedApi
from litemind.apis.model_features import ModelFeatures
from litemind.apis.providers.anthropic.anthropic_api import AnthropicApi
from litemind.apis.providers.google.google_api import GeminiApi
from litemind.apis.providers.ollama.ollama_api import OllamaApi
from litemind.apis.providers.openai.openai_api import OpenAIApi


def generate_readme(
    folder_path: str,
    context_text: str,
    api: Optional[BaseApi] = None,
    save_repo_to_file: bool = False,
) -> str:
    """
    Generate a README.md file for a Python repository.

    Parameters
    ----------
    folder_path: str
        The path to the folder containing the Python repository.
    context_text: str
        The context text to provide to the AI model.
    api: Optional[BaseApi]
        The API to use for generating the README. If None, the CombinedApi is used.
    save_repo_to_file: bool
        Whether to save the entire repository to a file for debugging purposes.

    Returns
    -------
    str
        The generated README content.

    """
    with asection("Generating README.md"):

        # Initialize the API
        if api is None:
            api = CombinedApi()

        # Initialize the agent
        from litemind.agent.agent import Agent

        agent = Agent(api=api, model_features=ModelFeatures.Document)

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
         - Concepts
         - More Code Examples
         - Caveats and Limitations
         - Code Health
         - Roadmap
         - Contributing
         - License
         But feel free to include additional sections as needed.
     
         The 'Summary' section should provide an enthusiastic and complete description of the library, its purpose, and philosophy.
         The 'Usage' section should consist of rich, striking and illustrative examples of the Agentic API in use with multimodal inputs.
         The 'Concepts' section should explain the concepts behind the library, the main classes and their purpose, and how they interact.
         The 'Code Health' section should include the results of unit test in file 'test_report.md'. Please also discuss which tests failed as logged in 'test_report_stdout.txt'. Please provide file names for failed tests and statistics about the number of failed tests and an analysis of what happened. 
         The 'Roadmap' section can use the contents of TODO.md as a starting point, keep the checkmarks.
         The 'More Code Examples' section further expands with many more also covering the wrapper API, uses ideas and code from the unit tests (no need to mention that).
            
         For code examples:
         - Please use markdown code blocks for code examples and other code snippets.
         - Make sure that the code examples are complete and can be run as-is.
         - Make sure that the code can run without errors, e.g. make sure that all required imports are included.
         - Make sure, if possible to include the expected output as comments in the code.
         - Make sure that comments are emphatically didactic.
         - Make sure that the code is well-formatted and follows PEP-8.
         - Make sure to include a variety of examples that cover different use cases and edge cases.
         - Make sure that the code examples are complete and can be run as-is.
         - Avoid putting multiple code examples in the same code block, unless they are chained.
         
         
         At the end of the README, include a note explaining that the README was generated with the help of AI.
         Avoid hallucinating things that are not in the repository.
         Please add badges for: pipy, license, and code coverage
         
         Your task is to follow the instructions above and generate a README.md file for the Python repository provided below:
     
         Here is some additional context about the repository:
         {context_text}
     
         Here is the the entire repository structure and its contents:
         """

        # Create a message with the prompt:
        message = Message(role="user")

        # Append the prompt and the folder contents to the message:
        message.append_text(prompt)
        message.append_folder(
            folder_path,
            allowed_extensions=[
                ".py",
                ".md",
                ".txt",
                ".toml",
                "LICENSE",
                ".tests",
                ".html",
            ],
            excluded_files=[
                "README.md",
                "litemind.egg-info",
                "dist",
                "build",
                "whole_repo.txt",
            ],
        )
        message.append_text(
            "Please generate a detailed, complete and informative README.md file for this repository without any preamble or postamble."
        )

        if save_repo_to_file:
            # Convert the message to a string
            message_str = str(message)

            # Save string to file:
            with open("whole_repo.txt", "w") as file:
                file.write(message_str)

        # Use the agent to generate the README.md content
        response = agent(message)

        # extract the README content from the response:
        readme_content = response[-1][0].content.strip()

        # If it starts with '```markdown', remove it:
        if readme_content.startswith("```markdown"):
            readme_content = readme_content[11:]

        # If it ends with '```', remove it:
        if readme_content.endswith("```"):
            readme_content = readme_content[:-3]

        # Write the generated content to README.md:
        readme_file_path = os.path.join(folder_path, "README.md")
        with open(readme_file_path, "w") as readme_file:
            readme_file.write(readme_content)

        return readme_content


def main():
    """Command-line entry point for generating the README."""
    parser = argparse.ArgumentParser(
        description="Generate a README.md file for a Python repository."
    )
    parser.add_argument(
        "model",
        choices=["gemini", "openai", "claude", "ollama", "combined"],
        default="combined",
        nargs="?",
        help="The model to use for generating the README. Default is 'combined'.",
    )

    # Add a boolean argument to save the repository to a file, --save-repo-to-file or -s
    parser.add_argument(
        "-s",
        "--save-repo-to-file",
        action="store_true",
        help="Save the entire repository to a file for debugging purposes.",
    )

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

    folder_path = "."
    context_text = (
        'This repository contains a Python project called "litemind" '
        "that provides a wrapper API around LLM Apis as well as an elegant API "
        "for fully multimodal agentic AI for building conversational agents "
        "and tools built upon them. Please make sure to explain the difference "
        "between the API wrapper layer versus the agentic API -- this should also"
        "be reflected in the examples."
    )
    generate_readme(
        folder_path, context_text, api=api, save_repo_to_file=args.save_repo_to_file
    )


if __name__ == "__main__":
    main()
