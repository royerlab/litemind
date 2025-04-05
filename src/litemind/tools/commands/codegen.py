import os
from typing import Optional

from arbol import aprint, asection

from litemind import CombinedApi
from litemind.agent.agent import Agent
from litemind.agent.messages.message import Message
from litemind.apis.base_api import BaseApi
from litemind.apis.model_features import ModelFeatures
from litemind.tools.commands.utils import default_folder_scanning_parameters, parse_yaml


def codegen(
    folder_path: str,
    api: Optional[BaseApi] = None,
    file_selection: Optional[str] = None,
) -> str:
    with asection(
        f"Generating files in folder: {folder_path} using API: {api} and selection: {file_selection}"
    ):

        # Locate folder '.codegen' within folder_path:
        codegen_folder = os.path.join(folder_path, ".codegen")

        # Strip file_selection:
        if file_selection is not None:
            file_selection = file_selection.strip()

        # For each file that ends in 'codegen.txt' in the '.codegen' folder:
        for file in os.listdir(codegen_folder):
            if file.endswith("codegen.yml"):

                # Check if file_selection is not None and if file_selection is not in file:
                if file_selection is not None:
                    if file_selection not in file:
                        # Skip this file:
                        continue

                # Get the file's absolute path:
                file = os.path.join(codegen_folder, file)

                # Parse yml file:
                parsed_data = parse_yaml(file)

                # Generate the file:
                generate(
                    prompt=parsed_data["prompt"],
                    input_folder=os.path.join(
                        folder_path, parsed_data["folder"]["path"]
                    ),
                    output_file=os.path.join(folder_path, parsed_data["file"]),
                    allowed_extensions=parsed_data.get("allowed_extensions", None),
                    excluded_files=parsed_data.get("excluded_files", None),
                    api=api,
                )


def generate(
    prompt: str,
    input_folder: str,
    output_file: str,
    allowed_extensions: Optional[list[str]] = None,
    excluded_files: Optional[list[str]] = None,
    api: Optional[BaseApi] = None,
) -> str:
    """
    Generates a file given a prompt and a folder containing the necessary files.

    Parameters
    ----------
    prompt : str
        The prompt to generate the file.
    input_folder : str
        The path to the folder containing the necessary files.
    output_file : str
        The path to the file to save the generated content to.
    allowed_extensions : Optional[list[str]]
        The list of allowed extensions for files to include in the README.
    excluded_files : Optional[list[str]]
        The list of files to exclude from the README.
    api : Optional[BaseApi]
        The API to use for generating the file.


    Returns
    -------
    str
        The generated content.

    """

    # Get just the filename from the output_file:
    output_file_name = os.path.basename(output_file)

    with asection(f"Generating file: {output_file_name}"):

        # If output file already exists, delete it:
        if os.path.exists(output_file):
            aprint(f"File {output_file} already exists. Deleting it.")
            os.remove(output_file)

        # Get the default folder scanning parameters:
        allowed_extensions, excluded_files = default_folder_scanning_parameters(
            allowed_extensions, excluded_files
        )

        # Print the parameters:
        aprint(f"Excluded files: {', '.join(excluded_files)}")
        aprint(f"Allowed extensions: {', '.join(allowed_extensions)}")

        # Initialize the API
        if api is None:
            api = CombinedApi()

        model_name = api.get_best_model(
            features=[ModelFeatures.TextGeneration],
            non_features=[ModelFeatures.Thinking],
        )

        # Initialize the agent
        agent = Agent(
            api=api,
            model_name=model_name,
            model_features=[ModelFeatures.TextGeneration, ModelFeatures.Document],
        )

        # Create a message:
        message = Message(role="user")

        # Append the prompt and the folder contents to the message:
        message.append_text(prompt)
        message.append_folder(
            input_folder,
            allowed_extensions=allowed_extensions,
            excluded_files=excluded_files,
        )
        message.append_text(
            f"\n\n\n# Task:\nPlease generate a detailed, complete and informative {output_file_name} file without any preamble or postamble: \n"
        )

        # Use the agent to generate the file contents:
        response = agent(message)

        # extract the file content from the response:
        main_response = response[-1][0].get_content()

        # Strip the response:
        text = main_response.strip()

        # If it starts with '```markdown' or '```md', remove it:
        if text.startswith("```markdown"):
            text = text[11:]
        if text.startswith("```md"):
            text = text[5:]

        # If it ends with '```', remove it:
        if text.endswith("```"):
            text = text[:-3]

        # Make sure that all folders leading to the file exist:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Write the generated content to output file:
        with open(output_file, "w") as readme_file:
            readme_file.write(text)

        return text
