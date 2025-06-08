import os
from typing import Optional

from arbol import aprint, asection

from litemind import CombinedApi
from litemind.agent.agent import Agent
from litemind.agent.messages.message import Message
from litemind.apis.base_api import BaseApi
from litemind.apis.model_features import ModelFeatures
from litemind.media.conversion.converters.document_converter_python_minify import (
    DocumentConverterPythonMinify,
)
from litemind.tools.commands.utils import default_folder_scanning_parameters, parse_yaml


def codegen(
    folder_path: str,
    api: Optional[BaseApi] = None,
    model_name: Optional[str] = None,
    file_selection: Optional[str] = None,
):
    """
    Generates files in the specified folder using the provided API and model name.
    This function looks for YAML files in the '.codegen' subfolder of the specified folder,
    parses them, and generates files based on the prompts and parameters defined in those YAML files.

    Parameters
    ----------
    folder_path: str
        The path to the folder where the '.codegen' subfolder is located.
    api: Optional[BaseApi]
        The API to use for generating the files. If None, a CombinedApi will be used.
    model_name: Optional[str]
        The name of the model to use for generating the files. If None, the best model will be selected.
    file_selection: Optional[str]
        A string to filter the files to be generated. If provided, only files that contain this string in their name will be processed.
        If None, all files will be processed.


    Returns
    -------
    None
        This function does not return anything. It generates files in the specified folder based on the prompts defined in the YAML files.
    """

    with asection(
        f"Generating files in folder: {folder_path} using model {model_name} fromAPI: {api} and selection: {file_selection}"
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

                # Print the parsed data:
                with asection(f"Parsed data from {file}:"):
                    aprint(parsed_data)

                allowed_extensions = parsed_data["folder"].get("extensions", None)
                aprint(allowed_extensions)

                excluded_files = parsed_data["folder"].get("excluded", None)
                aprint(excluded_files)

                # Generate the file:
                generate(
                    prompt=parsed_data["prompt"],
                    input_folder=os.path.join(
                        folder_path, parsed_data["folder"]["path"]
                    ),
                    output_file=os.path.join(folder_path, parsed_data["file"]),
                    allowed_extensions=allowed_extensions,
                    excluded_files=excluded_files,
                    api=api,
                    model_name=model_name,
                )


def generate(
    prompt: str,
    input_folder: str,
    output_file: str,
    allowed_extensions: Optional[list[str]] = None,
    excluded_files: Optional[list[str]] = None,
    api: Optional[BaseApi] = None,
    model_name: Optional[str] = None,
    thinking: bool = False,
    minify_python: bool = True,
    output_prompt_to_file: bool = True,
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
    model_name": Optional[str]
        The name of the model to use for generating the file. If None, the best model will be selected.
    minify_python: bool
        If True, Python files will be minified before being sent to the API.
        This is useful for reducing the size of the files and improving generation speed.
    output_prompt_to_file: bool
        If True, the prompt will be saved to a file for debugging purposes.

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
        aprint(
            f"Allowed extensions ({len(allowed_extensions)}): {', '.join(allowed_extensions)}"
        )
        aprint(f"Excluded files ({len(excluded_files)}): {', '.join(excluded_files)}")

        # Initialize the API
        if api is None:
            api = CombinedApi()

        if model_name is None:
            # If no model name is provided, get the best model for text generation and document processing but no thinking:
            model_name = api.get_best_model(
                features=[ModelFeatures.TextGeneration, ModelFeatures.Document],
                non_features=[ModelFeatures.Thinking] if not thinking else None,
            )

            # If no model is found we relax the requirements and try to get the best model for text generation and document processing:
            if model_name is None:
                model_name = api.get_best_model(
                    features=[ModelFeatures.TextGeneration, ModelFeatures.Document]
                )

        # Initialize the agent
        agent = Agent(api=api, model_name=model_name)

        # Create a message:
        message = Message(role="user")

        # Append the prompt and the folder contents to the message:
        message.append_text(prompt)
        message.append_folder(
            input_folder,
            allowed_extensions=allowed_extensions,
            excluded_files=excluded_files,
            date_and_times=False,
            file_sizes=False,
        )
        message.append_text(
            f"\n\n\n# Task:\nPlease generate a detailed, "
            f"complete and informative {output_file_name} file "
            f"without any preamble or postamble: \n"
        )

        from litemind.media.conversion.media_converter import MediaConverter

        # Instantiate the media converter:
        media_converter = MediaConverter()

        # Add the media converter for Python minification:
        if minify_python:
            media_converter.add_media_converter(DocumentConverterPythonMinify())

        # Add the media converter supporting APIs. This converter can use any model from the API that supports the required features.
        media_converter.add_default_converters()

        # Convert all files in the folder to test:
        message = message.convert_media(media_converter=media_converter)

        # Instantiate a text compressor:
        from litemind.utils.text_compressor import TextCompressor

        text_compressor = TextCompressor(
            schemes=["newlines", "comments", "repeats", "trailing"], max_repeats=10
        )

        # Compress the message to reduce its size:
        message = message.compress_text(text_compressor=text_compressor)

        # printout the message report:
        with asection(f"Message report:"):
            aprint(message.report())

        if output_prompt_to_file:
            # Save the message to a file for debugging:
            _save_prompt_to_file(message, output_file)

        try:
            # Use the agent to generate the file contents:
            response = agent(message)
        except Exception as e:
            aprint(f"Error generating file: {e}")
            # print stacktrace:
            import traceback

            aprint(traceback.format_exc())

            with asection("Message content before error:"):
                # aprint(message)

                _save_prompt_to_file(message, output_file)

                aprint(f"FAILED!!!")

                return f"Generation failed due to an error: {e}."

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


def _save_prompt_to_file(message, output_file):
    # Save the message to a file for debugging:
    debug_file = os.path.join(os.path.dirname(output_file), "debug_prompt.txt")
    with open(debug_file, "w") as f:
        f.write(str(message))
