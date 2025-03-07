import argparse
import os

from litemind.apis.combined_api import CombinedApi
from litemind.apis.providers.anthropic.anthropic_api import AnthropicApi
from litemind.apis.providers.google.google_api import GeminiApi
from litemind.apis.providers.ollama.ollama_api import OllamaApi
from litemind.apis.providers.openai.openai_api import OpenAIApi
from litemind.tools.commands.codegen import codegen
from litemind.tools.commands.export_repo import export_repo


def main():
    """
    The Litemind command-line tools consist of a series of subcommands that can be used to:
    - Generate files, such as a README.md for a Python repository, given a prompt, a folder, and some optional parameters.
    - Export the entire repository to a single file.


    """

    parser = argparse.ArgumentParser(description="Litemind command-line tool.")
    subparsers = parser.add_subparsers(dest="command")

    # Codegen subcommand
    codegen_parser = subparsers.add_parser(
        "codegen", help="Generate a README.md file for a Python repository."
    )
    codegen_parser.add_argument(
        "model",
        choices=["gemini", "openai", "claude", "ollama", "combined"],
        default="combined",
        nargs="?",
        help="The model to use for generating the README. Default is 'combined'.",
    )

    # Add the new argument to the codegen subcommand
    codegen_parser.add_argument(
        "-f",
        "--file",
        help="The specific file to generate. If not provided, all files will be generated.",
        default=None,
    )

    # Export subcommand
    export_parser = subparsers.add_parser(
        "export", help="Export the entire repository to a single file."
    )
    export_parser.add_argument(
        "-f",
        "--folder-path",
        help="The path to the folder containing the repository to export.",
        default=".",
    )
    export_parser.add_argument(
        "-o",
        "--output-file",
        default="exported.txt",
        #        required=True,
        help="The path to the file to save the entire repository to.",
    )
    export_parser.add_argument(
        "-e",
        "--extensions",
        default=None,
        nargs="*",
        help="The list of allowed extensions for files to include in the export.",
    )
    export_parser.add_argument(
        "-x",
        "--exclude",
        default=None,
        nargs="*",
        help="The list of files to exclude from the export.",
    )

    args = parser.parse_args()

    if args.command == "codegen":
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

        # By default the folder path is the current directory:
        folder_path = os.getcwd()

        # Generate the files:
        codegen(folder_path, api=api, file_selection=args.file)

    elif args.command == "export":
        export_repo(
            folder_path=args.folder_path,
            output_file=args.output_file,
            allowed_extensions=args.extensions,
            excluded_files=args.exclude,
        )


if __name__ == "__main__":
    main()
