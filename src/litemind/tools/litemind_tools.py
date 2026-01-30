import argparse
import os
from typing import List, Type

from litemind import API_IMPLEMENTATIONS, DefaultApi
from litemind.apis.base_api import BaseApi
from litemind.apis.combined_api import CombinedApi
from litemind.apis.providers.anthropic.anthropic_api import AnthropicApi
from litemind.apis.providers.google.google_api import GeminiApi
from litemind.apis.providers.ollama.ollama_api import OllamaApi
from litemind.apis.providers.openai.openai_api import OpenAIApi
from litemind.tools.commands.export_repo import export_repo
from litemind.tools.commands.scan import discover, scan, validate


def main():
    """
    The Litemind command-line tools consist of a series of subcommands that can be used to:
    - Export the entire repository to a single file.
    - Validate the model registry against live API responses.
    - Discover features for new/unknown models.

    """

    parser = argparse.ArgumentParser(description="Litemind command-line tool.")
    subparsers = parser.add_subparsers(dest="command")

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

    # Validate subcommand (recommended - validates registry against live API)
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate model registry against live API tests and report discrepancies.",
    )
    validate_parser.add_argument(
        "api",
        choices=["gemini", "openai", "claude", "ollama", "all"],
        default="all",
        nargs="+",
        help="The API(s) to validate. Can specify multiple APIs (gemini, openai, claude, ollama, all). Default is 'all'.",
    )
    validate_parser.add_argument(
        "-m",
        "--models",
        nargs="*",
        help="Specific model names to validate. If not provided, all registered models will be validated.",
        default=None,
    )
    validate_parser.add_argument(
        "-o",
        "--output-dir",
        help="Directory to save validation report. If not provided, report is only printed.",
        default=None,
    )

    # Discover subcommand (for new models not in registry)
    discover_parser = subparsers.add_parser(
        "discover",
        help="Discover features for new models not yet in the registry through live testing.",
    )
    discover_parser.add_argument(
        "api",
        choices=["gemini", "openai", "claude", "ollama"],
        nargs="+",
        help="The API(s) to test. Can specify multiple APIs (gemini, openai, claude, ollama).",
    )
    discover_parser.add_argument(
        "-m",
        "--models",
        nargs="+",
        required=True,
        help="Model names to discover features for. Required.",
    )
    discover_parser.add_argument(
        "-o",
        "--output-dir",
        help="Directory to save discovery results. If not provided, uses the default directory.",
        default=None,
    )

    # Scan subcommand (deprecated - kept for backward compatibility)
    scan_parser = subparsers.add_parser(
        "scan",
        help="[Deprecated] Scan models for features. Use 'validate' or 'discover' instead.",
    )
    scan_parser.add_argument(
        "api",
        choices=["gemini", "openai", "claude", "ollama", "all"],
        default="all",
        nargs="+",
        help="The API(s) to scan models from. Can specify multiple APIs (gemini, openai, claude, ollama, all). Default is 'all'.",
    )
    scan_parser.add_argument(
        "-m",
        "--models",
        nargs="*",
        help="Specific model names to scan. If not provided, all available models will be scanned.",
        default=None,
    )
    scan_parser.add_argument(
        "-o",
        "--output-dir",
        help="Directory to save scan results. If not provided, uses the default directory.",
        default=None,
    )

    args = parser.parse_args()

    if args.command == "export":
        export_repo(
            folder_path=args.folder_path,
            output_file=args.output_file,
            allowed_extensions=args.extensions,
            excluded_files=args.exclude,
        )

    elif args.command == "validate":
        api_classes: List[Type[BaseApi]] = _get_api_classes(args.api)
        validate(apis=api_classes, model_names=args.models, output_dir=args.output_dir)

    elif args.command == "discover":
        api_classes: List[Type[BaseApi]] = _get_api_classes(args.api)
        discover(apis=api_classes, model_names=args.models, output_dir=args.output_dir)

    elif args.command == "scan":
        print(
            "Note: 'scan' is deprecated. Consider using 'validate' or 'discover' instead."
        )
        api_classes: List[Type[BaseApi]] = _get_api_classes(args.api)
        scan(apis=api_classes, model_names=args.models, output_dir=args.output_dir)


def _get_api_classes(api_names: List[str]) -> List[Type[BaseApi]]:
    """Helper to convert API name strings to API classes."""
    api_classes: List[Type[BaseApi]] = []

    # If 'all' is in the list, use all available APIs
    if "all" in api_names:
        api_classes = list(API_IMPLEMENTATIONS)
        # Remove special API classes from the list:
        if CombinedApi in api_classes:
            api_classes.remove(CombinedApi)
        if DefaultApi in api_classes:
            api_classes.remove(DefaultApi)
    else:
        # Process each requested API
        for api_name in api_names:
            if api_name == "gemini":
                api_classes.append(GeminiApi)
            elif api_name == "openai":
                api_classes.append(OpenAIApi)
            elif api_name == "claude":
                api_classes.append(AnthropicApi)
            elif api_name == "ollama":
                api_classes.append(OllamaApi)
            else:
                print(f"Unrecognized API: {api_name}. Skipping.")

    return api_classes
