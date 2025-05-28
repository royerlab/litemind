from typing import List, Optional, Type

from arbol import aprint, asection

from litemind import CombinedApi
from litemind.apis.base_api import BaseApi
from litemind.apis.feature_scanner import ModelFeatureScanner


def scan(
    api: Optional[Type[BaseApi]] = None,
    model_names: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
) -> str:
    """
    Scan models from an API for supported features and generate a report.

    Parameters
    ----------
    api : Optional[Type[BaseApi]]
        The API class to scan. If None, uses CombinedApi.
    model_names : Optional[List[str]]
        The names of the models to scan. If None, scans all available models.
    output_dir : Optional[str]
        The directory to save the scan results. If None, uses the default directory.

    Returns
    -------
    str
        The path to the generated Markdown report file.
    """
    # Use CombinedApi if no API is provided
    if api is None:
        api = CombinedApi

    # Initialize the scanner
    with asection(
        f"Scanning model(s) {model_names if model_names else 'all'} from API: {api.__name__}"
    ):
        # Initialize the scanner
        scanner = ModelFeatureScanner(output_dir=output_dir)

        # Create an instance of the API class to use for model listing
        api_instance = api()

        try:
            # List all available models
            available_models = api_instance.list_models()

            # Give the list of all available models:
            with asection(f"Found {len(available_models)} models:"):
                for model in available_models:
                    aprint(f" - {model}")

        except Exception as e:
            aprint(f"Error listing models: {e}")
            available_models = []

        if not available_models:
            aprint("No models available for scanning.")
            return ""

        # Filter model list for scanning
        if model_names is not None:
            scan_models = [m for m in available_models if m in model_names]
            if len(scan_models) < len(model_names):
                missing = set(model_names) - set(scan_models)
                aprint(f"Warning: Some requested models are not available: {missing}")
        else:
            scan_models = available_models

        if not scan_models:
            aprint("No models to scan.")
            return ""

        # Scan the API class directly
        aprint(f"Scanning {len(scan_models)} models: {', '.join(scan_models)}")

        scanner.scan_apis([api], model_names=scan_models)

        # Generate and save the report
        report_text = scanner.generate_markdown_report()
        saved_files = scanner.save_results()

        # Print the report
        aprint("\n" + report_text)
