"""
Model Feature Scanning and Validation Commands

This module provides CLI commands for:
- validate: Compare registry data against live API tests
- discover: Discover features for new/unknown models
- scan: Original full model feature scanning (deprecated, use validate instead)
"""

from typing import List, Optional, Type

from arbol import aprint, asection

from litemind.apis.base_api import BaseApi
from litemind.apis.feature_validator import ModelFeatureValidator


def validate(
    apis: List[Type[BaseApi]],
    model_names: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
):
    """
    Validate registry data against live API for specified models.

    This command compares the curated model registry data against actual
    live API tests to identify discrepancies.

    Parameters
    ----------
    apis : List[Type[BaseApi]]
        The API classes to validate.
    model_names : Optional[List[str]]
        The names of the models to validate. If None, validates all registered models.
    output_dir : Optional[str]
        The directory to save the validation report. If None, uses the default directory.
    """
    if (
        apis is None
        or not isinstance(apis, list)
        or not all(isinstance(api, type) and issubclass(api, BaseApi) for api in apis)
    ):
        aprint("No valid APIs provided")
        return

    # Initialize the validator
    validator = ModelFeatureValidator(output_dir=output_dir)

    with asection("Validating model registry against live APIs"):
        # Run validation
        validation_results = validator.validate_apis(
            api_classes=apis,
            model_names=model_names,
        )

        # Generate and print the report
        report = validator.generate_validation_report(validation_results)
        aprint("\n" + report)

        # Save the report if output_dir is specified
        if output_dir:
            import os
            from datetime import datetime

            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = os.path.join(output_dir, f"validation_report_{timestamp}.md")
            with open(report_path, "w") as f:
                f.write(report)
            aprint(f"\nValidation report saved to: {report_path}")


def discover(
    apis: List[Type[BaseApi]],
    model_names: List[str],
    output_dir: Optional[str] = None,
):
    """
    Discover features for models not yet in the registry through live testing.

    This is useful for testing new models that haven't been added to the
    curated registry yet.

    Parameters
    ----------
    apis : List[Type[BaseApi]]
        The API classes to test.
    model_names : List[str]
        The names of the models to discover features for. Required.
    output_dir : Optional[str]
        The directory to save the discovery results. If None, uses the default directory.
    """
    if (
        apis is None
        or not isinstance(apis, list)
        or not all(isinstance(api, type) and issubclass(api, BaseApi) for api in apis)
    ):
        aprint("No valid APIs provided")
        return

    if not model_names:
        aprint(
            "No model names provided. Please specify models to discover features for."
        )
        return

    # Initialize the scanner
    scanner = ModelFeatureValidator(output_dir=output_dir)

    with asection(f"Discovering features for models: {', '.join(model_names)}"):
        for api in apis:
            with asection(f"API: {api.__name__}"):
                for model_name in model_names:
                    features = scanner.discover_features(api, model_name)
                    if features:
                        aprint(f"\nDiscovered features for {model_name}:")
                        for feature, supported in features.items():
                            icon = "✅" if supported else "❌"
                            aprint(f"  {icon} {feature.name}")
                    else:
                        aprint(f"\nNo features discovered for {model_name}")

        # Save results if any were found
        if scanner.scan_results:
            saved_files = scanner.save_results()
            aprint(f"\nResults saved to: {saved_files}")


def scan(
    apis: List[Type[BaseApi]],
    model_names: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
):
    """
    Scan models from an API for supported features and generate a report.

    .. deprecated::
        This function performs full feature scanning which is expensive and
        can produce false negatives. Consider using `validate()` to validate
        the curated registry instead, or `discover()` for new models.

    Parameters
    ----------
    apis : List[Type[BaseApi]]
        The API classes to scan.
    model_names : Optional[List[str]]
        The names of the models to scan. If None, scans all available models.
    output_dir : Optional[str]
        The directory to save the scan results. If None, uses the default directory.

    Returns
    -------
    None
        The function does not return anything. It prints the report and saves the results to the specified directory.

    """
    # Ensure that list of APis is not empty or None:
    if (
        apis is None
        or not isinstance(apis, list)
        or not all(isinstance(api, type) and issubclass(api, BaseApi) for api in apis)
    ):
        aprint("No valid APIs provided")
        return

    for api in apis:
        # Initialize the scanner
        with asection(
            f"Scanning model(s) {model_names if model_names else 'all'} from API: {api.__name__}"
        ):
            # Initialize the scanner
            scanner = ModelFeatureValidator(output_dir=output_dir)

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
                return

            # Filter model list for scanning
            if model_names is not None:
                scan_models = [m for m in available_models if m in model_names]
                if len(scan_models) < len(model_names):
                    missing = set(model_names) - set(scan_models)
                    aprint(
                        f"Warning: Some requested models are not available: {missing}"
                    )
            else:
                scan_models = available_models

            if not scan_models:
                aprint("No models to scan.")
                return

            # Scan the API class directly
            aprint(f"Scanning {len(scan_models)} models: {', '.join(scan_models)}")

            scanner.scan_apis([api], model_names=scan_models)

            # Generate and save the report
            report_text = scanner.generate_markdown_report()
            scanner.save_results()

            # Print the report
            aprint("\n" + report_text)
