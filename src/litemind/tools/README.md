# Tools Package

The `tools` package provides command-line interfaces for repository export and model feature scanning/validation.

## Package Structure

```
tools/
├── __init__.py
├── litemind_tools.py        # Main CLI entry point
└── commands/
    ├── __init__.py
    ├── export_repo.py       # Repository export to single file
    ├── scan.py              # Model feature scanning/validation
    └── utils.py             # Shared utilities
```

## CLI Commands

### Repository Export (`export`)

Exports entire repository contents to a single file:

```bash
# Export with defaults
litemind export --folder-path . --output-file exported.txt

# Custom extensions
litemind export -f . -o output.txt -e .py .md .toml

# Exclude patterns
litemind export -f . -o output.txt -x __pycache__ .git
```

### Model Feature Validation (`validate`)

Validates the model registry against live API responses:

```bash
# Validate all models for specified APIs
litemind validate --api openai anthropic

# Validate specific models
litemind validate --api openai --models gpt-4o gpt-4o-mini

# Custom output directory
litemind validate --api gemini -o ./reports
```

### Model Feature Discovery (`discover`)

Discovers features for new/unknown models:

```bash
# Discover features for new models
litemind discover --api openai --models new-model-name

# Multiple models
litemind discover --api anthropic -m claude-4 claude-4-vision
```

### Model Feature Scan (`scan`) - Deprecated

Full model feature scan (use `validate` or `discover` instead):

```bash
litemind scan --api openai gemini
```

## Python API

### Repository Export

```python
from litemind.tools.commands.export_repo import export_repo

content = export_repo(
    folder_path=".",
    allowed_extensions=[".py", ".md"],
    excluded_files=["__pycache__", ".git"],
    output_file="exported.txt"
)
```

### Model Scanning

```python
from litemind.tools.commands.scan import validate, discover
from litemind.apis.providers.openai.openai_api import OpenAIApi

# Validate registry accuracy
validate(
    apis=[OpenAIApi],
    model_names=["gpt-4o", "gpt-4o-mini"],
    output_dir="./reports"
)

# Discover features for new models
discover(
    apis=[OpenAIApi],
    model_names=["new-model"],
    output_dir="./reports"
)
```

## Features

- **Multi-API Support**: Works with OpenAI, Anthropic, Gemini, Ollama
- **Feature-Based Validation**: Validates model registry accuracy against live APIs
- **Feature Discovery**: Discovers capabilities of new/unknown models

## Docstring Coverage

All public functions have comprehensive numpy-style docstrings with Parameters, Returns, and detailed descriptions. Coverage is 100%.
