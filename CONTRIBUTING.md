# Contributing to LiteMind

First off, thank you for considering contributing to LiteMind!

## Development Process

1. Fork the repository
2. Create a new branch for your feature/fix: `git checkout -b feature-name`
3. Make your changes
4. Run tests: `pytest`
5. Push to your fork: `git push origin feature-name`
6. Submit a Pull Request

## Setting Up Development Environment

1. Clone your fork:

```bash
git clone https://github.com/royerlab/litemind.git
cd litemind
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: `venv\Scripts\activate`
```

3. Install development dependencies:

```bash
pip install -e ".[dev]"
```

## Code Style

We use several tools to maintain code quality:

- **Black** for code formatting
- **isort** for import sorting
- **flake8** for style guide enforcement
- **mypy** for static type checking

Run the following before committing:

```bash
isort .
black .
flake8 .
mypy src/litemind
```

And fix any issues that arise in the files you've modified.

## Testing

- Write tests for new features using pytest
- Ensure all tests pass: `pytest src/`
- Maintain or improve code coverage

## Documentation

- Use Numpy-style docstrings
- Update documentation when adding/modifying features
- Keep the README.md up to date

Example docstring:

```python
def function_name(param1: str, param2: int) -> bool:
    """
    Short description of function.

    Longer description if needed.

    Parameters
    ----------
    param1 : str
        Description of param1
    param2 : int
        Description of param2

    Returns
    -------
    bool
        Description of return value

    Raises
    ------
    ValueError
        Description of when this error occurs
    """
```

## Pull Request Process

1. Update documentation if needed
2. Add tests for new features
3. Ensure CI pipeline passes
4. Get review from at least one maintainer
5. Squash commits if requested

## Branching Strategy

- `main`: Stable, released code
- Feature branches: `feature/your-feature-name`
- Bug fixes: `fix/bug-description`

## Commit Messages

Follow conventional commits:

```
feat: add new feature
fix: correct bug
docs: update documentation
style: formatting changes
refactor: code restructuring
test: add/modify tests
chore: maintenance tasks
```

## Getting Help

- Open an issue for bugs
- Start a discussion for feature requests
- Tag issues appropriately (`bug`, `enhancement`, `documentation`, etc.)

## License

By contributing, you agree that your contributions will be licensed under the BSD-3-Clause License.
