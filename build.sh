#!/bin/bash
set -e

# Sort imports with isort:
isort .

# Format code with black:
black .

# Create folder for test reports:
mkdir -p test_reports

# Run tests and generate test report:
# filtering specific tests: -k "tools"
pytest --cov-report json:./test_reports/test_coverage.json --md-report --md-report-verbose=1 --md-report-output=./test_reports/test_report.md .

# Clean and build the project:
hatch clean
hatch build
