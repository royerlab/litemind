
# Sort imports with isort:
isort .

# Format code with black:
black .

# Create folder for test reports:
mkdir test_reports

# Run tests and generate test report:
# filtering specific tests: -k_dict "tools"
pytest  --cov-report json:./test_reports/test_coverage.json --md-report --md-report-verbose=1 --md-report-output=./test_reports/test_report.md .

# Generate updated ANALYSIS.md and README.md:
litemnd codegen gemini -f analysis
litemnd codegen gemini -f readme

# Clean ans build the project:
hatch clean
hatch build