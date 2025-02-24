
# Format code with black:
black .

# Create folder for test reports:
mkdir test_reports

# Run tests and generate test report:
# filtering specific tests: -k "tools"
pytest  --cov-report json:./test_reports/test_coverage.json --md-report --md-report-verbose=1 --md-report-output=./test_reports/test_report.md . > ./test_reports/test_report_stdout.txt

# Generate updated README.md:
readmegen gemini