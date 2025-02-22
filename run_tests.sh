mkdir test_reports
#pytest --html=/test_reports/test_report.html .
pytest  -k "tools" --md-report --md-report-verbose=1 --md-report-output=./test_reports/test_report.md . > ./test_reports/test_report_stdout.txt