import tempfile

import requests


def download_table_to_temp_file(url: str) -> str:
    """
    Downloads the table from an HTTP(S) URL and saves it to a temp file.
    Tries to infer extension from Content-Type or from the URL.
    """

    # If file is already local, just return the path:
    if url.startswith("file://"):
        return url.replace("file://", "")

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    }

    resp = requests.get(url, headers=headers)
    resp.raise_for_status()

    content_type = resp.headers.get("Content-Type", "")
    if content_type == "text/csv":
        extension = ".csv"
    elif content_type == "text/tab-separated-values":
        extension = ".tsv"
    elif content_type == "application/vnd.ms-excel":
        extension = ".xls"
    elif content_type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
        extension = ".xlsx"
    elif content_type == "application/vnd.oasis.opendocument.spreadsheet":
        extension = ".ods"
    elif content_type == "application/json":
        extension = ".json"
    else:
        urll = url.lower()
        if urll.endswith(".csv"):
            extension = ".csv"
        elif urll.endswith(".tsv"):
            extension = ".tsv"
        elif urll.endswith(".xls"):
            extension = ".xls"
        elif urll.endswith(".xlsx"):
            extension = ".xlsx"
        elif urll.endswith(".ods"):
            extension = ".ods"
        elif urll.endswith(".json"):
            extension = ".json"
        else:
            raise ValueError("Unknown table type")

    with tempfile.NamedTemporaryFile(suffix=extension,
                                     delete=False) as tmp_file:
        tmp_file.write(resp.content)
        tmp_path = tmp_file.name

    return tmp_path
