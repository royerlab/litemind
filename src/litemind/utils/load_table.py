from os import path

from arbol import aprint
from pandas import read_csv, read_excel

from litemind.utils.normalise_uri_to_local_file_path import uri_to_local_file_path


def load_table_from_uri(uri: str):
    """Load a table file (CSV, TSV, etc.) into a pandas DataFrame."""
    file_extension = path.splitext(uri)[1].lower()

    try:

        # Download the table file to a local temp file using download_table_to_temp_file:
        local_path = uri_to_local_file_path(uri)

        # Detect character encoding:
        import chardet

        with open(local_path, "rb") as f:
            result = chardet.detect(f.read())

        if file_extension == ".csv":
            # Load CSV with comma delimiter
            return read_csv(local_path, encoding=result["encoding"])
        elif file_extension in [".tsv", ".tab"]:
            # Load TSV with tab delimiter
            return read_csv(local_path, sep="\t", encoding=result["encoding"])
        elif file_extension == ".xlsx" or file_extension == ".xls":
            # Load Excel files
            return read_excel(local_path)
        else:
            # Try to infer the delimiter
            return read_csv(
                local_path, sep=None, engine="python", encoding=result["encoding"]
            )  # python engine will try to detect sep
    except Exception as e:
        aprint(f"Error loading file {uri}: {str(e)}")
        return None
