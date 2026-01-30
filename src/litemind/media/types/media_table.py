from typing import Optional, Union

from pandas import DataFrame

from litemind.media.media_uri import MediaURI
from litemind.media.types.media_text import Text
from litemind.utils.load_table import load_table_from_uri


class Table(MediaURI):
    """
    A media that stores a table.
    """

    def __init__(self, uri: str, extension: Optional[str] = None, **kwargs):
        """
        Create a new table media from a DataFrame.

        Parameters
        ----------
        uri : str
            The URI of the table file.
        extension: str
            The extension of the table file.
        kwargs: Optional
            Other parameters passed to MediaURI

        """

        # Check that the file extension is valid:
        if extension is not None:
            if not uri.endswith(extension):
                raise ValueError(
                    f"Invalid table URI: '{uri}' (must have a valid table file extension)"
                )

        super().__init__(uri=uri, extension=extension, **kwargs)

        self.dataframe: Optional[DataFrame] = None

    @classmethod
    def from_table(
        cls,
        table: Union["ndarray", "DataFrame", list, tuple],
        filepath: Optional[str] = None,
    ):

        if filepath is None:
            # Create temporary file:
            import tempfile

            from litemind.utils.temp_file_manager import register_temp_file

            filepath = register_temp_file(
                tempfile.NamedTemporaryFile(delete=False).name
            )

        # If dataframe is a numpy array, or array-like, convert to DataFrame:
        if isinstance(table, (list, tuple)) or hasattr(table, "__array__"):
            import pandas as pd

            table = pd.DataFrame(table)

        # At this point `table` should be a DataFrame:
        elif not isinstance(table, DataFrame):
            raise ValueError(
                f"Parameter `dataframe` must be a pandas DataFrame or an array-like object, not {type(table)}"
            )

        # Save dataframe to file:
        table.to_csv(
            filepath,
            index=False,  # avoid the synthetic index column
            na_rep="",  # match original null representation
            float_format=None,  # let 'round_trip' preserve original
        )

        # URI:
        table_uri = "file://" + filepath

        # Create Image from
        return Table(uri=table_uri)

    @classmethod
    def from_csv_string(cls, csv_table_str: str):
        """
        Create a table from a CSV string.

        Parameters
        ----------
        csv_table_str: str
            The CSV string representing the table.
        """

        from io import StringIO

        import pandas as pd

        # Check that it is really a string:
        if not isinstance(csv_table_str, str):
            raise ValueError(
                f"Parameter `csv_table_str` must be a string, not {type(csv_table_str)}"
            )

        dataframe = pd.read_csv(
            StringIO(csv_table_str),
            dtype=str,  # keep all columns textual
            keep_default_na=False,  # don't turn 'NA' into NaN
            float_precision="round_trip",  # exact float repr
        )

        # Create a table from the DataFrame:
        return cls.from_table(dataframe)

    def load_from_uri(self):
        """
        Load the table data from the URI.
        """

        # Check if the URI is provided
        if self.uri is None:
            raise ValueError("No URI provided to load the table data")

        # Load the table data from the URI
        self.dataframe = load_table_from_uri(self.uri)

    def to_dataframe(self):
        """
        Get the table as a DataFrame.
        """

        # If the dataframe is not loaded, load it from the URI:
        if self.dataframe is None:
            self.load_from_uri()

        return self.dataframe

    def to_markdown(self):

        #
        if self.dataframe is None:
            self.load_from_uri()

        # Convert DataFrame to markdown table string:
        from tabulate import tabulate

        table_markdown = tabulate(
            self.dataframe, headers="keys", tablefmt="github", showindex=True
        )

        # Convert table to markdown:
        markdown = f"```dataframe\n{table_markdown}\n```"

        return markdown

    def to_markdown_text_media(self):

        # Get Markdown:
        markdown = self.to_markdown()

        return Text(markdown)
