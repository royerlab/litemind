import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from litemind.media.types.media_table import Table
from litemind.media.types.media_text import Text
from litemind.ressources.media_resources import MediaResources


def test_table_initialization_with_uri():
    """Test that the Table is correctly initialized with a URI."""
    table = Table(uri="file://test.csv")
    assert table.uri == "file://test.csv"
    assert table.dataframe is None


def test_table_initialization_with_dataframe():
    """Test that the Table is correctly initialized with a DataFrame."""
    df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
    table = Table.from_table(df)
    table.load_from_uri()
    assert table.dataframe.equals(df)
    assert table.uri is not None


def test_table_initialization_no_uri_or_dataframe():
    """Test that the Table raises ValueError when neither URI nor DataFrame is provided."""
    with pytest.raises(TypeError):
        Table()


def test_from_dataframe():
    """Test that the Table is correctly created from a DataFrame."""
    df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
    table = Table.from_table(df)
    assert table.uri is not None
    assert table.dataframe is None


def test_from_dataframe_with_filepath():
    """Test that the Table is correctly created from a DataFrame with a specified filepath."""
    df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
        filepath = tmp_file.name
        table = Table.from_table(df, filepath=filepath)
        assert table.uri == f"file://{filepath}"
        assert table.dataframe is None
        os.remove(filepath)  # Clean up the temporary file


def test_load_from_uri(tmp_path):
    """Test that the table data is correctly loaded from the URI."""
    # Create a temporary CSV file for testing
    csv_data = "col1,col2\n1,2\n3,4"
    file_path = tmp_path / "test.csv"
    file_path.write_text(csv_data)
    uri = f"file://{file_path}"

    table = Table(uri=uri)
    table.load_from_uri()

    # Assert that the dataframe is loaded correctly
    assert table.dataframe is not None
    expected_df = pd.read_csv(file_path)
    pd.testing.assert_frame_equal(table.dataframe, expected_df)


def test_load_from_uri_no_uri():
    """Test that ValueError is raised when no URI is provided."""
    table = Table.from_table(pd.DataFrame({"col1": [1, 2], "col2": [3, 4]}))
    table.uri = None
    with pytest.raises(ValueError, match="No URI provided to load the table data"):
        table.load_from_uri()


def test_to_markdown_with_uri(tmp_path):
    """Test that the table is correctly converted to a markdown string when initialized with a URI."""
    # Create a temporary CSV file for testing
    csv_data = "col1,col2\n1,2\n3,4"
    file_path = tmp_path / "test.csv"
    file_path.write_text(csv_data)
    uri = f"file://{file_path}"

    table = Table(uri=uri)
    markdown_string = table.to_markdown()

    assert isinstance(markdown_string, str)
    assert "```dataframe" in markdown_string
    assert "|    |   col1 |   col2 |" in markdown_string
    assert "|----|--------|--------|" in markdown_string
    assert "|  0 |      1 |      2 |" in markdown_string
    assert "|  1 |      3 |      4 |" in markdown_string


def test_to_markdown_with_dataframe():
    """Test that the table is correctly converted to a markdown string when initialized with a DataFrame."""
    df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
    table = Table.from_table(df)
    markdown_string = table.to_markdown()

    assert isinstance(markdown_string, str)
    assert "```dataframe" in markdown_string
    assert "|    |   col1 |   col2 |" in markdown_string
    assert "|----|--------|--------|" in markdown_string
    assert "|  0 |      1 |      3 |" in markdown_string
    assert "|  1 |      2 |      4 |" in markdown_string


def test_to_markdown_text_media_with_dataframe():
    """Test that the table is correctly converted to a markdown Text media when initialized with a DataFrame."""
    df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
    table = Table.from_table(df)
    text_media = table.to_markdown_text_media()

    assert isinstance(text_media, Text)
    assert "```dataframe" in text_media.text
    assert "|    |   col1 |   col2 |" in text_media.text
    assert "|----|--------|--------|" in text_media.text
    assert "|  0 |      1 |      3 |" in text_media.text
    assert "|  1 |      2 |      4 |" in text_media.text


def test_to_markdown_text_media_with_uri(tmp_path):
    """Test that the table is correctly converted to a markdown Text media when initialized with a URI."""
    # Create a temporary CSV file for testing
    csv_data = "col1,col2\n1,2\n3,4"
    file_path = tmp_path / "test.csv"
    file_path.write_text(csv_data)
    uri = f"file://{file_path}"

    table = Table(uri=uri)
    text_media = table.to_markdown_text_media()

    assert isinstance(text_media, Text)
    assert "```dataframe" in text_media.text
    assert "|    |   col1 |   col2 |" in text_media.text
    assert "|----|--------|--------|" in text_media.text
    assert "|  0 |      1 |      2 |" in text_media.text
    assert "|  1 |      3 |      4 |" in text_media.text


def test_round_trip_csv(tmp_path):
    # Locate the original clusters.csv file
    table_uri = MediaResources.get_local_test_table_uri("clusters.csv")

    # Convert the URI to a Path object
    clusters_path = Path(table_uri.replace("file://", ""))

    # Ensure the clusters.csv file exists
    assert (
        clusters_path.exists()
    ), "clusters.csv does not exist in the expected location."

    # Create a URI for the Table initialization
    table = Table(uri=table_uri)
    table.load_from_uri()

    # Save the data loaded from clusters.csv to a new temporary CSV file
    round_trip_csv = tmp_path / "clusters_round_trip.csv"
    table.dataframe.to_csv(round_trip_csv, index=False)

    # Read both CSV files into DataFrames for comparison
    df_original = pd.read_csv(clusters_path)
    df_round_trip = pd.read_csv(round_trip_csv)

    # Check that the columns have the same names:
    assert list(df_original.columns) == list(
        df_round_trip.columns
    ), "Column names do not match."

    # Check that the indices are the same:
    assert df_original.index.equals(df_round_trip.index), "Indices do not match."

    # Check one-by-one that the values in the DataFrames are the same:
    for col in df_original.columns:
        # Check that the values in the column are the same, one-by-one:
        for i in range(len(df_original)):
            assert (
                df_original.at[i, col] == df_round_trip.at[i, col]
            ), f"Values in column '{col}' do not match at index {i}."
