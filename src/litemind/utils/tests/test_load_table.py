import os
from tempfile import NamedTemporaryFile

import pandas as pd
import pytest

from litemind.utils.load_table import load_table_from_uri


@pytest.fixture
def csv_file():
    with NamedTemporaryFile(suffix=".csv", delete=False) as f:
        f.write(b"col1,col2,col3\n1,2,3\n4,5,6")
        temp_path = f.name
    yield temp_path
    os.unlink(temp_path)


@pytest.fixture
def tsv_file():
    with NamedTemporaryFile(suffix=".tsv", delete=False) as f:
        f.write(b"col1\tcol2\tcol3\n1\t2\t3\n4\t5\t6")
        temp_path = f.name
    yield temp_path
    os.unlink(temp_path)


@pytest.fixture
def excel_file():
    df = pd.DataFrame({"col1": [1, 4], "col2": [2, 5], "col3": [3, 6]})
    with NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
        temp_path = f.name
    df.to_excel(temp_path, index=False)
    yield temp_path
    os.unlink(temp_path)


@pytest.fixture
def unknown_ext_file():
    with NamedTemporaryFile(suffix=".data", delete=False) as f:
        f.write(b"col1;col2;col3\n1;2;3\n4;5;6")
        temp_path = f.name
    yield temp_path
    os.unlink(temp_path)


@pytest.fixture
def rich_table_file():
    # Create a DataFrame with multiple columns, rows, and data types
    data = {
        "id": range(1, 21),  # 20 rows, integers
        "name": [f"Person {i}" for i in range(1, 21)],  # strings
        "score": [round(i * 3.14159, 2) for i in range(1, 21)],  # floats
        "is_active": [i % 2 == 0 for i in range(1, 21)],  # booleans
        "join_year": [2000 + i for i in range(1, 21)],  # integers (years)
    }
    df = pd.DataFrame(data)

    with NamedTemporaryFile(suffix=".csv", delete=False) as f:
        temp_path = f.name

    df.to_csv(temp_path, index=False)
    yield temp_path
    os.unlink(temp_path)


def test_load_csv(csv_file):
    df = load_table_from_uri(csv_file)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 3)
    assert list(df.columns) == ["col1", "col2", "col3"]
    assert df.iloc[0, 0] == 1
    assert df.iloc[1, 2] == 6


def test_load_tsv(tsv_file):
    df = load_table_from_uri(tsv_file)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 3)
    assert list(df.columns) == ["col1", "col2", "col3"]
    assert df.iloc[0, 0] == 1
    assert df.iloc[1, 2] == 6


def test_load_excel(excel_file):
    df = load_table_from_uri(excel_file)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 3)
    assert list(df.columns) == ["col1", "col2", "col3"]
    assert df.iloc[0, 0] == 1
    assert df.iloc[1, 2] == 6


def test_load_unknown_extension(unknown_ext_file):
    # Should infer delimiter
    df = load_table_from_uri(unknown_ext_file)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 3)
    assert list(df.columns) == ["col1", "col2", "col3"]


def test_load_nonexistent_file():
    df = load_table_from_uri("nonexistent_file.csv")
    assert df is None


def test_load_rich_table(rich_table_file):
    df = load_table_from_uri(rich_table_file)

    # Basic checks
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (20, 5)
    assert set(df.columns) == {"id", "name", "score", "is_active", "join_year"}

    # Check specific values from different parts of the table
    assert df.iloc[0]["id"] == 1
    assert df.iloc[19]["id"] == 20
    assert df.iloc[5]["name"] == "Person 6"
    assert abs(float(df.iloc[9]["score"]) - 31.42) < 0.01

    # Check data ranges and statistics
    assert df["id"].min() == 1
    assert df["id"].max() == 20
    assert len(df["name"].unique()) == 20
    assert df["join_year"].max() == 2020
