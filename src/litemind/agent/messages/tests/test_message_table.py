import pandas as pd
import pytest

from litemind.agent.messages.message import Message
from litemind.agent.messages.message_block import MessageBlock
from litemind.media.types.media_table import Table


def test_append_table_with_dataframe():
    # Create a DataFrame and append it as a table
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    message = Message(role="user")
    message.append_table(df)

    # Verify that the appended block is a Table and content is the same
    table_block = message.blocks[-1]
    assert isinstance(table_block.media, Table)
    loaded_df = table_block.media.to_dataframe()
    pd.testing.assert_frame_equal(loaded_df, df)


def test_append_table_with_list():
    # Create an array-like table using a list
    data = [[1, 2, 3], [4, 5, 6]]
    df = pd.DataFrame(data)
    message = Message(role="user")
    message.append_table(data)

    # Verify that the appended block is a Table and the conversion is correct
    table_block = message.blocks[-1]
    assert isinstance(table_block.media, Table)
    loaded_df = table_block.media.to_dataframe()
    pd.testing.assert_frame_equal(loaded_df, df)


def test_append_table_invalid_type():
    # Pass an invalid type (integer) and verify that ValueError is raised
    message = Message(role="user")
    with pytest.raises(ValueError):
        message.append_table(123)


def test_append_table_with_csv_uri(tmp_path):
    # Create a temporary CSV file
    csv_content = "A,B\n1,2\n3,4"
    file_path = tmp_path / "test.csv"
    file_path.write_text(csv_content)
    uri = "file://" + str(file_path)

    # Append the table using the CSV file URI
    message = Message(role="user")
    message.append_table(uri)

    table_block = message.blocks[-1]
    assert isinstance(table_block.media, Table)

    # Verify that table data is loaded correctly from the CSV file
    loaded_df = table_block.media.to_dataframe()
    expected_df = pd.read_csv(file_path)
    pd.testing.assert_frame_equal(loaded_df, expected_df)


def test_message_append_table_as_text():
    # Create a user message
    user_message = Message(role="user")

    # Append a text block before appending the table as text
    user_message.append_text("Can you describe what you see in the table?")

    # Create a pandas DataFrame as sample table
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})

    # Append table as text block
    block = user_message.append_table_as_text(df)

    # Ensure the returned block is a MessageBlock object
    assert isinstance(block, MessageBlock)

    # Validate that the markdown string contains a markdown fence indicating a DataFrame
    content = block.get_content()
    assert "```dataframe" in content
    assert "|   A" in content  # Check for header indicating column A present
