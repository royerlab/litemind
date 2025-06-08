from datetime import datetime, timedelta

import pytest

from litemind import API_IMPLEMENTATIONS
from litemind.agent.agent import Agent
from litemind.agent.tools.toolset import ToolSet
from litemind.apis.model_features import ModelFeatures

# Take API_IMPLEMENTATIONS and remove OllamaApi if present:
# Ollama is a bit slow and the tests take for ever...
API_IMPLEMENTATIONS_ADV_AGENT_TESTS = [
    api for api in API_IMPLEMENTATIONS if api.__name__ != "OllamaApi"
]


def get_current_date() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def multiply_by_two(x: int) -> int:
    return x * 2


def add_days_to_date(date_str: str, days: int) -> str:
    date = datetime.strptime(date_str, "%Y-%m-%d")
    new_date = date + timedelta(days=int(days))
    return new_date.strftime("%Y-%m-%d")


def get_day_of_week(year: int, month: int, day: int) -> str:
    """
    Returns the day of the week for a given date.

    Args:
        year: The year (e.g., 2025)
        month: The month (1-12)
        day: The day of month (1-31)

    Returns:
        The day of the week as a string (e.g., 'Monday', 'Tuesday', etc.)
    """
    date = datetime(int(year), int(month), int(day))
    # %A gives the full weekday name
    return date.strftime("%A")


@pytest.mark.parametrize("api_class", API_IMPLEMENTATIONS)
def test_agent_single_tool(api_class):
    # Initialize API
    api = api_class()

    # Get model name that supports text generation and tools:
    model_name = api.get_best_model(
        features=[ModelFeatures.TextGeneration, ModelFeatures.Tools]
    )

    # Check that a model name was returned, if not then skip test:
    if not model_name:
        pytest.skip("No model supports text generation and tools. Skipping test.")

    # Initialize ToolSet
    toolset = ToolSet()

    # Add a tool that returns the current date
    toolset.add_function_tool(get_current_date, "Returns the current date")

    # Create the agent
    agent = Agent(
        api=api,
        model_name=model_name,
        toolset=toolset,
        temperature=0.0,
    )

    # Use the agent to determine the date 15 days later
    response = agent("What date is today?")

    # Calculate the expected date day:
    expected_date = datetime.now().strftime("%Y-%m-%d")

    # Figure out the year, month and day in this format: 2025, March, 9
    year, month, day = expected_date.split("-")
    month = datetime.strptime(month, "%m").strftime("%B")

    # Remove leading zeros from the day
    if "0" in day:
        day = day.replace("0", "")

    # Convert day to int:
    day = int(day)

    # Convert response (list of messages) to string:
    response = str(response)

    # Check the response
    assert response is not None
    assert (expected_date in response) or (
        year in response
        and month in response
        and (
            str(day - 1) in response or str(day) in response or str(day + 1) in response
        )
    )


@pytest.mark.parametrize("api_class", API_IMPLEMENTATIONS_ADV_AGENT_TESTS)
def test_agent_multiple_tools(api_class):
    # Initialize API
    api = api_class()

    # If no model supports text generation and tools, skip the test
    if not api.has_model_support_for(
        features=[ModelFeatures.TextGeneration, ModelFeatures.Tools]
    ):
        pytest.skip("No model supports text generation and tools. Skipping test.")

    # Initialize ToolSet
    toolset = ToolSet()

    # Add tools
    toolset.add_function_tool(get_current_date, "Returns the current date")
    toolset.add_function_tool(multiply_by_two, "Multiply a number by 2")
    toolset.add_function_tool(
        add_days_to_date, "Add a certain number of days to a given date"
    )

    # Create the agent
    agent = Agent(api=api, toolset=toolset, temperature=0.0)

    # Use the agent to determine the date 15 days later
    response = agent("What date is 15 days from today?")
    expected_date = (datetime.now() + timedelta(days=15)).strftime("%Y-%m-%d")
    year, month, day = expected_date.split("-")
    month = datetime.strptime(month, "%m").strftime("%B")

    # Remove leading zeros from the day:
    if "0" in day:
        day = day.replace("0", "")

    # Convert day to int:
    day = int(day)

    # Convert response (list of messages) to string:
    response = str(response)

    assert response is not None
    assert (expected_date in response) or (
        year in response
        and month in response
        and (
            str(day - 1) in response or str(day) in response or str(day + 1) in response
        )
    )

    # Use the agent to multiply a number by 2
    response = agent("What is 21 multiplied by 2?")
    assert response is not None
    assert "42" in str(response)


@pytest.mark.parametrize("api_class", API_IMPLEMENTATIONS_ADV_AGENT_TESTS)
def test_agent_chained_tools(api_class):
    # Initialize API
    api = api_class()

    # If no model supports text generation and tools, skip the test
    if not api.has_model_support_for(
        features=[ModelFeatures.TextGeneration, ModelFeatures.Tools]
    ):
        pytest.skip("No model supports text generation and tools. Skipping test.")

    # Initialize ToolSet
    toolset = ToolSet()

    # Add tools
    toolset.add_function_tool(get_current_date, "Returns the current date")
    toolset.add_function_tool(
        add_days_to_date, "Add a certain number of days to a given date"
    )

    # Create the agent
    agent = Agent(api=api, toolset=toolset, temperature=0.0)

    # Use the agent to determine the date 15 days from today using chained tools
    response = agent("What date is 15 days from today?")

    # Calculate the expected date day:
    expected_date = (datetime.now() + timedelta(days=15)).strftime("%Y-%m-%d")
    year, month, day = expected_date.split("-")
    month = datetime.strptime(month, "%m").strftime("%B")

    # Remove leading zeros from the day:s
    if "0" in day:
        day = day.replace("0", "")

    # Convert day to int:
    day = int(day)

    # Convert response (list of messages) to string:
    response = str(response)

    assert response is not None
    assert (expected_date in response) or (
        year in response
        and month in response
        and (
            str(day - 1) in response or str(day) in response or str(day + 1) in response
        )
    )

    # Add get_day_of_week to toolset:
    toolset.add_function_tool(
        get_day_of_week, "Get the day of the week for a given date"
    )

    # Use the agent to determine the date 15 days from today using chained tools
    response = agent("Which day of the week is that?")

    # Check if the response contains a day fo the week:
    assert response is not None

    response_str = str(response)

    assert (
        "Monday" in response_str
        or "Tuesday" in response_str
        or "Wednesday" in response_str
        or "Thursday" in response_str
        or "Friday" in response_str
        or "Saturday" in response_str
        or "Sunday" in response_str
    )


@pytest.mark.parametrize("api_class", API_IMPLEMENTATIONS_ADV_AGENT_TESTS)
def test_agent_longer_dialog(api_class):
    # Initialize API
    api = api_class()

    # If no model supports text generation and tools, skip the test
    if not api.has_model_support_for(
        features=[ModelFeatures.TextGeneration, ModelFeatures.Tools]
    ):
        pytest.skip("No model supports text generation and tools. Skipping test.")

    # Get name of best model that supports text generation and tools:
    model_name = api.get_best_model(
        features=[ModelFeatures.TextGeneration, ModelFeatures.Tools]
    )

    # Initialize ToolSet
    toolset = ToolSet()

    # Add tools
    toolset.add_function_tool(get_current_date, "Returns the current date.")
    toolset.add_function_tool(
        add_days_to_date,
        "Add a certain number of days (positive or negative) to a given date.",
    )
    toolset.add_function_tool(multiply_by_two, "Multiply a number by 2.")

    # Create the agent
    agent = Agent(
        api=api,
        model_name=model_name,
        toolset=toolset,
        temperature=0.0,
    )

    # Use the agent to determine the date 15 days later
    response1 = agent("What date is 15 days from today?")

    # Calculate the expected date day:
    expected_date = (datetime.now() + timedelta(days=15)).strftime("%Y-%m-%d")

    # Figure out the year, month and day in this format: 2025, March, 9
    year, month, day = expected_date.split("-")
    month = datetime.strptime(month, "%m").strftime("%B")

    # Remove leading zeros from the day
    if "0" in day:
        day = day.replace("0", "")

    # Convert day to int:
    day = int(day)

    # Convert response (list of messages) to string:
    response1 = str(response1)

    # Check the response:
    assert response1 is not None
    assert (expected_date in response1) or (
        year in response1
        and month in response1
        and (
            str(day - 1) in response1
            or str(day) in response1
            or str(day + 1) in response1
        )
    )

    # Use the agent to multiply a number by 2
    response2 = agent("What is 21 multiplied by 2?")

    # Check the response:
    assert response2 is not None
    assert "42" in str(response2)

    # Use the agent to determine the current date
    response3 = agent("What is the double of the current date day number?")

    # Check the response:
    assert response3 is not None

    # Calculate the expected date day:
    todays_date = (datetime.now()).strftime("%Y-%m-%d")
    year, month, day = todays_date.split("-")

    # Convert response (list of messages) to string:
    response3 = str(response3)

    # Check the response:
    assert str(int(day) * 2) in response3

    # Final question
    response3 = agent("Do you know what happened exactly 5 years ago (365*5 days ago)?")

    # Check the response:
    assert response3 is not None

    print(response3)
