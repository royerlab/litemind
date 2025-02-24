from datetime import datetime, timedelta

from litemind.agent.react.react_agent import ReActAgent
from litemind.agent.tools.toolset import ToolSet
from litemind.apis.combined_api import CombinedApi
from litemind.apis.model_features import ModelFeatures


def get_current_date():
    return datetime.now().strftime("%Y-%m-%d")


def multiply_by_two(x):
    return x * 2


def add_days_to_date(date_str, days):
    date = datetime.strptime(date_str, "%Y-%m-%d")
    new_date = date + timedelta(days=days)
    return new_date.strftime("%Y-%m-%d")


def test_react_agent_construction_methods():
    # Initialize API
    api = CombinedApi()

    # Initialize ToolSet
    toolset = ToolSet()

    # Add a tool that returns the current date
    toolset.add_function_tool(get_current_date, "Returns the current date")

    # Method 1: Specify model features explicitly
    features = ModelFeatures.Audio
    agent1 = ReActAgent(api=api, toolset=toolset, model_features=features)
    assert api.has_model_support_for(model_name=agent1.model, features=features)

    # Method 2: Provide a model that is explicitly constructed with the right features
    features = [ModelFeatures.TextGeneration, ModelFeatures.Tools, ModelFeatures.Image]
    model = api.get_best_model(features=features)
    agent2 = ReActAgent(api=api, toolset=toolset, model=model)
    assert api.has_model_support_for(model_name=agent2.model, features=features)

    # Method 3: Without specifying features or model, relying on the default
    agent3 = ReActAgent(api=api, toolset=toolset)
    assert api.has_model_support_for(model_name=agent1.model, features=features)


def test_react_agent_single_tool():

    # Initialize API
    api = CombinedApi()

    # Initialize ToolSet
    toolset = ToolSet()

    # Add a tool that returns the current date
    toolset.add_function_tool(get_current_date, "Returns the current date")

    # Create the ReAct agent
    agent = ReActAgent(api=api, toolset=toolset, temperature=0.0, max_reasoning_steps=5)

    # Use the agent to determine the date 15 days later
    response = agent("What date is 15 days from today?")

    # Calculate the expected date day:
    expected_date = (datetime.now() + timedelta(days=15)).strftime("%Y-%m-%d")

    # Figure out the year, month and day in this format: 2025, March, 9
    year, month, day = expected_date.split("-")
    month = datetime.strptime(month, "%m").strftime("%B")

    # Remove leading zeros from the day
    if "0" in day:
        day = day.replace("0", "")

    # Check the response
    assert response is not None
    assert (expected_date in response) or (
        year in response and month in response and day in response
    )


def test_react_agent_multiple_tools():

    # Initialize API
    api = CombinedApi()

    # Initialize ToolSet
    toolset = ToolSet()

    # Add tools
    toolset.add_function_tool(get_current_date, "Returns the current date")
    toolset.add_function_tool(multiply_by_two, "Multiply a number by 2")

    # Create the ReAct agent
    agent = ReActAgent(api=api, toolset=toolset, temperature=0.0, max_reasoning_steps=5)

    # Use the agent to determine the date 15 days later
    response = agent("What date is 15 days from today?")
    expected_date = (datetime.now() + timedelta(days=15)).strftime("%Y-%m-%d")
    year, month, day = expected_date.split("-")
    month = datetime.strptime(month, "%m").strftime("%B")

    if "0" in day:
        day = day.replace("0", "")
    assert response is not None
    assert (expected_date in response) or (
        year in response and month in response and day in response
    )

    # Use the agent to multiply a number by 2
    response = agent("What is 21 multiplied by 2?")
    assert response is not None
    assert "42" in response


def test_react_agent_chained_tools():

    # Initialize API
    api = CombinedApi()

    # Initialize ToolSet
    toolset = ToolSet()

    # Add tools
    toolset.add_function_tool(get_current_date, "Returns the current date")
    toolset.add_function_tool(
        add_days_to_date, "Add a certain number of days to a given date"
    )

    # Create the ReAct agent
    agent = ReActAgent(api=api, toolset=toolset, temperature=0.0, max_reasoning_steps=5)

    # Use the agent to determine the date 15 days from today using chained tools
    response = agent("What date is 15 days from today?")
    expected_date = (datetime.now() + timedelta(days=15)).strftime("%Y-%m-%d")
    year, month, day = expected_date.split("-")
    month = datetime.strptime(month, "%m").strftime("%B")
    if "0" in day:
        day = day.replace("0", "")
    assert response is not None
    assert (expected_date in response) or (
        year in response and month in response and day in response
    )


def test_react_agent_longer_dialog():

    # Initialize API
    api = CombinedApi()

    # Initialize ToolSet
    toolset = ToolSet()

    # Add tools
    toolset.add_function_tool(get_current_date, "Returns the current date")
    toolset.add_function_tool(multiply_by_two, "Multiply a number by 2")

    # Create the ReAct agent
    agent = ReActAgent(api=api, toolset=toolset, temperature=0.0, max_reasoning_steps=5)

    # Use the agent to determine the date 15 days later
    response1 = agent("What date is 15 days from today?")

    # Calculate the expected date day:
    expected_date = (datetime.now() + timedelta(days=15)).strftime("%Y-%m-%d")

    # Figure out the year, month and day in this format: 2025, March, 9
    year, month, day = expected_date.split("-")
    month = datetime.strptime(month, "%m").strftime("%B")
    if "0" in day:
        day = day.replace("0", "")

    # Check the response:
    assert response1 is not None
    assert (expected_date in response1) or (
        year in response1 and month in response1 and day in response1
    )

    # Use the agent to multiply a number by 2
    response2 = agent("What is 21 multiplied by 2?")

    # Check the response:
    assert response2 is not None
    assert "42" in response2

    # Use the agent to determine the current date
    response3 = agent("What is the double of the current date day number?")

    # Check the response:
    assert response3 is not None

    # Calculate the expected date day:
    todays_date = (datetime.now()).strftime("%Y-%m-%d")
    year, month, day = todays_date.split("-")
    assert str(int(day) * 2) in response3
