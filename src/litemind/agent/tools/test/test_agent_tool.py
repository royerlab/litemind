import pytest

from litemind.agent.agent import Agent
from litemind.agent.message import Message
from litemind.agent.tools.agent_tool import AgentTool
from litemind.agent.tools.toolset import ToolSet
from litemind.apis.openai.openai_api import OpenAIApi
from litemind.apis.openai.utils.openai_api_key import \
    is_openai_api_key_available


@pytest.mark.skipif(not is_openai_api_key_available(),
                    reason="requires OpenAI API key to run")
def test_agent_tool_translation():
    # Initialize the OpenAIApi and Agent
    api = OpenAIApi()  # Assumes API key is available in the environment

    # Create a system message that defines the assistant's role as a translator
    system_message = Message(role='system',
                             text="You are a translator that translates English sentences to French.")

    # Initialize the Agent with the system message
    agent = Agent(api=api)
    agent += system_message

    # Initialize the AgentTool with the Agent instance
    agent_tool = AgentTool(agent, "Translation agent from English to French")

    # Define a user prompt for translation
    prompt = "Translate the following sentence: 'Hello, how are you?'"

    # Execute the AgentTool with the prompt
    agent_response = agent_tool.execute(prompt=prompt)

    # Validate that the response is an AgentResponse and that the result is in French
    assert isinstance(agent_response,
                      str), "AgentResponse result should be a string"
    assert agent_response, "AgentResponse result should not be empty"
    assert "comment" in agent_response.lower() or "bonjour" in agent_response.lower(), \
        "AgentTool response should be a French translation of the input"

    # Validate that the Agent's conversation has the correct sequence of messages
    assert len(
        agent.conversation) == 3, "Agent's conversation should have 3 messages: system, prompt, and assistant's response"
    assert agent.conversation[0].role == 'system'
    assert agent.conversation[1].role == 'user'
    assert agent.conversation[2].role == 'assistant'

    # Ensure the assistant's response is correct French for the given prompt
    assert 'bonjour' in agent_response.lower(), \
        f"Expected translation to contain 'Bonjour' but got '{agent_response}'"


# Test for AgentTool functionality, using an actual OpenAI API call
@pytest.mark.skipif(not is_openai_api_key_available(),
                    reason="requires OpenAI API key to run")
def test_agent_tool():
    # Initialize the OpenAIApi and Agent for the AgentTool
    api = OpenAIApi()  # Assumes API key is available in the environment
    agent = Agent(api=api)

    # Initialize the AgentTool with the Agent instance
    agent_tool = AgentTool(agent, "Assistant agent that processes prompts")

    # Check that the description is correctly set
    assert agent_tool.description == "Assistant agent that processes prompts"

    # Execute the AgentTool with a prompt
    prompt = "What is the status of my order?"
    agent_response = agent_tool.execute(prompt=prompt)

    # Validate the structure and content of the AgentResponse
    assert isinstance(agent_response,
                      str), "AgentResponse result should be a string"
    assert agent_response, "AgentResponse result should not be empty"
    assert "status" in agent_response.lower() or "order" in agent_response.lower(), \
        "AgentTool response should address the order status"


@pytest.mark.skipif(not is_openai_api_key_available(),
                    reason="requires OpenAI API key to run")
def test_agent_tool_with_internal_tool():
    # Define a simple function that will act as a tool within the Agent
    def check_order_status(order_id: str) -> str:
        """Mock function to check the status of an order."""
        return f"The status of order {order_id} is: Shipped"

    # Initialize ToolSet and add the FunctionTool
    toolset = ToolSet()
    toolset.add_function_tool(check_order_status,
                              "Check the status of an order by ID")

    # Initialize the OpenAIApi and Agent
    api = OpenAIApi()  # Assumes API key is available in the environment
    agent = Agent(api=api)
    agent.toolset = toolset  # Assign the toolset to the agent

    # Create and add a system message to set the agent's role as an assistant handling orders
    agent += Message(role='system',
                     text="You are an assistant that provides order status updates.")

    # Initialize the AgentTool with the Agent instance
    agent_tool = AgentTool(agent,
                           "Assistant agent that processes order inquiries and has a tool to check order status")

    # Define a user prompt that should trigger the internal tool for checking order status
    prompt = "Can you tell me the status of order #12345?"

    # Execute the AgentTool with the prompt
    agent_response = agent_tool.execute(prompt=prompt)

    # Validate that the response is an AgentResponse and that the result is from the FunctionTool
    assert isinstance(agent_response,
                      str), "AgentResponse result should be a string"
    assert "status of order 12345 is: Shipped" in agent_response, \
        "AgentTool response should include the status provided by FunctionTool"

    # Validate conversation structure
    assert len(
        agent.conversation) == 3, "Agent's conversation should have 3 messages: system, user and assistant's response"
    assert agent.conversation[0].role == 'system'
    assert agent.conversation[1].role == 'user'
    assert agent.conversation[2].role == 'assistant'

    # Ensure the assistant's response includes the expected output from the tool
    expected_status_message = "The status of order 12345 is: Shipped"
    assert expected_status_message in agent_response, \
        f"Expected response to include '{expected_status_message}', but got '{agent_response}'"
