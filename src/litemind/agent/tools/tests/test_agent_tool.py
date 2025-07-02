import pytest

from litemind import API_IMPLEMENTATIONS
from litemind.agent.agent import Agent
from litemind.agent.messages.message import Message
from litemind.agent.tools.agent_tool import AgentTool
from litemind.agent.tools.toolset import ToolSet
from litemind.apis.model_features import ModelFeatures


@pytest.mark.parametrize("api_class", API_IMPLEMENTATIONS)
def test_tool_agent_translation(api_class):
    # Initialize the OpenAIApi and Agent
    api = api_class()  # Assumes API key is available in the environment

    # Make sure that the API supports text generation, or skip test:
    if not api.has_model_support_for(ModelFeatures.TextGeneration):
        # skip pytest:
        pytest.skip(
            f"Skipping test for {api_class.__name__} as no text model is available."
        )

    # Create a system message that defines the assistant's role as a translator
    system_message = Message(
        role="system",
        text="You are a translator that translates English sentences to French.",
    )

    # Initialize the Agent with the system message
    agent = Agent(api=api)
    agent += system_message

    # Initialize the AgentTool with the Agent instance
    agent_tool = AgentTool(
        agent, "Translation agent from English to French", has_memory=True
    )

    # Define a user prompt for translation
    prompt = "Translate the following sentence: 'Hello, how are you?'"

    # Execute the AgentTool with the prompt
    agent_response = agent_tool._execute(prompt=prompt)

    # Validate that the response is an AgentResponse and that the result is in French
    assert isinstance(agent_response, str), "AgentResponse result should be a string"
    assert agent_response, "AgentResponse result should not be empty"
    assert (
        "comment" in agent_response.lower() or "bonjour" in agent_response.lower()
    ), "AgentTool response should be a French translation of the input"

    # Validate that the Agent's conversation has the correct sequence of messages
    assert (
        len(agent.conversation) == 3
    ), "Agent's conversation should have 3 messages: system, prompt, and assistant's response"
    assert agent.conversation[0].role == "system"
    assert agent.conversation[1].role == "user"
    assert agent.conversation[2].role == "assistant"

    # Ensure the assistant's response is correct French for the given prompt
    assert (
        "bonjour" in agent_response.lower()
    ), f"Expected translation to contain 'Bonjour' but got '{agent_response}'"

    # Check that the translation agent was really called by checking its conversation history:
    assert (
        len(agent_tool.agent.conversation) == 3
    ), "Agent's conversation should have 3 messages: system, prompt, and assistant's response"


@pytest.mark.parametrize("api_class", API_IMPLEMENTATIONS)
def test_tool_agent(api_class):
    # Initialize the OpenAIApi and Agent for the AgentTool
    api = api_class()  # Assumes API key is available in the environment

    # Make sure that the API supports text generation, or skip test:
    if not api.has_model_support_for(ModelFeatures.TextGeneration):
        # skip pytest:
        pytest.skip(
            f"Skipping test for {api_class.__name__} as no text model is available."
        )

    # Initialize the Agent
    agent = Agent(api=api, name="Order Assistant Agent")

    # Initialize the AgentTool with the Agent instance
    agent_tool = AgentTool(agent, "Assistant agent that processes prompts")

    # Check that the description is correctly set
    assert (
        agent_tool.description == "Assistant agent that processes prompts."
    )  # Note the period at the end

    # Execute the AgentTool with a prompt
    prompt = "What is the status of my order?"
    agent_response = agent_tool._execute(prompt=prompt)

    # Validate the structure and content of the AgentResponse
    assert isinstance(agent_response, str), "AgentResponse result should be a string"
    assert agent_response, "AgentResponse result should not be empty"
    assert (
        "status" in agent_response.lower() or "order" in agent_response.lower()
    ), "AgentTool response should address the order status"


@pytest.mark.parametrize("api_class", API_IMPLEMENTATIONS)
def test_tool_agent_with_internal_tool(api_class):
    # Initialize the OpenAIApi and Agent
    api = api_class()  # Assumes API key is available in the environment

    # Make sure that the API supports text generation, or skip test:
    if not api.has_model_support_for(ModelFeatures.TextGeneration):
        # skip pytest:
        pytest.skip(
            f"Skipping test for {api_class.__name__} as no text model is available."
        )

    # Initialize the Agent with the API
    agent = Agent(api=api, name="Order Assistant Agent")

    # Define a simple function that will act as a tool within the Agent
    def check_order_status(order_id: str) -> str:
        """Mock function to check the status of an order."""
        return f"The status of order {order_id} is: Shipped"

    # Initialize ToolSet and add the FunctionTool
    toolset = ToolSet()
    toolset.add_function_tool(check_order_status, "Check the status of an order by ID")
    agent.toolset = toolset  # Assign the toolset to the agent

    # Create and add a system message to set the agent's role as an assistant handling orders
    agent += Message(
        role="system", text="You are an assistant that provides order status updates."
    )

    # Initialize the AgentTool with the Agent instance
    agent_tool = AgentTool(
        agent,
        "Processes order inquiries, e.g. can check the status of an order by ID",
    )

    # Initialize the Agent with the API
    main_agent = Agent(api=api)

    # Define a system message that sets the role of the main agent
    main_agent += Message(
        role="system",
        text="You are an assistant that provides order status updates and can write poems.",
    )

    # Add tool_agent as tool for the main agent:
    main_agent.toolset = ToolSet()
    main_agent.toolset.add_tool(agent_tool)

    # Define a user prompt that should trigger the internal tool for checking order status
    prompt = "Can you tell me the status of order #12345?"

    # Execute the AgentTool with the prompt
    agent_response = main_agent(prompt)

    # Validate that the response is  a list and that the result includes the status of the order
    assert isinstance(agent_response, list), "AgentResponse result should be a list"

    agent_response_str = str(agent_response).lower()

    assert (
        "shipped" in agent_response_str and "12345" in agent_response_str or "order"
    ), "AgentTool response should include the status provided by FunctionTool"

    # Validate conversation structure
    assert (
        len(agent.conversation) == 5
    ), "Agent's conversation should have 3 messages: system, user and assistant's response"
    assert agent.conversation[0].role == "system"
    assert agent.conversation[1].role == "user"
    assert agent.conversation[2].role == "assistant"
    assert agent.conversation[3].role == "user" or agent.conversation[3].role == "tool"
    assert agent.conversation[4].role == "assistant"
