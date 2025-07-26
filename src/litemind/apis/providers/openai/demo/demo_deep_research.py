from arbol import aprint, asection

from litemind import OpenAIApi
from litemind.agent.agent import Agent
from litemind.agent.tools.toolset import ToolSet


def demo_deep_research_main():
    """
    Test deep research agent with OpenAI API.
    """

    # Initialize the OpenAI API:
    api = OpenAIApi()

    # get the list of models:
    models = api.list_models()

    # Print the available OpenAI models (uses the Arbol print library):
    with asection("Available OpenAI Models:"):
        for model in models:
            aprint(model)

    # Get the best model for text generation:
    model_name = "o4-mini-deep-research-medium"

    # Toolset with web search tool:
    toolset = ToolSet()
    toolset.add_builtin_web_search_tool()

    agent = Agent(api, model_name=model_name, toolset=toolset)

    system_message = """
    You are a professional researcher preparing a very small and concise weather reports.
    Don't sweat it. Be quick and to the point. 
    Don't write a lot of text, just the most important information.
    Ideally in a few lines.
    I know, normal reports are long, but this is a very small report!
    No need to search a lot of info!
    """

    # Append system message to the agent:
    agent.append_system_message(system_message)

    # Generate a response from the agent:
    response = agent("What is the weather like in SF?")

    # Print the response:
    for message in response:
        aprint(message.to_plain_text())


if __name__ == "__main__":
    demo_deep_research_main()
