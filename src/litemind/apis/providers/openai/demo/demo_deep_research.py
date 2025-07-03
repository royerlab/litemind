from litemind import OpenAIApi
from litemind.agent.messages.message import Message
from litemind.agent.tools.toolset import ToolSet


def demo():
    """
    Test a simple completion with the default or a known good model.
    """
    # Initialize the OpenAI API:
    api = OpenAIApi()

    # get the list of models:
    models = api.list_models()

    # Print the available models:
    for model in models:
        print(model)

    # Get the best model for text generation:
    model_name = "o4-mini-deep-research-medium"

    system_message = """
    You are a professional researcher preparing a very small and concise weather reports.
    Don't sweat it. Be quick and to the point. 
    Don't write a lot of text, just the most important information.
    Ideally in a few lines.
    I know, normal reports are long, but this is a very small report!
    No need to search a lot of info!
    """

    user_query = "What is the weather like in SF?"

    # A simple message:
    messages = [
        Message(
            role="system",
            text=system_message,
        ),
        Message(role="user", text=user_query),
    ]

    toolset = ToolSet()
    toolset.add_builtin_web_search_tool()

    # Get the completion:
    response = api.generate_text(
        model_name=model_name, messages=messages, toolset=toolset, stream=False
    )

    # Print the response:
    for message in response:
        print(message.to_plain_text())


if __name__ == "__main__":
    demo()
