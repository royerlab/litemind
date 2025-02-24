import os

from arbol import aprint

from litemind.apis.providers.openai.openai_api import OpenAIApi


def is_openai_api_key_available() -> bool:
    # check is OpenAI API key is available in the environment:
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key is None:
        aprint("OpenAI API key is not available in the environment!")
        return False

    # check if the OpenAI API key is valid:
    openai = OpenAIApi(api_key=openai_api_key)
    is_valid = openai.check_availability_and_credentials()
    if is_valid:
        aprint("OpenAI API key is valid.")
    else:
        aprint("OpenAI API key is invalid.")

    return is_valid
