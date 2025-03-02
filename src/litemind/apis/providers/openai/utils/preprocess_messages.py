import copy
from typing import List

from litemind.agent.messages.message import Message


def openai_preprocess_messages(
    model_name: str, messages: List[Message]
) -> List[Message]:
    """Preprocess messages for OpenAI API."""

    # Ensure that the messages are not modified, as we
    preprocessed_messages = copy.deepcopy(messages)

    # o1 models have special needs:
    if "o1" in model_name or "o3" in model_name:

        # ox models do not support system messages, make a copy of messages and recast system messages as user messages:
        for message in preprocessed_messages:
            if message.role == "system":
                message.role = "user"

    return preprocessed_messages
