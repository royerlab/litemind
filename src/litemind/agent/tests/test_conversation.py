from litemind.agent.messages.conversation import Conversation
from litemind.agent.messages.message import Message


def test_conversation():
    conversation = Conversation()

    system_message = Message(
        role="system", text="You are an omniscient all-knowing being called Ohmm"
    )
    conversation.append(system_message)

    # Check += operator:
    conversation += Message(role="user", text="Who are you?")

    assert len(conversation) == 2

    messages = conversation.get_all_messages()
    assert messages[0].role == "system"
    assert "You are an omniscient all-knowing being called Ohmm" in messages[0]
    assert messages[1].role == "user"
    assert "Who are you?" in messages[1]

    conversation.clear_all()
    assert len(conversation) == 0
