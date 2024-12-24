from litemind.agent.conversation import Conversation
from litemind.agent.message import Message


def test_conversation():

    conversation = Conversation()

    system_message = Message(role='system', text='You are an omniscient all-knowing being called Ohmm')
    conversation.append(system_message)

    # Check += operator:
    conversation += Message(role='user', text='Who are you?')

    assert len(conversation) == 2

    messages = conversation.get_all_messages()
    assert messages[0].role == 'system'
    assert messages[0].text == 'You are an omniscient all-knowing being called Ohmm'
    assert messages[1].role == 'user'
    assert messages[1].text == 'Who are you?'

    conversation.clear_all()
    assert len(conversation) == 0













