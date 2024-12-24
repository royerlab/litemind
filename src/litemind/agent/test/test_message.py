
from litemind.agent.message import Message


def test_message_text():

    system_message = Message(role='system')
    system_message.append_text('You are an omniscient all-knowing being called Ohmm')

    assert system_message.role == 'system'
    assert system_message.text == 'You are an omniscient all-knowing being called Ohmm'

    user_message = Message(role='user')
    user_message.append_text('Who are you?')

    assert user_message.role == 'user'
    assert user_message.text == 'Who are you?'

    # Checks contains operator:
    assert 'Who' in user_message.text
    assert 'omniscient' in system_message.text


def test_message_image():

    system_message = Message(role='system')
    system_message.append_text('You are an omniscient all-knowing being called Ohmm')

    assert system_message.role == 'system'
    assert system_message.text == 'You are an omniscient all-knowing being called Ohmm'


    user_message = Message(role='user')
    user_message.append_text('Can you describe what you see?')
    user_message.append_image_url('https://upload.wikimedia.org/wikipedia/commons/thumb/3/3e/Einstein_1921_by_F_Schmutzer_-_restoration.jpg/456px-Einstein_1921_by_F_Schmutzer_-_restoration.jpg')

    assert user_message.role == 'user'
    assert user_message.text == 'Can you describe what you see?'
    assert len(user_message.image_urls) == 1
    assert 'upload' in user_message.image_urls[0]

    # Checks contains operator:
    assert 'upload' in user_message







