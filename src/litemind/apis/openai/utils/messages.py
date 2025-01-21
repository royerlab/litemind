from litemind.agent.message_block_type import BlockType
from litemind.apis.utils.get_media_type_from_uri import get_media_type_from_uri
from litemind.apis.utils.read_file_and_convert_to_base64 import \
    read_file_and_convert_to_base64, base64_to_data_uri


def convert_messages_for_openai(messages):
    """
    Convert a list of Message objects into a format compatible with OpenAI's API.

    Parameters
    ----------
    messages : list of Message
        The list of Message objects to convert.

    Returns
    -------
    list of dict
        A list of dictionaries formatted for OpenAI's API.
    """

    # Initialize the list of formatted messages:
    openai_formatted_messages = []

    # Iterate over each message:
    for message in messages:

        # Start with the role of the message
        formatted_message = {
            "role": message.role,
            "content": []
        }

        # Iterate over each block in the message
        for block in message.blocks:
            if block.block_type == BlockType.Text:
                formatted_message["content"].append(
                    {"type": "text", "text": block.content})
            elif block.block_type == BlockType.Image:
                image_uri = block.content
                media_type = get_media_type_from_uri(image_uri)
                if image_uri.startswith("file://"):
                    local_path = image_uri.replace("file://", "")
                    base64_data = read_file_and_convert_to_base64(local_path)
                    image_uri = base64_to_data_uri(base64_data, media_type)
                formatted_message["content"].append(
                    {"type": "image_url", "image_url": {"url": image_uri}})
            elif block.block_type == BlockType.Audio:
                audio_uri = block.content
                media_type = get_media_type_from_uri(audio_uri)
                if audio_uri.startswith("file://"):
                    local_path = audio_uri.replace("file://", "")
                    base64_data = read_file_and_convert_to_base64(local_path)
                    audio_uri = base64_to_data_uri(base64_data, media_type)
                formatted_message["content"].append({"type": "input_audio",
                                                     "input_audio": {
                                                         "data": audio_uri,
                                                         "format":
                                                             media_type.split(
                                                                 '/')[1]}})
            # Add more block types as needed

        # Append the formatted message to the list
        openai_formatted_messages.append(formatted_message)

    return openai_formatted_messages
