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

        # Add text content, if any
        if message.text:
            formatted_message["content"].append(
                {"type": "text", "text": message.text})

        # Add each image URL in the required format
        for image_uri in message.image_uris:

            # OpenAI API requires the image to be a valid remote URL or in base64 format:

            # Determine media type:
            media_type = get_media_type_from_uri(image_uri)

            if image_uri.startswith("data:image/"):
                # All good
                pass

            elif image_uri.startswith("http://") or image_uri.startswith(
                    "https://"):
                # All good
                pass

            elif image_uri.startswith("file://"):
                # If it's a local file path, read the file and convert to base64:
                local_path = image_uri.replace("file://", "")
                base64_data = read_file_and_convert_to_base64(local_path)
                image_uri = base64_to_data_uri(base64_data, media_type)

            else:
                # raise exception:
                raise ValueError(
                    f"Invalid image URI: '{image_uri}' (must start with 'data:image/', 'http://', 'https://', or 'file://')")

            formatted_message["content"].append({
                "type": "image_url",
                "image_url": {"url": image_uri}
            })

        # Append the formatted message to the list
        openai_formatted_messages.append(formatted_message)

    return openai_formatted_messages
