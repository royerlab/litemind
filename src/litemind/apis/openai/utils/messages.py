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
    openai_formatted_messages = []

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
        for image_url in message.image_urls:
            formatted_message["content"].append({
                "type": "image_url",
                "image_url": {"url": image_url}
            })

        # Append the formatted message to the list
        openai_formatted_messages.append(formatted_message)

    return openai_formatted_messages
