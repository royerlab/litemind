def extract_outermost_markdown_block(text):
    """
    Extract content between the first ```markdown (or ```md) and the last ``` in the text.

    Parameters:
    -----------
    text : str
        The text containing markdown blocks

    Returns:
    --------
    str
        The extracted content between markdown delimiters, or the original text if no block found
    """
    # Find the start of the markdown block
    start_markers = ["```markdown", "```md"]
    start_index = -1
    marker_length = 0

    for marker in start_markers:
        pos = text.find(marker)
        if pos != -1 and (start_index == -1 or pos < start_index):
            start_index = pos
            marker_length = len(marker)

    if start_index == -1:
        return text  # No markdown block found

    # Find the last occurrence of ``` (end marker)
    end_index = text.rfind("```")

    if end_index <= start_index + marker_length:
        return text  # No valid end marker found after start marker

    # Extract the content between markers (excluding the markers themselves)
    content_start = start_index + marker_length
    content_end = end_index

    return text[content_start:content_end].strip()
