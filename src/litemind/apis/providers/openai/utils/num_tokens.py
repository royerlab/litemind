def estimate_num_input_tokens(
    max_num_input_tokens, max_num_output_tokens, preprocessed_messages
):
    # Estimate the number of input tokens:
    estimated_num_input_tokens = (
        sum([len(str(message).split()) for message in preprocessed_messages]) * 3
    )

    # Adjust the number of completion tokens accordingly:
    effective_max_output_tokens = min(
        max_num_output_tokens,
        max_num_input_tokens - estimated_num_input_tokens,
    )

    return effective_max_output_tokens
