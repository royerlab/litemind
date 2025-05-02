def _get_gemini_models_list():
    # Get the full list of models:
    model_list = []
    from google.generativeai import (
        list_models,
    )  # This is the function from your snippet

    for model_obj in list_models():
        # model_obj is a Model protobuf (or typed dict) with a .name attribute
        # e.g. "models/gemini-1.5-flash", "models/gemini-2.0-flash-exp", etc.
        if (
            ("gemini" in model_obj.name.lower() or "imagen" in model_obj.name.lower())
            and not "will be discontinued" in model_obj.description.lower()
            and not "deprecated" in model_obj.description.lower()
            and not "live" in model_obj.name.lower()
        ):
            model_list.append(model_obj.name)

    # FIX: For some reason this model is not listed:
    # Add 'models/text-embedding-004' to the end of the list:
    model_list.append("models/text-embedding-004")

    # Reverse the list so that the best models are at the beginning:
    model_list.reverse()

    return model_list
