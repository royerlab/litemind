from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from litemind import API_IMPLEMENTATIONS
from litemind.apis.base_api import ModelFeatures
from litemind.apis.tests.utils.levenshtein import levenshtein_distance
from litemind.ressources.media_resources import MediaResources
from litemind.utils.normalise_uri_to_local_file_path import uri_to_local_file_path


@pytest.mark.parametrize("api_class", API_IMPLEMENTATIONS)
class TestBaseApiImplementationsGenerateMultimodal(MediaResources):
    """
    A tests suite that runs the same tests on each ApiClass
    implementing the abstract BaseApi interface.
    These tests are for the generate methods of the API.
    """

    def test_generate_audio(self, api_class):
        # Test audio generation with generate_audio:
        api_instance = api_class()

        # Get the best audio generation model:
        audio_gen_model_name = api_instance.get_best_model(
            ModelFeatures.AudioGeneration
        )

        # Skip tests if the model does not support audio generation:
        if audio_gen_model_name is None:
            pytest.skip(
                f"{api_class.__name__} does not support audio generation. Skipping tests."
            )

        print("Audio generation model name: ", audio_gen_model_name)

        # Define the text to generate audio from:
        text = "Success is the ability to go from one failure to another with no loss of enthusiasm"

        # Print the text:
        print(f"Text to generate audio from: '{text}'")

        # Generate the audio:
        generated_audio_uri = api_instance.generate_audio(
            model_name=audio_gen_model_name,
            text=text,
        )

        # save URI to file:
        generated_audio_file = uri_to_local_file_path(generated_audio_uri)

        # Convert the path string to a file to check its properties:
        generated_audio_file = Path(generated_audio_file)

        # Check that the file exists:
        assert generated_audio_file.exists(), "The generated audio file should exist."
        # Check the audio file length:
        assert (
            generated_audio_file.stat().st_size > 0
        ), "The generated audio file should not be empty."
        # Check that it is an MP3 file:
        assert (
            generated_audio_file.suffix == ".mp3"
        ), "The generated audio file should be an MP3 file."

        # Transcribe the audio to check its contents:
        if api_instance.has_model_support_for(
            model_name=audio_gen_model_name, features=ModelFeatures.AudioConversion
        ):
            # Transcribe audio:
            transcription = api_instance.transcribe_audio(generated_audio_uri)

            # compute the edit distance between the two strings:

            edit_distance = levenshtein_distance(text, transcription)

            # Make sure that the distance is not more than the number of words in the text:
            assert edit_distance <= len(
                text.split()
            ), "The edit distance between the generated audio and the original text should be small."

            print(f"Transcription: \n'{transcription}'")

    def test_generate_image(self, api_class):
        api_instance = api_class()

        # Get the best image generation model:
        image_gen_model_name = api_instance.get_best_model(
            ModelFeatures.ImageGeneration
        )

        # Skip tests if the model does not support image generation:å
        if not api_instance.has_model_support_for(
            model_name=image_gen_model_name, features=ModelFeatures.ImageGeneration
        ):
            pytest.skip(
                f"{api_class.__name__} does not support image generation. Skipping tests."
            )

        print("Image generation model name: ", image_gen_model_name)

        # Define the prompt:å
        prompt = "Please draw me a white siamese cat"

        # Define the image dimensions:
        image_width = 1024
        image_height = 1024

        # Generate the image
        generated_image = api_instance.generate_image(
            model_name=image_gen_model_name,
            positive_prompt=prompt,
            image_width=image_width,
            image_height=image_height,
            preserve_aspect_ratio=True,
            allow_resizing=True,
        )

        assert isinstance(
            generated_image, Image.Image
        ), "The generated image should be a PIL Image instance."

        # Save the generated image to a BytesIO object
        image_bytes = BytesIO()
        generated_image.save(image_bytes, format="PNG")
        image_bytes.seek(0)

        # Save image to temp PNG file using tempfile:
        import tempfile

        temp_image_path = tempfile.NamedTemporaryFile(suffix=".png").name
        generated_image.save(temp_image_path, format="PNG")

        # Convert filepath to URI:
        image_uri = "file://" + temp_image_path

        # Get model with image support:
        image_model_name = api_instance.get_best_model(ModelFeatures.Image)
        print("Image model name: ", image_model_name)

        # If describe_image is available, use it to check the content of the generated image
        if image_model_name and api_instance.has_model_support_for(
            model_name=image_model_name, features=ModelFeatures.Image
        ):
            # Ask model to describe image:
            description = api_instance.describe_image(
                model_name=image_model_name, image_uri=image_uri
            )

            print(f"Description of the generated image: \n{description}")

            # Asserts:
            assert isinstance(description, str), "The description should be a string."
            assert len(description) > 0, "The description should not be empty."
            assert (
                "cat" in description.lower()
            ), "The description should mention a 'cat'."
            assert (
                "white" in description.lower() or "blue" in description.lower()
            ), "The description should mention 'white' or 'blue'."
