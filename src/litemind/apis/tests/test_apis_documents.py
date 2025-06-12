from typing import Dict

import pytest

from litemind import API_IMPLEMENTATIONS
from litemind.agent.messages.message import Message
from litemind.apis.base_api import ModelFeatures
from litemind.media.types.media_document import Document
from litemind.media.types.media_image import Image
from litemind.ressources.media_resources import MediaResources


@pytest.mark.parametrize("api_class", API_IMPLEMENTATIONS)
class TestBaseApiImplementationsDocuments(MediaResources):
    """
    A tests suite that runs the same tests on each ApiClass
    implementing the abstract BaseApi interface.
    These tests are for the document methods of the API.
    """

    def test_text_generation_with_pdf_document(self, api_class):

        # If OllamaApi we skip because opensource models are not yet strong enough:
        if api_class.__name__ == "OllamaApi":
            pytest.skip(
                f"{api_class.__name__} does not have strong enough models for this test. Skipping."
            )

        # Get an instance of the api class:
        api_instance = api_class()

        # Get the best model for text generation:
        best_model_name = api_instance.get_best_model(
            features=[ModelFeatures.TextGeneration],
            media_types=[Image, Document],
        )

        # Skip tests if the model does not support text generation:
        if best_model_name is None:
            pytest.skip(
                f"{api_class.__name__} does not support documents. Skipping documents tests."
            )

        print("\n" + best_model_name)

        messages = []

        # System message:
        system_message = Message(role="system")
        system_message.append_text(
            "You are a highly qualified scientist with extensive experience in Microscopy, Biology and Bioimage processing and analysis."
        )
        messages.append(system_message)

        # User message:
        user_message = Message(role="user")
        user_message.append_text(
            "Can you write a review for the provided paper below? Please break down your comments into major and minor comments."
        )
        doc_path = self.get_local_test_document_uri("intracktive_preprint.pdf")
        user_message.append_document(doc_path)

        messages.append(user_message)

        # Run agent:
        response = api_instance.generate_text(
            messages=messages, model_name=best_model_name
        )

        print("\n" + str(response))

        # Get the last, and possibly only, message in response:
        response = response[-1]

        # Make sure that the answer is not empty:
        assert (
            len(response) > 0
        ), f"{api_class.__name__}.completion() should return a non-empty string!"

        # Check response details:
        assert (
            "microscopy" in response.lower()
            or "biology" in response.lower()
            or "lineage" in response.lower()
        )

        # Check if the response is detailed and follows instructions:

        if not ("inTRACKtive" in response and "review" in response.lower()):
            # printout message that warns that the response miight lack in detail:
            print(
                "The response might lack in detail!! Please check the response for more details."
            )

    def test_text_generation_with_word_document(self, api_class):

        # If OllamaApi we skip because opensource models are not yet strong enough:
        if api_class.__name__ == "OllamaApi":
            pytest.skip(
                f"{api_class.__name__} does not have strong enough models for this test. Skipping."
            )

        # Get an instance of the api class:
        api_instance = api_class()

        # Get the best model for text generation:
        best_model_name = api_instance.get_best_model(
            features=[ModelFeatures.TextGeneration],
            media_types=[Image, Document],
        )

        # Skip tests if the model does not support text generation:
        if best_model_name is None:
            pytest.skip(
                f"{api_class.__name__} does not support documents. Skipping documents tests."
            )

        print("\n" + best_model_name)

        messages = []

        # System message:
        system_message = Message(role="system")
        system_message.append_text("You are a very very helpfull assistant.")
        messages.append(system_message)

        # User message:
        user_message = Message(role="user")
        user_message.append_text("Please summarise this document.")
        doc_path = self.get_local_test_document_uri("cartographers_of_life.docx")
        user_message.append_document(doc_path)

        messages.append(user_message)

        # Run agent:
        response = api_instance.generate_text(
            messages=messages, model_name=best_model_name
        )

        print("\n" + str(response))

        # Get the last, and possibly only, message in response:
        response = response[-1]

        # Make sure that the answer is not empty:
        assert (
            len(response) > 0
        ), f"{api_class.__name__}.completion() should return a non-empty string!"

        # Check response details:
        assert (
            "virology" in response.lower()
            or "biology" in response.lower()
            or "immunology" in response.lower()
        )

        # Check if the response is detailed and follows instructions:

        if not ("Takahashi" in response and "vitae" in response.lower()):
            # printout message that warns that the response miight lack in detail:
            print(
                "The response might lack in detail!! Please check the response for more details."
            )

    def test_text_generation_with_webpage(self, api_class):

        # If OllamaApi we skip because opensource models are not yet strong enough:
        if api_class.__name__ == "OllamaApi":
            pytest.skip(
                f"{api_class.__name__} does not have strong enough models for this test. Skipping."
            )

        # Get an instance of the api class:
        api_instance = api_class()

        # Get the best model for text generation:
        best_model_name = api_instance.get_best_model(
            features=[ModelFeatures.TextGeneration],
            media_types=[Image, Document],
        )

        # Skip tests if the model does not support text generation:
        if best_model_name is None:
            pytest.skip(
                f"{api_class.__name__} does not support documents. Skipping documents tests."
            )

        print("\n" + best_model_name)

        messages = []

        # System message:
        system_message = Message(role="system")
        system_message.append_text(
            "You are a highly qualified scientist with extensive experience in Zebrafish biology."
        )
        messages.append(system_message)

        # User message:
        user_message = Message(role="user")
        user_message.append_text(
            "Can you summarise the contents of the webpage and what is known about this gene in zebrafish? Which tissue do you think this gene is expressed in?"
        )
        user_message.append_document("https://zfin.org/ZDB-GENE-060606-1")

        messages.append(user_message)

        # Run agent:
        response = api_instance.generate_text(
            messages=messages, model_name=best_model_name
        )

        print("\n" + str(response))

        # Get the last, and possibly only, message in response:
        response = response[-1]

        # Make sure that the answer is not empty:
        assert (
            len(response) > 0
        ), f"{api_class.__name__}.completion() should return a non-empty string!"

        # Check response details:
        assert "ZFIN" in response or "COMP" in response or "zebrafish" in response

    def test_text_generation_with_json(self, api_class):

        api_instance = api_class()

        # Get the best model for text generation:
        best_model_name = api_instance.get_best_model(ModelFeatures.TextGeneration)

        # Skip tests if the model does not support text generation:
        if best_model_name is None:
            pytest.skip(
                f"{api_class.__name__} does not support text generation. Skipping tests."
            )

        print("\n" + best_model_name)

        messages = []

        # System message:
        system_message = Message(role="system")
        system_message.append_text(
            "You are a computer program that can read complex json strings and understand what they contain."
        )
        messages.append(system_message)

        # complex and long tests Json input:
        json_str = """
        {
            "name": "John Doe",
            "age": 30,
            "cars": {
                "car1": "Ford",
                "car2": "BMW",
                "car3": "Fiat"
            }
        }
        """

        # User message:
        user_message = Message(role="user")
        user_message.append_json(json_str)
        user_message.append_text("What is the age of John?")
        messages.append(user_message)

        # Run agent:
        response = api_instance.generate_text(
            messages=messages, model_name=best_model_name
        )

        print("\n" + str(response))

        # Get the last, and possibly only, message in response:
        response = response[-1]

        # Make sure that the answer is not empty:
        assert (
            len(response) > 0
        ), f"{api_class.__name__}.completion() should return a non-empty string!"

        # Check response details:
        assert "30" in response

    def test_text_generation_with_object(self, api_class):

        # Check if pydantic is installed otherwise skip tests:
        try:
            from pydantic import BaseModel
        except ImportError:
            pytest.skip("Pydantic is not installed. Skipping tests.")

        api_instance = api_class()

        # Get the best model for text generation:
        best_model_name = api_instance.get_best_model(ModelFeatures.TextGeneration)

        # Skip tests if the model does not support text generation:
        if best_model_name is None:
            pytest.skip(
                f"{api_class.__name__} does not support text generation. Skipping tests."
            )

        print("\n" + best_model_name)

        messages = []

        # System message:
        system_message = Message(role="system")
        system_message.append_text(
            "You are a computer program that can read complex json strings and understand what they contain."
        )
        messages.append(system_message)

        # complex pydantic object  (derives from BaseModel) for testing:
        from pydantic import BaseModel

        class TestObject(BaseModel):
            name: str
            age: int
            cars: Dict[str, str]

        # Create instance of object and fill with details:
        test_object = TestObject(
            name="John Doe",
            age=30,
            cars={"car1": "Ford", "car2": "BMW", "car3": "Fiat"},
        )

        # User message:
        user_message = Message(role="user")
        user_message.append_object(test_object)
        user_message.append_text("What is the age of John?")
        messages.append(user_message)

        # Run agent:
        response = api_instance.generate_text(
            messages=messages, model_name=best_model_name
        )

        print("\n" + str(response))

        # Get the last, and possibly only, message in response:
        response = response[-1]

        # Make sure that the answer is not empty:
        assert (
            len(response) > 0
        ), f"{api_class.__name__}.completion() should return a non-empty string!"

        # Check response details:
        assert "30" in response

    def test_text_generation_with_csv(self, api_class):

        api_instance = api_class()

        # Get the best model for text generation:
        best_model_name = api_instance.get_best_model(ModelFeatures.TextGeneration)

        # Skip tests if the model does not support text generation:
        if best_model_name is None:
            pytest.skip(
                f"{api_class.__name__} does not support text generation. Skipping tests."
            )

        print("\n" + best_model_name)

        messages = []

        # System message:
        system_message = Message(role="system")
        system_message.append_text("You are a highly qualified data scientist.")
        messages.append(system_message)

        # _get_local_test_table_uri
        table_path = self.get_local_test_table_uri("spreadsheet.csv")

        # User message:
        user_message = Message(role="user")
        user_message.append_text(
            "List all items sold by rep Carl Jackson in the provided table."
        )
        user_message.append_table(table_path)
        messages.append(user_message)

        # Run agent:
        response = api_instance.generate_text(
            messages=messages, model_name=best_model_name
        )

        print("\n" + str(response))

        # Get the last, and possibly only, message in response:
        response = response[-1]

        # Make sure that the answer is not empty:
        assert (
            len(response) > 0
        ), f"{api_class.__name__}.completion() should return a non-empty string!"

        # Check response details:
        assert "Binders" in response or "SAFCO" in response

    def test_text_generation_with_archive(self, api_class):

        # If OllamaApi we skip because opensource models are not yet strong enough:
        if api_class.__name__ == "OllamaApi":
            pytest.skip(
                f"{api_class.__name__} does not have strong enough models for this test. Skipping."
            )

        # Get an instance of the api class:
        api_instance = api_class()

        # Get the best model for text generation:
        best_model_name = api_instance.get_best_model(
            features=[ModelFeatures.TextGeneration],
            media_types=[Image, Document],
        )

        # Skip tests if the model does not support text generation:
        if best_model_name is None:
            pytest.skip(
                f"{api_class.__name__} does not support text generation. Skipping tests."
            )

        print("\n" + best_model_name)

        messages = []

        # System message:
        system_message = Message(role="system")
        system_message.append_text(
            "You are a highly qualified historian and literature expert"
        )
        messages.append(system_message)

        # _get_local_test_archive_uri
        archive_uri = self.get_local_test_archive_uri("alexander.zip")

        # User message:
        user_message = Message(role="user")
        user_message.append_text(
            "Make a one paragraph summary of the provided material, plus a list of all documents provided."
        )
        user_message.append_archive(archive_uri)
        messages.append(user_message)

        # Run agent:
        response = api_instance.generate_text(
            messages=messages, model_name=best_model_name
        )

        print("\n" + str(response))

        # Get the last, and possibly only, message in response:
        response = response[-1]

        # Make sure that the answer is not empty:
        assert (
            len(response) > 0
        ), f"{api_class.__name__}.completion() should return a non-empty string!"

        # If teh model is Ollama based, the relax the check, otherwise check for the details:
        if api_class.__name__ == "OllamaApi":
            assert (
                "image" in response or "landscape" in response or "artwork" in response
            )
        else:
            # Check response details:
            assert "Alexander" in response or "Aristotle" in response

    def test_text_generation_with_folder(self, api_class):

        api_instance = api_class()

        # Get the best model for text generation:
        best_model_name = api_instance.get_best_model(
            [ModelFeatures.TextGeneration],
            media_types=[Image, Document],
        )

        # Skip tests if the model does not support text generation:
        if best_model_name is None:
            pytest.skip(
                f"{api_class.__name__} does not support text generation. Skipping tests."
            )

        print("\n" + best_model_name)

        messages = []

        # System message:
        system_message = Message(role="system")
        system_message.append_text(
            "You are a highly qualified at comparing images and documents."
        )
        messages.append(system_message)

        # get the path of a folder at this relative path: './images' to this python file:
        folder_path = self.get_local_test_folder_path("images")

        # User message:
        user_message = Message(role="user")
        user_message.append_text(
            "Make a one paragraph summary of the provided material, compare the files provided, and make a list of all documents provided."
        )
        user_message.append_folder(folder_path)
        messages.append(user_message)

        # Run agent:
        response = api_instance.generate_text(
            messages=messages, model_name=best_model_name
        )

        print("\n" + str(response))

        # Get the last, and possibly only, message in response:
        response = response[-1]

        # Make sure that the answer is not empty:
        assert (
            len(response) > 0
        ), f"{api_class.__name__}.completion() should return a non-empty string!"

        if api_class.__name__ == "OllamaApi":
            # Check response details, ollama models are not strong enough to give detailed responses for all files
            assert (
                "artwork" in response
                or "futuristic" in response
                or "landscape" in response
                or "humanoid" in response
            )
        else:
            # Check response details:
            assert (
                "beach" in response
                or "diverse" in response
                or "Python" in response
                or "ball" in response
            )
