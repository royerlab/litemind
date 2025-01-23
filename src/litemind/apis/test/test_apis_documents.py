from typing import Dict

import pytest

from litemind.agent.message import Message
from litemind.apis.anthropic.anthropic_api import AnthropicApi
from litemind.apis.base_api import ModelFeatures
from litemind.apis.google.google_api import GeminiApi
from litemind.apis.ollama.ollama_api import OllamaApi
from litemind.apis.openai.openai_api import OpenAIApi
from litemind.apis.test.base_test import BaseTest

# Put all your implementations in this list:
API_IMPLEMENTATIONS = [
    OpenAIApi,
    OllamaApi,
    AnthropicApi,
    GeminiApi
]


@pytest.mark.parametrize("ApiClass", API_IMPLEMENTATIONS)
class TestBaseApiImplementations(BaseTest):
    """
    A test suite that runs the same tests on each ApiClass
    implementing the abstract BaseApi interface.
    """

    def test_text_generation_with_pdf_document(self, ApiClass):

        api_instance = ApiClass()

        default_model_name = api_instance.get_best_model(
            [ModelFeatures.TextGeneration, ModelFeatures.Documents,
             ModelFeatures.Image])
        if not default_model_name or not api_instance.has_model_support_for(
                model_name=default_model_name,
                features=[ModelFeatures.TextGeneration, ModelFeatures.Documents,
                          ModelFeatures.Image]):
            pytest.skip(
                f"{ApiClass.__name__} does not support documents. Skipping documents test.")

        print('\n' + default_model_name)

        messages = []

        # System message:
        system_message = Message(role='system')
        system_message.append_text(
            'You are a highly qualified scientist with extensive experience in Microscopy, Biology and Bioimage processing and analysis.')
        messages.append(system_message)

        # User message:
        user_message = Message(role='user')
        user_message.append_text(
            'Can you write a review for the provided paper? Please break down your comments into major and minor comments.')
        doc_path = self._get_local_test_document_uri('intracktive_preprint.pdf')
        user_message.append_document(doc_path)

        messages.append(user_message)

        # Run agent:
        response = api_instance.generate_text_completion(messages=messages,
                                                         model_name=default_model_name)

        print('\n' + str(response))

        # Make sure that the answer is not empty:
        assert len(response) > 0, (
            f"{ApiClass.__name__}.completion() should return a non-empty string!"
        )

        # Check response details:
        assert 'microscopy' in response.lower() or 'biology' in response.lower()

        # Check if the response is detailed and follows instructions:

        if not ('inTRACKtive' in response and 'review' in response.lower()):
            # printout message that warns that the response miight lack in detail:
            print(
                "The response might lack in detail!! Please check the response for more details.")

    def test_text_generation_with_webpage(self, ApiClass):

        api_instance = ApiClass()

        default_model_name = api_instance.get_best_model(
            [ModelFeatures.TextGeneration, ModelFeatures.Documents,
             ModelFeatures.Image])
        if not default_model_name or not api_instance.has_model_support_for(
                model_name=default_model_name,
                features=[ModelFeatures.TextGeneration, ModelFeatures.Documents,
                          ModelFeatures.Image]):
            pytest.skip(
                f"{ApiClass.__name__} does not support documents. Skipping documents test.")

        print('\n' + default_model_name)

        messages = []

        # System message:
        system_message = Message(role='system')
        system_message.append_text(
            'You are a highly qualified scientist with extensive experience in Zebrafish biology.')
        messages.append(system_message)

        # User message:
        user_message = Message(role='user')
        user_message.append_text(
            'Can you summarise the contents of the webpage and what is known about this gene in zebrafish? Which tissue do you think this gene is expressed in?')
        user_message.append_document("https://zfin.org/ZDB-GENE-060606-1")

        messages.append(user_message)

        # Run agent:
        response = api_instance.generate_text_completion(messages=messages,
                                                         model_name=default_model_name)

        print('\n' + str(response))

        # Make sure that the answer is not empty:
        assert len(response) > 0, (
            f"{ApiClass.__name__}.completion() should return a non-empty string!"
        )

        # Check response details:
        assert 'ZFIN' in response or 'COMP' in response or 'zebrafish' in response

    def test_text_generation_with_json(self, ApiClass):

        api_instance = ApiClass()

        default_model_name = api_instance.get_best_model(
            ModelFeatures.TextGeneration)
        if not default_model_name or not api_instance.has_model_support_for(
                model_name=default_model_name,
                features=ModelFeatures.TextGeneration):
            pytest.skip(
                f"{ApiClass.__name__} does not support text generation. Skipping test.")

        print('\n' + default_model_name)

        messages = []

        # System message:
        system_message = Message(role='system')
        system_message.append_text(
            'You are a computer program that can read complex json strings and understand what they contain.')
        messages.append(system_message)

        # complex and long test Json input:
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
        user_message = Message(role='user')
        user_message.append_json(json_str)
        user_message.append_text('What is the age of John?')
        messages.append(user_message)

        # Run agent:
        response = api_instance.generate_text_completion(messages=messages,
                                                         model_name=default_model_name)

        print('\n' + str(response))

        # Make sure that the answer is not empty:
        assert len(response) > 0, (
            f"{ApiClass.__name__}.completion() should return a non-empty string!"
        )

        # Check response details:
        assert '30' in response

    def test_text_generation_with_object(self, ApiClass):

        # Check if pydantic is installed otherwise skip test:
        try:
            from pydantic import BaseModel
        except ImportError:
            pytest.skip(
                "Pydantic is not installed. Skipping test."
            )

        api_instance = ApiClass()

        default_model_name = api_instance.get_best_model(
            ModelFeatures.TextGeneration)
        if not default_model_name or not api_instance.has_model_support_for(
                model_name=default_model_name,
                features=ModelFeatures.TextGeneration):
            pytest.skip(
                f"{ApiClass.__name__} does not support text generation. Skipping test.")

        print('\n' + default_model_name)

        messages = []

        # System message:
        system_message = Message(role='system')
        system_message.append_text(
            'You are a computer program that can read complex json strings and understand what they contain.')
        messages.append(system_message)

        # complex pydantic object  (derives from BaseModel) for testing:
        from pydantic import BaseModel
        class TestObject(BaseModel):
            name: str
            age: int
            cars: Dict[str, str]

        # Create instance of object and fill with details:
        test_object = TestObject(name='John Doe', age=30,
                                 cars={'car1': 'Ford', 'car2': 'BMW',
                                       'car3': 'Fiat'})

        # User message:
        user_message = Message(role='user')
        user_message.append_object(test_object)
        user_message.append_text('What is the age of John?')
        messages.append(user_message)

        # Run agent:
        response = api_instance.generate_text_completion(messages=messages,
                                                         model_name=default_model_name)

        print('\n' + str(response))

        # Make sure that the answer is not empty:
        assert len(response) > 0, (
            f"{ApiClass.__name__}.completion() should return a non-empty string!"
        )

        # Check response details:
        assert '30' in response

    def test_text_generation_with_csv(self, ApiClass):

        api_instance = ApiClass()

        default_model_name = api_instance.get_best_model(
            ModelFeatures.TextGeneration)
        if not default_model_name or not api_instance.has_model_support_for(
                model_name=default_model_name,
                features=ModelFeatures.TextGeneration):
            pytest.skip(
                f"{ApiClass.__name__} does not support text generation. Skipping test.")

        print('\n' + default_model_name)

        messages = []

        # System message:
        system_message = Message(role='system')
        system_message.append_text('You are a highly qualified data scientist.')
        messages.append(system_message)

        # _get_local_test_table_uri
        table_path = self._get_local_test_table_uri('spreadsheet.csv')

        # User message:
        user_message = Message(role='user')
        user_message.append_text(
            'List all items sold by rep Carl Jackson in the provided table.')
        user_message.append_table(table_path)
        messages.append(user_message)

        # Run agent:
        response = api_instance.generate_text_completion(messages=messages,
                                                         model_name=default_model_name)

        print('\n' + str(response))

        # Make sure that the answer is not empty:
        assert len(response) > 0, (
            f"{ApiClass.__name__}.completion() should return a non-empty string!"
        )

        # Check response details:
        assert 'Binders' in response or 'SAFCO' in response

    def test_text_generation_with_archive(self, ApiClass):

        api_instance = ApiClass()

        default_model_name = api_instance.get_best_model(
            ModelFeatures.TextGeneration)
        if not default_model_name or not api_instance.has_model_support_for(
                model_name=default_model_name,
                features=ModelFeatures.TextGeneration):
            pytest.skip(
                f"{ApiClass.__name__} does not support text generation. Skipping test.")

        print('\n' + default_model_name)

        messages = []

        # System message:
        system_message = Message(role='system')
        system_message.append_text(
            'You are a highly qualified historian and literature expert')
        messages.append(system_message)

        # _get_local_test_archive_uri
        archive_path = self._get_local_test_archive_uri('alexander.zip')

        # User message:
        user_message = Message(role='user')
        user_message.append_text(
            'Make a one paragraph summary of the provided material, plus a list of all documents provided.')
        user_message.append_archive(archive_path)
        messages.append(user_message)

        # Run agent:
        response = api_instance.generate_text_completion(messages=messages,
                                                         model_name=default_model_name)

        print('\n' + str(response))

        # Make sure that the answer is not empty:
        assert len(response) > 0, (
            f"{ApiClass.__name__}.completion() should return a non-empty string!"
        )

        # Check response details:
        assert 'Alexander' in response or 'Aristotle' in response

    def test_text_generation_with_folder(self, ApiClass):

        api_instance = ApiClass()

        default_model_name = api_instance.get_best_model(
            ModelFeatures.TextGeneration)
        if not default_model_name or not api_instance.has_model_support_for(
                model_name=default_model_name,
                features=ModelFeatures.TextGeneration):
            pytest.skip(
                f"{ApiClass.__name__} does not support text generation. Skipping test.")

        print('\n' + default_model_name)

        messages = []

        # System message:
        system_message = Message(role='system')
        system_message.append_text(
            'You are a highly qualified at comparing images and documents.')
        messages.append(system_message)

        # get the path of a folder at this relative path: './images' to this python file:
        folder_path = self._get_local_test_folder_path('images')

        # User message:
        user_message = Message(role='user')
        user_message.append_text(
            'Make a one paragraph summary of the provided material, compare the files provided, and make a list of all documents provided.')
        user_message.append_folder(folder_path)
        messages.append(user_message)

        # Run agent:
        response = api_instance.generate_text_completion(messages=messages,
                                                         model_name=default_model_name)

        print('\n' + str(response))

        # Make sure that the answer is not empty:
        assert len(response) > 0, (
            f"{ApiClass.__name__}.completion() should return a non-empty string!"
        )

        # Check response details:
        assert 'beach' in response or 'diverse' in response or 'Python' in response or 'ball' in response
