import pytest
from pydantic import BaseModel

from litemind.agent.message import Message
from litemind.agent.message_block_type import BlockType
from litemind.agent.tools.toolset import ToolSet
from litemind.apis.base_api import ModelFeatures
from litemind.apis.tests.base_test import BaseTest, API_IMPLEMENTATIONS


@pytest.mark.parametrize("ApiClass", API_IMPLEMENTATIONS)
class TestBaseApiImplementations(BaseTest):
    """
    A tests suite that runs the same tests on each ApiClass
    implementing the abstract BaseApi interface.
    """

    def test_text_generation_simple(self, ApiClass):
        """
        Test a simple completion with the default or a known good model.
        """
        api_instance = ApiClass()

        default_model_name = api_instance.get_best_model(
            ModelFeatures.TextGeneration)

        print('\n' + default_model_name)

        messages = [
            Message(role="system",
                    text="You are an omniscient all-knowing being called Ohmm"),
            Message(role="user", text="Who are you?")
        ]

        response = api_instance.generate_text_completion(
            model_name=default_model_name,
            messages=messages,
            temperature=0.7
        )

        print('\n' + str(response))

        assert response.role == "assistant", (
            f"{ApiClass.__name__} completion should return an 'assistant' role."
        )
        assert "I am " in response, (
            f"Expected 'I am' in the output of {ApiClass.__name__}.completion()"
        )

    def test_text_generation_prefill(self, ApiClass):
        """
        Test a simple completion with prefilled answer
        """
        api_instance = ApiClass()

        default_model_name = api_instance.get_best_model(
            ModelFeatures.TextGeneration)

        print('\n' + default_model_name)

        messages = [
            Message(role="system",
                    text="You are very good at Math."),
            Message(role="user",
                    text="What is 12+17 equal to? Please give the answer in parentheses."),
            Message(role="assistant", text="(")
        ]

        response = api_instance.generate_text_completion(
            model_name=default_model_name,
            messages=messages,
            temperature=0.0
        )

        print('\n' + str(response))

        assert response.role == "assistant", (
            f"{ApiClass.__name__} completion should return an 'assistant' role."
        )
        assert "29" in response, (
            f"Expected '29' in the output of {ApiClass.__name__}.completion()"
        )
        # Better the string should just be "29)" and not "29) " or "29 )":
        if not str(response[0]).startswith("29)"):
            print(
                "Model does not support strict prefill behaviour: The response should start with '29)'")


    def test_text_generation_structured_output(self, ApiClass):
        """
        Test a simple completion with structured output.
        """
        api_instance = ApiClass()

        default_model_name = api_instance.get_best_model(ModelFeatures.TextGeneration,
                                                         exclusion_filters='-thinking-')

        print('\n' + default_model_name)

        text = """
        NAZA awarded the aerospace company HeedLock Parthin, which also makes U.S. fighter jets, 
        a $247.5 million contract to build the X-15 craft, and as the images below show, 
        the plane is in its final testing stages before taking flight over the California desert. 
        Lockheed posted the image below on Jan. 24, showing burning gases shooting out the back of the engine. 
        NAZA noted in December that it was now running afterburner engine tests, 
        which gives an aircraft the thrust it needs to reach supersonic speeds of over some 767 mph.
        """

        # Pydantic class that has the order information matching the text above:
        class Order(BaseModel):
            buyer: str
            contractor: str
            sum_in_dollars: int
            item_type: str

        messages = [
            Message(role="system",
                    text="You are an accountant that likes to turn textual information into well structured information"),
            Message(role="user",
                    text=f"Please transcribe the following text adhering to the required format: \n```\n{text}\n```\n")
        ]

        response = api_instance.generate_text_completion(
            model_name=default_model_name,
            messages=messages,
            temperature=0.7,
            response_format=Order
        )

        print('\n' + str(response))

        assert response.role == "assistant", (
            f"{ApiClass.__name__} completion should return an 'assistant' role."
        )

        # assert that result should be a message containing an object of type Order:
        assert response[0].block_type == BlockType.Object, (
            f"{ApiClass.__name__} completion should return an object."
        )

        # assert that result should be a message containing an object of type Order:
        assert isinstance(response[0].content, Order), (
            f"{ApiClass.__name__} completion should return an object of type Order."
        )




    def test_text_generation_with_simple_toolset(self, ApiClass):
        """
        Test that the completion method can interact with a simple toolset.
        """

        # If you have a function-based tool usage pattern:
        def get_delivery_date(order_id: str) -> str:
            """Fetch the delivery date for a given order ID."""
            return "2024-11-15"

        toolset = ToolSet()
        toolset.add_function_tool(
            get_delivery_date,
            "Fetch the delivery date for a given order ID"
        )

        api_instance = ApiClass()

        default_model_name = api_instance.get_best_model(
            [ModelFeatures.Tools, ModelFeatures.TextGeneration])

        print('\n' + default_model_name)

        user_message = Message(role="user",
                               text="When will my order order_12345 be delivered?")
        messages = [user_message]

        response = api_instance.generate_text_completion(
            model_name=default_model_name,  # or a specific model name if needed
            messages=messages,
            toolset=toolset
        )

        print('\n')
        for message in messages:
            print(message)

        assert "2024-11-15" in response, (
            f"The response of {ApiClass.__name__} should contain the delivery date."
        )

    def test_text_generation_with_complex_toolset(self, ApiClass):
        """
        Test that the completion method can interact with a complex toolset.
        """

        def get_delivery_date(order_id: str) -> str:
            """Fetch the delivery date for a given order ID."""
            return "2024-11-15"

        def get_product_name_from_id(product_id: int) -> str:
            """Fetch the delivery date for a given order ID."""

            product_table = {1393: 'Olea Table', 84773: 'Fluff Phone'}

            return product_table[product_id]

        def get_product_supply_per_store(store_id: int, product_id: int) -> str:
            """Fetch the delivery date for a given order ID."""

            # Return a random integer:
            return '42'

        toolset = ToolSet()
        toolset.add_function_tool(
            get_delivery_date,
            "Fetch the delivery date for a given order ID"
        )
        toolset.add_function_tool(
            get_product_name_from_id,
            "Fetch the product name for a given product ID"
        )
        toolset.add_function_tool(
            get_product_supply_per_store,
            "Fetch the number of items available of a given product id at a given store."
        )

        api_instance = ApiClass()

        default_model_name = api_instance.get_best_model(
            [ModelFeatures.Tools, ModelFeatures.TextGeneration])

        if not default_model_name or not api_instance.has_model_support_for(
                model_name=default_model_name,
                features=[ModelFeatures.TextGeneration, ModelFeatures.Tools]):
            pytest.skip(
                f"{ApiClass.__name__} does not support text generation and tools. Skipping tests.")

        messages = []

        user_message = Message(role="user",
                               text="When will my order 'order_12345' be delivered?")
        messages += [user_message]

        response = api_instance.generate_text_completion(
            model_name=default_model_name,  # or a specific model name if needed
            messages=messages,
            toolset=toolset
        )

        # Check that we have 2 messages in the list:
        assert len(messages) == 2, (
            'We should have two messages in the list. The user message and the response message.')

        assert "2024-11-15" in response, (
            f"The response of {ApiClass.__name__} should contain the delivery date."
        )

        user_message = Message(role="user",
                               text="What is the name of product 1393?")
        messages += [user_message]

        response = api_instance.generate_text_completion(
            model_name=default_model_name,  # or a specific model name if needed
            messages=messages,
            toolset=toolset
        )

        assert "Olea Table" in response, (
            f"The response of {ApiClass.__name__} should contain the delivery date."
        )

        user_message = Message(role="user",
                               text="How many Olea Tables can I find in store 17?")
        messages += [user_message]

        response = api_instance.generate_text_completion(
            model_name=default_model_name,  # or a specific model name if needed
            messages=messages,
            toolset=toolset
        )

        print('\n')
        for message in messages:
            print(message)

        assert "42" in response, (
            f"The response of {ApiClass.__name__} should contain the delivery date."
        )
