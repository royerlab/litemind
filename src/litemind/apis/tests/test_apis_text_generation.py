from datetime import datetime

import pytest
from pydantic import BaseModel

from litemind import API_IMPLEMENTATIONS
from litemind.agent.messages.message import Message
from litemind.agent.messages.message_block_type import BlockType
from litemind.agent.tools.toolset import ToolSet
from litemind.apis.base_api import ModelFeatures
from litemind.apis.tests.base_test import BaseTest


@pytest.mark.parametrize("api_class", API_IMPLEMENTATIONS)
class TestBaseApiImplementationsTextGeneration(BaseTest):
    """
    A tests suite that runs the same tests on each ApiClass
    implementing the abstract BaseApi interface.
    """

    def test_text_generation_simple(self, api_class):
        """
        Test a simple completion with the default or a known good model.
        """
        api_instance = api_class()

        # Get the best model for text generation:
        default_model_name = api_instance.get_best_model(ModelFeatures.TextGeneration)

        # If the model does not support text generation, skip the test:
        if default_model_name is None:
            pytest.skip(
                f"{api_class.__name__} does not support text generation. Skipping tests."
            )

        # Print the model name:
        print("\n" + default_model_name)

        # A simple message:
        messages = [
            Message(
                role="system",
                text="You are an omniscient all-knowing being called Ohmm",
            ),
            Message(role="user", text="I am 'The User'."),
            Message(role="user", text="Who are you?"),
        ]

        # Get the completion:
        response = api_instance.generate_text(
            model_name=default_model_name, messages=messages, temperature=0.7
        )

        # There should be only one message in the response:
        assert len(response) == 1, f"Expected only one message in the response."

        # Extract the message from the list:
        response = response[0]

        # Print the response:
        print("\n" + str(response))

        assert (
            response.role == "assistant"
        ), f"{api_class.__name__} completion should return an 'assistant' role."
        assert (
            "I am " in response or "I'm" in response
        ), f"Expected 'I am' or 'I'm' in the output of {api_class.__name__}.completion()"

    def test_text_generation_prefill(self, api_class):
        """
        Test a simple completion with prefilled answer
        """
        api_instance = api_class()

        # Get the best model for text generation:
        default_model_name = api_instance.get_best_model(ModelFeatures.TextGeneration)

        # If the model does not support text generation, skip the test:
        if default_model_name is None:
            pytest.skip(
                f"{api_class.__name__} does not support text generation. Skipping tests."
            )

        # Print the model name:
        print("\n" + default_model_name)

        # A simple message list:
        messages = [
            Message(role="system", text="You are very good at Math."),
            Message(
                role="user",
                text="What is 12+17 equal to? Please give the answer in parentheses.",
            ),
            Message(role="assistant", text="("),
        ]

        # Get the completion:
        response = api_instance.generate_text(
            model_name=default_model_name, messages=messages, temperature=0.0
        )

        # Print all messages:
        print("\n")
        for message in messages:
            print(message)

        # Check that there is only one message in the response:
        assert len(response) == 1, f"Expected only one message in the response."

        # Extract the message from the list:
        response = response[0]

        # Check the response:
        assert (
            response.role == "assistant"
        ), f"{api_class.__name__} completion should return an 'assistant' role."
        assert (
            "29" in response
        ), f"Expected '29' in the output of {api_class.__name__}.completion()"
        # Better the string should just be "29)" and not "29) " or "29 )":
        if not str(response[0]).startswith("29)"):
            print(
                "Model does not support strict prefill behaviour: The response should start with '29)'"
            )

    def test_text_generation_structured_output(self, api_class):
        """
        Test a simple completion with structured output.
        """
        api_instance = api_class()

        # Get the best model for text generation:
        default_model_name = api_instance.get_best_model(
            [ModelFeatures.TextGeneration, ModelFeatures.StructuredTextGeneration]
        )

        # If the model does not support text generation, skip the test:
        if default_model_name is None:
            pytest.skip(
                f"{api_class.__name__} does not support text generation. Skipping tests."
            )

        # Print the model name:
        print("\n" + default_model_name)

        # A short article for testing:
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

        # Assemble the messages:
        messages = [
            Message(
                role="system",
                text="You are an accountant that likes to turn textual information into well structured information",
            ),
            Message(
                role="user",
                text=f"Please transcribe the following text adhering to the required format: \n```\n{text}\n```\n",
            ),
        ]

        # Get the completion:
        response = api_instance.generate_text(
            model_name=default_model_name,
            messages=messages,
            temperature=0.7,
            response_format=Order,
        )

        # Print all messages:
        print("\n")
        for message in messages:
            print(message)

        # There should be only one message in the response:
        assert len(response) == 1, f"Expected only one message in the response."

        # Extract the message from the list:
        response = response[0]

        # assert that result should be of role "assistant":
        assert (
            response.role == "assistant"
        ), f"{api_class.__name__} completion should return an 'assistant' role."

        # The last block contains the object:
        object_block = response[-1]

        # assert that result should be a message of type object:
        assert (
            object_block.block_type == BlockType.Object
        ), f"{api_class.__name__} completion should return an object."

        # assert that result should be a message containing an object of type Order:
        assert isinstance(
            object_block.content, Order
        ), f"{api_class.__name__} completion should return an object of type Order."

    def test_text_generation_with_simple_parameterless_tool(self, api_class):
        """
        Test that the completion method can interact with a simple toolset.
        """

        # Create an instance of the API class:
        api_instance = api_class()

        def get_current_date() -> str:
            return datetime.now().strftime("%Y-%m-%d")

        # Create a toolset and add the function tool:
        toolset = ToolSet()
        toolset.add_function_tool(get_current_date, "Fetch the current date")

        # Get the best model for text generation with tools:
        default_model_name = api_instance.get_best_model(
            [ModelFeatures.Tools, ModelFeatures.TextGeneration]
        )

        # If the model does not support text generation, skip the test:
        if default_model_name is None:
            pytest.skip(
                f"{api_class.__name__} does not support text generation and tools. Skipping tests."
            )

        # Print the model name:
        print("\n" + default_model_name)

        # A simple message:
        user_message = Message(
            role="user", text="What is the current date? (Reply in %Y-%m-%d format)"
        )
        messages = [user_message]

        # Get the completion:
        response = api_instance.generate_text(
            model_name=default_model_name,  # or a specific model name if needed
            messages=messages,
            toolset=toolset,
        )

        # Print all messages:
        print("\n")
        for message in messages:
            print(message)

        # There should be only one message in the response:
        assert len(response) == 3, f"Expected three message in the response."

        # Check that the second last message in response is a tool use message:
        assert response[-2][0].block_type == BlockType.Tool

        # Extract the last message from the list which contains the actual final response:
        response = response[-1]

        # Check that we have 2 messages in the list:
        assert (
            get_current_date() in response
        ), f"The response of {api_class.__name__} should contain the delivery date."

    def test_text_generation_with_simple_toolset(self, api_class):
        """
        Test that the completion method can interact with a simple toolset.
        """

        # Create an instance of the API class:
        api_instance = api_class()

        # If you have a function-based tool usage pattern:
        def get_delivery_date(order_id: str) -> str:
            """Fetch the delivery date for a given order ID."""

            # Return 10 days past get_current_date():
            return "2024-11-15"

        # Create a toolset and add the function tool:
        toolset = ToolSet()
        toolset.add_function_tool(
            get_delivery_date, "Fetch the delivery date for a given order ID"
        )

        # Get the best model for text generation with tools:
        default_model_name = api_instance.get_best_model(
            [ModelFeatures.Tools, ModelFeatures.TextGeneration]
        )

        # If the model does not support text generation, skip the test:
        if default_model_name is None:
            pytest.skip(
                f"{api_class.__name__} does not support text generation and tools. Skipping tests."
            )

        # Print the model name:
        print("\n" + default_model_name)

        # A simple message:
        user_message = Message(
            role="user", text="When will my order order_12345 be delivered?"
        )
        messages = [user_message]

        # Get the completion:
        response = api_instance.generate_text(
            model_name=default_model_name,  # or a specific model name if needed
            messages=messages,
            toolset=toolset,
        )

        # Print all messages:
        print("\n")
        for message in messages:
            print(message)

        # There should be only one message in the response:
        assert len(response) == 3, f"Expected three message in the response."

        # Check that the second last message in reponse is a tool use message:
        assert response[-2][0].block_type == BlockType.Tool

        # Extract the last message from the list which contains the actual final response:
        response = response[-1]

        # Check that we have 2 messages in the list:
        assert (
            "2024-11-15" in response
            or "November 15, 2024" in response
            or ("November" in response and "15" in response and "2024" in response)
        ), f"The response of {api_class.__name__} should contain the delivery date."

    def test_text_generation_with_simple_toolset_and_struct_output(self, api_class):
        """
        Test that the completion method can interact with a simple toolset.
        """

        api_instance = api_class()

        # If you have a function-based tool usage pattern:
        def get_delivery_date(order_id: str) -> str:
            """Fetch the delivery date for a given order ID."""
            return "2024-11-15"

        # Create a toolset and add the function tool:
        toolset = ToolSet()
        toolset.add_function_tool(
            get_delivery_date, "Fetch the delivery date for a given order ID"
        )

        # Get the best model for text generation with tools:
        default_model_name = api_instance.get_best_model(
            [ModelFeatures.Tools, ModelFeatures.TextGeneration]
        )

        # If the model does not support text generation, skip the test:
        if default_model_name is None:
            pytest.skip(
                f"{api_class.__name__} does not support text generation and tools. Skipping tests."
            )

        # Print the model name:
        print("\n" + default_model_name)

        class OrderInfo(BaseModel):
            order_id: str
            delivery_date: str

        # A simple message:
        user_message = Message(
            role="user", text="When will my order order_12345 be delivered?"
        )
        messages = [user_message]

        # Get the completion:
        response = api_instance.generate_text(
            model_name=default_model_name,  # or a specific model name if needed
            messages=messages,
            toolset=toolset,
            response_format=OrderInfo,
        )

        # Print all messages:
        print("\n")
        for message in messages:
            print(message)

        # There should be only one message in the response:
        assert len(response) >= 3, f"Expected three message in the response."

        # Check that the second last message in response is a tool use message:
        assert response[-2][0].block_type == BlockType.Tool

        # Extract the last message from the list which contains the actual final response:
        response = response[-1]

        # Convert the response to a string:
        response_str = str(response)

        # Check that the date is in the response:
        assert (
            "2024-11-15" in response_str
        ), f"The response of {api_class.__name__} should contain the delivery date."

        # The object is in the last block:
        object_block = response[-1]

        # Check that the response is of type OrderInfo:
        assert isinstance(
            object_block.content, OrderInfo
        ), f"The response of {api_class.__name__} should be of type OrderInfo."

    def test_text_generation_with_complex_toolset(self, api_class):
        """
        Test that the completion method can interact with a complex toolset.
        """

        api_instance = api_class()

        def get_delivery_date(order_id: str) -> str:
            """Fetch the delivery date for a given order ID."""
            return "2024-11-15"

        def get_product_name_from_id(product_id: int) -> str:
            """Fetch the delivery date for a given order ID."""

            # Normalise the product ID to int in case it is a string:
            product_id = int(product_id)

            product_table = {1393: "Olea Table", 84773: "Fluff Phone"}

            return product_table[product_id]

        def get_product_supply_per_store(store_id: int, product_id: int) -> str:
            """Fetch the delivery date for a given order ID."""

            # Return a random integer:
            return "42"

        # Create a toolset and add the function tools:
        toolset = ToolSet()
        toolset.add_function_tool(
            get_delivery_date, "Fetch the delivery date given the order ID"
        )
        toolset.add_function_tool(
            get_product_name_from_id, "Fetch the product name given the product ID"
        )
        toolset.add_function_tool(
            get_product_supply_per_store,
            "Fetch the number of items available given a product ID and store ID.",
        )

        # Get the best model for text generation with tools:
        default_model_name = api_instance.get_best_model(
            [ModelFeatures.Tools, ModelFeatures.TextGeneration]
        )

        if default_model_name is None:
            pytest.skip(
                f"{api_class.__name__} does not support text generation and tools. Skipping tests."
            )

        print("\n" + default_model_name)

        # Request message:
        user_message = Message(
            role="user", text="When will my order 'order_12345' be delivered?"
        )

        messages = [user_message]

        # Get the completion:
        response = api_instance.generate_text(
            model_name=default_model_name,  # or a specific model name if needed
            messages=messages,
            toolset=toolset,
        )

        # Check that we have 4 messages in the list:
        assert (
            len(messages) == 4
        ), "We should have four messages in the list. The user message and the response message."

        # Convert all messages in response to one string:
        response = str(response)

        assert (
            "2024-11-15" in response
            or "November 15, 2024" in response
            or ("November" in response and "15" in response and "2024" in response)
        ), f"The response of {api_class.__name__} should contain the delivery date."

        # Get the product name:
        user_message = Message(role="user", text="What is the name of product 1393?")
        messages += [user_message]

        # Get the completion:
        response = api_instance.generate_text(
            model_name=default_model_name,  # or a specific model name if needed
            messages=messages,
            toolset=toolset,
        )

        # Check that there is three messages in the response:
        assert len(messages) == 8, "We should have eight messages in the response."

        # Extract the last message from the response:
        response = response[-1]

        # Check that we get the correct product name:
        assert (
            "Olea Table" in response
        ), f"The response of {api_class.__name__} should contain the delivery date."

        # Get the product supply:
        user_message = Message(
            role="user", text="How many Olea Tables can I find in store 17?"
        )
        messages += [user_message]

        # Get the completion:
        response = api_instance.generate_text(
            model_name=default_model_name,  # or a specific model name if needed
            messages=messages,
            toolset=toolset,
        )

        # Check that the response is of length 3:
        assert len(response) == 3, "The response should contain 3 messages."

        # Extract the last message from the response:
        response = response[-1]

        # Check that we get the correct number of available tables:
        assert (
            "42" in response
        ), f"The response of {api_class.__name__} should contain the number of tables."

        # Printout the whole conversation:
        print("\n")
        for message in messages:
            print(message)

    #
    # def test_text_generation_with_multi_tool_usage(self, api_class):
    #     """
    #     Test that the completion method can call multiple tools.
    #     """
    #
    #     api_instance = api_class()
    #
    #     def get_product_name_from_id(product_id: int) -> str:
    #         """Fetch the delivery date for a given order ID."""
    #
    #         # Normalise the product ID to int in case it is a string:
    #         product_id = int(product_id)
    #
    #         product_table = {1393: 'Olea Table', 84773: 'Fluff Phone'}
    #
    #         return product_table[product_id]
    #
    #     def get_product_supply_per_store(store_id: int, product_name: str) -> str:
    #         """Fetch the product supply from a given sore and product."""
    #
    #         if store_id==17 and product_name=="Olea Table":
    #             # Return a random integer:
    #             return '42'
    #         else:
    #             raise
    #
    #     # Create a toolset and add the function tools:
    #     toolset = ToolSet()
    #
    #     toolset.add_function_tool(
    #         get_product_name_from_id,
    #         "Fetch the product name for a given product ID"
    #     )
    #     toolset.add_function_tool(
    #         get_product_supply_per_store,
    #         "Fetch the number of items available of a given product name at a given store."
    #     )
    #
    #     # Get the best model for text generation with tools:
    #     default_model_name = api_instance.get_best_model(
    #         [ModelFeatures.Tools, ModelFeatures.TextGeneration])
    #
    #     #
    #     if default_model_name is None:
    #         pytest.skip(
    #             f"{api_class.__name__} does not support text generation and tools. Skipping tests.")
    #
    #     print('\n' + default_model_name)
    #
    #     # Request message:
    #     user_message = Message(role="user",
    #                            text="How many Olea tables are present in store 17, and what is the name of product 1393 ?")
    #
    #     messages = [user_message]
    #
    #     # Get the completion:
    #     response = api_instance.generate_text(
    #         model_name=default_model_name,  # or a specific model name if needed
    #         messages=messages,
    #         toolset=toolset
    #     )
    #
    #     # Check that we have 2 messages in the list:
    #     assert len(messages) == 2, (
    #         'We should have two messages in the list. The user message and the response message.')
    #
    #     # Check that we get the correct number of available tables:
    #     assert "42" in response, (
    #         f"The response of {api_class.__name__} should contain the number of tables."
    #     )
    #
    #     #
    #
    #     # Printout the whole conversation:
    #     print('\n')
    #     for message in messages:
    #         print(message)
