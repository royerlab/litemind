from datetime import datetime

import pytest
from pydantic import BaseModel

from litemind import API_IMPLEMENTATIONS
from litemind.agent.messages.message import Message
from litemind.agent.tools.toolset import ToolSet
from litemind.apis.base_api import ModelFeatures
from litemind.media.types.media_action import Action
from litemind.media.types.media_object import Object
from litemind.media.types.media_text import Text
from litemind.ressources.media_resources import MediaResources


@pytest.mark.parametrize("api_class", API_IMPLEMENTATIONS)
class TestBaseApiImplementationsTextGeneration(MediaResources):
    """
    A tests suite that runs the same tests on each ApiClass
    implementing the abstract BaseApi interface.
    These tests are for the text generation methods of the API.
    """

    def test_text_generation_simple(self, api_class):
        """
        Test a simple completion with the default or a known good model.
        """
        api_instance = api_class()

        # Get the best model for text generation:
        model_name = api_instance.get_best_model(
            ModelFeatures.TextGeneration, non_features=ModelFeatures.Thinking
        )

        # If the model does not support text generation, skip the test:
        if model_name is None:
            pytest.skip(
                f"{api_class.__name__} does not support text generation. Skipping tests."
            )

        # Print the model name:
        print("\n" + model_name)

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
            model_name=model_name, messages=messages, temperature=0.7
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
        model_name = api_instance.get_best_model(
            ModelFeatures.TextGeneration, non_features=ModelFeatures.Thinking
        )

        # If the model does not support text generation, skip the test:
        if model_name is None:
            pytest.skip(
                f"{api_class.__name__} does not support text generation. Skipping tests."
            )

        # Print the model name:
        print("\n" + model_name)

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
            model_name=model_name, messages=messages, temperature=0.0
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
        model_name = api_instance.get_best_model(
            [ModelFeatures.TextGeneration, ModelFeatures.StructuredTextGeneration]
        )

        # If the model does not support text generation, skip the test:
        if model_name is None:
            pytest.skip(
                f"{api_class.__name__} does not support text generation. Skipping tests."
            )

        # Print the model name:
        print("\n" + model_name)

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
            model_name=model_name,
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
        assert object_block.has_type(
            Object
        ), f"{api_class.__name__} completion should return an object."

        # assert that result should be a message containing an object of type Order:
        assert isinstance(
            object_block.get_content(), Order
        ), f"{api_class.__name__} completion should return an object of type Order."

        # Check the content of the object:
        order_info = object_block.get_content()
        assert order_info.buyer == "NAZA"
        assert order_info.contractor == "HeedLock Parthin"
        assert order_info.sum_in_dollars == 247500000
        assert "X-15" in order_info.item_type

        print(order_info)

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
        assert response[-2][0].has_type(Action)

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
        model_name = api_instance.get_best_model(
            [ModelFeatures.Tools, ModelFeatures.TextGeneration]
        )

        # If the model does not support text generation, skip the test:
        if model_name is None:
            pytest.skip(
                f"{api_class.__name__} does not support text generation and tools. Skipping tests."
            )

        # Print the model name:
        print("\n" + model_name)

        # A simple message:
        user_message = Message(
            role="user", text="When will my order order_12345 be delivered?"
        )
        messages = [user_message]

        # Get the completion:
        response = api_instance.generate_text(
            model_name=model_name,  # or a specific model name if needed
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
        assert response[-2][0].has_type(Action)

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
        model_name = api_instance.get_best_model(
            [ModelFeatures.Tools, ModelFeatures.TextGeneration]
        )

        # If the model does not support text generation, skip the test:
        if model_name is None:
            pytest.skip(
                f"{api_class.__name__} does not support text generation and tools. Skipping tests."
            )

        # Print the model name:
        print("\n" + model_name)

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
            model_name=model_name,  # or a specific model name if needed
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
        assert response[-2][0].has_type(Action)

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
            object_block.get_content(), OrderInfo
        ), f"The response of {api_class.__name__} should be of type OrderInfo."

        # Check the content of the object:
        order_info = object_block.get_content()
        # Print the order info:
        print(order_info)
        # Check the order info:
        assert order_info.order_id == "order_12345"
        assert order_info.delivery_date == "2024-11-15"

        # Print the order info:
        print(order_info)

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
        model_name = api_instance.get_best_model(
            [ModelFeatures.Tools, ModelFeatures.TextGeneration]
        )

        if model_name is None:
            pytest.skip(
                f"{api_class.__name__} does not support text generation and tools. Skipping tests."
            )

        print("\n" + model_name)

        # Request message:
        user_message = Message(
            role="user", text="When will my order 'order_12345' be delivered?"
        )

        messages = [user_message]

        # Get the completion:
        response = api_instance.generate_text(
            model_name=model_name,  # or a specific model name if needed
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
            model_name=model_name,  # or a specific model name if needed
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
            model_name=model_name,  # or a specific model name if needed
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

    def test_api_text_generation_with_thinking(self, api_class):
        # Initialize the API instance
        api_instance = api_class()

        # Get the best model for text generation with thinking
        model_name = api_instance.get_best_model(
            [ModelFeatures.TextGeneration, ModelFeatures.Thinking]
        )
        # exclusion_filters=['deepseek','llava'])

        if model_name is None:
            # Skip the test if no model is found
            pytest.skip(
                f"{api_class.__name__} does not support text generation with thinking. Skipping tests."
            )

        # Print the model name
        print("\n" + model_name)

        # If the model does not support thinking, skip the test
        if model_name is None:
            print(f"{api_class.__name__} does not support thinking. Skipping test.")
            pytest.skip(
                f"{api_class.__name__} does not support thinking. Skipping test."
            )

        # A simple message
        messages = [
            Message(
                role="system",
                text="You are a helpful assistant that helps with any task.",
            ),
            Message(role="user", text="What is the integral of x^2 from 0 to 1?"),
        ]

        # Get the completion
        response = api_instance.generate_text(model_name=model_name, messages=messages)

        # get the last message in the response:
        response = response[0]  # Only one message is expected in list...
        print("\n" + str(response))

        # Check that we have a response
        assert response is not None, "No response received"

        # Exclude this test if the API is OpenAIApi:
        if not (
            api_class.__name__ == "OpenAIApi" or api_class.__name__ == "CombinedApi"
        ):
            # Check if there are MessageBlocks of type Thinking
            thinking_blocks = [
                block
                for block in response
                if block.has_type(Text) and block.is_thinking()
            ]
            assert len(thinking_blocks) > 0, "No thinking blocks found in the response"

        # Test with a follow-up question
        follow_up_messages = messages + [
            Message(role="user", text="Wait, are you sure?")
        ]
        follow_up_response = api_instance.generate_text(
            model_name=model_name, messages=follow_up_messages
        )

        # get the last message in the response:
        follow_up_response = follow_up_response[-1]

        # Print the response
        print("\n" + str(follow_up_response))

        # Check that we have a response
        assert follow_up_response is not None, "No follow-up response received"
