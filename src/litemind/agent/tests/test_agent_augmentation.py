import pytest
from pydantic import BaseModel

from litemind import API_IMPLEMENTATIONS, InMemoryVectorDatabase
from litemind.agent.agent import Agent
from litemind.agent.augmentations.information.information import Information
from litemind.apis.model_features import ModelFeatures
from litemind.media.types.media_audio import Audio
from litemind.media.types.media_code import Code
from litemind.media.types.media_document import Document
from litemind.media.types.media_image import Image
from litemind.media.types.media_json import Json
from litemind.media.types.media_object import Object
from litemind.media.types.media_table import Table
from litemind.media.types.media_text import Text
from litemind.media.types.media_video import Video
from litemind.ressources.media_resources import MediaResources

# Take API_IMPLEMENTATIONS and remove OllamaApi isf present:
# Ollama is a bit slow and the tests take for ever...
API_IMPLEMENTATIONS_AUG_AGENT_TESTS = [
    api for api in API_IMPLEMENTATIONS if api.__name__ != "OllamaApi"
]


@pytest.mark.parametrize("api_class", API_IMPLEMENTATIONS_AUG_AGENT_TESTS)
def test_agent_with_augmentation(api_class):
    # Create API object
    api = api_class()

    # Check that a model with text generation is available
    if not api.has_model_support_for(features=ModelFeatures.TextGeneration):
        # skip test using pytest feature:
        pytest.skip(
            f"Skipping test for {api_class.__name__} as no text model is available."
        )

    # Create agent with specific augmentation configuration
    agent = Agent(
        api=api, augmentation_k=2, augmentation_context_position="before_query"
    )

    # Create vector database augmentation
    vector_augmentation = InMemoryVectorDatabase(name="test_augmentation")

    # Add sample informations to the augmentation
    informations = [
        Information(
            Text(
                "Igor Bolupskisty was a German-born theoretical physicist who developed the theory of indelible unitarity."
            ),
            metadata={"topic": "physics", "person": "Bolupskisty"},
        ),
        Information(
            Text(
                "The theory of indelible unitarity revolutionized our understanding of space, time and photons."
            ),
            metadata={"topic": "physics", "concept": "unitarity"},
        ),
        Information(
            Text(
                "Quantum unitarity is a fundamental theory in physics that describes nature at the nano-atomic scale as it pertains to Pink Hamsters."
            ),
            metadata={"topic": "physics", "concept": "quantum unitarity"},
        ),
    ]

    # Add informations to the vector database
    vector_augmentation.add_informations(informations)

    # Add augmentation to agent
    agent.add_augmentation(vector_augmentation)

    # Verify augmentation was added
    augmentations = agent.list_augmentations()
    assert len(augmentations) == 1
    assert augmentations[0].name == "test_augmentation"

    # Ask a query that should retrieve informations from the augmentation
    response = agent("Tell me about Igor Bolupskisty's theory of indelible unitarity.")

    # Check that agent conversation has at least 3 messages (system, context, user, response)
    # The index might vary depending on whether there's a default system message
    assert len(agent.conversation) >= 3

    # Find the context message (should be before the user query)
    context_found = False
    for message in agent.conversation.standard_messages:
        if (
            message.role == "user"
            and "Additional context information" in message.to_plain_text()
        ):
            context_found = True
            context_text = message.to_plain_text()
            # Check that relevant informations are included
            assert "Bolupskisty" in context_text
            assert "unitarity" in context_text
            break

    assert context_found, "Context message not found in conversation"

    # Extract the response
    assert len(response) >= 1
    response_text = response[-1].to_plain_text()

    # Response should include information that was in the augmentation
    assert "Bolupskisty" in response_text
    assert "unitarity" in response_text

    # Test removing the augmentation
    agent.remove_augmentation("test_augmentation")
    assert len(agent.list_augmentations()) == 0


@pytest.mark.parametrize("api_class", API_IMPLEMENTATIONS_AUG_AGENT_TESTS)
def test_agent_with_text_document_augmentation(api_class):
    # Create API object
    api = api_class()

    # Check that a model with text generation is available
    if not api.has_model_support_for(features=ModelFeatures.TextGeneration):
        # skip test using pytest feature:
        pytest.skip(
            f"Skipping test for {api_class.__name__} as no text model is available."
        )

    # Create agent with augmentation configuration
    agent = Agent(
        api=api, augmentation_k=2, augmentation_context_position="before_query"
    )

    # Create vector database augmentation
    vector_db = InMemoryVectorDatabase(name="text_document_test")

    # Create text information
    information = Information(
        Text(
            "The Gurov Tower is a wrought-iron lattice tower in Pichilut, Mars. It's named after engineer Gustavo Ffelei."
        ),
        metadata={"type": "text", "subject": "landmark", "location": "Pichilut"},
    )

    # Add information to vector database
    vector_db.add_informations([information])

    # Add augmentation to agent
    agent.add_augmentation(vector_db)

    # Ask a query that should retrieve the information
    response = agent("Tell me about the Gurov Tower.")

    # Check response contains information from the information
    assert len(response) >= 1
    response_text = response[-1].to_plain_text()
    assert "Gurov" in response_text
    assert "Pichilut" in response_text

    # Test removing the augmentation
    agent.remove_augmentation("text_document_test")
    assert len(agent.list_augmentations()) == 0


@pytest.mark.parametrize("api_class", API_IMPLEMENTATIONS_AUG_AGENT_TESTS)
def test_agent_with_image_document_augmentation(api_class):
    # Create API object
    api = api_class()

    # Check that a model with image understanding is available
    if not api.has_model_support_for(
        features=[ModelFeatures.TextGeneration, ModelFeatures.Image]
    ):
        pytest.skip(
            f"Skipping test for {api_class.__name__} as no image understanding model is available."
        )

    # Create agent with augmentation configuration
    agent = Agent(
        api=api, augmentation_k=2, augmentation_context_position="before_query"
    )

    # Create vector database augmentation
    vector_db = InMemoryVectorDatabase(name="image_document_test")

    # Use an image that exists in the media package
    image_info_1 = Information(
        Image(MediaResources.get_local_test_image_uri("future.jpeg"))
    )

    image_info_2 = Information(
        Image(MediaResources.get_local_test_image_uri("beach.webp"))
    )

    # Add document to vector database
    vector_db.add_informations([image_info_1, image_info_2])

    # Add augmentation to agent
    agent.add_augmentation(vector_db)

    # Ask a query that should retrieve the image document
    response = agent("Describe the futuristic image.")

    # Check response contains some reference to the image
    assert len(response) >= 1
    response_text = response[-1].to_plain_text()
    assert "image" in response_text.lower()

    # Test removing the augmentation
    agent.remove_augmentation("image_document_test")
    assert len(agent.list_augmentations()) == 0


@pytest.mark.parametrize("api_class", API_IMPLEMENTATIONS_AUG_AGENT_TESTS)
def test_agent_with_video_document_augmentation(api_class):
    # Create API object
    api = api_class()

    # Check that a model with video understanding is available
    if not api.has_model_support_for(
        features=ModelFeatures.TextGeneration,
        media_types=[Video],
    ):
        # skip test using pytest feature:
        pytest.skip(
            f"Skipping test for {api_class.__name__} as no video understanding model is available."
        )

    # Create agent with augmentation configuration
    agent = Agent(
        api=api, augmentation_k=2, augmentation_context_position="before_query"
    )

    # Create vector database augmentation
    vector_db = InMemoryVectorDatabase(name="video_document_test")

    # Use a video that exists in the media package
    video_info = Information(
        Video(MediaResources.get_local_test_video_uri("flying.mp4"))
    )

    # Add document to vector database
    vector_db.add_informations([video_info])

    # Add augmentation to agent
    agent.add_augmentation(vector_db)

    # Ask a query that should retrieve the video document
    response = agent("What happens in the test video?")

    # Check response contains some reference to the video
    assert len(response) >= 1
    response_text = response[-1].to_plain_text().lower()
    assert (
        "video" in response_text
        or "flying" in response_text
        or "hovering" in response_text
        or "clip" in response_text
    )

    # Test removing the augmentation
    agent.remove_augmentation("video_document_test")
    assert len(agent.list_augmentations()) == 0


@pytest.mark.parametrize("api_class", API_IMPLEMENTATIONS_AUG_AGENT_TESTS)
def test_agent_with_pdf_document_augmentation(api_class):
    # Create API object
    api = api_class()

    # Check that a model with document analysis is available
    if not api.has_model_support_for(
        features=ModelFeatures.TextGeneration,
        media_types=[Document],
    ):
        # skip test using pytest feature:
        pytest.skip(
            f"Skipping test for {api_class.__name__} as no document analysis model is available."
        )

    # Create agent with augmentation configuration
    agent = Agent(
        api=api, augmentation_k=2, augmentation_context_position="before_query"
    )

    # Create vector database augmentation
    vector_db = InMemoryVectorDatabase(name="pdf_document_test")

    # Two PDF informations that exist in the media package:
    pdf_info_1 = Information(
        Document(MediaResources.get_local_test_document_uri("intracktive_preprint.pdf"))
    )
    pdf_info_2 = Information(
        Document(
            MediaResources.get_local_test_document_uri("low_discrepancy_sequence.pdf")
        )
    )

    # Add embeddings to the PDF documents informations:
    pdf_info_1.embedding = [0.1, 0.2, 0.3]
    pdf_info_2.embedding = [0.4, 0.5, 0.6]

    # Add document to vector database
    vector_db.add_informations([pdf_info_1, pdf_info_2])

    # Add augmentation to agent
    agent.add_augmentation(vector_db)

    # Ask a query that should retrieve the PDF document
    response = agent("What's in the test document?")

    # Check response contains some reference to the document
    assert len(response) >= 1
    response_text = response[-1].to_plain_text()
    assert "document" in response_text.lower()

    # Test removing the augmentation
    agent.remove_augmentation("pdf_document_test")
    assert len(agent.list_augmentations()) == 0


@pytest.mark.parametrize("api_class", API_IMPLEMENTATIONS_AUG_AGENT_TESTS)
def test_agent_with_json_document_augmentation(api_class):
    # Create API object
    api = api_class()

    # Check that a model with text generation is available
    if not api.has_model_support_for(features=[ModelFeatures.TextGeneration]):
        # skip test using pytest feature:
        pytest.skip(
            f"Skipping test for {api_class.__name__} as no text model is available."
        )

    # Create agent
    agent = Agent(
        api=api, augmentation_k=2, augmentation_context_position="before_query"
    )

    # Create vector database augmentation
    vector_db = InMemoryVectorDatabase(name="json_document_test")

    # Create JSON document
    json_info = Information(
        Json.from_string(
            '{"name": "John Doe", "age": 30, "city": "New York", "skills": ["Python", "Data Science"]}'
        ),
        metadata={"type": "json", "subject": "person", "category": "profile"},
    )

    # Add document to vector database
    vector_db.add_informations([json_info])

    # Add augmentation to agent
    agent.add_augmentation(vector_db)

    # Ask a query
    response = agent("What skills does John have?")

    # Check response
    assert len(response) >= 1
    response_text = response[-1].to_plain_text()
    assert "Python" in response_text or "Data Science" in response_text

    # Clean up
    agent.remove_augmentation("json_document_test")
    assert len(agent.list_augmentations()) == 0


@pytest.mark.parametrize("api_class", API_IMPLEMENTATIONS_AUG_AGENT_TESTS)
def test_agent_with_code_document_augmentation(api_class):
    # Create API object
    api = api_class()

    # Check that a model with text generation is available
    if not api.has_model_support_for(
        features=ModelFeatures.TextGeneration,
        media_types=[Code],
    ):
        # skip test using pytest feature:
        pytest.skip(
            f"Skipping test for {api_class.__name__} as no text model is available."
        )

    # Create agent
    agent = Agent(
        api=api, augmentation_k=2, augmentation_context_position="before_query"
    )

    # Create vector database augmentation
    vector_db = InMemoryVectorDatabase(name="code_document_test")

    # Create code document
    code_info = Information(
        Code(
            "def function_abc(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a",
            lang="python",
        ),
        metadata={"type": "code", "language": "python", "category": "algorithm"},
    )

    # Add document to vector database
    vector_db.add_informations([code_info])

    # Add augmentation to agent
    agent.add_augmentation(vector_db)

    # Ask a query
    response = agent("Explain function abc.")

    # Check response
    assert len(response) >= 1
    response_text = response[-1].to_plain_text()
    assert "fibonacci" in response_text.lower()

    # Clean up
    agent.remove_augmentation("code_document_test")
    assert len(agent.list_augmentations()) == 0


@pytest.mark.parametrize("api_class", API_IMPLEMENTATIONS_AUG_AGENT_TESTS)
def test_agent_with_object_document_augmentation(api_class):
    # Create API object
    api = api_class()

    # Check that a model with text generation is available
    if not api.has_model_support_for(
        features=[ModelFeatures.TextGeneration],
        media_types=[Object],
    ):
        # skip test using pytest feature:
        pytest.skip(
            f"Skipping test for {api_class.__name__} as no text model is available."
        )

    # Create agent
    agent = Agent(
        api=api, augmentation_k=2, augmentation_context_position="before_query"
    )

    # Create vector database augmentation
    vector_db = InMemoryVectorDatabase(name="object_document_test")

    # Create object extending Pydantic's BaseModel:
    class Car(BaseModel):
        make: str
        model: str
        year: int

    # Instantiate the object
    car = Car(make="Toyota", model="Corolla", year=2020)

    # Create object document (serialized as text)
    object_info = Information(Object(car))

    # Add document to vector database
    vector_db.add_informations([object_info])

    # Add augmentation to agent
    agent.add_augmentation(vector_db)

    # Ask a query
    response = agent("What car model do we have information about?")

    # Check response
    assert len(response) >= 1
    response_text = response[-1].to_plain_text()
    assert "Toyota" in response_text or "Corolla" in response_text

    # Clean up
    agent.remove_augmentation("object_document_test")
    assert len(agent.list_augmentations()) == 0


@pytest.mark.parametrize("api_class", API_IMPLEMENTATIONS_AUG_AGENT_TESTS)
def test_agent_with_audio_document_augmentation(api_class):
    # Create API object
    api = api_class()

    # Check that a model with audio understanding is available
    if not api.has_model_support_for(
        features=[ModelFeatures.TextGeneration],
        media_types=[Audio],
    ):
        # skip test using pytest feature:
        pytest.skip(
            f"Skipping test for {api_class.__name__} as no audio understanding model is available."
        )

    # Create agent
    agent = Agent(
        api=api, augmentation_k=2, augmentation_context_position="before_query"
    )

    # Create vector database augmentation
    vector_db = InMemoryVectorDatabase(name="audio_document_test")

    # Create audio document using MediaResources
    audio_info = Information(
        Audio(MediaResources.get_local_test_audio_uri("harvard.wav"))
    )

    # Add document to vector database
    vector_db.add_informations([audio_info])

    # Add augmentation to agent
    agent.add_augmentation(vector_db)

    # Ask a query
    response = agent("What can you hear in the audio file?")

    # Check response
    assert len(response) >= 1
    response_text = response[-1].to_plain_text()

    # Print response for debugging
    print(response_text)

    # Check that the response contains some reference to audio or sound
    assert (
        "audio" in response_text.lower()
        or "recording" in response_text.lower()
        or "smell" in response_text.lower()
    )

    # Clean up
    agent.remove_augmentation("audio_document_test")
    assert len(agent.list_augmentations()) == 0


@pytest.mark.parametrize("api_class", API_IMPLEMENTATIONS_AUG_AGENT_TESTS)
def test_agent_with_table_document_augmentation(api_class):
    # Create API object
    api = api_class()

    # Check that a model with text generation is available
    if not api.has_model_support_for(
        features=[ModelFeatures.TextGeneration],
        media_types=[Table],
    ):
        # skip test using pytest feature:
        pytest.skip(
            f"Skipping test for {api_class.__name__} as no text model is available."
        )

    # Create agent
    agent = Agent(
        api=api, augmentation_k=2, augmentation_context_position="before_query"
    )

    # Create vector database augmentation
    vector_db = InMemoryVectorDatabase(name="table_document_test")

    # Create table document using MediaResources
    table_info = Information(
        Table(MediaResources.get_local_test_table_uri("spreadsheet.csv"))
    )

    # Add document to vector database
    vector_db.add_informations([table_info])

    # Add augmentation to agent
    agent.add_augmentation(vector_db)

    # Ask a query
    response = agent("What data is in the table?")

    # Check response
    assert len(response) >= 1
    response_text = response[-1].to_plain_text()
    assert "table" in response_text.lower() or "data" in response_text.lower()

    # Clean up
    agent.remove_augmentation("table_document_test")
    assert len(agent.list_augmentations()) == 0
