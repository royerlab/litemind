import socket
import time

import pytest

from litemind import CombinedApi
from litemind.agent.agent import Agent
from litemind.agent.messages.message import Message
from litemind.apis.base_api import ModelFeatures
from litemind.media.types.media_audio import Audio
from litemind.remote.client import Client
from litemind.remote.server import Server

# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------


def find_free_port():
    """Find an available (host, port) tuple for testing."""
    sock = socket.socket()
    sock.bind(("localhost", 0))
    host, port = sock.getsockname()
    sock.close()
    return host, port


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture(scope="function")
def running_litemind_server():
    """Start a real LiteMind RPC server backed by a real Agent and yield host/port."""

    host, port = find_free_port()

    # ------------------------------------------------------------------
    # Server setup
    # ------------------------------------------------------------------
    server = Server(host=host, port=port)

    try:
        api = CombinedApi()

        # Try to find a single model that supports text generation + images + audio
        best_model = api.get_best_model(
            features=[ModelFeatures.TextGeneration, ModelFeatures.Image],
            media_types=[Audio],
        )
        if best_model is None:
            pytest.skip(
                "No model on this backend supports both images and audio. Skipping test."
            )

        real_agent = Agent(api=api, model=best_model)
    except Exception as e:
        pytest.skip(f"Failed to create agent with CombinedApi: {e}")

    server.expose("main_agent", real_agent)

    try:
        server.start(block=False)
    except Exception as e:
        pytest.skip(f"Failed to start server: {e}")

    # Wait until the port starts accepting connections so we don't race the tests
    max_wait, step = 5.0, 0.1
    waited = 0.0
    while waited < max_wait:
        try:
            with socket.create_connection((host, port), timeout=0.1):
                break
        except OSError:
            time.sleep(step)
            waited += step
    else:
        server.close()
        pytest.skip("Server failed to start within timeout")

    # Give the server a tiny bit more time so that rpyc is completely ready
    time.sleep(0.1)

    yield host, port

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------
    try:
        server.close()
    except Exception as e:
        print(f"Warning: Error during server cleanup: {e}")


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


def test_remote_agent_text_interaction(running_litemind_server):
    """Basic text‑only conversation sanity check."""
    host, port = running_litemind_server

    client = Client(host=host, port=port)
    if not client.is_connected():
        pytest.fail("Failed to connect to server")

    remote_agent = client["main_agent"]
    if remote_agent is None:
        client.close()
        pytest.fail("Failed to get remote agent")

    try:
        # --- Interact with the remote agent ----------------------------------
        remote_agent.append_system_message(
            "You are an omniscient all‑knowing being called Ohmm"
        )
        response = remote_agent("Who are you?")

        # --- Assertions -------------------------------------------------------
        assert response is not None, "Response should not be None"
        assert (
            len(response) == 1
        ), f"Expected 1 message in response, got {len(response)}"

        response_text = str(response[-1]).lower()
        assert (
            "ohmm" in response_text or "chatgpt" in response_text
        ), f"Response should contain 'ohmm', got: {response_text}"

        # Ensure the server maintained state
        remote_conversation = remote_agent.conversation
        assert (
            len(remote_conversation) == 3
        ), f"Expected 3 messages in conversation, got {len(remote_conversation)}"
        assert remote_conversation[0].role == "system" and "Ohmm" in str(
            remote_conversation[0]
        )
        assert remote_conversation[1].role == "user" and "Who are you?" in str(
            remote_conversation[1]
        )
        assert remote_conversation[2].role == "assistant"
    finally:
        client.close()


# -----------------------------------------------------------------------------
# New multimodal test replacing the old JSON / Pydantic test
# -----------------------------------------------------------------------------


def test_remote_agent_multimodal_conversation(running_litemind_server):
    """End‑to‑end test of a multimodal (image + audio) conversation with a remote Agent."""

    host, port = running_litemind_server
    client = Client(host=host, port=port)

    if not client.is_connected():
        pytest.fail("Failed to connect to server")

    remote_agent = client["main_agent"]
    if remote_agent is None:
        client.close()
        pytest.fail("Failed to get remote agent")

    try:
        # Build messages in the same style as the MediaResources unit‑tests
        messages = []

        system_msg = Message(role="system")
        system_msg.append_text("You are an omniscient all‑knowing being called Ohmm")
        messages.append(system_msg)

        user_msg = Message(role="user")
        user_msg.append_text(
            "Describe in detail both what you see in the image and what you hear in the audio clip."
        )
        image_url = (
            "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3e/Einstein_1921_by_"  # noqa: E501
            "F_Schmutzer_-_restoration.jpg/456px-Einstein_1921_by_F_Schmutzer_-_restoration.jpg"
        )
        audio_url = "https://salford.figshare.com/ndownloader/files/14630270"  # Harvard sentences wav
        user_msg.append_image(image_url)
        user_msg.append_audio(audio_url)
        messages.append(user_msg)

        # ------------------------------------------------------------------
        # Run the remote generation call *on the server* and bring back the
        # assistant response to the client side
        # ------------------------------------------------------------------
        response_messages = remote_agent(messages)
        # We only care about the assistant's reply
        assistant_reply = response_messages[-1]
        assistant_text = str(assistant_reply).lower()

        # ------------------------------------------------------------------
        # Assertions – we accept a variety of possible correct answers, just
        # like the upstream multimodal unit‑tests bundled with LiteMind.
        # ------------------------------------------------------------------
        assert (
            "einstein" in assistant_text
            or "black-and-white" in assistant_text
            or "photograph" in assistant_text
            or "smell" in assistant_text
            or "ham" in assistant_text
            or "reading" in assistant_text
            or "test passage" in assistant_text
        ), (
            "Assistant reply did not appear to describe either the image or the audio clip "
            f"(got: {assistant_text!r})"
        )

        # ------------------------------------------------------------------
        # Also make sure that the remote agent's conversation history grew as
        # expected (system + user + assistant).
        # ------------------------------------------------------------------
        remote_conversation = remote_agent.conversation
        assert (
            len(remote_conversation) == 3
        ), f"Expected 3 messages in remote conversation, got {len(remote_conversation)}"
        assert remote_conversation[0].role == "system"
        assert remote_conversation[1].role == "user"
        assert remote_conversation[2].role == "assistant"

    finally:
        client.close()


# -----------------------------------------------------------------------------
# Connection‑handling edge‑cases (unchanged)
# -----------------------------------------------------------------------------


def test_connection_failure():
    """Client should handle connection failures gracefully."""
    client = Client(host="localhost", port=99999)

    assert not client.is_connected()
    assert client.get("any_object") is None

    client.close()  # Should not raise


def test_object_not_found(running_litemind_server):
    """Missing objects on the server should be reported as None."""
    host, port = running_litemind_server
    client = Client(host=host, port=port)

    try:
        assert client.is_connected()
        missing = client["non_existent_object"]
        assert missing is None
    finally:
        client.close()
