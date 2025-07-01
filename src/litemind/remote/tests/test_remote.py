# tests/test_rpc.py

import socket
import time

import pytest

from litemind.remote.client import Client
from litemind.remote.server import Server


# A dummy class to simulate the litemind.agent.Agent for isolated testing.
class DummyAgent:
    """A mock agent class for testing purposes."""

    def __init__(self, name="Test Agent"):
        self.name = name
        self.call_count = 0

    def think(self, prompt):
        """A mock method that returns a predictable response."""
        self.call_count += 1
        return f"Agent '{self.name}' thought about: {prompt}"

    def get_call_count(self):
        """Returns how many times think() was called."""
        return self.call_count

    def __str__(self):
        return f"<DummyAgent: {self.name}>"


# Pytest Fixture to manage the server lifecycle for tests.
@pytest.fixture(scope="function")
def running_server():
    """
    A pytest fixture that starts the RPC Server in a background thread
    for a test and safely shuts it down afterward.
    """
    # Find a free port to avoid conflicts
    sock = socket.socket()
    sock.bind(("", 0))
    host, port = sock.getsockname()
    sock.close()

    # 1. Setup the Server
    server = Server(host=host, port=port)

    # Expose a few objects for testing
    main_agent = DummyAgent(name="Main")
    research_agent = DummyAgent(name="Researcher")
    simple_dict = {"key": "value", "number": 123}

    server.expose("main_agent", main_agent)
    server.expose("research_agent", research_agent)
    server.expose("config_dict", simple_dict)

    # Start the server in a non-blocking background thread
    server.start(block=False)

    # Give the server a moment to start up
    time.sleep(0.1)

    # Yield the server and client _connection details to the test
    yield server, host, port

    # 2. Teardown the Server
    server.close()


# --- Test Cases ---


def test_server_expose_objects():
    """
    Tests that the server correctly registers objects to be exposed.
    """
    server = Server()
    agent = DummyAgent()
    server.expose("test_agent", agent)

    # Check internal state to ensure the object is registered
    assert "test_agent" in server._exposed_objects
    assert server._exposed_objects["test_agent"] is agent


def test_client_connection_fails_gracefully():
    """
    Tests that the Client handles a _connection failure without crashing.
    """
    # Attempt to connect to a port where no server is running
    client = Client(port=9999)
    assert client._connection is None

    # Methods should not raise exceptions
    assert client.get("any_agent") is None
    client.close()  # Should not raise an error


def test_full_connection_and_object_retrieval(running_server):
    """
    Tests the full end-to-end _connection from client to server and
    retrieving a remote object.
    """
    server, host, port = running_server
    client = Client(host=host, port=port)

    assert client._connection is not None

    # Get an object using the .get() method
    remote_agent = client.get("main_agent")

    assert remote_agent is not None
    # rpyc wraps the object, so we check a known attribute
    assert remote_agent.name == "Main"

    client.close()


def test_remote_method_call(running_server):
    """
    Tests calling a method on a remote object and verifying the result.
    """
    server, host, port = running_server
    client = Client(host=host, port=port)

    remote_agent = client["main_agent"]
    prompt = "the meaning of life"
    expected_response = "Agent 'Main' thought about: the meaning of life"

    response = remote_agent.think(prompt)

    assert response == expected_response

    # Verify that the state on the server-side object was mutated
    assert remote_agent.get_call_count() == 1

    client.close()


def test_remote_attribute_access(running_server):
    """
    Tests accessing an attribute of a remote object.
    """
    server, host, port = running_server
    client = Client(host=host, port=port)

    remote_agent = client.get("research_agent")

    assert remote_agent.name == "Researcher"

    client.close()


def test_client_dict_style_access(running_server):
    """
    Tests the __getitem__ functionality for more elegant access.
    """
    server, host, port = running_server
    client = Client(host=host, port=port)

    remote_agent = client["main_agent"]

    assert remote_agent is not None
    assert remote_agent.name == "Main"

    client.close()


def test_get_non_existent_object(running_server):
    """
    Tests that retrieving an object that was not exposed returns None.
    """
    server, host, port = running_server
    client = Client(host=host, port=port)

    non_existent_agent = client.get("non_existent_agent")

    assert non_existent_agent is None

    client.close()


def test_expose_and_get_multiple_objects(running_server):
    """
    Tests that multiple, different objects can be exposed and retrieved correctly.
    """
    server, host, port = running_server
    client = Client(host=host, port=port)

    # Get the first agent
    main_agent = client["main_agent"]
    assert main_agent is not None
    assert main_agent.name == "Main"
    assert main_agent.think("test1") == "Agent 'Main' thought about: test1"

    # Get the second agent
    research_agent = client["research_agent"]
    assert research_agent is not None
    assert research_agent.name == "Researcher"
    assert research_agent.think("test2") == "Agent 'Researcher' thought about: test2"

    client.close()


def test_expose_non_class_object(running_server):
    """
    Tests that the server can expose objects that are not class instances,
    like dictionaries, and that they are retrieved correctly.
    """
    server, host, port = running_server
    client = Client(host=host, port=port)

    remote_dict = client["config_dict"]

    assert remote_dict is not None
    # rpyc may wrap the dict, but attribute/item access should work
    assert remote_dict["key"] == "value"
    assert remote_dict["number"] == 123

    client.close()
