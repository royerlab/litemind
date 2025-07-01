import pytest

from litemind.apis.providers.ollama.ollama_server import OllamaServer


@pytest.mark.timeout(20)
def test_ollama_server_lifecycle():
    # Use a non-default port to avoid conflicts
    test_port = 11500
    server = OllamaServer(port=test_port)

    # Ensure server is not running
    assert not server.is_running()

    # Start the server
    process = server.start()
    assert process is not None
    assert server.is_running()

    # Stop the server
    server.stop()
    # Give it a moment to shut down
    import time

    time.sleep(1)
    assert not server.is_running()
