import shutil

import pytest

from litemind.apis.providers.ollama.ollama_server import OllamaServer


def is_ollama_cli_available() -> bool:
    """Check if the ollama CLI is installed and available in PATH."""
    return shutil.which("ollama") is not None


@pytest.mark.timeout(20)
@pytest.mark.skipif(
    not is_ollama_cli_available(),
    reason="ollama CLI is not installed or not in PATH",
)
def test_ollama_server_lifecycle():
    # Use a non-default port to avoid conflicts
    test_port = 11500
    server = OllamaServer(port=test_port)

    # wait for the server to be ready
    import time

    time.sleep(3)

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
