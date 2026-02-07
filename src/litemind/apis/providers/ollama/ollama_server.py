import socket
import subprocess
from time import sleep

from arbol import aprint, asection


class OllamaServer:
    """Manage the local Ollama server process lifecycle."""

    def __init__(self, host="127.0.0.1", port=11434):
        """Initialize the OllamaServer instance.

        Parameters
        ----------
        host : str
            Host address for the Ollama server.
        port : int
            Port number for the Ollama server.
        """
        self.host = host
        self.port = port
        self._ollama_process = None

    def start(self):
        """
        Start the Ollama server if it is not already running.
        """
        if self.is_running():
            aprint("Ollama is already running!")
        else:
            with asection("Starting Ollama server!"):
                process = subprocess.Popen(
                    "ollama serve",
                    shell=True,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                sleep(1)
                aprint(process.stdout.readline())
                aprint(process.stdout.readline())
                self._ollama_process = process
                return process

    def stop(self):
        """
        Stop the Ollama server if it is running.
        """
        with asection("Stopping Ollama server."):
            if self._ollama_process:
                self._ollama_process.terminate()
                self._ollama_process = None

    def is_running(self):
        """Check if the Ollama server is running.

        Returns
        -------
        bool
            True if the server is listening on the configured host/port.
        """
        return self._is_listening(self.host, self.port)

    def _is_listening(self, ip, port):
        """Check if a TCP port is open on the given IP address.

        Parameters
        ----------
        ip : str
            IP address to check.
        port : int
            Port number to check.

        Returns
        -------
        bool
            True if the port is open, False otherwise.
        """
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            try:
                sock.connect((ip, port))
                return True
            except socket.error:
                return False
            finally:
                sock.close()
        except Exception:
            return False

    def get_models(self):
        """Retrieve the list of available Ollama models.

        Returns
        -------
        List[str]
            List of model names available on the server.
        """
        result = subprocess.run(["ollama", "list"], stdout=subprocess.PIPE, text=True)
        lines = result.stdout.strip().split("\n")
        models = [line.split()[0] for line in lines[1:]]
        return models
