import socket
import subprocess
import traceback
from time import sleep

from arbol import aprint, asection


class OllamaServer:
    """
    Class to manage the Ollama server process and related operations.
    """

    def __init__(self, host="127.0.0.1", port=11434):
        """
        Initialize the OllamaServer instance.

        :param host: Host address for the Ollama server.
        :param port: Port number for the Ollama server.
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
        """
        Check if the Ollama server is running.

        :return: True if the server is running, False otherwise.
        """
        return self._is_listening(self.host, self.port)

    def _is_listening(self, ip, port):
        """
        Check if a TCP port is open on the given IP address.

        :param ip: IP address to check.
        :param port: Port number to check.
        :return: True if the port is open, False otherwise.
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
            traceback.print_exc()
            return False

    def get_models(self):
        """
        Retrieve the list of available Ollama models.

        :return: List of model names.
        """
        result = subprocess.run(["ollama", "list"], stdout=subprocess.PIPE, text=True)
        lines = result.stdout.strip().split("\n")
        models = [line.split()[0] for line in lines[1:]]
        return models
