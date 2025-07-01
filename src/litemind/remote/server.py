# server.py
import threading
from typing import Dict, Optional

import rpyc
from arbol import aprint
from rpyc.utils.helpers import classpartial
from rpyc.utils.server import ThreadedServer

from litemind.utils.free_port import find_free_port


class ObjectService(rpyc.Service):
    """
    RPyC service that exposes registered objects.
    """

    def __init__(self, exposed_objects: Dict[str, object]):
        super().__init__()
        self._exposed_objects = exposed_objects
        aprint(f"🏗️ ObjectService initialized with {len(exposed_objects)} objects")

    def on_connect(self, conn):
        super().on_connect(conn)
        aprint(f"🔗 Client connected")

    def on_disconnect(self, conn):
        super().on_disconnect(conn)
        aprint("🔌 Client disconnected")

    def exposed_get_object(self, name: str):
        """Return an object by its registered name (or None if absent)."""
        try:
            obj = self._exposed_objects.get(name)
            if obj is not None:
                aprint(f"📤 Serving object '{name}' to client")
                # Add a debugging wrapper to see what arguments the agent receives
                return _DebuggingWrapper(obj)
            else:
                aprint(
                    f"❓ Object '{name}' not found in {list(self._exposed_objects.keys())}"
                )
            return obj
        except Exception as e:
            aprint(f"❌ Error serving object '{name}': {e}")
            return None

    def exposed_list_objects(self):
        """Return list of available object names."""
        return list(self._exposed_objects.keys())

    def exposed_ping(self):
        """Simple ping method for testing."""
        return "pong"


class Server:
    """A simple, elegant server for exposing litemind objects."""

    def __init__(self, host: Optional[str] = None, port: Optional[int] = None):
        """
        Initializes the server.

        Args:
            host (str): The hostname to bind to.
            port (int): The port to listen on.
        """

        # Get free port if needed:
        if host is None or port is None:
            hostname, free_port = find_free_port()
            host = host or hostname
            port = port or free_port

        self.host = host
        self.port = port
        self._exposed_objects: Dict[str, object] = {}
        self._server = None
        self._thread = None
        aprint(f"✨ LiteMind RPC Server initialized. Ready to expose objects.")

    def get_port(self) -> int:
        return self.port

    def expose(self, name: str, obj: object):
        """
        Makes an object available to remote clients under a specific name.

        Args:
            name (str): The name to register the object under.
            obj (object): The object instance to expose.
        """
        aprint(f"  📦 Exposing '{name}': {obj}")
        self._exposed_objects[name] = obj

    def start(self, block=True):
        """
        Starts the RPC server.

        Args:
            block (bool): If True (default), the server runs forever.
                          If False, it runs in a background thread.
        """

        # Use classpartial to create a service class with the exposed objects
        service_class = classpartial(ObjectService, self._exposed_objects)

        try:
            self._server = ThreadedServer(
                service_class,  # Pass the partial class
                hostname=self.host,
                port=self.port,
                protocol_config={
                    "allow_all_attrs": True,
                    "allow_pickle": True,
                    "sync_request_timeout": 30,
                },
            )

            aprint(f"\n🚀 Server starting on {self.host}:{self.port}")
            if block:
                self._server.start()
            else:
                # Start in a non-blocking background thread
                self._thread = threading.Thread(target=self._server.start, daemon=True)
                self._thread.start()
                aprint("   Server running in the background.")

                # Give the server a moment to fully start
                import time

                time.sleep(0.3)  # Increased wait time

        except Exception as e:
            aprint(f"❌ Failed to start server: {e}")
            raise

    def close(self):
        """Stops the server."""
        try:
            if self._server:
                aprint("🛑 Server shutting down.")
                self._server.close()

            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=1.0)

        except Exception as e:
            aprint(f"⚠️ Error during server shutdown: {e}")

    def is_running(self) -> bool:
        """Check if the server is currently running."""
        return self._server is not None and (
            self._thread is None or self._thread.is_alive()
        )


class _DebuggingWrapper:
    """
    Server-side wrapper to debug what arguments the agent receives.
    """

    def __init__(self, wrapped_obj):
        self._wrapped_obj = wrapped_obj

    def __call__(self, *args, **kwargs):
        """Debug arguments before calling the wrapped object."""
        import inspect

        aprint(f"🔍 Server received {len(args)} args and {len(kwargs)} kwargs")

        for key, value in kwargs.items():
            aprint(
                f"🔍 Server kwarg '{key}': {value} (type: {type(value)}, id: {id(value)})"
            )
            if key == "response_format":
                aprint(f"🔍 response_format is class? {inspect.isclass(value)}")
                if hasattr(value, "__module__"):
                    aprint(f"🔍 response_format module: {value.__module__}")
                if hasattr(value, "__name__"):
                    aprint(f"🔍 response_format name: {value.__name__}")

        # Call the wrapped object normally
        return self._wrapped_obj(*args, **kwargs)

    def __getattr__(self, name):
        """Forward all other attribute access to the wrapped object."""
        return getattr(self._wrapped_obj, name)
