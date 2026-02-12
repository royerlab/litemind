"""
RPyC server for exposing litemind objects to remote clients.

Provides the :class:`Server` class which registers arbitrary Python objects
(typically :class:`~litemind.agent.agent.Agent` instances) and serves them
over RPyC so that remote :class:`~litemind.remote.client.Client` instances
can call them as if they were local.
"""

import threading
from typing import Dict, Optional

import rpyc
from arbol import aprint
from rpyc.utils.helpers import classpartial
from rpyc.utils.server import ThreadedServer

from litemind.utils.free_port import find_free_port


class ObjectService(rpyc.Service):
    """RPyC service that exposes registered Python objects to remote clients.

    Each instance holds a dictionary of named objects and provides
    ``exposed_get_object``, ``exposed_list_objects``, and ``exposed_ping``
    endpoints that RPyC clients can call.

    Parameters
    ----------
    exposed_objects : Dict[str, object]
        Mapping of names to the Python objects to be served.
    """

    def __init__(self, exposed_objects: Dict[str, object]):
        super().__init__()
        self._exposed_objects = exposed_objects
        aprint(f"🏗️ ObjectService initialized with {len(exposed_objects)} objects")

    def on_connect(self, conn):
        """Handle a new client connection.

        Parameters
        ----------
        conn : rpyc.Connection
            The RPyC connection object for the newly connected client.
        """
        super().on_connect(conn)
        aprint("🔗 Client connected")

    def on_disconnect(self, conn):
        """Handle a client disconnection.

        Parameters
        ----------
        conn : rpyc.Connection
            The RPyC connection object for the disconnecting client.
        """
        super().on_disconnect(conn)
        aprint("🔌 Client disconnected")

    def exposed_get_object(self, name: str):
        """Return a server-wrapped object by its registered name.

        Parameters
        ----------
        name : str
            The registered name of the object.

        Returns
        -------
        _ServerWrapper or None
            The wrapped object, or ``None`` if no object is registered
            under *name*.
        """
        try:
            obj = self._exposed_objects.get(name)
            if obj is not None:
                aprint(f"📤 Serving object '{name}' to client")
                # Wrap the object to handle RPyC netref arguments
                return _ServerWrapper(obj)
            else:
                aprint(
                    f"❓ Object '{name}' not found in {list(self._exposed_objects.keys())}"
                )
            return obj
        except Exception as e:
            aprint(f"❌ Error serving object '{name}': {e}")
            return None

    def exposed_list_objects(self):
        """Return the list of registered object names.

        Returns
        -------
        list of str
            Names under which objects have been exposed.
        """
        return list(self._exposed_objects.keys())

    def exposed_ping(self):
        """Return ``"pong"`` -- used by clients to verify connectivity.

        Returns
        -------
        str
            The literal string ``"pong"``.
        """
        return "pong"


class Server:
    """RPyC server for exposing litemind objects to remote clients.

    Objects are registered with :meth:`expose` and become available to any
    ``Client`` that connects to this server. The server can run in the
    foreground (blocking) or in a background daemon thread.
    """

    def __init__(self, host: Optional[str] = None, port: Optional[int] = None):
        """
        Initialize the server.

        Parameters
        ----------
        host : str or None, optional
            The hostname to bind to. If ``None``, an available local
            address is selected automatically.
        port : int or None, optional
            The port to listen on. If ``None``, a free port is selected
            automatically.
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
        aprint("✨ LiteMind RPC Server initialized. Ready to expose objects.")

    def get_port(self) -> int:
        """Return the port the server is (or will be) listening on.

        Returns
        -------
        int
            The configured port number.
        """
        return self.port

    def expose(self, name: str, obj: object):
        """Register an object so that remote clients can access it by name.

        Parameters
        ----------
        name : str
            The name to register the object under. Clients use this name
            to retrieve the object via ``Client.get(name)``.
        obj : object
            The object instance to expose.
        """
        aprint(f"  📦 Exposing '{name}': {obj}")
        self._exposed_objects[name] = obj

    def start(self, block=True):
        """Start the RPyC server.

        Parameters
        ----------
        block : bool, optional
            If ``True`` (the default), the call blocks and the server
            runs in the foreground until stopped.  If ``False``, the
            server runs in a daemon background thread.
        """

        # Use classpartial to create a service class with the exposed objects
        service_class = classpartial(ObjectService, self._exposed_objects)

        try:
            self._server = ThreadedServer(
                service_class,  # Pass the partial class
                hostname=self.host,
                port=self.port,
                protocol_config={
                    "allow_pickle": True,  # Required for complex objects like Messages
                    "allow_all_attrs": True,  # Required for attribute access on netrefs
                    "allow_public_attrs": True,
                    "sync_request_timeout": 300,  # 5 minutes for long-running API calls
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
        """Stop the server and join the background thread if one exists."""
        try:
            if self._server:
                aprint("🛑 Server shutting down.")
                self._server.close()

            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=1.0)

        except Exception as e:
            aprint(f"⚠️ Error during server shutdown: {e}")

    def is_running(self) -> bool:
        """Check whether the server is currently running.

        Returns
        -------
        bool
            ``True`` if the server has been started and has not been
            closed, ``False`` otherwise.
        """
        return self._server is not None and (
            self._thread is None or self._thread.is_alive()
        )


class _ServerWrapper:
    """Server-side wrapper that materialises RPyC netref arguments before forwarding.

    When a client sends arguments to a remote callable, RPyC transmits them
    as proxy references (netrefs).  This wrapper intercepts ``__call__`` and
    uses ``rpyc.utils.classic.obtain()`` to pull every argument by value so
    that the wrapped object receives ordinary local Python objects.

    Parameters
    ----------
    wrapped_obj : object
        The local Python object to be wrapped.
    """

    def __init__(self, wrapped_obj):
        """
        Initialise the wrapper around a local object.

        Parameters
        ----------
        wrapped_obj : object
            The local Python object to wrap.
        """
        self._wrapped_obj = wrapped_obj

    def __call__(self, *args, **kwargs):
        """Materialise remote arguments and call the wrapped object.

        All positional and keyword arguments are pulled by value using
        ``rpyc.utils.classic.obtain()`` before being forwarded to the
        wrapped callable.

        Parameters
        ----------
        *args : object
            Positional arguments (potentially RPyC netrefs) to materialise.
        **kwargs : object
            Keyword arguments (potentially RPyC netrefs) to materialise.

        Returns
        -------
        object
            The return value of the wrapped callable.
        """
        import rpyc.utils.classic

        # Materialize all arguments from netrefs to local copies
        processed_args = []
        for arg in args:
            try:
                # obtain() pulls remote objects by value
                processed_args.append(rpyc.utils.classic.obtain(arg))
            except Exception:
                # Fallback if obtain fails
                processed_args.append(arg)

        processed_kwargs = {}
        for key, value in kwargs.items():
            try:
                processed_kwargs[key] = rpyc.utils.classic.obtain(value)
            except Exception:
                processed_kwargs[key] = value

        return self._wrapped_obj(*processed_args, **processed_kwargs)

    def __getattr__(self, name):
        """Forward attribute access to the wrapped object.

        Parameters
        ----------
        name : str
            The attribute name to look up on the wrapped object.

        Returns
        -------
        object
            The attribute value from the wrapped object.
        """
        return getattr(self._wrapped_obj, name)

    def __getitem__(self, key):
        """Forward item access to the wrapped object for dict-like interfaces.

        Parameters
        ----------
        key : object
            The key to look up on the wrapped object.

        Returns
        -------
        object
            The value associated with *key* on the wrapped object.
        """
        return self._wrapped_obj[key]
