import inspect

from arbol import aprint


class Client:
    """Client for interacting with remote litemind objects over RPyC.

    Connects to a ``Server`` instance and retrieves proxy objects that
    behave like local Python objects. Typical usage is to obtain a remote
    ``Agent`` and call it as if it were local.
    """

    def __init__(self, host="localhost", port=18861):
        """
        Initialize the client and connect to the RPyC server.

        Parameters
        ----------
        host : str, optional
            The hostname of the server, by default ``"localhost"``.
        port : int, optional
            The port of the server, by default ``18861``.
        """
        try:
            import rpyc

            self._connection = rpyc.connect(
                host,
                port,
                config={
                    "allow_pickle": True,  # Required for complex objects like Messages
                    "allow_all_attrs": True,  # Required for attribute access on netrefs
                    "allow_public_attrs": True,
                    "sync_request_timeout": 300,  # 5 minutes for long-running API calls
                },
            )
            self._remote_service = self._connection.root
            aprint(f"✨ Connected to LiteMind RPC Server at {host}:{port}")
        except ConnectionRefusedError:
            aprint(f"❌ Connection refused. Is the server running at {host}:{port}?")
            self._connection = None
        except Exception as e:
            aprint(f"❌ Connection failed: {e}")
            self._connection = None

    def is_connected(self) -> bool:
        """
        Check if the client is connected to the server.

        Returns
        -------
        bool
            True if connected, False otherwise.
        """
        try:
            return self._connection is not None and not self._connection.closed
        except Exception:
            return False

    def get(self, name):
        """
        Retrieve a proxy object for a named remote object.

        Pings the server to verify connectivity, lists available objects
        for diagnostics, then fetches and wraps the requested object so
        that class arguments are transmitted by value rather than as RPyC
        proxies.

        Parameters
        ----------
        name : str
            The registered name of the remote object to retrieve.

        Returns
        -------
        _ValueWrapper or None
            A proxy wrapper that behaves like the remote object, or
            ``None`` if the object was not found or the client is
            disconnected.
        """
        if not self.is_connected():
            aprint("❌ Not connected to server")
            return None

        try:
            aprint(f"  🔎 Requesting remote object '{name}'...")

            # First, let's test basic connectivity with a simple ping
            try:
                ping_result = self._remote_service.ping()
                aprint(f"  🏓 Ping test successful: {ping_result}")
            except Exception as ping_e:
                aprint(f"  ❌ Ping test failed: {ping_e}")
                return None

            # Then try to list available objects for debugging
            try:
                available_objects = self._remote_service.list_objects()
                aprint(f"  📋 Available objects on server: {available_objects}")
            except Exception as debug_e:
                aprint(f"  ⚠️ Couldn't list objects: {debug_e}")

            remote_obj = self._remote_service.get_object(name)

            if remote_obj is not None:
                # Return a wrapper that forces by-value transmission for classes
                return _ValueWrapper(remote_obj)

            aprint(f"  ⚠️ Object '{name}' not found on server.")
            return None
        except Exception as e:
            aprint(f"❌ Error retrieving object '{name}': {e}")
            return None

    def __getitem__(self, name):
        """
        Allow dictionary-style access to remote objects.

        Parameters
        ----------
        name : str
            The registered name of the remote object.

        Returns
        -------
        _ValueWrapper or None
            The proxy-wrapped remote object, or ``None`` if not found.
        """
        return self.get(name)

    def close(self):
        """Close the connection to the server.

        Safe to call multiple times; does nothing if already closed.
        """
        if self._connection and not self._connection.closed:
            try:
                self._connection.close()
                aprint("🔌 Connection closed.")
            except Exception:
                pass


class _ValueWrapper:
    """
    Wrapper that forces by-value transmission for classes using RPyC's obtain().
    This prevents classes from being sent as proxies, which would fail issubclass() checks.
    """

    def __init__(self, remote_obj):
        self._remote_obj = remote_obj

    def __call__(self, *args, **kwargs):
        """Handle method calls, using obtain() to force by-value transmission for classes."""

        # Process arguments to force by-value transmission for classes
        processed_args = list(args)  # Keep args as-is for now
        processed_kwargs = {}

        for key, value in kwargs.items():
            if inspect.isclass(value):
                try:
                    # Use RPyC's obtain() to force by-value transmission
                    import rpyc.utils.classic

                    obtained_value = rpyc.utils.classic.obtain(value)
                    processed_kwargs[key] = obtained_value
                except Exception:
                    # Fallback to original value
                    processed_kwargs[key] = value
            else:
                processed_kwargs[key] = value

        return self._remote_obj(*processed_args, **processed_kwargs)

    def __getattr__(self, name):
        """Forward all other attribute access to the remote object."""
        return getattr(self._remote_obj, name)

    def __getitem__(self, key):
        """Forward item access to the remote object (for dict-like objects)."""
        return self._remote_obj[key]
