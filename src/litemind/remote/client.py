import inspect

from arbol import aprint


class Client:
    """A simple, elegant client for interacting with remote litemind objects."""

    def __init__(self, host="localhost", port=18861):
        """
        Initializes the client and connects to the server.

        Args:
            host (str): The hostname of the server.
            port (int): The port of the server.
        """
        try:
            import rpyc

            self._connection = rpyc.connect(
                host,
                port,
                config={
                    "allow_pickle": True,
                    "allow_all_attrs": True,
                    "sync_request_timeout": 30,
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
        except:
            return False

    def get(self, name):
        """
        Retrieves a proxy object for a remote agent.

        Args:
            name (str): The name of the remote object.

        Returns:
            A proxy object that behaves like the remote object, or None if not found.
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
        Allows dictionary-style access to remote objects.
        e.g., `client['main_agent']`
        """
        return self.get(name)

    def close(self):
        """Closes the connection to the server."""
        if self._connection and not self._connection.closed:
            try:
                self._connection.close()
                aprint("🔌 Connection closed.")
            except:
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

                    # Add detailed debugging
                    aprint(
                        f"🔍 Original class {key}: {value} (type: {type(value)}, id: {id(value)})"
                    )
                    aprint(
                        f"🔍 Obtained class {key}: {obtained_value} (type: {type(obtained_value)}, id: {id(obtained_value)})"
                    )
                    aprint(
                        f"🔍 Is obtained value a class? {inspect.isclass(obtained_value)}"
                    )
                    aprint(f"🔍 Are they the same object? {value is obtained_value}")

                    processed_kwargs[key] = obtained_value
                    aprint(
                        f"🔄 Used obtain() to get local copy of class {value.__name__}"
                    )
                except Exception as e:
                    aprint(f"⚠️ Failed to obtain class {value.__name__}: {e}")
                    # Fallback to original value
                    processed_kwargs[key] = value
            else:
                processed_kwargs[key] = value

        aprint(
            f"🔄 Processed {len(processed_args)} args and {len(processed_kwargs)} kwargs"
        )
        return self._remote_obj(*processed_args, **processed_kwargs)

    def __getattr__(self, name):
        """Forward all other attribute access to the remote object."""
        return getattr(self._remote_obj, name)
