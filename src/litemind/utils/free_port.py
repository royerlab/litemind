from typing import Tuple


def find_free_port() -> Tuple[str, int]:
    """Finds and returns a free port and its associated host.

    This utility function temporarily binds a socket to an ephemeral port
    chosen by the operating system (by specifying port 0). It then
    retrieves the assigned port number, closes the socket to release it,
    and returns the port number. This is useful for running tests or services
    on a dynamically assigned port to avoid conflicts.

    Returns
    -------
    Tuple[str, int]
        A tuple containing the host address (str) and a free port number (int).

    """
    # Import Python's built-in library for low-level network operations.
    import socket

    # 1. Create a new socket object.
    # AF_INET specifies the address family for IPv4.
    # SOCK_STREAM specifies the socket type for TCP (as opposed to UDP).
    sock = socket.socket()

    # 2. Set the SO_REUSEADDR option.
    # This is a crucial step for test servers. It allows the kernel to reuse
    # a local socket in TIME_WAIT state, which can happen if the test server
    # is stopped and restarted quickly. This prevents "Address already in
    # use" errors during rapid test cycles.
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    # 3. Bind the socket to an address. This is the key step.
    # The first argument, '', is a shorthand for '0.0.0.0', meaning the socket
    # will be accessible from any of the machine's network interfaces.
    # The second argument, 0, is the special port number that tells the
    # operating system to find and assign any currently available ephemeral port.
    sock.bind(("", 0))

    # 4. Retrieve the assigned address and port.
    # Now that the OS has assigned a port, getsockname() returns a tuple
    # containing the host IP and the dynamically assigned port number.
    host, port = sock.getsockname()

    # 5. Close the socket.
    # This immediately releases the port, making it available for our actual
    # server to use right away. The OS is unlikely to assign this same port
    # to another process in the tiny fraction of a second before our test
    # server grabs it.
    sock.close()

    # The 'port' variable now holds the number of a port that was free
    # just moments ago, making it safe to use for the test server.
    return host, port
