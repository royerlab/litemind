from functools import lru_cache


@lru_cache()
def is_rpyc_available() -> bool:
    """
    Check if the rpyc library is installed and working.

    Returns
    -------
    bool
        True if rpyc can be imported, False otherwise.
    """
    try:
        import rpyc  # noqa: F401

        return True
    except Exception:
        return False
