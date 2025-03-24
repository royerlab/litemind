from functools import lru_cache


@lru_cache
def is_chromadb_available() -> bool:

    try:
        import importlib.util

        return importlib.util.find_spec("chromadb") is not None
    
    except ImportError:
        return False
    