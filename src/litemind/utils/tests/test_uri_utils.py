from litemind.utils.uri_utils import is_uri


def test_standard_http_urls():
    """Test standard HTTP URLs."""
    assert is_uri("http://example.com") is True
    assert is_uri("http://example.com/path") is True
    assert is_uri("http://example.com/path?query=value") is True
    assert is_uri("http://user:pass@example.com:8080/path") is True


def test_https_urls():
    """Test HTTPS URLs."""
    assert is_uri("https://example.com") is True
    assert is_uri("https://subdomain.example.com/path") is True


def test_other_common_protocols():
    """Test other common protocol URIs."""
    assert is_uri("ftp://ftp.example.com") is True
    assert is_uri("sftp://user@server.com") is True
    assert is_uri("ssh://user@server.com") is True
    assert is_uri("mongodb://localhost:27017") is True


def test_file_uris():
    """Test file protocol URIs."""
    assert is_uri("file:///path/to/file.txt") is True
    assert is_uri("file://localhost/path/to/file.txt") is True


def test_custom_protocol_uris():
    """Test custom protocol URIs."""
    assert is_uri("custom://example.com") is True
    assert is_uri("git://github.com/user/repo.git") is True
    assert is_uri("s3://bucket-name/key") is True


def test_incomplete_uris():
    """Test URIs that are missing required components."""
    assert is_uri("example.com") is False  # Missing scheme
    assert is_uri("http://") is False  # Missing netloc and path
    assert is_uri("://example.com") is False  # Missing scheme
    assert is_uri("scheme:") is False  # Missing netloc or path


def test_invalid_inputs():
    """Test invalid inputs."""
    assert is_uri("") is False
    assert is_uri(" ") is False
    assert is_uri("http:\\\\example.com") is False  # Incorrect slashes


def test_edge_cases():
    """Test edge cases."""
    assert is_uri("mailto:user@example.com") is True  # mailto has path, not netloc
    assert is_uri("data:text/plain;base64,SGVsbG8=") is True  # data URI
    assert is_uri("tel:+1234567890") is True  # telephone URI


def test_specific_cases():
    uri = "file:///Users/loic.royer/workspace/python/litemind/src/litemind/ressources/documents/intracktive_preprint.pdf"

    assert is_uri(uri) is True  # telephone UR
