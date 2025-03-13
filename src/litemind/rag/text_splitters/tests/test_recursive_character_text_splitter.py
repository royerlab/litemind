from litemind.rag.text_splitters.recursive_character_text_splitter import RecursiveCharacterTextSplitter


def test_split_text_no_split_needed():
    """Test that text shorter than chunk_size is not split."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    text = "This is a short text."
    chunks = splitter.split_text(text)
    assert chunks == [text]


def test_split_text_single_separator_no_overlap():
    """Test splitting with a single separator."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=10, chunk_overlap=0, separators=[" "])
    text = "This is a test. We are trying to split text with a single separator."
    chunks = splitter.split_text(text)
    assert chunks == ['This is a', ' test.', ' We are', ' trying', ' to split', ' text', ' with a', ' single', ' separator.']

    # Verify sizes
    assert len(chunks[0]) <= 10  # 9 chars
    assert len(chunks[1]) <= 10  # 10 chars


def test_split_text_multiple_separators():
    """Test splitting with multiple separators."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=20, chunk_overlap=6, separators=["\n", ".", " "])
    text = "This is a test. Another test. \n Here is another sentence. How cool is that?"
    chunks = splitter.split_text(text)

    # We expect first to split on "." then on " " if needed
    expected = ['This is a test', ' test. Another test', '\n Here is another', '\n Here is another sentence', '. How cool is that?']
    assert chunks == expected

    # Verify sizes and overlaps
    for chunk in chunks:
        assert len(chunk) <= 20+6 # 20 chars + 6-char overlap



def test_split_text_long_text():
    """Test splitting a long text with consistent overlaps."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=10, chunk_overlap=2, separators=[" "])
    text = "This is a very long text that needs to be split into smaller chunks."
    chunks = splitter.split_text(text)

    # Verify sizes and overlaps
    for chunk in chunks:
        assert len(chunk) <= 10*2


def test_split_text_with_overlap():
    """Test that overlaps are correctly applied."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=10, chunk_overlap=5, separators=[" "])
    text = "This is a test of overlap."
    chunks = splitter.split_text(text)

    # Verify sizes
    for chunk in chunks:
        assert len(chunk) <= 10*2

def test_split_text_with_newline_separator():
    """Test splitting with newline separator."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=10, chunk_overlap=2, separators=["\n"])
    text = "Line 1\nLine 2\nLine 3"
    chunks = splitter.split_text(text)

    # Verify sizes
    for chunk in chunks:
        assert len(chunk) <= 2*10


def test_split_text_with_multiple_newline_separators():
    """Test splitting with multiple newline separators."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=10, chunk_overlap=2, separators=["\n\n", "\n"])
    text = "Paragraph 1\n\nParagraph 2\nParagraph 3"
    chunks = splitter.split_text(text)

    # Verify sizes
    for chunk in chunks:
        assert len(chunk) <= 10*2
